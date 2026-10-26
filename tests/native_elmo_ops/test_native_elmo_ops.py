"""Synthetic native effects; real published service pins and runtime bytes are tested."""
import copy
import os
import tempfile
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OPS=Path(os.environ.get('ELMO_TEST_OPS_ROOT',REPO/'scripts/train/joint/native_elmo_ops')).resolve()
SOURCE=Path(os.environ.get('ELMO_TEST_SERVICE_ROOT',REPO)).resolve()
NAMES=('execution_spec','phase_observer','wrapper_common','host_guard')


def load(name,path):
    loader=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);return module

# Export only the maintained service closure. No campaign receipts or models.
service_temp=tempfile.TemporaryDirectory(prefix='native-elmo-ops-service-')
SERVICE=Path(service_temp.name).resolve()
bootstrap=load('_portable_ops_service_export',SOURCE/'scripts/train/joint/service_bootstrap.py')
for name in bootstrap.service_files(SOURCE):
    target=SERVICE/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,target)
(SERVICE/'FILES.sha256').write_text(''.join(hashlib.sha256((SERVICE/name).read_bytes()).hexdigest()+'  '+name+'\n' for name in bootstrap.service_files(SERVICE)))
support=load('_portable_execution_fixture',HERE/'execution_fixture.py')
observer_support=load('_portable_observer_fixture',HERE/'observer_fixture.py')

@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    for name in NAMES:monkeypatch.delitem(sys.modules,name,raising=False)
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('Unintercepted subprocess forbidden'))
    yield
    for name in NAMES:sys.modules.pop(name,None)


def fixture(tmp_path,monkeypatch,mode='inspect',arm='plain',accept=True):
    src=load('_composite_fixture_source',OPS/'execution_spec.py')
    assert src.SERVICE_SHA==src.sha(SERVICE/'FILES.sha256')
    support.e=src;support.HERE=OPS;support.SERVICE=SERVICE
    f=support.fixture(tmp_path,mode,arm);root=f['root']
    for source in OPS.iterdir():
        if source.is_file():shutil.copy2(source,root/source.name)
        elif not (root/source.name).exists():(root/source.name).mkdir()
    f['spec']['release_observer_path']=str(root/'observer-specs/phase.json')
    f['spec']['ops_manifest']['sha256']=src.sha(root/'FILES.sha256')
    support.put(f['path'],f['spec'])
    f['args']=(str(f['path']),src.sha(f['path']),mode,src.sha(root/'FILES.sha256'))
    e=load('execution_spec',root/'execution_spec.py');monkeypatch.setitem(sys.modules,'execution_spec',e)
    monkeypatch.setattr(e,'PINS',src.PINS)
    assert e.SERVICE_SHA==e.sha(SERVICE/'FILES.sha256')
    support.e=e
    if accept:support.accepted(f,monkeypatch)
    o=e.operations_module(f['spec'],'phase_observer')
    wc=e.operations_module(f['spec'],'wrapper_common')
    assert (wc.SERVICE_COMMIT,wc.SERVICE_SHA)==(e.SERVICE_COMMIT,e.SERVICE_SHA)
    h=e.operations_module(f['spec'],'host_guard')
    execution=dict(path=str(f['path']),sha256=f['args'][1],mode=mode,ops_manifest_sha256=f['args'][3])
    observer_spec=o.build_spec(execution);sp=Path(f['spec']['release_observer_path']);support.put(sp,observer_spec)
    f.update(e=e,o=o,h=h,wc=wc,observer_spec=observer_spec,observer_path=sp,
        observer_args=(str(sp),e.sha(sp),f['args'][3]),observer_id='900002')
    observer_support.o=o;observer_support.support=support
    return f


def accept_observer(f,monkeypatch):return observer_support.accept(f,monkeypatch)

def capture(f,monkeypatch):
    observer_support.ready_capture_after_accept(f,monkeypatch)
    return f['o'].capture(*f['observer_args'])


def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


@pytest.mark.parametrize('mode',['capture','inspect','diagnostic','finalize','full'])
@pytest.mark.parametrize('arm',['plain','fused','boundary'])
def test_manifest_cycle_free_held_observer_release(tmp_path,monkeypatch,mode,arm):
    f=fixture(tmp_path,monkeypatch,mode,arm);e=f['e'];o=f['o']
    accept_observer(f,monkeypatch)
    # Real composed release guard invokes real observer dependency checks.
    calls=[]
    monkeypatch.setattr(e.subprocess,'run',lambda argv,**kw:(calls.append(argv) or subprocess.CompletedProcess(argv,0,'','')))
    e.release_bound(*f['args'])
    assert calls==[['/usr/local/bin/scontrol','release','900001']]
    release=e.read_json(e.paths(f['spec'],f['args'][1])['spec_claim']/'RELEASE_ATTEMPT.json')
    release_held=e.paths(f['spec'],f['args'][1])['spec_claim']/'RELEASE_HELD_NATIVE.txt'
    assert release_held.read_text()==support.native(f)
    assert e.sha(release_held)==release['held_native_sha256']
    assert release['observer_readiness']['producer_job_id']=='900001'
    assert release['observer_readiness']['observer_job_id']=='900002'
    assert release['observer_readiness']['observer_pending_query']['stdout']
    assert e._verified_ops_member_sha256==e.sha(f['root']/'execution_spec.py')
    assert o._verified_ops_member_sha256==e.sha(f['root']/'phase_observer.py')
    assert 'EXECUTION_SHA' not in (f['root']/'phase_observer.py').read_text()


@pytest.mark.parametrize('conflict',['missing','unknown','not-dependency','wrong-job'])
def test_release_requires_actual_matching_observer(tmp_path,monkeypatch,conflict):
    f=fixture(tmp_path,monkeypatch);e=f['e'];o=f['o']
    if conflict!='missing':accept_observer(f,monkeypatch)
    if conflict=='unknown':
        a=o.paths(dict(f['observer_spec'],producer_job_id='900001'),f['observer_args'][1])['claim']/'acceptance.json'
        a.unlink() # Disposable synthetic fixture only.
    if conflict=='not-dependency':
        monkeypatch.setattr(o,'query',lambda argv:observer_support.record(argv,stdout=support.native(f) if argv[3]=='900001' else observer_support.observer_native(f,Reason='Resources')))
    if conflict=='wrong-job':
        monkeypatch.setattr(o,'query',lambda argv:observer_support.record(argv,stdout=support.native(f) if argv[3]=='900001' else observer_support.observer_native(f,JobId='900099')))
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('release reached without observer proof'))
    with pytest.raises((ValueError,FileNotFoundError)):e.release_bound(*f['args'])
    assert not (e.paths(f['spec'],f['args'][1])['spec_claim']/'RELEASE_ATTEMPT.json').exists()


@pytest.mark.parametrize('member',['phase_observer.py','execution_spec.py','observe-phase.sbatch','wrapper_common.py'])
def test_manifest_changed_member_rejected_before_import(tmp_path,monkeypatch,member):
    f=fixture(tmp_path,monkeypatch);p=f['root']/member;p.write_text(p.read_text()+'\n# drift\n')
    with pytest.raises(ValueError):f['o'].engine(f['root'],f['args'][3])
    with pytest.raises(ValueError):f['e'].operations_module(f['spec'],'phase_observer')


def test_foreign_cached_execution_not_adopted(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    monkeypatch.setitem(sys.modules,'execution_spec',SimpleNamespace(__file__='/wrong/path.py'))
    with pytest.raises(ValueError,match='Foreign'):f['o'].engine(f['root'],f['args'][3])


def test_phase_verify_is_maintained_delegate_with_exact_identity(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);accept_observer(f,monkeypatch);result=capture(f,monkeypatch)
    original=f['e'].service_module;calls=[]
    def verify(binding,**kwargs):calls.append((binding,kwargs));return {'result':result}
    monkeypatch.setattr(f['e'],'service_module',lambda spec,name:SimpleNamespace(verify_observer=verify) if name=='phase_observer_proof' else original(spec,name))
    assert f['o'].verify_captured(*f['observer_args'])==result
    assert calls[0][0]==bind(f['root']/'status/native-900001.observer.json')
    assert calls[0][1]['producer_native']['JobId']=='900001'
    assert calls[0][1]['producer_request']==result['producer_request']
    assert calls[0][1]['terminal_native']==result['producer_native']
    result['producer_job_id']='999'
    with pytest.raises(ValueError,match='another producer'):f['o'].verify_captured(*f['observer_args'])


def terminal_route(f,monkeypatch):
    accept_observer(f,monkeypatch);result=capture(f,monkeypatch)
    original=f['e'].service_module
    monkeypatch.setattr(f['e'],'service_module',lambda spec,name:SimpleNamespace(verify_observer=lambda *a,**k:{'result':copy.deepcopy(result)},check_owned_steps=original(spec,name).check_owned_steps) if name=='phase_observer_proof' else original(spec,name))
    adapter=original(f['spec'],'original_observer')
    record={'schema':'native-elmo-native-query-v1','argv':adapter.step_argv('paphiwetsa3'),'returncode':0,'stdout':'','stderr':'','observed_at_utc':datetime.now(timezone.utc).isoformat()}
    # Use adapter's own query builder with intercepted subprocess for its exact schema.
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'',''))
    return result,adapter


def test_execution_census_requires_observer_final_commit_and_fresh_steps(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);result,adapter=terminal_route(f,monkeypatch)
    assert f['e'].execution_claim_census(f['spec'])[0]['observer_spec']==result['observer_spec']
    monkeypatch.setattr(f['o'],'verify_captured',lambda *a:(_ for _ in ()).throw(ValueError('missing final commit')))
    with pytest.raises(ValueError,match='final commit'):f['e'].execution_claim_census(f['spec'])
    monkeypatch.setattr(f['o'],'verify_captured',lambda *a:result)
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'900001.batch|RUNNING\n',''))
    with pytest.raises(ValueError):f['e'].execution_claim_census(f['spec'])


def test_host_claim_census_uses_same_observer_consumer(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch,'diagnostic');result,adapter=terminal_route(f,monkeypatch)
    scope={'user':'paphiwetsa3','scientific_job_names':[f['spec']['native_contract']['JobName']],
        'scientific_name_prefixes':['elmo-large314m-continuous-'],
        'claim_roots':[{'path':str(f['root']/'scientific-claims'),'terminal_directory':str(f['root']/'status')}]}
    proof=f['h'].quiescence(f['spec'],scope,'run-id')
    assert proof['prior_claims'][0]['route']=='maintained-phase-observer'
    monkeypatch.setattr(f['o'],'verify_captured',lambda *a:(_ for _ in ()).throw(ValueError('corrupt observer ledger')))
    with pytest.raises(ValueError,match='observer ledger'):f['h'].quiescence(f['spec'],scope,'run-id')


def test_unknown_execution_claim_blocks_before_other_raw_terminals(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch,accept=False)
    c=f['root']/'execution-claims'/f['args'][1];c.mkdir()
    support.put(c/'PREPARED.json',dict(spec_sha256=f['args'][1],arm='plain',parent_request=f['spec']['parent_request'],spec_path=str(f['path'])))
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'',''))
    with pytest.raises(ValueError,match='Unresolved'):f['e'].execution_claim_census(f['spec'])


def test_source_publication_pins_match_exact_published_service():
    e=load('_publication_source',OPS/'execution_spec.py');wc=load('_publication_wrapper',OPS/'wrapper_common.py')
    expected=('abbb00499096faabf7035fde23bf4b1eba984377',
              'adf45524de3c2e6dba186572e7151d48bfca0b3adc8e6cf9defd686b47744c87')
    assert (e.SERVICE_COMMIT,e.SERVICE_SHA)==expected==(wc.SERVICE_COMMIT,wc.SERVICE_SHA)
    assert e.sha(SERVICE/'FILES.sha256')==expected[1]
    binding=dict(root=str(SERVICE),manifest=str(SERVICE/'FILES.sha256'),manifest_sha256=expected[1])
    module=e.service_module(dict(operational_service=binding),'phase_observer_proof')
    assert Path(module.__file__).resolve()==SERVICE/'scripts/train/joint/phase_observer_proof.py'


@pytest.mark.parametrize('mode',['inspect','diagnostic'])
@pytest.mark.parametrize('running',[False,True])
def test_actual_current_claim_interface_held_and_running(tmp_path,monkeypatch,mode,running):
    import os
    f=fixture(tmp_path,monkeypatch,mode);e=f['e'];h=f['h']
    if running:
        accept_observer(f,monkeypatch)
        monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'',''))
        e.release_bound(*f['args'])
        monkeypatch.setattr(os,'getuid',lambda:41003)
        monkeypatch.setenv('SLURM_JOB_ID','900001');monkeypatch.setenv('SLURMD_NODENAME','gpu11')
        monkeypatch.delenv('SLURM_JOB_GPUS',raising=False);monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    state='RUNNING' if running else 'PENDING'
    raw=support.native(f,state)
    monkeypatch.setattr(e,'command',lambda argv:raw)
    def query(argv,**kw):
        out=raw if argv[0].endswith('scontrol') else '' if '--steps' in argv else '900001|'+f['spec']['native_contract']['JobName']+'|'+state+'|\n'
        return subprocess.CompletedProcess(argv,0,out,'')
    monkeypatch.setattr(subprocess,'run',query)
    _,qpath,q,job,claim,held=e.read_binding(*f['args'])
    scope={'user':'paphiwetsa3','scientific_job_names':[f['spec']['native_contract']['JobName']],
        'scientific_name_prefixes':['elmo-large314m-continuous-'],
        'claim_roots':[{'path':str(claim.parent),'terminal_directory':str(f['root']/'status')}]}
    # No stubs for own_current, read_binding, load_execution or native identity.
    proof=h.quiescence(f['spec'],scope,'run-id',job,e.sha(qpath),running=running)
    assert proof['own_current_claim']==str(claim)
    assert proof['prior_claims']==[]


@pytest.mark.parametrize('area',['execution-json','observer-json','preserved-query'])
def test_metadata_and_query_bounds(tmp_path,monkeypatch,area):
    f=fixture(tmp_path,monkeypatch)
    if area=='preserved-query':
        row=observer_support.record(['/native'],stdout='x'*(1048576+1))
        with pytest.raises(ValueError,match='bounds'):f['o'].successful(row,['/native'])
    else:
        p=tmp_path/'oversize.json';p.write_bytes(b' '*(4*1024*1024+1))
        module=f['e'] if area=='execution-json' else f['o']
        with pytest.raises(ValueError,match='4 MiB'):module.read_json(p)


def actual_service_capture(f,monkeypatch):
    accept_observer(f,monkeypatch)
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'',''))
    f['e'].release_bound(*f['args'])
    result=capture(f,monkeypatch)
    proof=f['e'].service_module(f['spec'],'phase_observer_proof')
    # Synthetic original-parent profiles are adjusted only in test memory.
    # No deployed request or production profile is altered.
    profiles=copy.deepcopy(proof.ORIGINAL_OBSERVERS)
    run_id=f['e'].read_json(Path(f['spec']['parent_request']['path']))['run_id']
    profiles[run_id]['request_path']=f['spec']['parent_request']['path']
    profiles[run_id]['request_sha256']=f['spec']['parent_request']['sha256']
    monkeypatch.setattr(proof,'ORIGINAL_OBSERVERS',profiles)
    return result,proof


@pytest.mark.parametrize('mode',['capture','inspect','diagnostic','finalize','full'])
@pytest.mark.parametrize('arm',['plain','fused','boundary'])
def test_actual_composite_capture_through_actual_maintained_service(tmp_path,monkeypatch,mode,arm):
    f=fixture(tmp_path,monkeypatch,mode,arm);result,proof=actual_service_capture(f,monkeypatch)
    # No stub for observer verify_captured, maintained verify_observer, binding,
    # claim readers, native semantics, final commit or release readiness.
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('Verifier made native mutation/query'))
    assert f['o'].verify_captured(*f['observer_args'])==result
    ctx=proof.verify_observer(bind(f['root']/'status/native-900001.observer.json'),
        producer_request=result['producer_request'],producer_native=dict(f['spec']['native_contract'],JobId='900001'),
        terminal_native=result['producer_native'])
    assert ctx['running_admission_status']=='VALIDATED'
    assert ctx['job']=='900001'


@pytest.mark.parametrize('broken',['release-held','final-commit','release-readiness','owned-step'])
def test_actual_service_rejects_incomplete_or_conflicting_composite_proof(tmp_path,monkeypatch,broken):
    f=fixture(tmp_path,monkeypatch);result,proof=actual_service_capture(f,monkeypatch)
    sc=f['e'].paths(f['spec'],f['args'][1])['spec_claim']
    if broken=='release-held':(sc/'RELEASE_HELD_NATIVE.txt').write_text(support.native(f,SubmitTime='2026-01-01T00:00:00'))
    elif broken=='final-commit':(f['root']/'status/native-900001.observer.json').unlink()
    elif broken=='release-readiness':
        release=f['e'].read_json(sc/'RELEASE_ATTEMPT.json');release['observer_readiness']['observer_pending_query']['stdout']=observer_support.observer_native(f,Reason='Resources');support.put(sc/'RELEASE_ATTEMPT.json',release)
    else:
        path=Path(result['owned_steps']['path']);r=f['e'].read_json(path);r['stdout']='900001.batch|RUNNING\n';support.put(path,r)
        result['owned_steps']=bind(path);support.put(f['root']/'status/native-900001.observer.json',result)
    with pytest.raises((ValueError,FileNotFoundError)):f['o'].verify_captured(*f['observer_args'])

@pytest.mark.parametrize('consumer',['execution','scientific'])
def test_both_censuses_consume_actual_service_proof(tmp_path,monkeypatch,consumer):
    f=fixture(tmp_path,monkeypatch,'diagnostic');result,proof=actual_service_capture(f,monkeypatch)
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'',''))
    scope={'user':'paphiwetsa3','scientific_job_names':[f['spec']['native_contract']['JobName']],
        'scientific_name_prefixes':['elmo-large314m-continuous-'],
        'claim_roots':[{'path':str(f['root']/'scientific-claims'),'terminal_directory':str(f['root']/'status')}]}
    def census():
        return f['e'].execution_claim_census(f['spec']) if consumer=='execution' else f['h'].quiescence(f['spec'],scope,'run-id')
    assert census()
    final=f['root']/'status/native-900001.observer.json';final.unlink()
    assert (f['root']/'status/native-900001.txt').is_file()
    with pytest.raises((ValueError,FileNotFoundError)):census()


def test_original_parent_never_falls_back_to_raw_phase_terminal(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch,'full');e=f['e'];h=f['h'];parent_job=f['spec']['original_parent_job_id']
    root=tmp_path/'original-scientific-claims';root.mkdir();claim=root/f['spec']['parent_request']['sha256'];claim.mkdir()
    support.put(claim/'PREPARED.json',{'request_sha256':claim.name});(claim/'PARENT_JOB_ID').write_text(parent_job+'\n')
    running=support.native(f,'RUNNING',job=parent_job)
    support.put(claim/'RUNNING_ADMISSION.json',{'job_id':parent_job,'request_sha256':claim.name,'native':running})
    terminal=running.replace('JobState=RUNNING','JobState=TIMEOUT')+' EndTime=2026-10-27T02:00:00 ExitCode=0:15\n'
    (f['root']/'status'/('native-'+parent_job+'.txt')).write_text(terminal)
    scope={'user':'paphiwetsa3','scientific_job_names':['original'], 'scientific_name_prefixes':['elmo-large314m-continuous-'],
        'claim_roots':[{'path':str(root),'terminal_directory':str(f['root']/'status')}]}
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'',''))
    with pytest.raises(ValueError,match='separately validated'):h.quiescence(f['spec'],scope,'run-id')
    out=h.quiescence(f['spec'],scope,'run-id',original_terminal=terminal)
    assert out['prior_claims'][0]['route']=='validated-original-parent'
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,parent_job+'.batch|RUNNING\n' if '--steps' in argv else '',''))
    with pytest.raises(ValueError):h.quiescence(f['spec'],scope,'run-id',original_terminal=terminal)


def controller_history_fixture(monkeypatch):
    c=load('_composite_controller',OPS/'controller.py');monkeypatch.setitem(sys.modules,'controller',c)
    helper=load('_controller_virtual_fixture',HERE/'controller_fixture.py')
    helper.c=c;helper.ROOT=c.PREFIX+'synthetic-prior-controller'
    oldfiles,old,args=helper.fixture()
    def save(files,loc,value):
        files[loc]=c.encode(value);return {'path':loc,'sha256':c.digest(files[loc])}
    def producer(files,spec,changes):
        data=c.parse(files[spec['producer_execution']['path']]);data.update(changes)
        spec['producer_execution'].update(save(files,spec['producer_execution']['path'],data))
        return data
    oldroot={'path':old['root']+'/execution-claims','terminal_directory':old['terminal_directory']}
    oldproducer=producer(oldfiles,old,dict(mode='diagnostic',execution_claim_roots=[oldroot]))
    old['producer_execution']['mode']='diagnostic'
    oldsb=save(oldfiles,args[0],old);oldargs=(oldsb['path'],oldsb['sha256'],args[2])
    previous=c.ledger(old);argv=c.controller_argv(old,*oldargs)
    held=helper.native(old,Comment='elmo-controller:'+oldargs[1]);oldfiles[previous+'/HELD_NATIVE.txt']=held.encode()
    prepared=dict(schema='native-elmo-controller-prepared-v1',spec_path=oldargs[0],spec_sha256=oldargs[1],ops_manifest_sha256=oldargs[2],
        producer_spec_sha256=old['producer_execution']['sha256'],argv=argv)
    save(oldfiles,previous+'/PREPARED.json',prepared)
    save(oldfiles,previous+'/acceptance.json',dict(schema='native-elmo-controller-acceptance-v1',job_id='900001',spec_sha256=oldargs[1],
        producer_spec_sha256=old['producer_execution']['sha256'],argv=argv,held_native_sha256=c.digest(held.encode())))
    for name,value in [('PARENT_JOB_ID',b'900001\n'),('sbatch.stdout',b'900001\n'),('sbatch.exit',b'0\n')]:oldfiles[previous+'/'+name]=value
    save(oldfiles,previous+'/RESULT.json',dict(schema='native-elmo-controller-result-v1',controller_job_id='900001',
        producer_spec_sha256=old['producer_execution']['sha256'],status='CHILD_RELEASE_ACCEPTED_NOT_COMPLETION'))
    helper.ROOT=c.PREFIX+'synthetic-current-controller';newfiles,current,newargs=helper.fixture()
    current['parent_request']=old['parent_request']
    newproducer=c.parse(newfiles[current['producer_execution']['path']]);newproducer['parent_request']=old['parent_request']
    newproducer['execution_claim_roots']=[oldroot,{'path':current['root']+'/execution-claims','terminal_directory':current['terminal_directory']}]
    newproducer['runtime']['admission']['quiescence']['claim_roots']=[{'path':old['root']+'/scientific-claims','terminal_directory':old['terminal_directory']}]
    current['producer_execution'].update(save(newfiles,current['producer_execution']['path'],newproducer))
    allfiles={**oldfiles,**newfiles}
    remote=SimpleNamespace(census=lambda _:old['producer_execution']['sha256']+'|d\n')
    return c,allfiles,current,old,previous,newproducer,remote


@pytest.mark.parametrize('omit',['execution-root','terminal-directory','scientific-root'])
def test_gateway_new_root_cannot_omit_purged_prior_child(monkeypatch,omit):
    c,files,current,old,ledger,newproducer,remote=controller_history_fixture(monkeypatch)
    assert c.prior_census(c.Reader(files.__getitem__),remote,current)
    if omit=='execution-root':newproducer['execution_claim_roots']=newproducer['execution_claim_roots'][1:]
    elif omit=='terminal-directory':newproducer['execution_claim_roots'][0]['terminal_directory']=current['terminal_directory']
    else:newproducer['runtime']['admission']['quiescence']['claim_roots']=[]
    files[current['producer_execution']['path']]=c.encode(newproducer)
    current['producer_execution']['sha256']=c.digest(files[current['producer_execution']['path']])
    # Empty native queue / disappeared child is immaterial: history cannot be omitted.
    with pytest.raises(ValueError,match='omits'):c.prior_census(c.Reader(files.__getitem__),remote,current)


@pytest.mark.parametrize('broken',['prepared','spec','parent','acceptance','held','unknown-result'])
def test_gateway_prior_intent_and_acceptance_are_reopened(monkeypatch,broken):
    c,files,current,old,ledger,newproducer,remote=controller_history_fixture(monkeypatch)
    if broken=='prepared':files[ledger+'/PREPARED.json']=b'{}\n'
    elif broken=='spec':
        p=c.parse(files[ledger+'/PREPARED.json'])['spec_path'];files[p]+=b' '
    elif broken=='parent':
        p=c.parse(files[ledger+'/PREPARED.json']);s=c.parse(files[p['spec_path']]);s['parent_request']['sha256']='f'*64
        files[p['spec_path']]=c.encode(s);p['spec_sha256']=c.digest(files[p['spec_path']]);files[ledger+'/PREPARED.json']=c.encode(p)
    elif broken=='acceptance':files[ledger+'/PARENT_JOB_ID']=b'900099\n'
    elif broken=='held':files[ledger+'/HELD_NATIVE.txt']+=b' conflict\n'
    else:files.pop(ledger+'/RESULT.json')
    with pytest.raises((ValueError,KeyError)):c.prior_census(c.Reader(files.__getitem__),remote,current)

@pytest.mark.parametrize('consumer',['execution','scientific'])
def test_fresh_array_alias_cannot_escape_complete_step_census(tmp_path,monkeypatch,consumer):
    f=fixture(tmp_path,monkeypatch,'diagnostic');result,proof=actual_service_capture(f,monkeypatch)
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:subprocess.CompletedProcess(argv,0,'900001_7.batch|RUNNING\n' if '--steps' in argv else '',''))
    scope={'user':'paphiwetsa3','scientific_job_names':[f['spec']['native_contract']['JobName']],
        'scientific_name_prefixes':['elmo-large314m-continuous-'],
        'claim_roots':[{'path':str(f['root']/'scientific-claims'),'terminal_directory':str(f['root']/'status')}]}
    with pytest.raises(ValueError,match='native workers'):
        if consumer=='execution':f['e'].execution_claim_census(f['spec'])
        else:f['h'].quiescence(f['spec'],scope,'run-id')


@pytest.mark.parametrize('member',['native_scontrol.sh','native_squeue.sh'])
def test_native_file_mount_wrappers_preserve_executable_mode(tmp_path,monkeypatch,member):
    f=fixture(tmp_path,monkeypatch)
    assert (OPS/member).stat().st_mode & 0o111
    client=load('_composite_native_client_mode_check',f['root']/'native_client.py')
    client.executable_wrappers(f['root'])
    (f['root']/member).chmod(0o644)
    with pytest.raises(ValueError,match='not executable'):client.executable_wrappers(f['root'])
