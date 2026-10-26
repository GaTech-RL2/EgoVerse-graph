"""Portable synthetic CPU tests: no campaign-relative data, native or ML calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('ELMO_TEST_SERVICE_ROOT',HERE.parent/'service')).resolve()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,value):
    path=Path(path);path.write_text(value if isinstance(value,str) else json.dumps(value,sort_keys=True,indent=2)+'\n')
    return dict(path=str(path),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())


@pytest.fixture(scope='session')
def service(tmp_path_factory):
    source=SOURCE
    if not (source/'scripts/train/joint/phase_observer_proof.py').is_file():
        spec=importlib.util.find_spec('scripts.train.joint.phase_observer_proof')
        assert spec and spec.origin;source=Path(spec.origin).resolve().parents[3]
    loader=importlib.util.spec_from_file_location('portable_service_export',source/'scripts/train/joint/service_bootstrap.py')
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    root=tmp_path_factory.mktemp('portable-service')
    for rel in module.service_files(source):
        dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/rel,dst)
    names=module.service_files(root)
    (root/'FILES.sha256').write_text(''.join(sha(root/n)+'  '+n+'\n' for n in sorted(names)))
    return root


def environment(service):return dict(os.environ,PYTHONPATH=str(service),PYTHONDONTWRITEBYTECODE='1')


def fixture(tmp_path,service,mode='inspect',arm='plain',running=True,state='COMPLETED'):
    run=subprocess.run([sys.executable,'-B',str(HERE/'proof_fixture.py'),str(tmp_path/'ops'),mode,arm,'yes' if running else 'no',state],
        env=environment(service),cwd=service,capture_output=True,text=True,check=False)
    assert run.returncode==0,run.stderr
    return json.loads(run.stdout)


def execute(f,service,operation='verify',**extra):
    payload=dict(f,operation=operation,**extra)
    program='''import builtins,json,sys,subprocess
from scripts.train.joint import phase_observer_proof as p
f=json.loads(sys.argv[1]);p.ORIGINAL_OBSERVERS=f['profiles']
from scripts.train.joint import original_observer
original_observer.ORIGINAL_OBSERVERS=f['profiles']
original=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in {'torch','lightning','egomimic','numpy'}: raise AssertionError('ML import forbidden')
 return original(name,*a,**k)
builtins.__import__=guarded
subprocess.run=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Native side effect forbidden'))
b=f['bundle']
if f['operation']=='manifest':
 print(json.dumps(sorted(p.manifest(f['root'],f['ops_manifest']))))
elif f['operation']=='claim':
 v=p.verify_claim_terminal(f['request_claim'],f.get('terminal_directory',str(__import__('pathlib').Path(f['root'])/'status')),next(iter(f['profiles'])))
 print(json.dumps(v))
elif f['operation']=='quiescence':
 from scripts.train.joint import interruption_admission as gate
 root=__import__('pathlib').Path(f['root']);gate.command=lambda argv:'900999.batch|RUNNING'+chr(10) if '--steps' in argv else ''
 spec=dict(user='paphiwetsa3',claim_roots=[dict(path=str(__import__('pathlib').Path(f['request_claim']).parent),terminal_directory=f.get('terminal_directory',str(root/'status')))],scientific_job_names=['synthetic'],scientific_name_prefixes=['elmo-large314m-continuous-'])
 print(json.dumps(gate.quiescence(spec,next(iter(f['profiles'])))))
elif f['operation']=='verify':
 v=p.verify_observer(b['observer'],producer_request=b['request'],terminal_native=b['native'])
 print(json.dumps({'result':v['result'],'running':v['running_admission_status'],'job':v['job'],'mode':v['spec']['mode']}))
else:
 from scripts.train.joint.interruption_admission import completed_producer
 q,r=completed_producer(b,f.get('call_schema',f['result_schema']),f.get('call_status',f['status']))
 print(json.dumps({'request_schema':q['schema'],'result_schema':r['schema']}))
'''
    return subprocess.run([sys.executable,'-B','-c',program,json.dumps(payload)],env=environment(service),cwd=service,capture_output=True,text=True,check=False)


def refresh_observer(f):
    path=Path(f['bundle']['observer']['path']);f['bundle']['observer']['sha256']=sha(path)


@pytest.mark.parametrize('mode',['capture','inspect','diagnostic','finalize','full'])
@pytest.mark.parametrize('arm',['plain','fused','boundary'])
def test_all_phases_full_semantic_closure_without_external_code(tmp_path,service,mode,arm):
    f=fixture(tmp_path,service,mode,arm);r=execute(f,service);assert r.returncode==0,r.stderr
    out=json.loads(r.stdout);assert out['mode']==mode and out['result']['observer_own_terminal']=='UNKNOWN_NOT_REQUIRED'
    assert out['result']['phase_completion_proven'] is False


@pytest.mark.parametrize('mode',['capture','inspect','diagnostic','finalize'])
def test_four_successful_recovery_producers_require_full_observer(tmp_path,service,mode):
    f=fixture(tmp_path,service,mode);r=execute(f,service,'complete');assert r.returncode==0,r.stderr
    del f['bundle']['observer'];r=execute(f,service,'complete');assert r.returncode!=0


@pytest.mark.parametrize('mode',['capture','inspect','diagnostic','finalize'])
def test_unknown_result_status_or_mixed_phase_cannot_bypass(tmp_path,service,mode):
    f=fixture(tmp_path,service,mode)
    assert execute(f,service,'complete',call_schema='unknown-result').returncode!=0
    assert execute(f,service,'complete',call_status='PASS').returncode!=0
    other='inspect' if mode!='inspect' else 'capture'
    schemas=dict(capture='native-elmo-interruption-evidence-v1',inspect='native-elmo-checkpoint-inspection-result-v1')
    statuses=dict(capture='CAPTURED_REQUIRES_NATIVE_PRODUCER_COMPLETION',inspect='INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION')
    assert execute(f,service,'complete',call_schema=schemas[other],call_status=statuses[other]).returncode!=0


def test_missing_running_is_terminal_only_not_success_or_zero_progress(tmp_path,service):
    f=fixture(tmp_path,service,running=False);r=execute(f,service);assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['running']=='NOT_RECORDED'
    assert execute(f,service,'complete').returncode!=0


def test_failed_full_terminal_remains_failed_not_completion(tmp_path,service):
    f=fixture(tmp_path,service,'full',running=False,state='FAILED');r=execute(f,service);assert r.returncode==0,r.stderr
    result=json.loads(r.stdout)['result'];assert result['producer_terminal_exit']=='1:0' and result['phase_completion_proven'] is False
    assert execute(f,service,'complete').returncode!=0


@pytest.mark.parametrize('field',['job_id','producer_job_id'])
def test_result_job_keys_cannot_contradict_observed_producer(tmp_path,service,field):
    f=fixture(tmp_path,service);path=Path(f['bundle']['result']['path']);d=read(path);d[field]='999999';f['bundle']['result']=put(path,d)
    assert execute(f,service,'complete').returncode!=0


@pytest.mark.parametrize('field',['request','native','host_exit','result'])
def test_bundle_cannot_relocate_observed_or_canonical_evidence(tmp_path,service,field):
    f=fixture(tmp_path,service);src=Path(f['bundle'][field]['path']);target=tmp_path/('elsewhere-'+src.name);target.write_bytes(src.read_bytes())
    f['bundle'][field]=dict(path=str(target),sha256=sha(target))
    assert execute(f,service,'complete').returncode!=0


@pytest.mark.parametrize('key',['producer_query','owned_steps','observer_running_query'])
def test_rehashed_query_errors_do_not_become_complete_commits(tmp_path,service,key):
    f=fixture(tmp_path,service);orb=Path(f['bundle']['observer']['path']);result=read(orb);path=Path(result[key]['path']);d=read(path);d['stderr']='authentication failure'
    result[key]=put(path,d);put(orb,result);refresh_observer(f)
    assert execute(f,service).returncode!=0


@pytest.mark.parametrize('replacement',['900001.batch|RUNNING\n','900001.extern|COMPLETING\n','900001.0|COMPLETED\n','900001_7.batch|RUNNING\n','malformed'])
def test_rehashed_steps_reject_surviving_parent_or_unknown_inventory(tmp_path,service,replacement):
    f=fixture(tmp_path,service);orb=Path(f['bundle']['observer']['path']);result=read(orb);path=Path(result['owned_steps']['path']);d=read(path);d['stdout']=replacement
    result['owned_steps']=put(path,d);put(orb,result);refresh_observer(f)
    assert execute(f,service).returncode!=0


@pytest.mark.parametrize('target',['result','query','steps'])
def test_partial_missing_capture_never_admits_raw_native(tmp_path,service,target):
    f=fixture(tmp_path,service);result=read(f['bundle']['observer']['path'])
    path=f['bundle']['observer']['path'] if target=='result' else result['producer_query' if target=='query' else 'owned_steps']['path']
    Path(path).unlink();assert execute(f,service,'complete').returncode!=0


@pytest.mark.parametrize('file,field,value',[
    ('PREPARED.json','spec_sha256','f'*64),('acceptance.json','job_id','999999'),
    ('RELEASE_ATTEMPT.json','observer_readiness',{}),('RELEASE_ATTEMPT.json','held_native_sha256','a'*64)])
def test_exact_producer_claim_and_release_provenance_is_required(tmp_path,service,file,field,value):
    f=fixture(tmp_path,service);path=Path(f['execution_claim'])/file;d=read(path);d[field]=value;put(path,d)
    assert execute(f,service).returncode!=0


@pytest.mark.parametrize('file', ['PRODUCER_HELD_QUERY.json','ACCEPTANCE_QUERY.json'])
def test_original_observer_queries_must_really_have_succeeded(tmp_path,service,file):
    f=fixture(tmp_path,service);path=Path(f['observer_claim'])/file;d=read(path);d['returncode']=1;put(path,d)
    assert execute(f,service).returncode!=0


@pytest.mark.parametrize('change',['SubmitTime','Comment'])
def test_present_running_claim_cannot_contradict_exact_acceptance(tmp_path,service,change):
    f=fixture(tmp_path,service);path=Path(f['request_claim'])/'RUNNING_ADMISSION.json';d=read(path)
    d['native']=d['native'].replace('SubmitTime=2026-01-01T01:00:00','SubmitTime=2026-01-02T01:00:00') if change=='SubmitTime' else d['native'].replace('Comment=elmo-execution:','Comment=contradictory:')
    put(path,d);assert execute(f,service).returncode!=0


def test_host_exit_is_not_native_query_exit(tmp_path,service):
    f=fixture(tmp_path,service);path=Path(f['bundle']['host_exit']['path']);f['bundle']['host_exit']=put(path,'1\n')
    assert execute(f,service,'complete').returncode!=0


def test_oversized_proof_payload_rejects_before_hashing_or_loading(tmp_path,service):
    f=fixture(tmp_path,service);path=tmp_path/'oversized.ckpt';path.write_bytes(b'x'*(4*1024*1024+1))
    f['bundle']['observer']=dict(path=str(path),sha256=sha(path));r=execute(f,service);assert r.returncode!=0
    assert 'Bounded regular proof metadata' in r.stderr


def test_manifest_member_limit_and_unsafe_paths_reject(tmp_path,service):
    f=fixture(tmp_path,service);path=Path(f['ops_manifest']['path']);path.write_text(path.read_text()+'a'*64+'  ../outside.py\n')
    assert execute(f,service).returncode!=0


def test_source_contains_no_external_import_or_submission_entrypoint(service):
    import ast
    source=service/'scripts/train/joint/phase_observer_proof.py'
    tree=ast.parse(source.read_text());names={n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)}
    assert not names & {'subprocess','importlib','torch','numpy','egomimic'}
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'exec_module','run_module','load','loads_checkpoint'} for n in ast.walk(tree))


@pytest.mark.parametrize('mode',['capture','inspect','diagnostic','finalize','full'])
def test_new_claim_census_requires_complete_commit_not_native_only(tmp_path,service,mode):
    f=fixture(tmp_path,service,mode);r=execute(f,service,'claim');assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['provenance']=='COMPLETE_PHASE_OBSERVER_COMMIT'
    Path(f['bundle']['observer']['path']).unlink()
    assert execute(f,service,'claim').returncode!=0


def test_maintained_quiescence_consumes_new_commit_and_fresh_owned_steps(tmp_path,service):
    f=fixture(tmp_path,service,'full');r=execute(f,service,'quiescence');assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['claims'][0]['provenance']=='COMPLETE_PHASE_OBSERVER_COMMIT'
    Path(f['bundle']['observer']['path']).unlink()
    assert execute(f,service,'quiescence').returncode!=0


def test_generic_claim_census_does_not_hide_failed_early_wrapper(tmp_path,service):
    f=fixture(tmp_path,service,'full',running=False,state='FAILED');r=execute(f,service,'claim');assert r.returncode==0,r.stderr
    d=json.loads(r.stdout);assert d['terminal_state']=='FAILED' and d['running_admission_status']=='NOT_RECORDED'


def test_external_active_service_root_mismatch_rejects(tmp_path,service):
    f=fixture(tmp_path,service);other=tmp_path/'another-service';shutil.copytree(service,other)
    r=execute(f,other);assert r.returncode!=0 and 'active proof implementation' in r.stderr


def test_tampered_active_service_member_rejects_before_external_code(tmp_path,service):
    other=tmp_path/'tampered-service';shutil.copytree(service,other);f=fixture(tmp_path,other)
    member=other/'scripts/train/joint/phase_observer_proof.py';member.write_text(member.read_text()+'\n# unauthorized drift\n')
    r=execute(f,other);assert r.returncode!=0 and 'Manifest member changed' in r.stderr


def test_exact_original_claim_route_is_preserved_without_new_phase_commit(tmp_path,service):
    f=fixture(tmp_path,service,'original');r=execute(f,service,'claim');assert r.returncode==0,r.stderr
    d=json.loads(r.stdout);assert d['provenance']=='EXACT_ORIGINAL_AFTERANY' and d['terminal_state']=='TIMEOUT'
    assert not (Path(f['root'])/'native-111.observer.json').exists()
    (Path(f['root'])/'native-111.exit').write_text('1\n')
    assert execute(f,service,'claim').returncode!=0


def test_original_job_id_does_not_license_an_arbitrary_claim_path(tmp_path,service):
    f=fixture(tmp_path,service,'original');other=tmp_path/'relocated-claim';shutil.copytree(f['request_claim'],other)
    f['request_claim']=str(other);assert execute(f,service,'claim').returncode!=0


def test_manifest_member_bound_is_exercised_directly(tmp_path,service):
    f=fixture(tmp_path,service);root=Path(f['root']);manifest=Path(f['ops_manifest']['path']);rows=manifest.read_text()
    for i in range(124):
        member=root/('metadata-'+str(i)+'.txt');member.write_text('small fixture\n');rows+=sha(member)+'  '+member.name+'\n'
    f['ops_manifest']=put(manifest,rows)
    result=execute(f,service,'manifest');assert result.returncode!=0 and 'member bound' in result.stderr


def test_duplicate_final_commit_keys_reject_even_with_new_hash(tmp_path,service):
    f=fixture(tmp_path,service);path=Path(f['bundle']['observer']['path']);raw=path.read_text()
    path.write_text('{"producer_job_id":"999999",'+raw.lstrip()[1:]);refresh_observer(f)
    r=execute(f,service);assert r.returncode!=0 and 'Duplicate proof JSON keys' in r.stderr


def test_symlink_commit_cannot_relocate_a_valid_proof(tmp_path,service):
    f=fixture(tmp_path,service);path=Path(f['bundle']['observer']['path']);outside=tmp_path/'external-commit.json';outside.write_bytes(path.read_bytes())
    path.unlink();path.symlink_to(outside)
    r=execute(f,service);assert r.returncode!=0 and 'Physical existing proof path' in r.stderr
