"""Service-owned read-only semantic closure for new native phase observers.

No ops module is imported or executed, no native query is issued, and no model
or checkpoint is loaded. Ops code and dynamic records are data with hashes.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from scripts.train.joint.native_interruption import fields, native_identity, NATIVE_KEYS, TERMINAL
from scripts.train.joint.original_observer import ORIGINAL_OBSERVERS, strict_step_rows

MODES = {
    'capture': ('native-elmo-interruption-capture-v1','2','1G','00:10:00',''),
    'inspect': ('native-elmo-checkpoint-inspection-v1','4','24G','00:20:00',''),
    'diagnostic': ('native-elmo-production-recovery-request-v1','32','192G','01:00:00','gres/gpu:h100:4'),
    'finalize': ('native-elmo-interruption-finalizer-v1','4','24G','00:20:00',''),
    'full': ('joint-training-v2','32','192G','7-00:00:00','gres/gpu:h100:4'),
}
RESULT_MODES = {
    'native-elmo-interruption-evidence-v1':'capture',
    'native-elmo-checkpoint-inspection-result-v1':'inspect',
    'native-elmo-production-recovery-diagnostic-v1':'diagnostic',
    'native-elmo-interruption-finalized-v1':'finalize',
}
OBSERVER_KEYS={'schema','root','producer_execution','terminal_directory'}
MAX_BYTES=4*1024*1024
MAX_MEMBERS=128


def need(value,message):
    if not value: raise ValueError(message)


def physical(path):
    p=Path(path)
    need(p.is_absolute() and '..' not in p.parts and p.resolve(strict=True)==p,'Physical existing proof path required')
    return p


def data(path):
    p=physical(path);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=MAX_BYTES,'Bounded regular proof metadata required')
    with p.open('rb') as stream:payload=stream.read(MAX_BYTES+1)
    need(len(payload)<=MAX_BYTES,'Proof input exceeds metadata bound')
    return payload


def text(path):return data(path).decode('utf-8')


def sha(path):return hashlib.sha256(data(path)).hexdigest()


def read_json(path):
    def pairs(items):
        need(len(items)==len(dict(items)),'Duplicate proof JSON keys')
        return dict(items)
    return json.loads(text(path),object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite proof JSON')))


def bound(binding):
    need(set(binding)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',binding['sha256']),'Exact proof path/hash binding required')
    path=physical(binding['path']);need(sha(path)==binding['sha256'],'Proof bytes changed')
    return path


def binding(path):return {'path':str(path),'sha256':sha(path)}


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def manifest(root,proof):
    root=physical(root);path=bound(proof);need(path==root/'FILES.sha256','Manifest escaped exact root')
    names=set()
    for line in text(path).splitlines():
        match=re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
        need(match is not None,'Malformed proof manifest')
        digest,name=match.groups();relative=Path(name)
        need(name not in names and not relative.is_absolute() and '..' not in relative.parts,'Unsafe/duplicate manifest member')
        need(sha(root/relative)==digest,'Manifest member changed')
        names.add(name)
        need(len(names)<=MAX_MEMBERS,'Proof manifest exceeds metadata member bound')
    need(names,'Empty proof manifest');return names


def service(binding_value):
    root=physical(binding_value['root'])
    need(root==Path(__file__).resolve().parents[3] and Path(binding_value['manifest'])==root/'FILES.sha256',
         'Producer service is not the active proof implementation')
    names=manifest(root,{'path':binding_value['manifest'],'sha256':binding_value['manifest_sha256']})
    from scripts.train.joint import native_interruption, original_observer, service_bootstrap
    for module in (native_interruption,original_observer,service_bootstrap):
        need(Path(module.__file__).resolve()==root/'scripts/train/joint'/(module.__name__.rsplit('.',1)[1]+'.py'),
             'Proof dependency imported outside active service')
    need(names==set(service_bootstrap.service_files(root)) and 'scripts/train/joint/phase_observer_proof.py' in names,
         'Active service export closure differs')
    return root


def native_argv(job):return ['/usr/local/bin/scontrol','show','job',str(job),'-o','-dd']


def successful(record,argv):
    need(set(record)=={'schema','argv','returncode','stdout','stderr','observed_at_utc'}
         and record['schema']=='native-elmo-phase-observer-query-v1' and record['argv']==argv
         and type(record['returncode']) is int and record['returncode']==0
         and isinstance(record['stdout'],str) and isinstance(record['stderr'],str) and not record['stderr'].strip()
         and len(record['stdout'].encode())<=1048576 and len(record['stderr'].encode())<=65536,
         'Actual clean successful observer query required')
    stamp=datetime.fromisoformat(record['observed_at_utc'])
    need(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'Missing/future proof query time')
    return record['stdout']


def held_native(raw,spec,spec_sha,job):
    f=fields(raw);n=spec['native_contract']
    need(f.get('JobId')==job and f.get('JobState')=='PENDING' and f.get('Reason')=='JobHeldUser'
         and all(f.get(k,'')==v for k,v in n.items() if k!='NodeList')
         and f.get('NodeList','') in {'','(null)',n['NodeList']} and f.get('NumNodes') in {'1','1-1'}
         and f.get('NumTasks')=='1' and f.get('Comment')=='elmo-execution:'+spec_sha
         and f.get('SubmitTime') not in {None,'Unknown','None'},'Wrong actual held execution identity')
    need(f.get('AllocTRES','') in {'','(null)'} and f.get('GRES','') in {'','(null)'}
         and f.get('StartTime') in {None,'Unknown','N/A'},'Held execution already allocated/started')
    items=[p.split('=',1) for p in f.get('ReqTRES','').split(',')]
    need(all(len(v)==2 for v in items) and len(items)==len(dict(items)),'Ambiguous held TRES')
    tres=dict(items);memory=n['MinMemoryNode']
    need(tres.get('cpu')==n['NumCPUs'] and tres.get('node')=='1'
         and tres.get('mem') in {memory,str(int(memory[:-1])*1024)+'M'},'Held CPU/memory/node TRES differ')
    gpu={k+'='+v for k,v in tres.items() if 'gpu' in k.lower()}
    need(gpu<=({'gres/gpu=4','gres/gpu:h100=4'} if n['TresPerNode'] else set()),'Held GPU TRES conflict')
    if n['TresPerNode']:need('JOB_GRES' not in f or f['JOB_GRES']=='gpu:h100:4','Held JOB_GRES conflict')
    else:need(f.get('JOB_GRES','') in {'','(null)'},'CPU held allocation has GPU')
    for k,v in f.items():
        if k in {'ReqTRES','AllocTRES','TresPerNode','JOB_GRES','GRES'}:continue
        if k.lower().startswith('tresper') or 'gpu' in k.lower():
            need('gpu' not in k.lower() and 'gpu' not in v.lower(),'Auxiliary held GPU resources')
            if k=='TresPerTask':need(v=='cpu:'+n['NumCPUs'],'Held per-task CPU differs')
    return f


def producer_argv(spec,spec_path,spec_sha,ops_sha):
    n=spec['native_contract'];root=Path(spec['root'])
    return ['/usr/local/bin/sbatch','--parsable','--hold','--no-requeue','--export=NONE','--account='+n['Account'],
        '--qos='+n['QOS'],'--partition='+n['Partition'],'--job-name='+n['JobName'],'--nodelist='+n['ReqNodeList'],
        '--nodes=1','--ntasks=1','--cpus-per-task='+n['NumCPUs'],'--mem='+n['MinMemoryNode'],'--time='+n['TimeLimit'],
        '--gres='+('gpu:h100:4' if n['TresPerNode'] else 'none'),'--comment=elmo-execution:'+spec_sha,'--chdir='+str(root),
        '--output='+str(root/'logs/%x-%j.out'),'--error='+str(root/'logs/%x-%j.err'),str(root/'allocated.sbatch'),str(root),
        spec['mode'],str(spec_path),spec_sha,ops_sha]


def execution(ex):
    need(set(ex)=={'path','sha256','mode','ops_manifest_sha256'} and ex['mode'] in MODES,'Wrong exact phase execution binding')
    path=bound({'path':ex['path'],'sha256':ex['sha256']});spec=read_json(path);root=physical(spec['root']);mode=ex['mode']
    need(spec['schema']=='native-elmo-execution-spec-v1' and spec['mode']==mode and path.parent==root/'execution'
         and spec['arm'] in {'plain','fused','boundary'},'Execution schema/root/arm differs')
    ops={'path':str(root/'FILES.sha256'),'sha256':ex['ops_manifest_sha256']}
    need(spec['ops_manifest']==ops,'Execution/observer operations manifest differs')
    names=manifest(root,ops)
    need({'execution_spec.py','allocated.sbatch','host_guard.py','phase_observer.py','observe-phase.sbatch'}<=names
         and all(Path(n).parts[0] not in {'execution','execution-claims','requests','scientific-claims','producer-claims','observer-specs','observer-claims','status','logs'} for n in names),
         'Missing ops role closure or circular dynamic manifest')
    parent=read_json(bound(spec['parent_request']));run_id='mot-elmo-pickplace-continuous-large314m-'+spec['arm']+'-robot-human-camera-subtask-s42-300k-20261026-v1'
    profile=ORIGINAL_OBSERVERS.get(run_id)
    need(profile is not None and spec['parent_request']=={'path':profile['request_path'],'sha256':profile['request_sha256']}
         and str(spec['original_parent_job_id'])==profile['parent_id'] and parent['run_id']==run_id
         and parent['schema']=='joint-training-v2' and parent['kind']=='native-subtask-world-model-v1' and parent['phase']=='full'
         and parent['target_steps']==300000 and parent['parameters']==314187064 and parent['world_size']==4,'Exact original scientific parent differs')
    service(spec['operational_service'])
    template_path=bound(spec['request_template']);need(template_path.parent==root/'templates','Template escaped immutable namespace')
    template=read_json(template_path);schema,cpus,memory,limit,gres=MODES[mode]
    need(template['schema']==schema and template['operational_service']==spec['operational_service'],'Producer phase/schema/service differs')
    parent_binding=template.get('parent_request') if mode!='full' else template['interruption']['parent_request']
    need(parent_binding==spec['parent_request'],'Template belongs to another original parent')
    n=spec['native_contract']
    need(set(n)==(set(NATIVE_KEYS)|{'NodeList','TresPerNode'})-{'JobId'} and all(isinstance(v,str) for v in n.values()),'Complete ID-free native contract required')
    need(n['UserId']=='paphiwetsa3(41003)' and n['Account']=='loaner' and n['QOS']=='loaner-normal' and n['Partition']=='rl2'
         and n['ReqNodeList']==n['NodeList']==profile['node'] and n['NumCPUs']==cpus and n['MinMemoryNode']==memory
         and n['TimeLimit']==limit and n['TresPerNode']==gres and n['Requeue']==n['Restarts']=='0'
         and n['Command']==str(root/'allocated.sbatch'),'Exact native phase owner/resources/Command differ')
    prefix=('elmo-large314m-continuous-' if mode in {'full','diagnostic'} else 'elmo-native-recovery-')+spec['arm']+'-'+mode+'-'
    need(n['JobName'].startswith(prefix) and re.fullmatch('[A-Za-z0-9_-]{1,128}',n['JobName']),'Wrong native phase name')
    if mode=='full':
        from scripts.train.joint.interruption_admission import compatible
        need('producer_native' not in template and template['job_name']==n['JobName'],'Full template gains producer fields')
        compatible(parent,template)
    else:
        need(template['producer_native']==n,'Template may omit only producer JobId')
        if mode!='capture':
            for k in ('source','source_revision','source_manifest','source_manifest_sha256','data_manifest','data_manifest_sha256','lambda_runtime','data_cache'):
                need(template.get(k)==parent.get(k),'Producer changed parent source/runtime/cache: '+k)
        else:need(str(template['parent_job_id'])==profile['parent_id'] and template['parent_native']['JobId']==profile['parent_id'],'Capture original-parent ID differs')
    entries=spec['execution_claim_roots']
    need(entries and len({v['path'] for v in entries})==len(entries),'Complete unique execution-claim roots required')
    own=[v for v in entries if v['path']==str(root/'execution-claims')]
    need(len(own)==1,'Unique own terminal namespace required')
    for row in entries:need(physical(row['path']).is_dir() and physical(row['terminal_directory']).is_dir(),'Unavailable registered execution root')
    sc=root/'execution-claims'/ex['sha256'];a=read_json(sc/'acceptance.json');job=a['job_id']
    need(type(job) is str and re.fullmatch('[0-9]+',job),'Actual accepted producer job ID missing')
    argv=producer_argv(spec,path,ex['sha256'],ex['ops_manifest_sha256'])
    request_path=root/'requests'/(ex['sha256']+'-'+job+'.json')
    actual=bound(a['request']);request_sha=sha(actual)
    request_claim=root/('scientific-claims' if mode in {'diagnostic','full'} else 'producer-claims')/request_sha
    need(set(a)=={'schema','spec_sha256','ops_manifest_sha256','job_id','argv','request','request_claim','held_native'}
         and a['schema']=='native-elmo-held-execution-binding-v1' and a['spec_sha256']==ex['sha256']
         and a['ops_manifest_sha256']==ex['ops_manifest_sha256'] and a['argv']==argv and actual==request_path
         and a['request_claim']==str(request_claim) and a['held_native']['path']==str(sc/'HELD_NATIVE.txt'),'Producer acceptance/spec/argv/request differs')
    held=held_native(text(bound(a['held_native'])),spec,ex['sha256'],job)
    q=read_json(actual);expected=copy.deepcopy(template)
    if mode!='full':expected['producer_native']['JobId']=job;expected_bytes=encoded(expected)
    else:expected_bytes=data(template_path)
    need(data(actual)==expected_bytes and q['operational_service']==spec['operational_service'],'Actual request changed beyond sole actual JobId binding')
    for directory in (sc,request_claim):need(physical(directory).is_dir() and text(directory/'PARENT_JOB_ID').strip()==job,'Spec/request actual IDs differ')
    prepared=read_json(sc/'PREPARED.json')
    need(set(prepared)=={'spec_sha256','ops_manifest_sha256','spec_path','arm','parent_request','argv','original_parent_job_id','created_at'}
         and prepared['spec_sha256']==ex['sha256'] and prepared['ops_manifest_sha256']==ex['ops_manifest_sha256']
         and prepared['spec_path']==str(path) and prepared['arm']==spec['arm'] and prepared['parent_request']==spec['parent_request']
         and prepared['argv']==argv and str(prepared['original_parent_job_id'])==profile['parent_id'],'Original producer submission intent differs')
    canonical=read_json(request_claim/'PREPARED.json')
    need(canonical==dict(request_sha256=request_sha,spec_sha256=ex['sha256'],spec_path=str(path),actual_request_path=str(actual),argv=argv,
         original_parent_job_id=profile['parent_id']),'Canonical materialized request-SHA claim differs')
    need(text(sc/'sbatch.stdout').strip()==job and text(sc/'sbatch.exit').strip()=='0','Producer submission acceptance unresolved')
    text(sc/'sbatch.stderr')
    return dict(spec=spec,spec_path=path,root=root,request_path=actual,request=q,request_claim=request_claim,held=held,job=job,
                execution=ex,terminal_directory=physical(own[0]['terminal_directory']),run_id=run_id)


def observer_name(ctx):
    spec=ctx['spec'];value='elmo-native-phase-'+spec['arm']+'-'+spec['mode']+'-'+ctx['execution']['sha256'][:16]
    scope=spec.get('runtime',{}).get('admission',{}).get('quiescence',{})
    need(value!=spec['native_contract']['JobName'] and value not in scope.get('scientific_job_names',[])
         and not any(value.startswith(p) for p in scope.get('scientific_name_prefixes',[]))
         and not value.startswith(('joint-','elmo-large314m-continuous-')),'Observer collides with scientific writer scope')
    return value


def observer_native(ctx):
    n=ctx['spec']['native_contract']
    return dict(JobName=observer_name(ctx),UserId=n['UserId'],Account=n['Account'],QOS=n['QOS'],Partition=n['Partition'],
        ReqNodeList=n['ReqNodeList'],NodeList=n['NodeList'],NumCPUs='2',MinMemoryNode='1G',TimeLimit='00:10:00',Requeue='0',Restarts='0',
        Command=str(ctx['root']/'observe-phase.sbatch'),TresPerNode='')


def observer_argv(ctx,path,spec_sha):
    n=observer_native(ctx);root=ctx['root'];comment='elmo-phase-observer:'+spec_sha
    need(comment!=ctx['request'].get('run_id'),'Observer comment is scientific identity')
    return ['/usr/local/bin/sbatch','--parsable','--no-requeue','--export=NONE','--dependency=afterany:'+ctx['job'],
        '--account='+n['Account'],'--qos='+n['QOS'],'--partition='+n['Partition'],'--job-name='+n['JobName'],
        '--nodelist='+n['ReqNodeList'],'--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=1G','--time=00:10:00','--gres=none',
        '--comment='+comment,'--chdir='+str(root),'--output='+str(root/'logs/%x-%j.out'),'--error='+str(root/'logs/%x-%j.err'),
        str(root/'observe-phase.sbatch'),str(root),str(path),spec_sha,ctx['execution']['ops_manifest_sha256']]


def pending_native(raw,ctx,observer,spec_sha):
    f=fields(raw);n=observer_native(ctx)
    need(f.get('JobId')==observer and f.get('JobState')=='PENDING' and f.get('Reason')=='Dependency'
         and all(f.get(k,'')==v for k,v in n.items() if k not in {'NodeList','TresPerNode'})
         and f.get('NodeList','') in {'','(null)',n['NodeList']} and f.get('NumNodes') in {'1','1-1'} and f.get('NumTasks')=='1'
         and re.fullmatch('afterany:'+ctx['job']+r'\((?:unfulfilled|unsatisfied)\)',f.get('Dependency',''))
         and f.get('Comment')=='elmo-phase-observer:'+spec_sha and f.get('SubmitTime') not in {None,'Unknown','None'}
         and f.get('WorkDir')==str(ctx['root']),'Observer accepted pending native differs')
    for k,suffix in [('StdOut','.out'),('StdErr','.err')]:
        need(f.get(k)==str(ctx['root']/'logs'/(n['JobName']+'-'+observer+suffix)),'Observer output namespace differs')
    items=[p.split('=',1) for p in f.get('ReqTRES','').split(',')]
    need(all(len(v)==2 for v in items) and len(items)==len(dict(items)),'Ambiguous observer TRES')
    tres=dict(items);need(tres.get('cpu')=='2' and tres.get('node')=='1' and tres.get('mem') in {'1G','1024M'}
         and f.get('AllocTRES','') in {'','(null)'},'Observer pending requested/allocated TRES differ')
    for k,v in f.items():
        if k in {'ReqTRES','AllocTRES'}:need('gpu' not in v.lower(),'Observer has GPU TRES')
        if k.lower().startswith(('gres','job_gres','tresper')):
            need(v.lower() in {'','(null)','none','n/a','0'} or k=='TresPerTask' and v=='cpu:2','Observer GPU/GRES conflict')
    return f


def observer_context(observer_binding):
    result_path=bound(observer_binding);result=read_json(result_path)
    os_binding=result['observer_spec'];os_path=bound(os_binding);ospec=read_json(os_path)
    need(set(ospec)==OBSERVER_KEYS and ospec['schema']=='native-elmo-phase-observer-spec-v1','Wrong ID-free observer spec')
    ctx=execution(ospec['producer_execution']);root=ctx['root'];ps=ctx['execution']['sha256'];osh=os_binding['sha256']
    need(ospec['root']==str(root) and os_path.parent==root/'observer-specs' and ctx['spec']['release_observer_path']==str(os_path)
         and ospec['terminal_directory']==str(ctx['terminal_directory']),'Observer spec escaped actual producer namespace')
    ledger=physical(root/'observer-claims'/ps);job=ctx['job'];observer=result['observer_job_id']
    need(type(observer) is str and re.fullmatch('[0-9]+',observer) and observer!=job,'Wrong actual observer ID')
    expected_argv=observer_argv(ctx,os_path,osh);a=read_json(ledger/'acceptance.json')
    need(a==dict(schema='native-elmo-phase-observer-acceptance-v1',observer_spec=os_binding,
        ops_manifest_sha256=ctx['execution']['ops_manifest_sha256'],producer_execution=ctx['execution'],
        producer_request=binding(ctx['request_path']),producer_job_id=job,observer_job_id=observer,argv=expected_argv,
        pending_native=binding(ledger/'PENDING_NATIVE.txt')),'Observer acceptance belongs to another exact producer')
    intent=read_json(ledger/'PREPARED.json')
    need(intent==dict(schema='native-elmo-phase-observer-intent-v1',observer_spec=os_binding,
        ops_manifest_sha256=ctx['execution']['ops_manifest_sha256'],producer_execution=ctx['execution'],
        producer_request=binding(ctx['request_path']),producer_job_id=job,argv=expected_argv),'Original observer submission intent differs')
    need(text(ledger/'sbatch.stdout').strip()==observer and text(ledger/'sbatch.exit').strip()=='0'
         and text(ledger/'OBSERVER_JOB_ID').strip()==observer,'Observer actual submission acceptance unresolved')
    text(ledger/'sbatch.stderr')
    pending_raw=successful(read_json(ledger/'ACCEPTANCE_QUERY.json'),native_argv(observer))
    need(pending_raw==text(bound(a['pending_native'])),'Observer acceptance query/native bytes differ')
    pending=pending_native(pending_raw,ctx,observer,osh)
    original=held_native(successful(read_json(ledger/'PRODUCER_HELD_QUERY.json'),native_argv(job)),ctx['spec'],ps,job)
    need(original['SubmitTime']==ctx['held']['SubmitTime'],'Observer submitted against another actual held producer')
    release=read_json(root/'execution-claims'/ps/'RELEASE_ATTEMPT.json')
    need(set(release)=={'job_id','spec_sha256','request_sha256','held_native_sha256','observer_readiness'}
         and release['job_id']==job and release['spec_sha256']==ps and release['request_sha256']==sha(ctx['request_path']),
         'Producer release intent differs from accepted execution')
    release_held_path=root/'execution-claims'/ps/'RELEASE_HELD_NATIVE.txt'
    need(sha(release_held_path)==release['held_native_sha256'],'Actual held native release record changed')
    released_held=held_native(text(release_held_path),ctx['spec'],ps,job)
    need(released_held['SubmitTime']==ctx['held']['SubmitTime'],'Released held allocation differs from acceptance')
    ready=release['observer_readiness']
    need(set(ready)=={'schema','producer_job_id','producer_request_sha256','observer_job_id','observer_spec_sha256',
                       'observer_pending_query','observer_own_terminal','production_completion_proven'}
         and ready['schema']=='native-elmo-phase-observer-ready-v1' and ready['producer_job_id']==job
         and ready['producer_request_sha256']==sha(ctx['request_path']) and ready['observer_job_id']==observer
         and ready['observer_spec_sha256']==osh and ready['observer_own_terminal']=='UNKNOWN_NOT_REQUIRED'
         and ready['production_completion_proven'] is False,'Exact observer-before-release readiness missing')
    current=pending_native(successful(ready['observer_pending_query'],native_argv(observer)),ctx,observer,osh)
    need(current['SubmitTime']==pending['SubmitTime'],'Observer changed before producer release')
    ctx.update(result=result,result_path=result_path,observer_spec=os_binding,observer_spec_path=os_path,observer_id=observer,
               observer_ledger=ledger,observer_acceptance=a,pending=pending)
    return ctx


def check_owned_steps(raw,job):
    strict_step_rows(raw)
    for line in raw.splitlines():
        if not line.strip():continue
        step=line.split('|',1)[0].strip()
        need(step!=job and not step.startswith(job+'.') and not step.startswith(job+'_'),
             'Producer still has native workers')


def verify_observer(observer_binding,*,producer_request=None,producer_native=None,terminal_native=None):
    """Reopen all evidence, returning verified context; never issues a query."""
    ctx=observer_context(observer_binding);result=ctx['result'];job=ctx['job'];root=ctx['root'];terminal_dir=ctx['terminal_directory']
    canonical=dict(native=terminal_dir/('native-'+job+'.txt'),query=terminal_dir/('native-'+job+'.query.json'),
        query_exit=terminal_dir/('native-'+job+'.exit'),query_stderr=terminal_dir/('native-'+job+'.stderr'),
        steps=terminal_dir/('native-'+job+'.steps-query.json'),result=terminal_dir/('native-'+job+'.observer.json'))
    need(ctx['result_path']==canonical['result'],'Observer final commit is not canonical producer destination')
    own_query=read_json(ctx['observer_ledger']/'RUNNING_NATIVE_QUERY.json')
    own=native_identity(successful(own_query,native_argv(ctx['observer_id'])),dict(observer_native(ctx),JobId=ctx['observer_id']),'RUNNING')
    osh=ctx['observer_spec']['sha256']
    need(own.get('SubmitTime')==ctx['pending']['SubmitTime'] and own.get('Comment')=='elmo-phase-observer:'+osh,'Own observer differs from accepted afterany allocation')
    query=read_json(canonical['query']);raw=successful(query,native_argv(job));state=fields(raw).get('JobState')
    need(state in TERMINAL,'Producer is not genuinely terminal')
    expected_native=dict(ctx['spec']['native_contract'],JobId=job)
    terminal=native_identity(raw,expected_native,state)
    need(terminal.get('SubmitTime')==ctx['held']['SubmitTime'] and terminal.get('Comment')=='elmo-execution:'+ctx['execution']['sha256'],
         'Producer terminal contradicts exact accepted held identity')
    need(datetime.fromisoformat(terminal['StartTime'])<=datetime.fromisoformat(terminal['EndTime'])<=datetime.fromisoformat(own['StartTime']),
         'Producer terminal/afterany time order differs')
    running_path=ctx['request_claim']/'RUNNING_ADMISSION.json';running_binding=None
    if running_path.exists() or running_path.is_symlink():
        running=read_json(running_path)
        need(running['job_id']==job and running['request_sha256']==sha(ctx['request_path']) and running['spec_sha256']==ctx['execution']['sha256'],'Producer RUNNING claim differs')
        actual=native_identity(running['native'],expected_native,'RUNNING')
        need(actual.get('SubmitTime')==ctx['held']['SubmitTime'] and actual.get('Comment')=='elmo-execution:'+ctx['execution']['sha256']
             and actual['StartTime']==terminal['StartTime'],'Producer RUNNING contradicts held/terminal identity')
        running_binding=binding(running_path)
    steps=read_json(canonical['steps']);step_argv=['/usr/local/bin/squeue','--steps','--noheader','--user','paphiwetsa3','--format=%i|%T']
    step_raw=successful(steps,step_argv);check_owned_steps(step_raw,job)
    need(text(canonical['native'])==raw and text(canonical['query_exit']).strip()=='0' and not text(canonical['query_stderr']),
         'Canonical native/query artifacts disagree')
    stamp=datetime.fromisoformat(result['captured_at_utc']);need(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'Missing/future final commit time')
    expected=dict(schema='native-elmo-phase-observer-result-v1',status='GENUINE_PRODUCER_TERMINAL_CAPTURED',producer_execution=ctx['execution'],
        producer_request=binding(ctx['request_path']),producer_job_id=job,producer_phase=ctx['spec']['mode'],producer_terminal_state=terminal['JobState'],
        producer_terminal_exit=terminal['ExitCode'],producer_native=binding(canonical['native']),producer_query=binding(canonical['query']),
        producer_query_exit=binding(canonical['query_exit']),producer_query_stderr=binding(canonical['query_stderr']),owned_steps=binding(canonical['steps']),
        producer_running_admission=running_binding,producer_running_admission_status='VALIDATED' if running_binding else 'NOT_RECORDED',
        observer_spec=ctx['observer_spec'],observer_acceptance=binding(ctx['observer_ledger']/'acceptance.json'),
        observer_running_query=binding(ctx['observer_ledger']/'RUNNING_NATIVE_QUERY.json'),observer_job_id=ctx['observer_id'],
        observer_own_terminal='UNKNOWN_NOT_REQUIRED',observer_host_exit_is_native_proof=False,phase_completion_proven=False,zero_work_proven=False,
        captured_at_utc=result['captured_at_utc'],automatic_recovery_authorized=False)
    need(result==expected,'Final observer commit/evidence semantic closure differs')
    if producer_request is not None:need(producer_request==binding(ctx['request_path']),'Observer certifies another producer request')
    if producer_native is not None:need(producer_native==expected_native,'Observer certifies another native contract')
    if terminal_native is not None:need(terminal_native==binding(canonical['native']),'Observer certifies another terminal native binding')
    ctx.update(terminal=terminal,expected_native=expected_native,running_admission=running_binding,
               running_admission_status='VALIDATED' if running_binding else 'NOT_RECORDED')
    return ctx


def verify_claim_terminal(claim,terminal_directory,run_id):
    """Exact old-original route or complete new spec-bound phase observer."""
    claim=physical(claim);terminal_directory=physical(terminal_directory)
    prepared=read_json(claim/'PREPARED.json');job=text(claim/'PARENT_JOB_ID').strip()
    need(job.isdecimal() and prepared['request_sha256']==claim.name,'Unknown/incorrect accepted request claim')
    profile=ORIGINAL_OBSERVERS.get(run_id)
    need(profile is not None,'Claim outside exact supported original identity')
    if job==profile['parent_id']:
        root=Path(profile['root'])
        need(claim==root/'claims'/profile['request_sha256'] and terminal_directory==root,'Unreviewed original parent/claim/terminal route')
        running_binding=binding(claim/'RUNNING_ADMISSION.json');running=read_json(claim/'RUNNING_ADMISSION.json')
        f=fields(running['native']);expected={k:f.get(k,'') for k in (*NATIVE_KEYS,'NodeList','TresPerNode')}
        paths=dict(files_manifest=root/'FILES.sha256',capture_script=root/'capture-native.sbatch',guard=root/'guard_full.py',
            submitter=root/'submit_full_once.py',parent_script=root/'full.sbatch',prepared=claim/'PREPARED.json',
            parent_id=claim/'PARENT_JOB_ID',observer_id=claim/'CAPTURE_JOB_ID',pending_native=claim/'CAPTURE_PENDING_NATIVE.txt',
            running_admission=claim/'RUNNING_ADMISSION.json',terminal_native=root/('native-'+job+'.txt'),
            query_exit=root/('native-'+job+'.exit'),query_stderr=root/('native-'+job+'.stderr'))
        if profile['has_smoke_identity']:paths['smoke_identity']=root/'smoke-identity.json'
        proof=dict(schema='native-elmo-original-observer-v1',**{k:binding(v) for k,v in paths.items()})
        q=dict(parent_request={'path':profile['request_path'],'sha256':profile['request_sha256']},parent_job_id=job,
            parent_native=expected,parent_running_admission=running_binding,original_observer=proof)
        from scripts.train.joint.original_observer import validate_original
        def reader(b,json_value=True):return read_json(bound(b)) if json_value else text(bound(b))
        raw,provenance=validate_original(q,reader)
        return dict(job_id=job,terminal_native_sha256=sha(paths['terminal_native']),terminal_state=fields(raw)['JobState'],
                    provenance='EXACT_ORIGINAL_AFTERANY',observer_own_terminal=provenance['observer_own_terminal'])
    need('spec_path' in prepared and 'spec_sha256' in prepared and 'actual_request_path' in prepared,
         'New accepted claim lacks exact execution/materialized request provenance')
    commit=terminal_directory/('native-'+job+'.observer.json')
    verified=verify_observer(binding(commit),producer_request={'path':prepared['actual_request_path'],'sha256':claim.name},
        terminal_native=binding(terminal_directory/('native-'+job+'.txt')))
    need(verified['request_claim']==claim and verified['job']==job and verified['run_id']==run_id
         and verified['execution']['path']==prepared['spec_path'] and verified['execution']['sha256']==prepared['spec_sha256'],
         'New phase terminal observer belongs to another exact claim/spec/identity')
    return dict(job_id=job,terminal_native_sha256=verified['result']['producer_native']['sha256'],
        terminal_state=verified['terminal']['JobState'],provenance='COMPLETE_PHASE_OBSERVER_COMMIT',
        observer_commit=binding(commit),observer_own_terminal='UNKNOWN_NOT_REQUIRED',
        running_admission_status=verified['running_admission_status'])
