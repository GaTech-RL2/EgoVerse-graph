"""Exact evidence-only phase observers; stdlib only, no implicit retries or ML.

The controller must verify the accepted observer immediately before releasing
its genuinely held producer. This module never releases or cancels either job.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MODES = {'capture', 'inspect', 'diagnostic', 'finalize', 'full'}
KEYS = {'schema','root','producer_execution','terminal_directory'}
TERMINAL = {'COMPLETED','FAILED','TIMEOUT','NODE_FAIL','PREEMPTED','CANCELLED','OUT_OF_MEMORY'}

_LOADED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()



def need(value,message):
    if not value: raise ValueError(message)


def sha(path):
    p=absolute(path);need(p.is_file() and not p.is_symlink(),'Physical hash input required')
    need(p.stat().st_size<=4*1024*1024,'Observer metadata exceeds 4 MiB')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def absolute(path):
    p=Path(path)
    need(p.is_absolute() and '..' not in p.parts and p.resolve()==p,'Noncanonical or symlink observer path')
    return p


def bound(value):
    need(set(value)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',value['sha256']),'Exact path/hash required')
    p=absolute(value['path'])
    need(p.is_file() and not p.is_symlink() and sha(p)==value['sha256'],'Changed bound observer input')
    return p


def text(path):
    p=absolute(path);need(p.is_file() and not p.is_symlink(),'Physical observer input file required')
    need(p.stat().st_size<=4*1024*1024,'Observer metadata exceeds 4 MiB')
    return p.read_text()


def read_json(path):
    def pairs(items):
        need(len(items)==len(dict(items)),'Duplicate observer JSON field')
        return dict(items)
    return json.loads(text(path),object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite observer JSON')))


def encoded(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def write_once(path,value):
    p=absolute(path);need(p.parent.is_dir(),'Observer output parent missing')
    payload=value if isinstance(value,bytes) else encoded(value)
    with p.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    need(p.read_bytes()==payload,'Observer exclusive write readback failed')


def mkdir_once(path):
    p=absolute(path);p.mkdir();fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def now(): return datetime.now(timezone.utc).isoformat()


def query(argv):
    result=subprocess.run(argv,capture_output=True,text=True,timeout=30,check=False)
    need(len(result.stdout.encode())<=1048576 and len(result.stderr.encode())<=65536,'Observer query exceeded bound')
    return dict(schema='native-elmo-phase-observer-query-v1',argv=argv,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr,observed_at_utc=now())


def native_query(job): return ['/usr/local/bin/scontrol','show','job',str(job),'-o','-dd']


def successful(record,argv):
    need(set(record)=={'schema','argv','returncode','stdout','stderr','observed_at_utc'}
         and record['schema']=='native-elmo-phase-observer-query-v1' and record['argv']==argv
         and type(record['returncode']) is int and record['returncode']==0
         and isinstance(record['stdout'],str) and isinstance(record['stderr'],str) and not record['stderr'].strip(),
         'Actual clean native query success required')
    need(len(record['stdout'].encode())<=1048576 and len(record['stderr'].encode())<=65536,'Preserved native query exceeded bounds')
    t=datetime.fromisoformat(record['observed_at_utc'])
    need(t.tzinfo is not None and t<=datetime.now(timezone.utc),'Missing/future observer query time')
    return record['stdout']


def engine(root, manifest_sha):
    root=absolute(root);manifest=bound({'path':str(root/'FILES.sha256'),'sha256':manifest_sha})
    names=set()
    lines=manifest.read_text().splitlines();need(0<len(lines)<=128,'Unbounded operations manifest')
    for line in lines:
        digest,name=line.split('  ',1)
        need(re.fullmatch('[0-9a-f]{64}',digest) and name not in names
             and not Path(name).is_absolute() and '..' not in Path(name).parts,'Invalid operations manifest')
        bound({'path':str(root/name),'sha256':digest});names.add(name)
    need({'execution_spec.py','phase_observer.py','observe-phase.sbatch'}<=names
         and sha(root/'phase_observer.py')==sha(__file__),'Observer/binder implementation not bound to immutable manifest')
    path=root/'execution_spec.py';existing=sys.modules.get('execution_spec')
    if existing is not None:
        need(getattr(existing,'__file__',None)==str(path)
             and getattr(existing,'_LOADED_IMPLEMENTATION_SHA256',None)==sha(path)
             and getattr(existing,'_verified_ops_manifest_sha256',None) in {None,manifest_sha},
             'Foreign or unverified execution module already imported')
        existing._verified_ops_manifest_sha256=manifest_sha
        existing._verified_ops_member_sha256=sha(path)
        return existing
    loader=importlib.util.spec_from_file_location('execution_spec',path)
    module=importlib.util.module_from_spec(loader)
    module._verified_ops_manifest_sha256=manifest_sha
    module._verified_ops_member_sha256=sha(path)
    sys.modules['execution_spec']=module
    try:loader.loader.exec_module(module)
    except BaseException:
        del sys.modules['execution_spec'];raise
    return module


def producer_binding(execution):
    need(set(execution)=={'path','sha256','mode','ops_manifest_sha256'} and execution['mode'] in MODES,
         'Exact phase execution binding required')
    p=bound({'path':execution['path'],'sha256':execution['sha256']})
    root=p.parent.parent;e=engine(root,execution['ops_manifest_sha256'])
    values=e.read_binding(str(p),execution['sha256'],execution['mode'],execution['ops_manifest_sha256'])
    return e,values


def build_spec(execution):
    """Return ID-free data only; actual acceptance is reopened before submission."""
    need(set(execution)=={'path','sha256','mode','ops_manifest_sha256'} and execution['mode'] in MODES,
         'Exact phase execution binding required')
    p=bound({'path':execution['path'],'sha256':execution['sha256']});e=engine(p.parent.parent,execution['ops_manifest_sha256'])
    producer,_=e.read_spec(str(p),execution['sha256'],execution['mode'],execution['ops_manifest_sha256'])
    roots=[row for row in producer['execution_claim_roots'] if row['path']==str(Path(producer['root'])/'execution-claims')]
    need(len(roots)==1,'Unique own execution terminal directory required')
    return dict(schema='native-elmo-phase-observer-spec-v1',root=producer['root'],producer_execution=execution,
        terminal_directory=roots[0]['terminal_directory'])


def read_spec(path,spec_sha,manifest_sha):
    p=bound({'path':str(path),'sha256':spec_sha});s=read_json(p)
    need(set(s)==KEYS and s['schema']=='native-elmo-phase-observer-spec-v1','Wrong observer schema/fields')
    root=absolute(s['root']);need(p.parent==root/'observer-specs','Observer spec outside immutable dynamic namespace')
    execution=s['producer_execution']
    need(execution['ops_manifest_sha256']==manifest_sha,'Observer/producer operations manifest differs')
    e,values=producer_binding(execution);producer,request,q,job,claim,held=values
    need(s==build_spec(execution) and root==Path(producer['root']),'Observer spec does not bind exact accepted phase')
    need(e.sha(root/'phase_observer.py')==sha(__file__),'Observer implementation drift')
    names=e.manifest(root,producer['ops_manifest'])
    need({'phase_observer.py','observe-phase.sbatch'}<=names
         and all(Path(n).parts[0] not in {'observer-specs','observer-claims'} for n in names),
         'Manifest omits observer closure or includes circular dynamic artifacts')
    need(absolute(root/'observer-claims').is_dir() and absolute(root/'logs').is_dir()
         and absolute(s['terminal_directory']).is_dir(),'Observer ledger/log/terminal directories missing')
    derived=dict(s,producer_request={'path':str(request),'sha256':e.sha(request)},producer_job_id=job,producer_request_claim=str(claim))
    return derived,e,producer,request,q,job,claim,held


def paths(s,spec_sha):
    root=Path(s['root']);job=s['producer_job_id'];ps=s['producer_execution']['sha256']
    terminal=Path(s['terminal_directory'])
    return dict(claim=root/'observer-claims'/ps,native=terminal/('native-'+job+'.txt'),
        query_exit=terminal/('native-'+job+'.exit'),query_stderr=terminal/('native-'+job+'.stderr'),
        query=terminal/('native-'+job+'.query.json'),steps=terminal/('native-'+job+'.steps-query.json'),
        result=terminal/('native-'+job+'.observer.json'))


def name(s,producer):
    value='elmo-native-phase-'+producer['arm']+'-'+producer['mode']+'-'+s['producer_execution']['sha256'][:16]
    need(value.startswith('elmo-native-phase-') and not value.startswith(('elmo-large314m-continuous-','joint-')),
         'Observer must be outside scientific prefixes')
    scope=producer.get('runtime',{}).get('admission',{}).get('quiescence',{})
    need(value != producer['native_contract']['JobName'] and value not in scope.get('scientific_job_names',[]) and
         not any(value.startswith(p) for p in scope.get('scientific_name_prefixes',[])),
         'Observer name collides with scoped scientific writer')
    return value


def comment(producer,spec_sha):
    need(re.fullmatch('[0-9a-f]{64}',spec_sha),'Exact observer spec SHA required')
    value='elmo-phase-observer:'+spec_sha
    e=engine(producer['root'],producer['ops_manifest']['sha256']);parent=e.read_json(e.bound(producer['parent_request']))
    need(value!=parent['run_id'],'Observer comment must not be a scientific run identity')
    return value


def native_contract(s,producer):
    n=producer['native_contract'];return dict(JobName=name(s,producer),UserId=n['UserId'],Account=n['Account'],QOS=n['QOS'],
        Partition=n['Partition'],ReqNodeList=n['ReqNodeList'],NodeList=n['NodeList'],NumCPUs='2',MinMemoryNode='1G',
        TimeLimit='00:10:00',Requeue='0',Restarts='0',Command=str(Path(s['root'])/'observe-phase.sbatch'),TresPerNode='')


def argv(s,producer,spec_path,spec_sha,manifest_sha):
    n=native_contract(s,producer);root=Path(s['root'])
    return ['/usr/local/bin/sbatch','--parsable','--no-requeue','--export=NONE','--dependency=afterany:'+s['producer_job_id'],
        '--account='+n['Account'],'--qos='+n['QOS'],'--partition='+n['Partition'],'--job-name='+n['JobName'],
        '--nodelist='+n['ReqNodeList'],'--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=1G','--time=00:10:00','--gres=none',
        '--comment='+comment(producer,spec_sha),'--chdir='+str(root),'--output='+str(root/'logs/%x-%j.out'),
        '--error='+str(root/'logs/%x-%j.err'),str(root/'observe-phase.sbatch'),str(root),str(spec_path),spec_sha,manifest_sha]


def pending_native(raw,s,producer,spec_sha,observer):
    e=engine(s['root'],s['producer_execution']['ops_manifest_sha256']);f=e.fields(raw);n=native_contract(s,producer)
    need(f.get('JobId')==observer and f.get('JobState')=='PENDING' and f.get('Reason')=='Dependency'
         and f.get('NodeList','') in {'','(null)',n['NodeList']} and f.get('NumNodes') in {'1','1-1'}
         and f.get('NumTasks')=='1' and all(f.get(k,'')==v for k,v in n.items() if k not in {'NodeList','TresPerNode'}),
         'Actual observer PENDING identity/resources differ')
    need(re.fullmatch('afterany:'+s['producer_job_id']+r'\((?:unfulfilled|unsatisfied)\)',f.get('Dependency',''))
         and f.get('Comment')==comment(producer,spec_sha)
         and f.get('SubmitTime') not in {None,'Unknown','None'} and f.get('WorkDir')==s['root'],
         'Actual observer accepted dependency/spec differs')
    for key in ('StdOut','StdErr'):
        suffix='.out' if key=='StdOut' else '.err'
        need(f.get(key)==str(Path(s['root'])/'logs'/(n['JobName']+'-'+observer+suffix)),'Observer output namespace differs')
    parts=f.get('ReqTRES','').split(',');items=[item.split('=',1) for item in parts]
    need(all(len(item)==2 for item in items) and len(items)==len(dict(items)),'Ambiguous observer requested TRES')
    tres=dict(items)
    need(tres.get('cpu')=='2' and tres.get('node')=='1' and tres.get('mem') in {'1G','1024M'}
         and f.get('AllocTRES','') in {'','(null)'},'Observer requested/allocated CPU TRES differ')
    for k,v in f.items():
        if k in {'ReqTRES','AllocTRES'}:need('gpu' not in v.lower(),'Observer acquired/requested GPU')
        if k.lower().startswith(('gres','job_gres','tresper')):
            need(v.lower() in {'','(null)','none','n/a','0'} or k=='TresPerTask' and v=='cpu:2','Observer GPU/GRES conflict')
    return f


def held_producer(s,e,producer,job,held):
    ps=s['producer_execution']['sha256'];r=query(native_query(job));raw=successful(r,native_query(job))
    current=e.held_native(raw,producer,ps,job)
    need(current['SubmitTime']==held['SubmitTime'],'Held producer changed since actual acceptance')
    return r


def read_acceptance(path,spec_sha,manifest_sha):
    data=read_spec(path,spec_sha,manifest_sha);s,e,producer,request,q,job,claim,held=data;p=paths(s,spec_sha);ledger=p['claim']
    a=read_json(ledger/'acceptance.json');expected_argv=argv(s,producer,path,spec_sha,manifest_sha)
    need(set(a)=={'schema','observer_spec','ops_manifest_sha256','producer_execution','producer_request','producer_job_id','observer_job_id','argv','pending_native'}
         and a['schema']=='native-elmo-phase-observer-acceptance-v1' and a['observer_spec']=={'path':str(path),'sha256':spec_sha}
         and a['ops_manifest_sha256']==manifest_sha and a['producer_execution']==s['producer_execution']
         and a['producer_request']==s['producer_request'] and a['producer_job_id']==job and a['argv']==expected_argv,
         'Observer acceptance belongs to another producer/spec')
    observer=a['observer_job_id'];need(type(observer) is str and observer.isdecimal() and observer!=job,'Wrong observer actual job ID')
    prepared=read_json(ledger/'PREPARED.json')
    need(prepared==dict(schema='native-elmo-phase-observer-intent-v1',observer_spec=a['observer_spec'],ops_manifest_sha256=manifest_sha,
         producer_execution=s['producer_execution'],producer_request=s['producer_request'],producer_job_id=job,argv=expected_argv),
         'Original immutable observer submission intent changed')
    need(text(ledger/'sbatch.stdout').strip()==observer and text(ledger/'sbatch.exit').strip()=='0'
         and text(ledger/'OBSERVER_JOB_ID').strip()==observer,'Original observer submission acceptance unresolved')
    need(a['pending_native']['path']==str(ledger/'PENDING_NATIVE.txt'),'Observer pending provenance escaped ledger')
    pending_raw=bound(a['pending_native']).read_text()
    accepted_query=read_json(ledger/'ACCEPTANCE_QUERY.json')
    need(successful(accepted_query,native_query(observer))==pending_raw,'Accepted observer query and native record differ')
    pending=pending_native(pending_raw,s,producer,spec_sha,observer)
    original_held=read_json(ledger/'PRODUCER_HELD_QUERY.json')
    old=e.held_native(successful(original_held,native_query(job)),producer,s['producer_execution']['sha256'],job)
    need(old['SubmitTime']==held['SubmitTime'],'Original observer submission did not see actual accepted producer held')
    return data,a,pending


def submit_once(path,spec_sha,manifest_sha):
    data=read_spec(path,spec_sha,manifest_sha);s,e,producer,request,q,job,claim,held=data
    held_record=held_producer(s,e,producer,job,held);p=paths(s,spec_sha);ledger=p['claim']
    # Never replace an original capture or consume an old terminal namespace.
    need(all(not out.exists() and not out.is_symlink() for key,out in p.items() if key!='claim'),'Producer native destinations already exist')
    mkdir_once(ledger)
    command=argv(s,producer,path,spec_sha,manifest_sha)
    write_once(ledger/'PREPARED.json',dict(schema='native-elmo-phase-observer-intent-v1',observer_spec={'path':str(path),'sha256':spec_sha},
        ops_manifest_sha256=manifest_sha,producer_execution=s['producer_execution'],producer_request=s['producer_request'],producer_job_id=job,argv=command))
    write_once(ledger/'PRODUCER_HELD_QUERY.json',held_record)
    try:result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=False)
    except (OSError,subprocess.TimeoutExpired) as error:
        write_once(ledger/'SUBMISSION_UNKNOWN.json',dict(schema='native-elmo-phase-observer-submission-unknown-v1',error=type(error).__name__,automatic_retry_authorized=False))
        raise RuntimeError('Observer submission unknown; inspect exact ledger/native, never retry') from error
    write_once(ledger/'sbatch.stdout',result.stdout.encode());write_once(ledger/'sbatch.stderr',result.stderr.encode())
    write_once(ledger/'sbatch.exit',(str(result.returncode)+'\n').encode())
    observer=result.stdout.strip()
    need(result.returncode==0 and observer.isdecimal() and observer!=job,'Observer submission unresolved; inspect ledger/native, never retry')
    write_once(ledger/'OBSERVER_JOB_ID',(observer+'\n').encode())
    observation=query(native_query(observer));write_once(ledger/'ACCEPTANCE_QUERY.json',observation)
    raw=successful(observation,native_query(observer));pending_native(raw,s,producer,spec_sha,observer)
    write_once(ledger/'PENDING_NATIVE.txt',raw.encode())
    acceptance=dict(schema='native-elmo-phase-observer-acceptance-v1',observer_spec={'path':str(path),'sha256':spec_sha},ops_manifest_sha256=manifest_sha,
        producer_execution=s['producer_execution'],producer_request=s['producer_request'],producer_job_id=job,observer_job_id=observer,
        argv=command,pending_native={'path':str(ledger/'PENDING_NATIVE.txt'),'sha256':sha(ledger/'PENDING_NATIVE.txt')})
    write_once(ledger/'acceptance.json',acceptance)
    return acceptance


def verify_ready_for_release(path,spec_sha,manifest_sha):
    data,a,pending=read_acceptance(path,spec_sha,manifest_sha);s,e,producer,request,q,job,claim,held=data
    held_producer(s,e,producer,job,held)
    observation=query(native_query(a['observer_job_id']));raw=successful(observation,native_query(a['observer_job_id']))
    current=pending_native(raw,s,producer,spec_sha,a['observer_job_id'])
    need(current['SubmitTime']==pending['SubmitTime'],'Observer changed after accepted dependency')
    return dict(schema='native-elmo-phase-observer-ready-v1',producer_job_id=job,producer_request_sha256=sha(request),
        observer_job_id=a['observer_job_id'],observer_spec_sha256=spec_sha,observer_pending_query=observation,
        observer_own_terminal='UNKNOWN_NOT_REQUIRED',production_completion_proven=False)


def strict_steps(raw,parent):
    for line in raw.splitlines():
        if not line.strip():continue
        row=[v.strip() for v in line.split('|')]
        need(len(row)==2 and re.fullmatch(r'[0-9]+(?:_[0-9]+)?(?:\.[A-Za-z0-9_-]+)?',row[0])
             and row[1] in TERMINAL|{'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','STOPPED','RESIZING'},'Ambiguous complete owned step inventory')
        need(row[0]!=parent and not row[0].startswith(parent+'.') and not row[0].startswith(parent+'_'),'Producer still has native workers')


def validate_terminal(data,pending,own_query,observation,steps,spec_sha):
    s,e,producer,request,q,job,claim,held=data
    native=e.service_module(producer,'native_interruption')
    observer_query=read_json(paths(s,spec_sha)['claim']/'acceptance.json')['observer_job_id']
    own_raw=successful(own_query,native_query(observer_query))
    own=native.native_identity(own_raw,dict(native_contract(s,producer),JobId=observer_query),'RUNNING')
    need(own['SubmitTime']==pending['SubmitTime'] and own.get('Comment')==comment(producer,spec_sha),
         'Actual observer is not the accepted afterany allocation')
    raw=successful(observation,native_query(job));state=e.fields(raw).get('JobState')
    need(state in TERMINAL,'Actual producer is not terminal')
    terminal=native.native_identity(raw,dict(producer['native_contract'],JobId=job),state)
    need(terminal.get('SubmitTime')==held['SubmitTime'] and terminal.get('Comment')=='elmo-execution:'+s['producer_execution']['sha256'],
         'Terminal producer differs from accepted held identity')
    start=datetime.fromisoformat(terminal['StartTime']);end=datetime.fromisoformat(terminal['EndTime'])
    need(start<=end<=datetime.fromisoformat(own['StartTime']),'Terminal interval/afterany observer order differs')
    running_path=claim/'RUNNING_ADMISSION.json';running_binding=None
    if running_path.exists() or running_path.is_symlink():
        running_path=absolute(running_path);need(running_path.is_file(),'Producer RUNNING provenance is not a physical file')
        running=read_json(running_path)
        need(running['job_id']==job and running['request_sha256']==sha(request) and running['spec_sha256']==s['producer_execution']['sha256'],
             'Producer actual RUNNING provenance differs')
        actual_running=native.native_identity(running['native'],dict(producer['native_contract'],JobId=job),'RUNNING')
        need(terminal['StartTime']==actual_running['StartTime'] and actual_running.get('SubmitTime')==held['SubmitTime']
             and actual_running.get('Comment')=='elmo-execution:'+s['producer_execution']['sha256'],
             'Producer actual RUNNING allocation contradicts accepted held/terminal identity')
        running_binding={'path':str(running_path),'sha256':sha(running_path)}
    step_argv=['/usr/local/bin/squeue','--steps','--noheader','--user','paphiwetsa3','--format=%i|%T']
    strict_steps(successful(steps,step_argv),job)
    return terminal,running_binding


def result_value(data,a,spec_path,spec_sha,terminal,running_binding,captured_at):
    s,e,producer,request,q,job,claim,held=data;p=paths(s,spec_sha)
    at=datetime.fromisoformat(captured_at);need(at.tzinfo is not None and at<=datetime.now(timezone.utc),'Missing/future capture commit time')
    need(text(p['native'])==successful(read_json(p['query']),native_query(job))
         and text(p['query_exit']).strip()=='0' and not text(p['query_stderr']),
         'Canonical producer native/query artifacts disagree')
    def binding(path):return {'path':str(path),'sha256':sha(path)}
    return dict(schema='native-elmo-phase-observer-result-v1',status='GENUINE_PRODUCER_TERMINAL_CAPTURED',producer_execution=s['producer_execution'],
        producer_request=s['producer_request'],producer_job_id=job,producer_phase=producer['mode'],producer_terminal_state=terminal['JobState'],
        producer_terminal_exit=terminal['ExitCode'],producer_native=binding(p['native']),producer_query=binding(p['query']),
        producer_query_exit=binding(p['query_exit']),producer_query_stderr=binding(p['query_stderr']),owned_steps=binding(p['steps']),
        producer_running_admission=running_binding,producer_running_admission_status='VALIDATED' if running_binding else 'NOT_RECORDED',
        observer_spec={'path':str(spec_path),'sha256':spec_sha},observer_acceptance=binding(p['claim']/'acceptance.json'),
        observer_running_query=binding(p['claim']/'RUNNING_NATIVE_QUERY.json'),observer_job_id=a['observer_job_id'],
        observer_own_terminal='UNKNOWN_NOT_REQUIRED',observer_host_exit_is_native_proof=False,
        phase_completion_proven=False,zero_work_proven=False,captured_at_utc=captured_at,automatic_recovery_authorized=False)


def capture(path,spec_sha,manifest_sha):
    data,a,pending=read_acceptance(path,spec_sha,manifest_sha);s,e,producer,request,q,job,claim,held=data;p=paths(s,spec_sha)
    observer=a['observer_job_id'];native=e.service_module(producer,'native_interruption');expected=dict(native_contract(s,producer),JobId=observer)
    native.process_identity(expected,cpu=True)
    need(os.environ.get('SLURM_RESTART_COUNT','0')=='0','Fresh no-requeue observer required')
    own_query=query(native_query(observer));own_raw=successful(own_query,native_query(observer));own=native.native_identity(own_raw,expected,'RUNNING')
    need(own['SubmitTime']==pending['SubmitTime'] and own.get('Comment')==comment(producer,spec_sha),
         'Actual observer is not the accepted afterany allocation')
    write_once(p['claim']/'RUNNING_NATIVE_QUERY.json',own_query)
    # Preserve failed queries too. Native query exit is not the observer host exit.
    observation=query(native_query(job));write_once(p['query'],observation)
    write_once(p['native'],observation['stdout'].encode());write_once(p['query_exit'],(str(observation['returncode'])+'\n').encode())
    write_once(p['query_stderr'],observation['stderr'].encode())
    successful(observation,native_query(job))
    step_argv=['/usr/local/bin/squeue','--steps','--noheader','--user','paphiwetsa3','--format=%i|%T']
    steps=query(step_argv);write_once(p['steps'],steps)
    terminal,running_binding=validate_terminal(data,pending,own_query,observation,steps,spec_sha)
    result=result_value(data,a,path,spec_sha,terminal,running_binding,now())
    write_once(p['result'],result)  # final exclusive fsynced commit, no phase success claim
    return result


def verify_captured(path,spec_sha,manifest_sha):
    """Read-only maintained proof verification; operations never certify themselves."""
    data=read_spec(path,spec_sha,manifest_sha)
    s,e,producer,request,q,job,claim,held=data
    final=paths(s,spec_sha)['result'];terminal=paths(s,spec_sha)['native']
    verified=e.service_module(producer,'phase_observer_proof').verify_observer(
        {'path':str(final),'sha256':sha(final)}, producer_request={'path':str(request),'sha256':sha(request)},
        producer_native=dict(producer['native_contract'],JobId=job),
        terminal_native={'path':str(terminal),'sha256':sha(terminal)})
    result=verified['result']
    need(result['producer_execution']==s['producer_execution']
         and result['producer_request']==s['producer_request']
         and result['producer_job_id']==job,'Maintained observer proof returned another producer')
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['plan','submit-once','verify-ready','capture','verify-captured'])
    parser.add_argument('spec_path');parser.add_argument('spec_sha');parser.add_argument('ops_manifest_sha');args=parser.parse_args()
    if args.operation=='plan':
        s,e,p,*_=read_spec(args.spec_path,args.spec_sha,args.ops_manifest_sha)
        result=dict(argv=argv(s,p,args.spec_path,args.spec_sha,args.ops_manifest_sha),submitted=False)
    elif args.operation=='capture':
        capture(args.spec_path,args.spec_sha,args.ops_manifest_sha)
        return  # final fsynced capture commit is the last operation
    else:result={'submit-once':submit_once,'verify-ready':verify_ready_for_release,'verify-captured':verify_captured}[args.operation](args.spec_path,args.spec_sha,args.ops_manifest_sha)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
