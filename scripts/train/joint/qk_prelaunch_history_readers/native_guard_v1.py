"""Phase-specific full parser adapted from sealed GPU smoke v2; no ML."""
import re
from pathlib import Path
from datetime import datetime, timezone
REMOTE=Path('/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/full-v1')
REQUEST_NAME='elmo-large314m-qknorm-swiglu-rmsnorm-full-v1'
CAPTURE_NAME='elmo-native-qknorm-swiglu-rmsnorm-full-v1'

def need(ok,message):
    if not ok:raise ValueError(message)

def fields(raw):
    pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)',raw)
    need(len(pairs)==len(dict(pairs)),'Duplicate native fields')
    return dict(pairs)

def native_time(value):
    need(isinstance(value,str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d',value),'Missing exact native timestamp')
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)

def same_attempt(*rows,terminal=None):
    need(rows and len({r.get('JobId') for r in rows})==1,'Native job identities differ')
    submitted=[native_time(r.get('SubmitTime')) for r in rows]
    need(len(set(submitted))==1,'Native SubmitTime differs across accepted attempt')
    active=[r for r in rows if r.get('JobState') in {'RUNNING','COMPLETED'}]
    starts=[native_time(r.get('StartTime')) for r in active]
    need(not starts or len(set(starts))==1 and starts[0]>=submitted[0],'Native StartTime differs across accepted attempt')
    if terminal is not None:
        need(terminal in rows and terminal.get('JobState')=='COMPLETED','Successful same-attempt terminal required')
        need(native_time(terminal.get('EndTime'))>=native_time(terminal.get('StartTime')),'Terminal interval is invalid')

def no_owned_steps(raw,job):
    need(isinstance(raw,str) and len(raw.encode())<=1<<20,'Invalid complete native step output')
    if not raw.strip() or raw.strip()=='No job steps exist in the system.':return
    for line in raw.splitlines():
        row=fields(line);step=row.get('StepId')
        need(step is not None,'Unparseable complete native step inventory')
        need(not(step==job or step.startswith(job+'.') or step.startswith(job+'_')),'Owned worker or array-alias step remains')

def release_observation(record,job,request_sha,terminal):
    need(record.get('schema')=='native-owned-step-absence-v1' and record.get('parent_job_id')==job
         and record.get('request_sha256')==request_sha and record.get('uid')==41003 and record.get('host')=='lambda'
         and record.get('argv')==['/usr/local/bin/scontrol','show','step','-o']
         and record.get('exit_code')==0 and record.get('stderr')=='','Actual complete native step query required')
    at=datetime.fromisoformat(record['observed_at_utc'])
    need(at.tzinfo is not None and at>=native_time(terminal.get('EndTime')),'Step absence must be observed after terminal completion')
    no_owned_steps(record['stdout'],job)

def check_native(raw,*,job,state,capture=False,parent=None,cpu_source=None,request_sha=None):
    f=fields(raw)
    if cpu_source:
        name,cpus,mem,limit=cpu_source
    else:
        name,cpus,mem,limit=(CAPTURE_NAME,'2','1G','00:10:00') if capture else (REQUEST_NAME,'32','192G','7-00:00:00')
    need(f.get('JobId')==str(job) and f.get('JobName')==name and f.get('UserId')=='paphiwetsa3(41003)'
         and f.get('Account')=='loaner' and f.get('Partition')=='rl2' and f.get('ReqNodeList')=='gpu21'
         and f.get('NumNodes') in {'1','1-1'} and f.get('NumTasks')=='1' and f.get('NumCPUs')==cpus
         and f.get('MinMemoryNode') in {mem,str(int(mem[:-1])*1024)+'M',str(int(mem[:-1])*1024)}
         and f.get('TimeLimit')==limit and f.get('JobState')==state
         and f.get('Requeue')=='0' and f.get('Restarts')=='0','Wrong owned native name/node/resources/state')
    native_time(f.get('SubmitTime'))
    if state=='PENDING':
        need((parent and re.search(r'(?:^|,)afterany:'+re.escape(str(parent))+r'(?:\(|,|$)',f.get('Dependency','')))
             if capture else f.get('Reason')=='JobHeldUser','Exact held acceptance/afterany required')
    if state in {'RUNNING','COMPLETED'}:
        need(f.get('NodeList')=='gpu21' and f.get('NumNodes')=='1','Wrong actual native node')
        need(native_time(f.get('StartTime'))>=native_time(f.get('SubmitTime')),'Invalid actual start time')
    if state=='COMPLETED':need(f.get('ExitCode')=='0:0' and native_time(f.get('EndTime'))>=native_time(f.get('StartTime')),'True successful terminal proof required')
    gpu=not capture and not cpu_source
    for key in ('ReqTRES','AllocTRES'):
        parts=f.get(key,'').split(',')
        if state in {'RUNNING','COMPLETED'} or key=='ReqTRES':
            need('cpu='+cpus in parts and 'node=1' in parts and any(x in parts for x in {'mem='+mem,'mem='+str(int(mem[:-1])*1024)+'M'}),'Missing exact native TRES')
        for part in parts:
            if 'gpu' in part.lower():need(gpu and part in {'gres/gpu=4','gres/gpu:h100=4'},'Unexpected GPU resources')
    if gpu:
        need(f.get('TresPerNode')=='gres/gpu:h100:4',
             'Exact four-H100 native request required')
        if 'JOB_GRES' in f:
            need(f['JOB_GRES']=='gpu:h100:4','Conflicting native job GPU request')
        if state in {'RUNNING','COMPLETED'}:
            need(f.get('JOB_GRES')=='gpu:h100:4','Actual native job GPU request missing')
        # Native terminal records can clear released device indices. Completion
        # separately requires the same job's saved RUNNING admission below.
        if state == 'RUNNING' or (state == 'COMPLETED' and f.get('GRES')):
            actual=re.fullmatch(r'gpu:h100:4\(IDX:([0-9,-]+)\)',f.get('GRES',''))
            need(actual is not None,'Actual four-H100 allocation missing')
            indices=[]
            for item in actual[1].split(','):
                ends=[int(x) for x in item.split('-')]
                need(len(ends) in (1,2) and ends[-1]>=ends[0],'Invalid native GPU indices')
                indices.extend(range(ends[0],ends[-1]+1))
            need(len(indices)==len(set(indices))==4,'Actual four distinct allocated GPU indices required')
    if request_sha is not None:
        need(f.get('Command')==str(REMOTE/('capture-native.sbatch' if capture else 'full.sbatch')),
             'Immutable full native command changed')
        need(f.get('Comment')==('elmo-full-observer:'+request_sha+':'+str(parent) if capture else 'elmo-full:'+request_sha),
             'Immutable full native request comment changed')
    if not gpu:
        for key,value in f.items():
            if key.lower().startswith(('gres','job_gres','tresper')):
                need(value.lower() in {'','(null)','n/a','none','0'} or
                     (key=='TresPerTask' and value=='cpu:'+cpus),'CPU observer has auxiliary GPU resources')
    need(f.get('QOS')=='loaner-normal','Changed QoS')
    return f
