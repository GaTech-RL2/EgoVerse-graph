"""Pure reader of the exact 412831/412832 full-v5 afterany protocol.

This does not execute the original wrapper or issue queries. Fresh native,
owned-step and same-identity claim censuses remain mandatory at admission.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from scripts.train.joint.qk_recovery_profile import PROFILE, ROOT, PARENT_ID, OBSERVER_ID, REQUEST_SHA, FILES_SHA, validate_parent, need
from scripts.train.joint.native_interruption import fields, native_identity, TERMINAL

SCHEMA = 'native-elmo-qk-original-observer-v2'
CODE_PINS = {
    'capture_script': ('capture-native.sbatch', PROFILE['capture_script_sha256']),
    'submitter': ('submit_full_once.py', PROFILE['submitter_sha256']),
    'guard': ('guard_full.py','93bc99eb0e344ff9e45825227926d3fce1e3e50456a5ac8abbf202f6b5f34eae'),
    'native_guard': ('native_guard.py','a310af36f89e0aa7386edc0ac8a3b06c28ad0969746026c23f2f03b92a2c14e5'),
    'parent_script': ('full.sbatch','b0e899f0670d4a08789ccb7cd7deb1a716146ac5e9bc81fc58949d3244c282ec'),
    'early_reader': ('early_native_capture.py','85f0cae29cf487f7118fca26588797a99702251f0b2696586564f258dd58e3ab'),
    'hash_compat': ('host_hashlib_compat.py','257df596faafd7d3855547d5417d8f2cdd465a413e92af602643e89744561efe'),
}


def proof_paths():
    root=Path(ROOT);claim=root/'claims'/REQUEST_SHA
    paths={key:root/relative for key,(relative,_) in CODE_PINS.items()}
    paths.update(early_observation=root/('native-'+PARENT_ID+'.early.json'),
        authenticated_commit=root/('native-'+PARENT_ID+'.authenticated.json'),files_manifest=root/'FILES.sha256', prepared=claim/'PREPARED.json',
        parent_id=claim/'PARENT_JOB_ID',observer_id=claim/'CAPTURE_JOB_ID',
        request_claim=root/'request-claims'/PROFILE['parent_name']/'REQUEST_SHA256',
        held_native=claim/'HELD_NATIVE.txt',release_held_native=claim/'RELEASE_HELD_NATIVE.txt',
        pending_native=claim/'CAPTURE_PENDING_NATIVE.txt',release_pending_native=claim/'RELEASE_CAPTURE_PENDING_NATIVE.txt',
        release_attempt=claim/'RELEASE_ATTEMPT.json',running_admission=claim/'RUNNING_ADMISSION.json',
        observer_running=root/('capture-running-'+OBSERVER_ID+'.txt'),
        terminal_native=root/('native-'+PARENT_ID+'.txt'),query_exit=root/('native-'+PARENT_ID+'.exit'),
        query_stderr=root/('native-'+PARENT_ID+'.stderr'),steps=root/('native-'+PARENT_ID+'.steps.json'))
    for label in ('parent-sbatch','capture-sbatch'):
        for key,suffix in [('argv','argv.json'),('stdout','stdout'),('stderr','stderr'),('exit','exit')]:
            paths[label.replace('-','_')+'_'+key]=claim/(label+'.'+suffix)
    return paths


def timestamp(value):
    need(isinstance(value,str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d',value),
         'Exact native timestamp required')
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def expected_native(capture=False):
    return dict(JobId=OBSERVER_ID if capture else PARENT_ID,
        JobName=PROFILE['observer_name' if capture else 'parent_name'],UserId='paphiwetsa3(41003)',
        Account='loaner',QOS='loaner-normal',Partition='rl2',ReqNodeList='gpu21',NodeList='gpu21',
        NumCPUs='2' if capture else '32',MinMemoryNode='1G' if capture else '192G',
        TimeLimit='00:10:00' if capture else '7-00:00:00',Requeue='0',Restarts='0',
        Command=ROOT+('/capture-native.sbatch' if capture else '/full.sbatch'),
        TresPerNode='' if capture else 'gres/gpu:h100:4')


def native(raw,state,*,capture=False):
    expected=expected_native(capture);f=fields(raw)
    comment=('elmo-full-observer:'+REQUEST_SHA+':'+PARENT_ID if capture else 'elmo-full:'+REQUEST_SHA)
    need(f.get('Comment')==comment and f.get('WorkDir')==ROOT,'Original comment/workdir changed')
    for suffix in ('Out','Err'):
        ext='out' if suffix=='Out' else 'err'
        need(f.get('Std'+suffix)==ROOT+'/logs/'+expected['JobName']+'-'+expected['JobId']+'.'+ext,
             'Original native log namespace changed')
    timestamp(f.get('SubmitTime'))
    if state=='PENDING':
        need(f.get('JobState')==state and all(f.get(k,'')==v for k,v in expected.items() if k!='NodeList')
             and f.get('NodeList','') in {'','(null)','gpu21'} and f.get('NumNodes') in {'1','1-1'}
             and f.get('NumTasks')=='1' and f.get('StartTime') in {None,'Unknown','N/A'},'Wrong original pending identity')
        if capture:
            need(re.fullmatch(r'afterany:'+PARENT_ID+r'\((?:unfulfilled|unsatisfied)\)',f.get('Dependency','')),
                 'Original afterany dependency differs')
        else:need(f.get('Reason')=='JobHeldUser','Original producer was not held')
        parts=f.get('ReqTRES','').split(',');mem=expected['MinMemoryNode']
        need('cpu='+expected['NumCPUs'] in parts and 'node=1' in parts and
             bool(set(parts)&{'mem='+mem,'mem='+str(int(mem[:-1])*1024)+'M'}),'Original held TRES changed')
        need(f.get('AllocTRES','') in {'','(null)'} and f.get('GRES','') in {'','(null)'},'Pending already allocated')
    else:
        need(state=='RUNNING' or state in TERMINAL,'Unrecognized original native state')
        native_identity(raw,expected,state)
        need(timestamp(f.get('StartTime'))>=timestamp(f['SubmitTime']),'Original start precedes submission')
        if state in TERMINAL:need(timestamp(f.get('EndTime'))>=timestamp(f['StartTime']),'Invalid original terminal interval')
    if not capture and 'JOB_GRES' in f:
        need(f['JOB_GRES']=='gpu:h100:4','Original held/active JOB_GRES conflicts')
    for key,value in f.items():
        if not capture and key not in {'ReqTRES','AllocTRES','TresPerNode','JOB_GRES','GRES'} and (key.lower().startswith('tresper') or 'gpu' in key.lower()):
            need('gpu' not in key.lower() and 'gpu' not in value.lower(),'Original auxiliary GPU resources conflict')
            if key=='TresPerTask':need(value=='cpu:32','Original auxiliary CPU resources conflict')
        if key in {'ReqTRES','AllocTRES'}:
            gpu={p for p in value.split(',') if 'gpu' in p.lower()}
            need(gpu<=({'gres/gpu=4','gres/gpu:h100=4'} if not capture else set()),'Original GPU TRES conflict')
        if capture and (key.lower().startswith(('gres','job_gres','tresper')) or 'gpu' in key.lower()):
            need(value.lower() in {'','(null)','none','n/a','0'} or key=='TresPerTask' and value=='cpu:2',
                 'Original observer owns GPU resources')
    return f


def accepted_argv(capture=False):
    if capture:
        return ['/usr/local/bin/sbatch','--parsable','--no-requeue','--account=loaner','--qos=loaner-normal','--partition=rl2',
            '--job-name='+PROFILE['observer_name'],'--nodelist=gpu21','--nodes=1','--ntasks=1','--cpus-per-task=2',
            '--mem=1G','--gres=none','--time=00:10:00','--export=NONE','--dependency=afterany:'+PARENT_ID,
            '--comment=elmo-full-observer:'+REQUEST_SHA+':'+PARENT_ID,'--chdir='+ROOT,
            '--output='+ROOT+'/logs/%x-%j.out','--error='+ROOT+'/logs/%x-%j.err',ROOT+'/capture-native.sbatch',ROOT,PARENT_ID,REQUEST_SHA,FILES_SHA]
    return ['/usr/local/bin/sbatch','--parsable','--hold','--no-requeue','--account=loaner','--qos=loaner-normal','--partition=rl2',
        '--comment=elmo-full:'+REQUEST_SHA,'--job-name='+PROFILE['parent_name'],'--nodelist=gpu21','--nodes=1','--ntasks=1',
        '--cpus-per-task=32','--mem=192G','--gres=gpu:h100:4','--time=7-00:00:00','--export=NONE','--chdir='+ROOT,
        '--output='+ROOT+'/logs/%x-%j.out','--error='+ROOT+'/logs/%x-%j.err',ROOT+'/full.sbatch',ROOT,REQUEST_SHA,FILES_SHA]


def validate(q,reader):
    parent=reader(q['parent_request']);validate_parent(parent)
    need(q['parent_request']==dict(path=PROFILE['request_path'],sha256=REQUEST_SHA) and str(q['parent_job_id'])==PARENT_ID,
         'Only original412831 has this observer protocol')
    paths=proof_paths();proof=q['original_observer']
    need(proof.get('schema')==SCHEMA and set(proof)==set(paths)|{'schema'},'Complete full-v5 original protocol required')
    raw={}
    for key,path in paths.items():
        b=proof[key]
        need(set(b)=={'path','sha256'} and b['path']==str(path) and re.fullmatch('[0-9a-f]{64}',b['sha256']),
             'Wrong exact original proof binding: '+key)
        value=reader(b,False)
        need(isinstance(value,str) and len(value.encode())<=4*1024*1024 and hashlib.sha256(value.encode()).hexdigest()==b['sha256'],
             'Original proof bytes changed or exceed metadata bound')
        raw[key]=value
    need(proof['files_manifest']['sha256']==FILES_SHA and proof['running_admission']==q['parent_running_admission'],
         'Original manifest/running evidence changed')
    entries={}
    for line in raw['files_manifest'].splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
        need(m and m[2] not in entries and not Path(m[2]).is_absolute() and '..' not in Path(m[2]).parts,'Unsafe original manifest')
        entries[m[2]]=m[1]
    for key,(name,digest) in CODE_PINS.items():
        need(proof[key]['sha256']==digest==entries.get(name),'Original wrapper code pin differs: '+key)
    need(entries.get(str(Path(PROFILE['request_path']).relative_to(ROOT)))==REQUEST_SHA,'Original request missing from manifest')
    prepared=json.loads(raw['prepared']);release=json.loads(raw['release_attempt']);running=json.loads(raw['running_admission'])
    need(prepared['request_sha256']==REQUEST_SHA and prepared['manifest_sha256']==FILES_SHA
         and prepared['parent_argv']==accepted_argv() and release['parent']==PARENT_ID and release['capture']==OBSERVER_ID
         and release['request_sha256']==REQUEST_SHA and raw['parent_id'].strip()==PARENT_ID
         and raw['observer_id'].strip()==OBSERVER_ID and raw['request_claim'].strip()==REQUEST_SHA,'Original accepted claim differs')
    for label,capture,job in [('parent_sbatch',False,PARENT_ID),('capture_sbatch',True,OBSERVER_ID)]:
        need(json.loads(raw[label+'_argv'])==accepted_argv(capture) and raw[label+'_stdout'].strip()==job
             and raw[label+'_exit'].strip()=='0','Original submission acceptance/argv unresolved')
        # Slurm warnings are preserved but are not acceptance authority.
    need(running['status']=='ACTUAL_BOUND_FULL_NATIVE_ALLOCATION_PASS' and running['job_id']==PARENT_ID
         and running['request_sha256']==REQUEST_SHA,'Actual full-v5 RUNNING admission required')
    held=native(raw['held_native'],'PENDING');released=native(raw['release_held_native'],'PENDING')
    active=native(running['native'],'RUNNING');state=fields(raw['terminal_native']).get('JobState')
    need(state in TERMINAL,'Original parent is not terminal')
    terminal=native(raw['terminal_native'],state)
    need(q['parent_native']==expected_native(),'Bound original native contract differs')
    need(len({r['SubmitTime'] for r in (held,released,active,terminal)})==1
         and active['StartTime']==terminal['StartTime'],'Original native attempt identity changed')
    pending=native(raw['pending_native'],'PENDING',capture=True)
    pending2=native(raw['release_pending_native'],'PENDING',capture=True)
    observer=native(raw['observer_running'],'RUNNING',capture=True)
    need(len({r['SubmitTime'] for r in (pending,pending2,observer)})==1
         and timestamp(observer['StartTime'])>=timestamp(terminal['EndTime']), 'Original observer attempt/afterany interval differs')
    need(raw['query_exit'].strip()=='0' and not raw['query_stderr'].strip(),'Original parent capture did not succeed')
    steps=json.loads(raw['steps'])
    need(steps.get('schema')=='native-owned-step-absence-v1' and steps.get('parent_job_id')==PARENT_ID
         and steps.get('request_sha256')==REQUEST_SHA and steps.get('uid')==41003 and steps.get('host')=='lambda'
         and steps.get('argv')==['/usr/local/bin/scontrol','show','step','-o']
         and type(steps.get('exit_code')) is int and steps['exit_code']==0 and steps.get('stderr')=='',
         'Actual successful complete native step inventory required')
    at=datetime.fromisoformat(steps['observed_at_utc'])
    need(at.tzinfo is not None and timestamp(terminal['EndTime'])<=at<=datetime.now(timezone.utc), 'Step observation precedes terminal/is future')
    text=steps['stdout'];need(isinstance(text,str) and len(text.encode())<=1<<20,'Step inventory is not bounded')
    if text.strip() and text.strip()!='No job steps exist in the system.':
        for line in text.splitlines():
            sid=fields(line).get('StepId')
            need(sid and re.fullmatch(r'[0-9]+(?:_[0-9]+)?(?:\.[A-Za-z0-9_-]+)?',sid),'Malformed complete native step inventory')
            need(not(sid==PARENT_ID or sid.startswith(PARENT_ID+'.') or sid.startswith(PARENT_ID+'_')),'Original parent step remains')
    from scripts.train.joint.qk_early_capture_proof import verify_commit
    authenticated = verify_commit(raw)
    return raw['terminal_native'],dict(schema='native-elmo-qk-original-observer-validation-v1',
        original_parent_id=PARENT_ID,original_observer_id=OBSERVER_ID,observer_own_terminal='UNKNOWN_NOT_REQUIRED',
        capture_query_exit=0,scientific_parent_state=state,scientific_parent_exit=terminal['ExitCode'],
        authenticated_capture=authenticated,proofs=proof)
