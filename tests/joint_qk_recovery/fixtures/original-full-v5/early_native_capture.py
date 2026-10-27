"""CPU-only native evidence capture before slow NFS closure checks.

Raw observation and authenticated provenance are separate immutable artifacts.
No missing/error query becomes terminal proof, zero-work proof or completion.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path('/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/full-v5')
CLIENT='/usr/local/bin/scontrol'
LIMIT=1<<20
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def need(value,message):
    if not value:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw.encode()).hexdigest()


def write(path,value,*,text=False):
    with Path(path).open('x') as stream:
        stream.write(value if text else json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def query(argv):
    start=datetime.now(timezone.utc).isoformat();failure=None
    try:
        call=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        stdout,stderr,code=call.stdout,call.stderr,call.returncode
    except subprocess.TimeoutExpired as error:
        decode=lambda value:value.decode(errors='replace') if isinstance(value,bytes) else value or ''
        stdout,stderr,code=decode(error.stdout),decode(error.stderr),None
        failure='NATIVE_QUERY_TIMEOUT'
    except OSError as error:
        stdout,stderr,code='','',None;failure=type(error).__name__+':'+str(error.errno)
    truncated=len(stdout.encode())>LIMIT or len(stderr.encode())>65536
    stdout,stderr=stdout.encode()[:LIMIT].decode(errors='replace'),stderr.encode()[:65536].decode(errors='replace')
    return dict(argv=argv,start_utc=start,end_utc=datetime.now(timezone.utc).isoformat(),
                exit_code=code,stdout=stdout,stderr=stderr,failure=failure,output_truncated=truncated)


def collect_queries(parent,capture,scratch):
    # Do not access campaign claims/manifests/modules before these observations.
    result={}
    for key,argv in [('parent',[CLIENT,'show','job',parent,'-o','-dd']),
                     ('steps',[CLIENT,'show','step','-o']),
                     ('observer',[CLIENT,'show','job',capture,'-o','-dd'])]:
        result[key]=query(argv)
        # A later query or shared-filesystem failure cannot discard parent raw.
        write(Path(scratch)/(key+'.query.json'),result[key])
    return result


def collect(stage,parent,request_sha,files_sha,capture):
    need(Path(stage)==ROOT and parent.isdecimal() and capture.isdecimal() and parent!=capture,'Exact syntactic capture binding required')
    need(re.fullmatch('[0-9a-f]{64}',request_sha) and re.fullmatch('[0-9a-f]{64}',files_sha),'Exact immutable hashes required')
    need(os.getuid()==41003 and os.environ.get('SLURMD_NODENAME')=='gpu21'
         and os.environ.get('SLURM_JOB_ID')==capture and os.environ.get('SLURM_RESTART_COUNT','0')=='0','Exact owned CPU observer allocation required')
    # Private working files remain on any publication/authentication/cleanup error.
    scratch=Path(tempfile.mkdtemp(prefix='elmo-native-'+capture+'-',dir='/tmp'))
    os.chmod(scratch,0o700)
    print('EARLY_NATIVE_PRIVATE_DIR='+str(scratch),file=sys.stderr,flush=True)
    observed=collect_queries(parent,capture,scratch)
    payloads={f'native-{parent}.txt':observed['parent']['stdout'],
              f'native-{parent}.stderr':observed['parent']['stderr'],
              f'native-{parent}.exit':str(observed['parent']['exit_code'])+'\n',
              f'capture-running-{capture}.txt':observed['observer']['stdout']}
    steps=dict(schema='native-owned-step-absence-v1',parent_job_id=parent,request_sha256=request_sha,
               uid=os.getuid(),host='lambda',observed_at_utc=observed['steps']['end_utc'],
               argv=observed['steps']['argv'],exit_code=observed['steps']['exit_code'],
               stdout=observed['steps']['stdout'],stderr=observed['steps']['stderr'])
    payloads[f'native-{parent}.steps.json']=json.dumps(steps,indent=2,sort_keys=True,allow_nan=False)+'\n'
    record=dict(schema='early-full-native-observation-v1',status='RAW_UNAUTHENTICATED_NO_TERMINAL_AUTHORITY',
        parent_job_id=parent,observer_job_id=capture,request_sha256=request_sha,files_manifest_sha256=files_sha,
        uid=os.getuid(),node=os.environ.get('SLURMD_NODENAME'),queries=observed,
        canonical_sha256={name:digest(value) for name,value in payloads.items()},scratch=str(scratch),
        zero_work_proven=False,production_completion_proven=False)
    # Save a recoverable private copy before potentially slow shared writes.
    write(scratch/'raw-observation.json',record)
    for name,value in payloads.items():write(scratch/name,value,text=True)
    need(ROOT.resolve(strict=True)==ROOT and ROOT.stat().st_uid==41003,'Unsafe native destination root')
    for name,value in payloads.items():write(ROOT/name,value,text=True)
    write(ROOT/f'native-{parent}.early.json',record)
    print(str(scratch),flush=True)
    return record


def bounded_read(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=LIMIT,'Unsafe/oversized native proof')
    return path.read_text()


def verify_record(record,parent,capture,request_sha,files_sha,read,api):
    need(record['schema']=='early-full-native-observation-v1' and record['status']=='RAW_UNAUTHENTICATED_NO_TERMINAL_AUTHORITY'
         and record['parent_job_id']==parent and record['observer_job_id']==capture
         and record['request_sha256']==request_sha and record['files_manifest_sha256']==files_sha
         and record['uid']==41003 and record['node']=='gpu21','Wrong raw observation identity')
    expected={'parent':[CLIENT,'show','job',parent,'-o','-dd'],
              'steps':[CLIENT,'show','step','-o'],
              'observer':[CLIENT,'show','job',capture,'-o','-dd']}
    need(set(record['queries'])==set(expected),'Incomplete early native query closure')
    for key,argv in expected.items():
        row=record['queries'][key]
        need(row['argv']==argv and row['exit_code']==0 and row['stderr']=='' and row['failure'] is None and row.get('output_truncated') is False,'Actual native query failed/unknown/truncated; raw is not terminal proof')
        start=datetime.fromisoformat(row['start_utc']);end=datetime.fromisoformat(row['end_utc'])
        need(start.tzinfo is not None and end.tzinfo is not None and end>=start
             and end<=datetime.now(timezone.utc)+timedelta(seconds=5),'Invalid/future actual query interval')
    names={f'native-{parent}.txt',f'native-{parent}.stderr',f'native-{parent}.exit',
           f'native-{parent}.steps.json',f'capture-running-{capture}.txt'}
    need(set(record['canonical_sha256'])==names,'Changed canonical early evidence set')
    for name,sha in record['canonical_sha256'].items():need(digest(read(ROOT/name))==sha,'Changed early raw bytes')
    need(read(ROOT/f'native-{parent}.txt')==record['queries']['parent']['stdout']
         and read(ROOT/f'capture-running-{capture}.txt')==record['queries']['observer']['stdout']
         and read(ROOT/f'native-{parent}.stderr')=='' and read(ROOT/f'native-{parent}.exit')=='0\n','Raw native files differ from actual query')
    claim=ROOT/'claims'/request_sha
    need(read(claim/'PARENT_JOB_ID').strip()==parent and read(claim/'CAPTURE_JOB_ID').strip()==capture,'Wrong accepted claim IDs')
    for name,jid in [('parent-sbatch.stdout',parent),('capture-sbatch.stdout',capture),('parent-sbatch.exit','0'),('capture-sbatch.exit','0')]:
        need(read(claim/name).strip()==jid,'Actual accepted query/claim outcome differs')
    prepared=json.loads(read(claim/'PREPARED.json'))
    need(prepared['request_sha256']==request_sha and prepared['manifest_sha256']==files_sha,'Prepared request/code closure changed')
    release=json.loads(read(claim/'RELEASE_ATTEMPT.json'))
    need(release.get('parent')==parent and release.get('capture')==capture and release.get('request_sha256')==request_sha,'Wrong immutable release binding')
    parent_raw=record['queries']['parent']['stdout'];parent_fields=api.fields(parent_raw)
    need(parent_fields.get('JobState') in TERMINAL,'Parent is not genuinely terminal')
    terminal=api.check_native(parent_raw,job=parent,state=parent_fields['JobState'],request_sha=request_sha)
    need(terminal.get('NodeList')=='gpu21' and terminal.get('NumNodes')=='1'
         and api.native_time(terminal['EndTime'])>=api.native_time(terminal['StartTime'])>=api.native_time(terminal['SubmitTime'])
         and re.fullmatch('[0-9]+:[0-9]+',terminal.get('ExitCode','')),'Actual terminal node/interval/exit absent')
    active=api.check_native(record['queries']['observer']['stdout'],job=capture,state='RUNNING',capture=True,parent=parent,request_sha=request_sha)
    parent_end=api.native_time(terminal['EndTime'])
    need(api.native_time(active['StartTime'])>=parent_end,'Observer started before parent terminal; not genuine afterany')
    for row in record['queries'].values():
        need(datetime.fromisoformat(row['start_utc'])>=parent_end,'Native query predates parent terminal')
    need(datetime.fromisoformat(record['queries']['parent']['end_utc'])<=datetime.fromisoformat(record['queries']['steps']['start_utc'])
         and datetime.fromisoformat(record['queries']['steps']['end_utc'])<=datetime.fromisoformat(record['queries']['observer']['start_utc'])
         and datetime.fromisoformat(record['queries']['observer']['end_utc'])>=api.native_time(active['StartTime']),
         'Early native query order/observer interval differs')
    api.same_attempt(*[api.check_native(read(claim/name),job=parent,state='PENDING',request_sha=request_sha) for name in ['HELD_NATIVE.txt','RELEASE_HELD_NATIVE.txt']],terminal)
    api.same_attempt(*[api.check_native(read(claim/name),job=capture,state='PENDING',capture=True,parent=parent,request_sha=request_sha) for name in ['CAPTURE_PENDING_NATIVE.txt','RELEASE_CAPTURE_PENDING_NATIVE.txt']],active)
    steps=json.loads(read(ROOT/f'native-{parent}.steps.json'))
    need(steps['stdout']==record['queries']['steps']['stdout'] and steps['stderr']==record['queries']['steps']['stderr']
         and steps['observed_at_utc']==record['queries']['steps']['end_utc'],'Canonical complete steps differs')
    api.release_observation(steps,parent,request_sha,terminal)
    return dict(schema='authenticated-early-full-native-observation-v1',status='AUTHENTICATED_TERMINAL_EVIDENCE_ONLY',
        parent_job_id=parent,observer_job_id=capture,parent_state=terminal['JobState'],parent_exit_code=terminal['ExitCode'],
        parent_start=terminal['StartTime'],parent_end=terminal['EndTime'],request_sha256=request_sha,
        files_manifest_sha256=files_sha,canonical_sha256=record['canonical_sha256'],
        zero_work_proven=False,production_completion_proven=False,observer_own_terminal='UNKNOWN_NOT_INFERRED')


def authenticate(stage,parent,request_sha,files_sha,capture):
    need(Path(stage)==ROOT and os.getuid()==41003 and os.environ.get('SLURM_JOB_ID')==capture,'Exact observer required')
    need(ROOT.resolve(strict=True)==ROOT and ROOT.stat().st_uid==41003,'Unsafe authentication destination root')
    # Shell has already checked every member. Rebind this reader and native parser
    # explicitly before importing the reviewed native parser; never import ML.
    manifest=bounded_read(ROOT/'FILES.sha256');need(digest(manifest)==files_sha,'Static manifest changed')
    members={name:sha for sha,name in (line.split('  ',1) for line in manifest.splitlines())}
    for name in ['early_native_capture.py','native_guard.py','capture-native.sbatch']:
        need(members.get(name)==digest(bounded_read(ROOT/name)),'Authentication code not in verified static closure')
    request=ROOT/'ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v5.json'
    need(digest(bounded_read(request))==request_sha,'Actual request changed')
    spec=importlib.util.spec_from_file_location('exact_early_native_guard',ROOT/'native_guard.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    need(api.REMOTE==ROOT,'Native parser namespace differs')
    raw=bounded_read(ROOT/f'native-{parent}.early.json');record=json.loads(raw)
    result=verify_record(record,parent,capture,request_sha,files_sha,bounded_read,api)
    result['early_observation_sha256']=digest(raw)
    write(ROOT/f'native-{parent}.authenticated.json',result)
    print(json.dumps(result,sort_keys=True),flush=True)
    return result


def reopen_authenticated(stage,parent,capture,request_sha,files_sha,api):
    """Read-only consumer: early canonical files alone never supply authority."""
    stage=Path(stage).resolve(strict=True)
    need(api.REMOTE==ROOT,'Consumer native parser namespace differs')
    def read(logical):
        logical=Path(logical)
        need(logical.is_relative_to(ROOT) and '..' not in logical.parts,'Capture binding escaped original namespace')
        return bounded_read(stage/logical.relative_to(ROOT))
    raw=read(ROOT/f'native-{parent}.early.json')
    expected=verify_record(json.loads(raw),parent,capture,request_sha,files_sha,read,api)
    expected['early_observation_sha256']=digest(raw)
    actual=json.loads(read(ROOT/f'native-{parent}.authenticated.json'))
    need(actual==expected,'Missing/changed authenticated capture commit; raw native files are insufficient')
    return actual


if __name__=='__main__':
    mode,stage,parent,request_sha,files_sha=sys.argv[1:]
    capture=os.environ.get('SLURM_JOB_ID','')
    need(mode in ('collect','authenticate'),'Exact capture phase required')
    (collect if mode=='collect' else authenticate)(stage,parent,request_sha,files_sha,capture)
