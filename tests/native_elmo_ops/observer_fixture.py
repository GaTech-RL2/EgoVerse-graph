"""Portable synthetic native observer fixtures, never native queries."""
from datetime import datetime,timezone
import os, subprocess
from pathlib import Path

o=support=None

def observer_native(f,state='PENDING',**changes):
    s,e,p,*_=o.read_spec(*f['observer_args']);n=o.native_contract(s,p)
    row=dict(n,JobId=f['observer_id'],JobState=state,Reason='Dependency',NumNodes='1-1',NumTasks='1',NodeList='',
        ReqTRES='cpu=2,mem=1G,node=1,billing=2',AllocTRES='(null)',SubmitTime='2026-10-27T01:00:01',StartTime='Unknown',
        Dependency='afterany:900001(unfulfilled)',Comment='elmo-phase-observer:'+f['observer_args'][1],WorkDir=str(f['root']),
        StdOut=str(f['root']/'logs'/(n['JobName']+'-'+f['observer_id']+'.out')),
        StdErr=str(f['root']/'logs'/(n['JobName']+'-'+f['observer_id']+'.err')))
    if state=='RUNNING':row.update(NodeList=n['NodeList'],NumNodes='1',AllocTRES='cpu=2,mem=1G,node=1,billing=2',StartTime='2026-10-27T02:00:00',Reason='None',Dependency='(null)')
    row.update(changes);return ' '.join(k+'='+v for k,v in row.items())+'\n'

def record(argv,code=0,stdout='',stderr=''):
    return dict(schema='native-elmo-phase-observer-query-v1',argv=argv,returncode=code,stdout=stdout,stderr=stderr,
        observed_at_utc=datetime.now(timezone.utc).isoformat())

def accept(f,monkeypatch):
    monkeypatch.setattr(o,'query',lambda argv:record(argv,stdout=support.native(f) if argv[3]=='900001' else observer_native(f)))
    monkeypatch.setattr(o.subprocess,'run',lambda argv,**kwargs:subprocess.CompletedProcess(argv,0,f['observer_id']+'\n',''))
    return o.submit_once(*f['observer_args'])

def terminal(f,**changes):
    row=support.e.fields(support.native(f,'RUNNING'))
    row.update(JobState='TIMEOUT',Reason='TimeLimit',EndTime='2026-10-27T02:00:00',ExitCode='0:15')
    if row.get('GRES'):row['GRES']=''
    row.update(changes);return ' '.join(k+'='+v for k,v in row.items())+'\n'

def ready_capture_after_accept(f,monkeypatch):
    _,path,q,job,claim,held=support.e.read_binding(*f['args'])
    support.e.write_once(claim/'RUNNING_ADMISSION.json',dict(job_id=job,request_sha256=o.sha(path),spec_sha256=f['args'][1],native=support.native(f,'RUNNING')))
    monkeypatch.setattr(os,'getuid',lambda:41003)
    monkeypatch.setenv('SLURM_JOB_ID',f['observer_id']);monkeypatch.setenv('SLURMD_NODENAME',f['spec']['native_contract']['NodeList'])
    monkeypatch.delenv('SLURM_JOB_GPUS',raising=False);monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    monkeypatch.setattr(o,'query',lambda argv:record(argv,stdout=('900999.batch|RUNNING\n' if argv[0].endswith('squeue') else observer_native(f,'RUNNING') if argv[3]==f['observer_id'] else terminal(f))))
