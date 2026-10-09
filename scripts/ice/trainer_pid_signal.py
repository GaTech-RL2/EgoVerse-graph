#!/usr/bin/env python3
"""Single-rank save-only signal adapter; never signal a Slurm step or supervisor."""
from __future__ import annotations
import hashlib,json,os,pathlib,re,signal,subprocess,sys,time

def candidate(pid, *, repo, python, uid, namespace, job, proc=pathlib.Path('/proc')):
    assert pid > 0
    p=proc/str(pid)
    status=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
    assert [int(x) for x in status['Uid'].split()][:2] == [uid,uid]
    assert (p/'ns/pid').stat().st_ino == namespace
    assert re.search(r'(?:^|/)job_'+re.escape(job)+r'(?:/|$)',(p/'cgroup').read_text(),re.M)
    argv=(p/'cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in argv if x]
    assert any(argv[i:i+2]==['-m','egomimic.trainHydra'] for i in range(len(argv)-1))
    assert (p/'cwd').resolve() == repo
    assert (p/'exe').resolve() == python
    bit=1 << (signal.SIGUSR2-1)
    assert int(status['SigCgt'].strip(),16)&bit
    assert not int(status['SigIgn'].strip(),16)&bit
    fields=(p/'stat').read_text().rpartition(') ')[2].split()
    start=int(fields[19]);assert start > 0
    return {'pid':pid,'start_time_ticks':start,'uid':uid,'pid_namespace_inode':namespace,'source_repo':str(repo),'python':str(python),'job_id':job,'handler':'SIGUSR2 caught; not ignored','command':argv}

def main():
    audit = len(sys.argv)==3 and sys.argv[1]=='--audit-only'
    assert audit or (len(sys.argv)==3 and sys.argv[1]=='--signal=USR2'), 'Only exact save-only or audit invocation allowed'
    job=sys.argv[2];assert re.fullmatch(r'[0-9]+',job) and job==os.environ['SLURM_JOB_ID']
    assert os.environ.get('SLURM_STEP_ID') is not None
    assert int(os.environ['SLURM_NTASKS'])==1, 'Single-rank route only'
    repo=pathlib.Path(os.environ['ICE_TRAINER_REPO']).resolve(strict=True)
    python=pathlib.Path(os.environ['ICE_TRAINER_PYTHON']).resolve(strict=True)
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();assert head==os.environ['ICE_TRAINER_HEAD']
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all'],text=True).strip()
    clean={k:v for k,v in os.environ.items() if k!='SLURM_CLUSTERS'}
    output=subprocess.check_output(['scontrol','listpids',job],text=True,env=clean)
    pids=[]
    for line in output.splitlines():
        row=line.split()
        if len(row)>=2 and row[0].isdigit() and row[1]==job: pids.append(int(row[0]))
    assert pids, 'Native scheduler returned no current job PIDs'
    context=dict(repo=repo,python=python,uid=os.getuid(),namespace=pathlib.Path('/proc/self/ns/pid').stat().st_ino,job=job)
    candidates=[]
    for pid in pids:
        try: candidates.append(candidate(pid,**context))
        except (OSError,AssertionError,ValueError,KeyError,IndexError): pass
    assert len(candidates)==1, ('Exactly one verified trainer required',len(candidates))
    proof=candidates[0];proof['source_head']=head;proof['checked_at_unix']=time.time()
    if audit:
        proof['status']='TRAINER_PID_ROUTE_AUDITED'
    else:
        assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'), 'Race-safe Linux PID handle required'
        fd=os.pidfd_open(proof['pid'])
        try:
            current=candidate(proof['pid'],**context)
            assert all(current[k]==proof[k] for k in current), 'Trainer identity changed before signal'
            signal.pidfd_send_signal(fd,signal.SIGUSR2)
        finally: os.close(fd)
        proof['status']='SAVE_ONLY_TRAINER_PID_SIGNALED'
    print(json.dumps(proof,sort_keys=True),flush=True)
if __name__=='__main__': main()
