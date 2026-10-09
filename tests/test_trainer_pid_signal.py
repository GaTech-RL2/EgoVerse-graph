from pathlib import Path
import runpy,signal
import pytest

@pytest.fixture
def fixture(tmp_path):
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/ice/trainer_pid_signal.py'))
    p=tmp_path/'proc'/'123';(p/'ns').mkdir(parents=True);ns=p/'ns/pid';ns.write_text('namespace')
    repo=tmp_path/'repo';repo.mkdir();python=tmp_path/'python';python.touch()
    (p/'cwd').symlink_to(repo);(p/'exe').symlink_to(python)
    bit=1<<(signal.SIGUSR2-1)
    (p/'status').write_text(f'Uid:\t42\t42\t42\t42\nSigCgt:\t{bit:016x}\nSigIgn:\t0000000000000000\n')
    (p/'cgroup').write_text('0::/slurm/uid_42/job_456/step_0\n')
    (p/'cmdline').write_bytes(b'python\0-m\0egomimic.trainHydra\0')
    (p/'stat').write_text('123 (python worker) '+' '.join(['0']*19+['100']+['0']*20))
    kwargs=dict(repo=repo,python=python,uid=42,namespace=ns.stat().st_ino,job='456',proc=tmp_path/'proc')
    return api,p,kwargs

def test_verified_identity(fixture):
    api,p,kw=fixture;assert api['candidate'](123,**kw)['start_time_ticks']==100

@pytest.mark.parametrize('field',['uid','namespace','job','source','python','handler','ignored','command','start'])
def test_fail_closed(fixture,field):
    api,p,kw=fixture
    if field=='uid':kw['uid']=99
    if field=='namespace':kw['namespace']+=1
    if field=='job':kw['job']='999'
    if field=='source':kw['repo']=kw['repo'].parent
    if field=='python':kw['python']=kw['python'].parent
    if field=='handler':(p/'status').write_text('Uid: 42 42 42 42\nSigCgt: 0\nSigIgn: 0\n')
    if field=='ignored':(p/'status').write_text(f"Uid: 42 42 42 42\nSigCgt: {1<<(signal.SIGUSR2-1):x}\nSigIgn: {1<<(signal.SIGUSR2-1):x}\n")
    if field=='command':(p/'cmdline').write_bytes(b'python\0ice_requeue_runner.py\0')
    if field=='start':(p/'stat').write_text('123 (python) '+' '.join(['0']*40))
    with pytest.raises(AssertionError):api['candidate'](123,**kw)

def test_pid_handle_python_api(monkeypatch):
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/ice/trainer_pid_signal.py'))
    import os
    monkeypatch.setattr(os,'pidfd_open',lambda pid: 77 if pid==123 else None,raising=False)
    monkeypatch.setattr(signal,'pidfd_send_signal',lambda fd,sig: (fd,sig),raising=False)
    assert api['open_pidfd'](123)==77
    assert api['send_pidfd'](77,signal.SIGUSR2)==(77,signal.SIGUSR2)

def test_missing_api_rejects_unverified_platform(monkeypatch):
    api=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/ice/trainer_pid_signal.py'))
    import os,sys
    monkeypatch.delattr(os,'pidfd_open',raising=False)
    monkeypatch.setattr(sys,'platform','unverified')
    with pytest.raises(AssertionError,match='ABI'): api['open_pidfd'](123)
