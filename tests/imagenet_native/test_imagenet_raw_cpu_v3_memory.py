"""Synthetic v3 source envelopes only; no real data/native/memory certification.

The original26-case raw chain file and receipt remain historical unchanged.
Only the pending cache-native interface is patched for full raw-reader tests.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.train.joint import imagenet
from scripts.train.joint import imagenet_memory_policy as memory_policy
from test_imagenet_raw_cpu_chain import SyntheticRawChain, digest, write


CGROUP = Path(__file__).parent/'fixtures/observed-skynet-cgroup.conf'
REVIEWED_POLICY_SHA = 'a9ef364c6714e35be0a1937fef46cab2325b4f3e221ddfd2186aa133983518d9'


def observation():
    return dict(schema='skynet-cr-cpu-memory-observation-v1',node='sky1',
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        controller_config='SelectType = select/cons_tres\nSelectTypeParameters = CR_CPU\n'
            'TaskPlugin = task/cgroup,task/affinity\nProctrackType = proctrack/cgroup\n',
        cgroup_config=CGROUP.read_text(),meminfo='MemFree: 5000 kB\nMemAvailable: 250514316 kB\n')


def limits(finite=64*1024**3):
    return memory_policy.cgroup_limits('0::/job/step\n',
        '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
        lambda path:str(finite) if str(path)=='/sys/fs/cgroup/job/memory.max' else 'max')


def replace_values(value,old,new):
    if isinstance(value,dict):return {key:replace_values(item,old,new) for key,item in value.items()}
    if isinstance(value,list):return [replace_values(item,old,new) for item in value]
    return value.replace(old,new) if isinstance(value,str) else value


def rebind_request(fixture):
    """Rebind synthetic claim/file graph, preserving actual semantic checks."""
    old_request,old_files=fixture.qsha,fixture.fsha
    write(fixture.root/'request.json',fixture.q);new_request=digest(fixture.root/'request.json')
    entries={line.split('  ',1)[1]:line.split('  ',1)[0]
        for line in (fixture.root/'FILES.sha256').read_text().splitlines()}
    entries['request.json']=new_request
    entries['memory_policy.py']=digest(fixture.root/'memory_policy.py')
    write(fixture.root/'FILES.sha256',''.join(value+'  '+name+'\n' for name,value in entries.items()))
    new_files=digest(fixture.root/'FILES.sha256')
    for path in fixture.root.rglob('*'):
        if path.is_file() and path.name not in ('request.json','FILES.sha256'):
            text=path.read_text().replace(old_request,new_request).replace(old_files,new_files)
            path.write_text(text)
    new_claim=fixture.root/'claims'/new_request
    if fixture.claim!=new_claim:fixture.claim.rename(new_claim)
    fixture.claim=new_claim
    for key in ('result','capture','completion'):
        value=getattr(fixture,key)
        value=replace_values(replace_values(value,old_request,new_request),old_files,new_files)
        setattr(fixture,key,value)
    fixture.qsha,fixture.fsha=new_request,new_files


def refresh(fixture):
    parent=fixture.bind('memory-policy-123.json');observer=fixture.bind('memory-policy-124.json')
    fixture.result['memory_policy']=parent;fixture.capture['memory_policy']=observer
    fixture.completion['memory_policy']=parent;fixture.completion['observer_memory_policy']=observer
    fixture.refresh()


def v3_fixture(tmp_path):
    fixture=SyntheticRawChain(tmp_path,'v3')
    fixture.q['memory_policy']=copy.deepcopy(memory_policy.POLICY)
    # Copy the reviewed static helper as a fixture closure member; never import
    # anything from the sealed operational packet or execute its test module.
    helper=Path(memory_policy.__file__)
    (fixture.root/'memory_policy.py').write_bytes(helper.read_bytes())
    rebind_request(fixture)
    for path in fixture.root.rglob('*.txt'):
        text=path.read_text()
        if 'JobId=' in text:
            text=text.replace('AllocTRES=cpu=4,mem=24G,','AllocTRES=cpu=4,')
            text=text.replace('AllocTRES=cpu=2,mem=1G,','AllocTRES=cpu=2,')
            path.write_text(text+' Mem=0')
    for job,capture in (('123',False),('124',True)):
        actual=observation()
        write(fixture.root/('memory-policy-'+job+'.json'),dict(schema='allocated-imagenet-memory-policy-v1',
            request_sha256=fixture.qsha,job_id=job,capture=capture,observation=actual,
            summary=memory_policy.check_observation(fixture.q,actual,capture=capture),cgroup=limits(),
            synthetic_fixture_only=True))
    prepared=json.loads((fixture.claim/'PREPARED.json').read_text());prepared['memory']=observation()
    write(fixture.claim/'PREPARED.json',prepared)
    write(fixture.claim/'RELEASE_MEMORY.json',observation())
    fixture.completion.update(memory_semantics='requested_not_consumable_reserved',reserved_memory_mib=None)
    refresh(fixture)
    return fixture


def test_static_memory_policy_has_exact_reviewed_source_hash():
    assert hashlib.sha256(Path(memory_policy.__file__).read_bytes()).hexdigest()==REVIEWED_POLICY_SHA
    assert hashlib.sha256(CGROUP.read_bytes()).hexdigest()==memory_policy.POLICY['cgroup_config_sha256']


def test_exact_v3_cr_cpu_raw_reader_accepts_complete_both_role_memory_records(tmp_path,monkeypatch):
    fixture=v3_fixture(tmp_path);calls=[]
    monkeypatch.setattr(imagenet,'verify_cache_native',lambda *args,**kwargs:calls.append(kwargs))
    assert imagenet.verify_dataset(fixture.request)==({'train':1281167,'val':50000},1000)
    assert calls==[{'allocated':False}]
    for job,capture in (('123',False),('124',True)):
        record=json.loads((fixture.root/('memory-policy-'+job+'.json')).read_text())
        assert record['capture'] is capture and record['request_sha256']==fixture.qsha
        assert record['summary']['reserved_memory_mib'] is None
        assert record['summary']['memory_semantics']=='requested_not_consumable_reserved'
        assert len(record['cgroup']['observations'])==3 and record['cgroup']['effective_finite_limit_bytes']==64*1024**3
    raw=(fixture.root/'native-123.txt').read_text()
    assert 'ReqTRES=cpu=4,mem=24G,node=1,billing=4' in raw
    assert 'AllocTRES=cpu=4,node=1,billing=4' in raw and 'Mem=0' in raw


@pytest.mark.parametrize('capture',[False,True])
@pytest.mark.parametrize('damage',['wrong_job','wrong_request','wrong_role','bad_memavailable','reserved_summary',
    'configured_enforcement','tampered_ancestor','low_ancestor','wrong_config'])
def test_raw_reader_rejects_rehashed_allocated_memory_proof_damage(tmp_path,monkeypatch,capture,damage):
    fixture=v3_fixture(tmp_path);job='124' if capture else '123'
    path=fixture.root/('memory-policy-'+job+'.json');record=json.loads(path.read_text())
    if damage=='wrong_job':record['job_id']='125'
    if damage=='wrong_request':record['request_sha256']='f'*64
    if damage=='wrong_role':record['capture']=not capture
    if damage=='bad_memavailable':
        minimum=memory_policy.POLICY['minimum_memavailable_kib']['observer' if capture else 'producer']
        record['observation']['meminfo']='MemAvailable: '+str(minimum-1)+' kB\n'
    if damage=='reserved_summary':record['summary']['reserved_memory_mib']=1024 if capture else 24576
    if damage=='configured_enforcement':record['summary']['configured_ram_enforcement']=True
    if damage=='tampered_ancestor':
        cgroup=record['cgroup'];cgroup['observations']=[row for row in cgroup['observations'] if row['path']!='/sys/fs/cgroup/job/memory.max']
        cgroup['effective_finite_limit_bytes']=None;cgroup['enforcement']='UNKNOWN_OR_UNLIMITED_NOT_CERTIFIED'
    if damage=='low_ancestor':record['cgroup']=limits(512*1024**2 if capture else 8*1024**3)
    if damage=='wrong_config':record['observation']['cgroup_config']+='ConstrainRAMSpace=yes\n'
    write(path,record);refresh(fixture)
    monkeypatch.setattr(imagenet,'verify_cache_native',lambda *args,**kwargs:pytest.fail('Invalid raw memory proof reached pending cache gate'))
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)


@pytest.mark.parametrize('damage',['wrong_policy','reserved_completion','wrong_completion_semantics','prepared_headroom'])
def test_v3_requested_memory_never_becomes_a_consumable_reservation_claim(tmp_path,monkeypatch,damage):
    fixture=v3_fixture(tmp_path)
    if damage=='wrong_policy':
        fixture.q['memory_policy']['select_type_parameters']='CR_CPU_Memory';rebind_request(fixture)
    if damage=='reserved_completion':fixture.completion['reserved_memory_mib']=24576
    if damage=='wrong_completion_semantics':fixture.completion['memory_semantics']='consumable_reserved'
    if damage=='prepared_headroom':
        path=fixture.claim/'PREPARED.json';prepared=json.loads(path.read_text())
        prepared['memory']['meminfo']='MemAvailable: 1000 kB\n';write(path,prepared)
    refresh(fixture)
    monkeypatch.setattr(imagenet,'verify_cache_native',lambda *args,**kwargs:pytest.fail('Invalid memory policy reached pending cache gate'))
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)


def test_v3_default_raw_chain_stays_stdlib_without_payload_or_torch_reads(tmp_path):
    fixture=v3_fixture(tmp_path)
    code='''import importlib.abc,json,pathlib,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=="torch" or fullname.startswith("torch."):
   raise AssertionError("Raw v3 gateway imported Torch")
sys.meta_path.insert(0,NoTorch())
from scripts.train.joint import imagenet,contract
request,blocked=json.loads(sys.stdin.read());blocked={pathlib.Path(p) for p in blocked};original=contract.sha
def spy(path):
 assert pathlib.Path(path) not in blocked,"Gateway hashed raw payload"
 return original(path)
contract.sha=spy
original_open=pathlib.Path.open
def open_guard(self,*args,**kwargs):
 assert self not in blocked,"Gateway opened raw payload"
 return original_open(self,*args,**kwargs)
pathlib.Path.open=open_guard
calls=[]
imagenet.verify_cache_native=lambda *args,**kwargs:calls.append(kwargs)
assert imagenet.verify_dataset(request)==({"train":1281167,"val":50000},1000)
assert calls==[{"allocated":False}] and "torch" not in sys.modules
assert not any(k.startswith("egomimic.") for k in sys.modules)
print("PASS_SYNTHETIC_V3_MEMORY_CHAIN_ONLY")
'''
    source=Path(imagenet.__file__).resolve().parents[3]
    result=subprocess.run([sys.executable,'-c',code],cwd=source,
        input=json.dumps([fixture.request,[str(path) for path in fixture.blocked]]),
        env=dict(os.environ,PYTHONPATH=str(source)),text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='PASS_SYNTHETIC_V3_MEMORY_CHAIN_ONLY'
