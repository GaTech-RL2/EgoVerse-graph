"""Synthetic source-CPU v4 proof envelopes, never actual corpus/native proof.

The two beside-test fixtures are exact copied wheel inspection metadata and the
sealed producer helper bytes. The helper is only a never-executed manifest
member: the service must run its own stdlib validator, not import producer code.
No wheel/archive/image bytes or extracted native libraries are needed or read.
The existing source-CPU v1/v2/v3 tests remain separate and unchanged.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.train.joint import imagenet
from scripts.train.joint import imagenet_memory_policy as memory_policy
from scripts.train.joint import imagenet_wheel_overlay as wheel_policy
from test_imagenet_raw_cpu_chain import SyntheticRawChain, digest, write
from test_imagenet_raw_cpu_v3_memory import observation, limits, rebind_request, refresh

FIXTURES = Path(__file__).parent/'fixtures'
MEMBERS = FIXTURES/'pillow-cp311-wheel-members.json'
PRODUCER_HELPER = FIXTURES/'source-cpu-v4-wheel-overlay.py'
MEMBERS_SHA = '41bcda8b0eb02836d8cf1beef291bc3d6f6fa4aeb689c68dda7562f82539a5ce'
PRODUCER_SHA = '8aa731d672b1a034cdfd17e9356cc409a36e5ee68fe8c22e479a6eb1fa36f631'
WHEEL_NAME = 'pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
WHEEL_SHA = 'a625df9a2a5597fa5e5266e01aefc82604bb85d291e3547981e863560d75da6c'
PYTHON = '/coc/flash7/paphiwetsa3/projects/EgoVerse-refactor/.venv/bin/python'
PYTHON_REAL = '/nethome/paphiwetsa3/flash/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11'


def runtime_proof(fixture):
    inspection=json.loads(MEMBERS.read_text())['inspection']
    overlay=fixture.original/'output/pillow-overlay'
    installation=dict(schema='pillow-private-wheel-install-v1',status='EXACT_RECORD_AND_EXTRACTED_BYTES_PASS',
        overlay=str(overlay),wheel_sha256=WHEEL_SHA,metadata_sha256=inspection['metadata_sha256'],
        record_sha256=inspection['record_sha256'],wheel_metadata_sha256=inspection['wheel_metadata_sha256'],
        files=inspection['files'],bytes=inspection['bytes'],records=copy.deepcopy(inspection['records']),
        source_stat_unchanged=True,all_extracted_files_reopened_sha256=True,files_and_directories_fsynced=True,
        pip_used=False,network_used=False,historical_environment_mutated=False)
    return dict(pillow_version='13.0.0',pillow_wheel=copy.deepcopy(fixture.q['pillow_wheel']),
        real_executable=PYTHON_REAL,executable=PYTHON,python_version='3.11.14',
        python_implementation='cpython',system='Linux',machine='x86_64',
        pillow_origin=str(overlay/'PIL/__init__.py'),image_origin=str(overlay/'PIL/Image.py'),
        native_decoder_origin=str(overlay/'PIL/_imaging.cpython-311-x86_64-linux-gnu.so'),
        pillow_installation=installation,synthetic_fixture_only=True)


def v4_fixture(tmp_path):
    fixture=SyntheticRawChain(tmp_path,'v4')
    fixture.q['memory_policy']=copy.deepcopy(memory_policy.POLICY)
    fixture.q['runtime']=dict(python=PYTHON,python_real=PYTHON_REAL,
        python_version='3.11.14',pillow_version='13.0.0')
    old_id=fixture.q['request_id'];fixture.q['request_id']='imagenet-qwen-and-raw-20261026-v4'
    (fixture.root/'request-claims'/old_id).rename(fixture.root/'request-claims'/fixture.q['request_id'])
    fixture.q.update(workers=4,max_inflight=16)
    fixture.q['pillow_wheel']=dict(path=str(fixture.original/'inputs'/WHEEL_NAME),sha256=WHEEL_SHA)
    (fixture.root/'memory_policy.py').write_bytes(Path(memory_policy.__file__).read_bytes())
    (fixture.root/'wheel_overlay.py').write_bytes(PRODUCER_HELPER.read_bytes())
    entries={line.split('  ',1)[1]:line.split('  ',1)[0]
        for line in (fixture.root/'FILES.sha256').read_text().splitlines()}
    del entries['inputs/pillow-fixture.whl']
    entries['inputs/'+WHEEL_NAME]=WHEEL_SHA
    entries['wheel_overlay.py']=digest(fixture.root/'wheel_overlay.py')
    write(fixture.root/'FILES.sha256',''.join(value+'  '+name+'\n' for name,value in entries.items()))
    fixture.blocked.remove(fixture.root/'inputs/pillow-fixture.whl')
    fixture.blocked.add(fixture.root/'inputs'/WHEEL_NAME)
    rebind_request(fixture)
    # CR_CPU has an exact request but no consumable memory reservation. Both
    # actual-role observations below are entirely synthetic fixture payloads.
    for path in fixture.root.rglob('*.txt'):
        raw=path.read_text()
        if 'JobId=' in raw:
            raw=raw.replace('AllocTRES=cpu=4,mem=24G,','AllocTRES=cpu=4,')
            raw=raw.replace('AllocTRES=cpu=2,mem=1G,','AllocTRES=cpu=2,')
            if path.name=='capture-running-124.txt':
                raw=raw.replace('StartTime=2026-10-26T23:00:00','StartTime=2026-10-26T23:30:01')
            path.write_text(raw+' Mem=0')
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
    write(fixture.root/'output/runtime.json',runtime_proof(fixture))
    refresh(fixture)
    return fixture


def reject_before_cache(monkeypatch):
    monkeypatch.setattr(imagenet,'verify_cache_native',lambda *args,**kwargs:
        pytest.fail('Invalid synthetic source-CPU v4 reached pending target cache gate'))


def test_fixtures_are_exact_sealed_metadata_and_never_executed_producer_bytes():
    assert digest(MEMBERS)==MEMBERS_SHA
    assert digest(PRODUCER_HELPER)==PRODUCER_SHA
    fixture=json.loads(MEMBERS.read_text())
    assert fixture['actual_linux_import_proven'] is False
    assert fixture['inspection']['files']==139 and fixture['inspection']['bytes']==20385837


def test_v4_complete_synthetic_envelope_accepts_only_pending_external_cache(tmp_path,monkeypatch):
    fixture=v4_fixture(tmp_path);calls=[]
    monkeypatch.setattr(imagenet,'verify_cache_native',lambda *args,**kwargs:calls.append(kwargs))
    assert imagenet.verify_dataset(fixture.request)==({'train':1281167,'val':50000},1000)
    assert calls==[{'allocated':False}]
    runtime=json.loads((fixture.root/'output/runtime.json').read_text())
    # The general source-owned helper preserves the exact cp311 default. It is
    # semantic replay of metadata, not a new extraction or native import proof.
    assert wheel_policy.validate_runtime(runtime,fixture.q['pillow_wheel'],
        fixture.original/'output/pillow-overlay',PYTHON_REAL)==runtime
    assert not any(path.exists() for path in fixture.blocked)


@pytest.mark.parametrize('damage',['missing_helper','changed_helper','wrong_wheel_sha','cp312_wheel',
    'wrong_wheel_path','wrong_q_python','wrong_q_python_real','wrong_q_python_version'])
def test_v4_rejects_rehashed_wrong_producer_or_request_contract(tmp_path,monkeypatch,damage):
    fixture=v4_fixture(tmp_path)
    if damage in ('missing_helper','changed_helper'):
        entries={line.split('  ',1)[1]:line.split('  ',1)[0]
            for line in (fixture.root/'FILES.sha256').read_text().splitlines()}
        if damage=='missing_helper':
            del entries['wheel_overlay.py']
        else:
            write(fixture.root/'wheel_overlay.py','# SYNTHETIC ALTERED PRODUCER HELPER NEVER EXECUTED\n')
            entries['wheel_overlay.py']=digest(fixture.root/'wheel_overlay.py')
        write(fixture.root/'FILES.sha256',''.join(value+'  '+name+'\n' for name,value in entries.items()))
    if damage=='wrong_wheel_sha':fixture.q['pillow_wheel']['sha256']='f'*64
    if damage=='cp312_wheel':
        fixture.q['pillow_wheel']=dict(path=str(fixture.original/'inputs'/WHEEL_NAME.replace('cp311','cp312')),
            sha256='42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca')
    if damage=='wrong_wheel_path':fixture.q['pillow_wheel']['path']=str(fixture.original/'inputs/other.whl')
    if damage=='wrong_q_python':fixture.q['runtime']['python']='/tmp/python'
    if damage=='wrong_q_python_real':fixture.q['runtime']['python_real']='/tmp/python3.11'
    if damage=='wrong_q_python_version':fixture.q['runtime']['python_version']='3.12.12'
    # Rebind even known large declarations to reach semantic v4 checks rather
    # than merely rejecting a stale input declaration or request digest.
    if damage in ('wrong_wheel_sha','cp312_wheel','wrong_wheel_path'):
        entries={line.split('  ',1)[1]:line.split('  ',1)[0]
            for line in (fixture.root/'FILES.sha256').read_text().splitlines()}
        del entries['inputs/'+WHEEL_NAME]
        name=str(Path(fixture.q['pillow_wheel']['path']).relative_to(fixture.original))
        entries[name]=fixture.q['pillow_wheel']['sha256']
        write(fixture.root/'FILES.sha256',''.join(value+'  '+name+'\n' for name,value in entries.items()))
    rebind_request(fixture);refresh(fixture);reject_before_cache(monkeypatch)
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)


@pytest.mark.parametrize('field,value',[('real_executable','/tmp/python3.11'),('python_version','3.11.13'),
    ('python_implementation','pypy'),('system','Darwin'),('machine','aarch64'),('pillow_version','12.0.0'),
    ('pillow_origin','/other/PIL/__init__.py'),('image_origin','/other/PIL/Image.py'),
    ('native_decoder_origin','/other/PIL/_imaging.cpython-311-x86_64-linux-gnu.so')])
def test_v4_rejects_rehashed_actual_runtime_identity_damage(tmp_path,monkeypatch,field,value):
    fixture=v4_fixture(tmp_path);path=fixture.root/'output/runtime.json';runtime=json.loads(path.read_text())
    runtime[field]=value;write(path,runtime);refresh(fixture);reject_before_cache(monkeypatch)
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)


@pytest.mark.parametrize('damage',['missing_record','changed_record_sha','changed_record_bytes','metadata_hash',
    'record_hash','wheel_metadata_hash','files','bytes','overlay','wheel_sha','status','schema',
    'source_changed','readback_false','fsync_false','pip_used','network_used','historical_mutation'])
def test_v4_rejects_rehashed_offline_installation_damage(tmp_path,monkeypatch,damage):
    fixture=v4_fixture(tmp_path);path=fixture.root/'output/runtime.json';runtime=json.loads(path.read_text())
    install=runtime['pillow_installation'];first=next(iter(install['records']))
    if damage=='missing_record':del install['records'][first]
    if damage=='changed_record_sha':install['records'][first]['sha256']='0'*64
    if damage=='changed_record_bytes':install['records'][first]['bytes']+=1
    if damage=='metadata_hash':install['metadata_sha256']='0'*64
    if damage=='record_hash':install['record_sha256']='0'*64
    if damage=='wheel_metadata_hash':install['wheel_metadata_sha256']='0'*64
    if damage=='files':install['files']=138
    if damage=='bytes':install['bytes']+=1
    if damage=='overlay':install['overlay']='/tmp/other-overlay'
    if damage=='wheel_sha':install['wheel_sha256']='0'*64
    if damage=='status':install['status']='PENDING'
    if damage=='schema':install['schema']='different'
    if damage=='source_changed':install['source_stat_unchanged']=False
    if damage=='readback_false':install['all_extracted_files_reopened_sha256']=False
    if damage=='fsync_false':install['files_and_directories_fsynced']=False
    if damage=='pip_used':install['pip_used']=True
    if damage=='network_used':install['network_used']=True
    if damage=='historical_mutation':install['historical_environment_mutated']=True
    write(path,runtime);refresh(fixture);reject_before_cache(monkeypatch)
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)


def test_v4_explicit_cp312_runtime_does_not_broaden_source_default(tmp_path,monkeypatch):
    fixture=v4_fixture(tmp_path);path=fixture.root/'output/runtime.json';runtime=json.loads(path.read_text())
    runtime.update(python_version='3.12.12',native_decoder_origin=str(
        fixture.original/'output/pillow-overlay/PIL/_imaging.cpython-312-x86_64-linux-gnu.so'))
    write(path,runtime);refresh(fixture);reject_before_cache(monkeypatch)
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)


def test_v4_default_gateway_reads_bounded_proofs_without_payload_native_import_or_producer_code(tmp_path):
    fixture=v4_fixture(tmp_path)
    code='''import importlib.abc,json,pathlib,sys
class NoPayloadImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in ("torch","PIL","egomimic","wheel_overlay") or fullname.startswith(("torch.","PIL.","egomimic.")):
   raise AssertionError("Gateway imported payload/native/producer module: "+fullname)
sys.meta_path.insert(0,NoPayloadImports())
from scripts.train.joint import imagenet,contract
request,blocked=json.loads(sys.stdin.read());blocked={pathlib.Path(p) for p in blocked};real_sha=contract.sha;hashed=[]
def hash_guard(path):
 assert pathlib.Path(path) not in blocked,"Gateway hashed a large payload"
 hashed.append(str(path));return real_sha(path)
contract.sha=hash_guard
original_open=pathlib.Path.open
def open_guard(self,*args,**kwargs):
 assert self not in blocked,"Gateway opened a large payload or extracted native library"
 return original_open(self,*args,**kwargs)
pathlib.Path.open=open_guard
calls=[]
imagenet.verify_cache_native=lambda *args,**kwargs:calls.append(kwargs)
assert imagenet.verify_dataset(request)==({"train":1281167,"val":50000},1000)
assert calls==[{"allocated":False}] and "torch" not in sys.modules and "PIL" not in sys.modules
assert "wheel_overlay" not in sys.modules and "egomimic" not in sys.modules and not any(n.startswith("egomimic.") for n in sys.modules)
assert any(p.endswith("/output/runtime.json") for p in hashed),"Missing bounded runtime proof replay"
assert any(p.endswith("/wheel_overlay.py") for p in hashed),"Producer helper byte closure was skipped"
print("PASS_SYNTHETIC_V4_ONLY_NO_PAYLOAD_OR_NATIVE_IMPORT")
'''
    source=Path(imagenet.__file__).resolve().parents[3]
    # Deny all actual large files plus plausible original/imported native paths.
    blocked=set(fixture.blocked)
    blocked.add(fixture.root/'output/pillow-overlay/PIL/_imaging.cpython-311-x86_64-linux-gnu.so')
    blocked.add(fixture.original/'output/pillow-overlay/PIL/_imaging.cpython-311-x86_64-linux-gnu.so')
    result=subprocess.run([sys.executable,'-c',code],cwd=source,input=json.dumps([fixture.request,
        [str(path) for path in blocked]]),env=dict(os.environ,PYTHONPATH=str(source)),
        text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='PASS_SYNTHETIC_V4_ONLY_NO_PAYLOAD_OR_NATIVE_IMPORT'
