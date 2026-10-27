"""Synthetic V5 source envelopes; no actual data, interpreter or native proof.

V4 fixtures and their exact cp311 helper stay unchanged. This new fixture only
rebases the small native/claim graph and separates configured aliases from the
explicit physical interpreter string. The gateway must never resolve that
remote string on the machine interpreting the metadata.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.train.joint import imagenet
from scripts.train.joint import imagenet_wheel_overlay as wheel_policy
from test_imagenet_raw_cpu_chain import digest, write
from test_imagenet_raw_cpu_v3_memory import rebind_request, refresh
from test_imagenet_raw_cpu_v4_wheel import (
    PYTHON, PYTHON_REAL, PRODUCER_HELPER, PRODUCER_SHA, v4_fixture,
)

PYTHON_PHYSICAL = '/coc/flash7/paphiwetsa3/uv-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11'


def v5_fixture(tmp_path):
    """Reuse V4's synthetic leaves, rebind every native/claim namespace to V5."""
    fixture = v4_fixture(tmp_path)
    old_root = str(fixture.original); new_root = old_root.removesuffix('v4')+'v5'
    old_request_id = fixture.q['request_id']; new_request_id = old_request_id.removesuffix('v4')+'v5'
    replacements = ((old_root,new_root),
        ('elmo-imagenet-cpu-preparation-v4','elmo-imagenet-cpu-preparation-v5'),
        ('elmo-native-imagenet-cpu-v4','elmo-native-imagenet-cpu-v5'),
        (old_request_id,new_request_id))
    def replace(value):
        if isinstance(value,dict):return {key:replace(item) for key,item in value.items()}
        if isinstance(value,list):return [replace(item) for item in value]
        if isinstance(value,str):
            for old,new in replacements:value=value.replace(old,new)
        return value
    for path in fixture.root.rglob('*'):
        if path.is_file():write(path,replace(path.read_text()))
    (fixture.root/'request-claims'/old_request_id).rename(fixture.root/'request-claims'/new_request_id)
    for key in ('q','result','capture','completion','request'):
        setattr(fixture,key,replace(getattr(fixture,key)))
    fixture.original = Path(new_root)
    fixture.q['runtime']['python_physical'] = PYTHON_PHYSICAL
    path = fixture.root/'output/runtime.json'; runtime = json.loads(path.read_text())
    runtime.update(real_executable=PYTHON_PHYSICAL,configured_executable=PYTHON,
        configured_target=PYTHON_REAL,physical_executable_expected=PYTHON_PHYSICAL)
    write(path,runtime)
    rebind_request(fixture); refresh(fixture)
    return fixture


def reject_before_pending_cache(monkeypatch):
    monkeypatch.setattr(imagenet,'verify_cache_native',lambda *args,**kwargs:
        pytest.fail('Invalid synthetic V5 source proof reached external pending cache gate'))


def test_v5_explicit_physical_string_accepts_only_complete_synthetic_source_envelope(tmp_path,monkeypatch):
    fixture = v5_fixture(tmp_path); calls = []
    monkeypatch.setattr(imagenet,'verify_cache_native',lambda *args,**kwargs:calls.append(kwargs))
    assert imagenet.verify_dataset(fixture.request) == ({'train':1281167,'val':50000},1000)
    assert calls == [{'allocated':False}]
    assert fixture.q['root'].endswith('/imagenet-cpu-v5')
    assert fixture.q['runtime'] == dict(python=PYTHON,python_real=PYTHON_REAL,
        python_physical=PYTHON_PHYSICAL,python_version='3.11.14',pillow_version='13.0.0')
    assert fixture.completion['job_name'] == 'elmo-imagenet-cpu-preparation-v5'
    assert 'elmo-native-imagenet-cpu-v5' in (fixture.root/'capture-running-124.txt').read_text()
    assert str(fixture.original) in (fixture.claim/'HELD_NATIVE.txt').read_text()
    runtime = json.loads((fixture.root/'output/runtime.json').read_text())
    assert runtime['configured_executable'] == PYTHON and runtime['configured_target'] == PYTHON_REAL
    assert wheel_policy.validate_runtime(runtime,fixture.q['pillow_wheel'],
        fixture.original/'output/pillow-overlay',PYTHON_PHYSICAL,profile='cp311') == runtime
    assert digest(PRODUCER_HELPER) == PRODUCER_SHA
    assert not any(path.exists() for path in fixture.blocked)


@pytest.mark.parametrize('damage',['missing_physical','wrong_physical','alias_as_physical','extra_runtime_field',
    'wrong_configured_python','wrong_configured_target','wrong_version'])
def test_v5_rehashed_request_rejects_missing_or_changed_explicit_physical_contract(tmp_path,monkeypatch,damage):
    fixture = v5_fixture(tmp_path)
    runtime = fixture.q['runtime']
    if damage == 'missing_physical':runtime.pop('python_physical')
    if damage == 'wrong_physical':runtime['python_physical']='/other/bin/python3.11'
    if damage == 'alias_as_physical':runtime['python_physical']=PYTHON_REAL
    if damage == 'extra_runtime_field':runtime['assume_alias_resolution']=True
    if damage == 'wrong_configured_python':runtime['python']='/other/.venv/bin/python'
    if damage == 'wrong_configured_target':runtime['python_real']=PYTHON_PHYSICAL
    if damage == 'wrong_version':runtime['python_version']='3.11.13'
    rebind_request(fixture); refresh(fixture); reject_before_pending_cache(monkeypatch)
    with pytest.raises(ValueError,match='reviewed decoder/interpreter'):imagenet.verify_dataset(fixture.request)


@pytest.mark.parametrize('damage',['missing_configured_executable','missing_configured_target',
    'missing_physical_expected','missing_real','wrong_configured_executable','wrong_configured_target',
    'wrong_physical_expected','alias_as_real','wrong_real','wrong_version'])
def test_v5_rehashed_actual_runtime_keeps_configured_and_physical_paths_distinct(tmp_path,monkeypatch,damage):
    fixture = v5_fixture(tmp_path); path = fixture.root/'output/runtime.json'; runtime = json.loads(path.read_text())
    if damage == 'missing_configured_executable':runtime.pop('configured_executable')
    if damage == 'missing_configured_target':runtime.pop('configured_target')
    if damage == 'missing_physical_expected':runtime.pop('physical_executable_expected')
    if damage == 'missing_real':runtime.pop('real_executable')
    if damage == 'wrong_configured_executable':runtime['configured_executable']='/other/.venv/bin/python'
    if damage == 'wrong_configured_target':runtime['configured_target']=PYTHON_PHYSICAL
    if damage == 'wrong_physical_expected':runtime['physical_executable_expected']=PYTHON_REAL
    if damage == 'alias_as_real':runtime['real_executable']=PYTHON_REAL
    if damage == 'wrong_real':runtime['real_executable']='/other/bin/python3.11'
    if damage == 'wrong_version':runtime['python_version']='3.11.13'
    write(path,runtime); refresh(fixture); reject_before_pending_cache(monkeypatch)
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)


def test_v5_gateway_is_offline_stdlib_and_never_resolves_remote_interpreter_alias(tmp_path):
    fixture = v5_fixture(tmp_path)
    code = '''import importlib.abc,json,pathlib,sys
class NoMLOrProducer(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in ('torch','PIL','numpy','transformers','egomimic','wheel_overlay') or fullname.startswith(('torch.','PIL.','numpy.','transformers.','egomimic.')):
   raise AssertionError('Gateway imported payload/native/producer module: '+fullname)
sys.meta_path.insert(0,NoMLOrProducer())
from scripts.train.joint import imagenet,contract
request,blocked,interpreters=json.loads(sys.stdin.read());blocked={pathlib.Path(p) for p in blocked}
interpreters={pathlib.Path(p) for p in interpreters};real_sha=contract.sha;hashed=[]
def hash_guard(path):
 assert pathlib.Path(path) not in blocked,'Gateway hashed large payload or remote executable'
 hashed.append(str(path));return real_sha(path)
contract.sha=hash_guard
real_open=pathlib.Path.open
real_resolve=pathlib.Path.resolve
def open_guard(self,*args,**kwargs):
 assert self not in blocked,'Gateway opened payload or native executable'
 return real_open(self,*args,**kwargs)
def resolve_guard(self,*args,**kwargs):
 assert self not in interpreters,'Gateway locally resolved a configured/physical remote interpreter string'
 return real_resolve(self,*args,**kwargs)
pathlib.Path.open=open_guard;pathlib.Path.resolve=resolve_guard
calls=[]
imagenet.verify_cache_native=lambda *args,**kwargs:calls.append(kwargs)
assert imagenet.verify_dataset(request)==({'train':1281167,'val':50000},1000)
assert calls==[{'allocated':False}]
assert any(p.endswith('/output/runtime.json') for p in hashed)
assert any(p.endswith('/wheel_overlay.py') for p in hashed)
assert not any(n in sys.modules for n in ('torch','PIL','numpy','transformers','egomimic','wheel_overlay'))
print('PASS_SYNTHETIC_V5_EXPLICIT_PHYSICAL_NO_ALIAS_RESOLUTION_NO_PAYLOAD_IO')
'''
    source = Path(imagenet.__file__).resolve().parents[3]
    blocked = set(fixture.blocked) | {Path(PYTHON),Path(PYTHON_REAL),Path(PYTHON_PHYSICAL)}
    for prefix in (fixture.root,fixture.original):
        blocked.add(prefix/'output/pillow-overlay/PIL/_imaging.cpython-311-x86_64-linux-gnu.so')
    result = subprocess.run([sys.executable,'-c',code],cwd=source,
        input=json.dumps([fixture.request,[str(path) for path in blocked],[PYTHON,PYTHON_REAL,PYTHON_PHYSICAL]]),
        env=dict(os.environ,PYTHONPATH=str(source)),text=True,capture_output=True,timeout=20)
    assert result.returncode == 0,result.stderr
    assert result.stdout.strip() == 'PASS_SYNTHETIC_V5_EXPLICIT_PHYSICAL_NO_ALIAS_RESOLUTION_NO_PAYLOAD_IO'
