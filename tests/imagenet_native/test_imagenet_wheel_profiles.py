"""Two exact Pillow profiles; byte/ZIP tests do not import native code.

Official archive tests require explicit IMAGENET_CP311_WHEEL and
IMAGENET_CP312_WHEEL paths. No download, resolver or native import occurs.
Malicious ZIP fixtures patch only the archive hash and synthetic METADATA hash
to reach the structural parser; no such bypass exists in the public helper API.
"""
import base64
import copy
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import warnings
import zipfile

import pytest

from scripts.train.joint import imagenet_wheel_overlay as wheel

PINS = {
    'cp311':'a625df9a2a5597fa5e5266e01aefc82604bb85d291e3547981e863560d75da6c',
    'cp312':'42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca',
}
LEGACY_SHA = '8aa731d672b1a034cdfd17e9356cc409a36e5ee68fe8c22e479a6eb1fa36f631'
SYNTHETIC_METADATA = b'Metadata-Version: 2.4\nName: pillow\nVersion: 13.0.0\n'


def official(profile):
    value=os.environ.get('IMAGENET_'+profile.upper()+'_WHEEL')
    if value is None:pytest.skip('Official '+profile+' wheel path is an explicit local test input')
    path=Path(value).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==PINS[profile]
    return path


def make_fixture(path,profile,mutate=None,record_mutate=None,extras=()):
    core='PIL/_imaging.cpython-'+profile[2:]+'-x86_64-linux-gnu.so'
    tags=sorted(wheel._PROFILES[profile]['tags'])
    entries={wheel.DIST+'/METADATA':SYNTHETIC_METADATA,
        wheel.DIST+'/WHEEL':('Wheel-Version: 1.0\nRoot-Is-Purelib: false\n'+''.join('Tag: '+tag+'\n' for tag in tags)).encode(),
        'PIL/__init__.py':b'# synthetic test only\n','PIL/Image.py':b'# synthetic test only\n',
        core:b'SYNTHETIC NOT A NATIVE LIBRARY'}
    if mutate:mutate(entries)
    rows=[[name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))]
        for name,data in entries.items()]
    rows.append([wheel.DIST+'/RECORD','',''])
    if record_mutate:record_mutate(rows)
    stream=io.StringIO();csv.writer(stream,lineterminator='\n').writerows(rows)
    entries[wheel.DIST+'/RECORD']=stream.getvalue().encode()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,data in entries.items():archive.writestr(name,data)
            for info,data in extras:archive.writestr(info,data)


def inspect_synthetic(path,profile,monkeypatch):
    original=wheel.sha
    monkeypatch.setattr(wheel,'sha',lambda item:PINS[profile] if Path(item)==path else original(item))
    monkeypatch.setattr(wheel,'METADATA_SHA',hashlib.sha256(SYNTHETIC_METADATA).hexdigest())
    return wheel.inspect_wheel(path,PINS[profile],profile=profile)


@pytest.mark.parametrize('profile',['cp311','cp312'])
def test_structural_fixture_baseline_reaches_complete_record_validation(tmp_path,monkeypatch,profile):
    path=tmp_path/'synthetic.whl';make_fixture(path,profile)
    proof=inspect_synthetic(path,profile,monkeypatch)
    assert proof['files']==6 and wheel._PROFILES[profile]['native_core'] in proof['records']


@pytest.mark.parametrize('profile',['cp311','cp312'])
def test_official_inspection_and_exclusive_full_extraction_without_native_imports(tmp_path,profile):
    path=official(profile);target=tmp_path/'private-overlay'
    expected_bytes=20385837 if profile=='cp311' else 20409333
    proof=wheel.install(path,target,PINS[profile],profile=profile)
    assert proof['files']==139 and proof['bytes']==expected_bytes
    assert wheel.validate_installation(proof,PINS[profile],target,profile=profile) is proof
    assert set(proof['records'])=={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
    for name,row in proof['records'].items():
        assert wheel.sha(target/name)==row['sha256'] and (target/name).stat().st_size==row['bytes']
    with pytest.raises(ValueError,match='already exists'):wheel.install(path,target,PINS[profile],profile=profile)
    if profile=='cp311':assert wheel.inspect_wheel(path,PINS[profile])==wheel.inspect_wheel(path,PINS[profile],profile='cp311')


def test_default_cp311_inspection_equals_sealed_source_v4_helper_when_supplied():
    value=os.environ.get('IMAGENET_CP311_LEGACY_HELPER')
    if value is None:pytest.skip('Sealed legacy helper is an explicit local parity-test input')
    source=Path(value).resolve();assert hashlib.sha256(source.read_bytes()).hexdigest()==LEGACY_SHA
    spec=importlib.util.spec_from_file_location('synthetic_readonly_legacy_wheel_overlay',source)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    path=official('cp311')
    assert wheel.inspect_wheel(path,PINS['cp311'])==legacy.inspect_wheel(path,PINS['cp311'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==LEGACY_SHA


@pytest.mark.parametrize('profile',['cp311','cp312'])
def test_exact_official_installation_validates_only_matching_runtime_profile(tmp_path,profile):
    path=official(profile);target=tmp_path/'private-overlay';pin=PINS[profile]
    installation=wheel.install(path,target,pin,profile=profile)
    supplied=dict(path=str(path),sha256=pin);python='/fixture-only/exact-python'
    runtime=dict(synthetic_fixture_only=True,pillow_version='13.0.0',pillow_wheel=supplied,
        real_executable=python,python_version='3.11.14' if profile=='cp311' else '3.12.3',
        python_implementation='cpython',system='Linux',machine='x86_64',
        pillow_origin=str(target/'PIL/__init__.py'),image_origin=str(target/'PIL/Image.py'),
        native_decoder_origin=str(target/wheel._PROFILES[profile]['native_core']),pillow_installation=installation)
    assert wheel.validate_runtime(runtime,supplied,target,python,profile=profile) is runtime
    if profile=='cp311':assert wheel.validate_runtime(runtime,supplied,target,python) is runtime
    for key,value in [('python_version','3.11.13' if profile=='cp311' else '3.13.0'),
        ('python_implementation','pypy'),('system','Darwin'),('machine','aarch64'),
        ('native_decoder_origin',str(target/'PIL/_imaging.cpython-310-x86_64-linux-gnu.so'))]:
        damaged=copy.deepcopy(runtime);damaged[key]=value
        with pytest.raises(ValueError):wheel.validate_runtime(damaged,supplied,target,python,profile=profile)
    other='cp312' if profile=='cp311' else 'cp311'
    with pytest.raises(ValueError):wheel.validate_runtime(runtime,supplied,target,python,profile=other)


@pytest.mark.parametrize('profile',['cp310','CP311','cp313','pypy',None,True])
def test_unknown_profiles_reject_before_any_source_read_or_destination_write(tmp_path,monkeypatch,profile):
    monkeypatch.setattr(wheel,'sha',lambda path:pytest.fail('Unknown profile read a file'))
    target=tmp_path/'overlay'
    for call in (lambda:wheel.inspect_wheel(tmp_path/'absent.whl',PINS['cp311'],profile=profile),
        lambda:wheel.install(tmp_path/'absent.whl',target,PINS['cp311'],profile=profile),
        lambda:wheel.validate_installation({},PINS['cp311'],target,profile=profile),
        lambda:wheel.validate_runtime({},dict(sha256=PINS['cp311']),target,'python',profile=profile)):
        with pytest.raises(ValueError,match='Unknown exact'):call()
    assert not target.exists()


@pytest.mark.parametrize('profile',['cp311','cp312'])
@pytest.mark.parametrize('pin',['a'*64,None])
def test_unknown_archive_hashes_are_not_an_arbitrary_install_api(tmp_path,monkeypatch,profile,pin):
    monkeypatch.setattr(wheel,'sha',lambda path:pytest.fail('Unreviewed hash read a file'))
    with pytest.raises(ValueError,match='Unreviewed wheel checksum'):
        wheel.install(tmp_path/'absent.whl',tmp_path/'overlay',pin,profile=profile)
    assert not (tmp_path/'overlay').exists()


@pytest.mark.parametrize('profile',['cp311','cp312'])
@pytest.mark.parametrize('damage',['traversal','absolute','backslash','noncanonical','other_root','duplicate',
    'symlink','fifo','wrong_abi','metadata','missing_core','record_missing','record_duplicate','record_hash',
    'record_size','record_self_hash','record_content','parent_shadow'])
def test_structural_record_and_abi_guards_reject_malicious_synthetic_archives(tmp_path,monkeypatch,profile,damage):
    path=tmp_path/'synthetic.whl';mutate=None;record_mutate=None;extras=[]
    if damage in ('traversal','absolute','backslash','noncanonical','other_root'):
        name={'traversal':'PIL/../escape','absolute':'/PIL/escape','backslash':'PIL\\escape',
            'noncanonical':'PIL//escape','other_root':'other/escape'}[damage];extras=[(name,b'x')]
    if damage=='duplicate':extras=[('PIL/Image.py',b'different')]
    if damage in ('symlink','fifo'):
        info=zipfile.ZipInfo('PIL/unsafe');info.external_attr=((stat.S_IFLNK|0o777) if damage=='symlink' else (stat.S_IFIFO|0o600))<<16
        extras=[(info,b'target')]
    if damage=='wrong_abi':mutate=lambda entries:entries.update({wheel.DIST+'/WHEEL':entries[wheel.DIST+'/WHEEL'].replace(profile.encode(),b'cp310')})
    if damage=='metadata':mutate=lambda entries:entries.update({wheel.DIST+'/METADATA':entries[wheel.DIST+'/METADATA']+b'changed'})
    if damage=='missing_core':mutate=lambda entries:entries.pop(wheel._PROFILES[profile]['native_core'])
    if damage=='record_missing':record_mutate=lambda rows:rows.pop(0)
    if damage=='record_duplicate':record_mutate=lambda rows:rows.append(rows[0])
    if damage=='record_hash':record_mutate=lambda rows:rows[0].__setitem__(1,'md5=abc')
    if damage=='record_size':record_mutate=lambda rows:rows[0].__setitem__(2,'99999')
    if damage=='record_self_hash':record_mutate=lambda rows:rows[-1].__setitem__(1,'sha256='+'a'*43)
    if damage=='record_content':record_mutate=lambda rows:rows[0].__setitem__(1,'sha256='+base64.urlsafe_b64encode(b'\0'*32).decode().rstrip('='))
    if damage=='parent_shadow':mutate=lambda entries:entries.update({'PIL/sub':b'file','PIL/sub/child.py':b'child'})
    make_fixture(path,profile,mutate,record_mutate,extras)
    with pytest.raises((ValueError,zipfile.BadZipFile)):inspect_synthetic(path,profile,monkeypatch)


@pytest.mark.parametrize('profile',['cp311','cp312'])
def test_source_link_and_unsafe_destination_parent_reject(tmp_path,profile):
    path=official(profile);alias=tmp_path/'alias.whl';alias.symlink_to(path)
    with pytest.raises(ValueError,match='physical'):wheel.install(alias,tmp_path/'overlay',PINS[profile],profile=profile)
    unsafe=tmp_path/'unsafe';unsafe.mkdir();unsafe.chmod(0o777)
    with pytest.raises(ValueError,match='physical'):wheel.install(path,unsafe/'overlay',PINS[profile],profile=profile)
    assert not (unsafe/'overlay').exists()


def test_official_archive_operations_import_no_pillow_or_torch_in_child(tmp_path):
    paths={profile:str(official(profile)) for profile in PINS}
    code='''import importlib.abc,json,pathlib,sys
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('PIL','torch'):
   raise AssertionError('Wheel byte operation imported native runtime')
sys.meta_path.insert(0,NoNative())
from scripts.train.joint import imagenet_wheel_overlay as w
paths,pins,target=json.loads(sys.stdin.read())
for profile,path in paths.items():
 proof=w.install(path,pathlib.Path(target)/profile,pins[profile],profile=profile)
 w.validate_installation(proof,pins[profile],pathlib.Path(target)/profile,profile=profile)
assert 'PIL' not in sys.modules and 'torch' not in sys.modules
print('PASS_OFFICIAL_WHEEL_BYTES_ONLY_NO_NATIVE_IMPORT')
'''
    source=Path(wheel.__file__).resolve().parents[3]
    result=subprocess.run([sys.executable,'-c',code],cwd=source,input=json.dumps([paths,PINS,str(tmp_path)]),
        env=dict(os.environ,PYTHONPATH=str(source)),text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='PASS_OFFICIAL_WHEEL_BYTES_ONLY_NO_NATIVE_IMPORT'


def test_public_api_profiles_are_keyword_only_and_no_resolver_exists():
    for name in ('inspect_wheel','install','validate_installation','validate_runtime'):
        parameter=inspect.signature(getattr(wheel,name)).parameters['profile']
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default=='cp311'
    assert not {'PIL','torch','subprocess','pip'} & set(wheel.__dict__)
