"""Exact offline Pillow wheel extraction into a fresh private overlay; no pip."""
import base64
import csv
from email.parser import Parser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

DIST = 'pillow-13.0.0.dist-info'
METADATA_SHA = 'dba7fdfe24e302f46bae562f55d66cc367c0618abae9894a1ebec93dc8f0e511'
TAGS = {'cp311-cp311-manylinux_2_27_x86_64', 'cp311-cp311-manylinux_2_28_x86_64'}
MAX_TOTAL = 64 << 20
MAX_MEMBER = 32 << 20


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def member_name(name, *, directory=False):
    value = name[:-1] if directory and name.endswith('/') else name
    need(isinstance(value, str) and value and len(value) <= 4096
         and '\\' not in value and not any(ord(c) < 32 for c in value), 'Unsafe wheel member')
    path = PurePosixPath(value)
    need(not path.is_absolute() and path.as_posix() == value and '..' not in path.parts
         and path.parts[0] in {'PIL', 'pillow.libs', DIST}, 'Wheel member escaped exact package roots')
    return value


def inspect_wheel(path, expected_sha):
    """Validate archive structure and complete RECORD before any destination write."""
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path,
         'Wheel must be a physical regular file')
    before = signature(path)
    need(re.fullmatch('[0-9a-f]{64}', expected_sha or '') and sha(path) == expected_sha,
         'Pinned wheel checksum differs')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        need(0 < len(entries) <= 512 and sum(x.file_size for x in entries) <= MAX_TOTAL,
             'Wheel exceeds bounded member/byte inventory')
        members = {}
        directories = set()
        for item in entries:
            name = member_name(item.filename, directory=item.is_dir())
            need(name not in members and name not in directories, 'Duplicate wheel member')
            kind = stat.S_IFMT(item.external_attr >> 16)
            need(not item.flag_bits & 1 and item.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                 and 0 <= item.file_size <= MAX_MEMBER, 'Unsupported wheel entry encoding or size')
            if item.is_dir():
                need(kind in {0, stat.S_IFDIR} and item.file_size == 0, 'Invalid wheel directory')
                directories.add(name)
            else:
                need(kind in {0, stat.S_IFREG}, 'Linked or special wheel member forbidden')
                members[name] = item
        required = {DIST+'/WHEEL', DIST+'/METADATA', DIST+'/RECORD',
                    'PIL/__init__.py', 'PIL/Image.py', 'PIL/_imaging.cpython-311-x86_64-linux-gnu.so'}
        need(required <= set(members), 'Pillow wheel runtime/metadata closure incomplete')
        for name in members:
            need(not any(str(parent) in members for parent in PurePosixPath(name).parents),
                 'Wheel file shadows a parent directory')
        metadata = archive.read(DIST+'/METADATA')
        need(hashlib.sha256(metadata).hexdigest() == METADATA_SHA, 'Pillow METADATA differs from matching Lambda package')
        parsed = Parser().parsestr(metadata.decode('utf-8'))
        need(parsed.get_all('Name') == ['pillow'] and parsed.get_all('Version') == ['13.0.0'],
             'Wrong Pillow distribution/version')
        wheel_raw = archive.read(DIST+'/WHEEL')
        wheel = Parser().parsestr(wheel_raw.decode('utf-8'))
        need(wheel.get_all('Wheel-Version') == ['1.0'] and wheel.get_all('Root-Is-Purelib') == ['false']
             and len(wheel.get_all('Tag', [])) == 2 and set(wheel.get_all('Tag', [])) == TAGS,
             'Wheel ABI/platform tags differ')
        record_raw = archive.read(DIST+'/RECORD')
        records = {}
        for row in csv.reader(io.StringIO(record_raw.decode('utf-8'))):
            need(len(row) == 3, 'Malformed wheel RECORD row')
            name, encoded, size = row
            member_name(name)
            need(name not in records and name in members, 'Duplicate or unlisted wheel RECORD member')
            if name == DIST+'/RECORD':
                need(encoded == size == '', 'Only RECORD may omit its own checksum')
                records[name] = {'sha256': hashlib.sha256(record_raw).hexdigest(), 'bytes': len(record_raw)}
                continue
            need(re.fullmatch(r'sha256=[A-Za-z0-9_-]{43}', encoded or '')
                 and re.fullmatch(r'0|[1-9][0-9]*', size or ''), 'Missing or malformed wheel RECORD hash/size')
            checksum = base64.urlsafe_b64decode(encoded[7:]+'=')
            need(len(checksum) == 32 and base64.urlsafe_b64encode(checksum).decode().rstrip('=') == encoded[7:],
                 'Noncanonical RECORD digest')
            need(int(size) == members[name].file_size, 'Wheel RECORD size differs')
            records[name] = {'sha256': checksum.hex(), 'bytes': int(size)}
        need(set(records) == set(members), 'Wheel RECORD does not cover every regular member')
        for name, expected in records.items():
            h = hashlib.sha256()
            total = 0
            with archive.open(members[name]) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    total += len(chunk)
                    need(total <= expected['bytes'], 'Expanded wheel member exceeds RECORD size')
                    h.update(chunk)
            need(total == expected['bytes'] and h.hexdigest() == expected['sha256'],
                 'Wheel member content differs from RECORD')
    need(signature(path) == before, 'Wheel changed during verification')
    return dict(records=records, wheel_sha256=expected_sha, metadata_sha256=METADATA_SHA,
                record_sha256=hashlib.sha256(record_raw).hexdigest(), wheel_metadata_sha256=hashlib.sha256(wheel_raw).hexdigest(),
                files=len(records), bytes=sum(x['bytes'] for x in records.values()), source_signature=before)


def install(path, target, expected_sha):
    """No dependency resolution, executable setup, environment mutation or bytecode."""
    path = Path(path)
    proof = inspect_wheel(path, expected_sha)
    target = Path(target)
    parent = target.parent
    need(target.is_absolute() and parent.resolve(strict=True) == parent and parent.is_dir()
         and parent.stat().st_uid == os.getuid() and not parent.stat().st_mode & 0o022,
         'Private physical overlay parent required')
    need(not target.exists() and not target.is_symlink(), 'Pillow overlay already exists; preserve unknown outcome')
    target.mkdir(mode=0o700)
    sync_dir(parent)
    directories = {target}
    with zipfile.ZipFile(path) as archive:
        for name, expected in sorted(proof['records'].items()):
            destination = target/name
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            directories.update(parent for parent in destination.parents if parent == target or target in parent.parents)
            item = archive.getinfo(name)
            mode = 0o700 if (item.external_attr >> 16) & 0o111 else 0o600
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), mode)
            with os.fdopen(fd, 'wb') as output, archive.open(item) as source:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            need(destination.stat().st_size == expected['bytes'] and sha(destination) == expected['sha256'],
                 'Extracted Pillow member failed readback')
    need(signature(path) == proof['source_signature'] and sha(path) == expected_sha, 'Wheel changed during extraction')
    actual = {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
    need(actual == set(proof['records']), 'Unexpected overlay content')
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        sync_dir(directory)
    return dict(schema='pillow-private-wheel-install-v1', status='EXACT_RECORD_AND_EXTRACTED_BYTES_PASS',
                overlay=str(target), wheel_sha256=expected_sha, metadata_sha256=METADATA_SHA,
                record_sha256=proof['record_sha256'], wheel_metadata_sha256=proof['wheel_metadata_sha256'],
                files=proof['files'], bytes=proof['bytes'], records=proof['records'],
                source_stat_unchanged=True, all_extracted_files_reopened_sha256=True,
                files_and_directories_fsynced=True, pip_used=False, network_used=False,
                historical_environment_mutated=False)


def validate_installation(proof, wheel_sha, overlay):
    """Reopenable semantic check of the exact official wheel installation report."""
    need(proof.get('schema') == 'pillow-private-wheel-install-v1'
         and proof.get('status') == 'EXACT_RECORD_AND_EXTRACTED_BYTES_PASS'
         and proof.get('overlay') == str(overlay) and proof.get('wheel_sha256') == wheel_sha
         and wheel_sha == 'a625df9a2a5597fa5e5266e01aefc82604bb85d291e3547981e863560d75da6c',
         'Wrong exact private wheel installation')
    expected = dict(metadata_sha256=METADATA_SHA,
        record_sha256='13a95d8009c913208878c3f95f0d18940b17d9e9f206a8b547126aedd4e6ac5d',
        wheel_metadata_sha256='1b8b73fc43743c69ec1937814a98114f9f4e48fe1c545296313c0518c95f7978',
        files=139, bytes=20385837)
    need(all(proof.get(k) == v for k, v in expected.items()), 'Pillow wheel closure differs')
    need(all(proof.get(k) is True for k in ('source_stat_unchanged','all_extracted_files_reopened_sha256','files_and_directories_fsynced'))
         and all(proof.get(k) is False for k in ('pip_used','network_used','historical_environment_mutated')),
         'Missing exact offline installation/readback contract')
    records = proof.get('records')
    need(isinstance(records, dict) and len(records) == 139 and hashlib.sha256(
         json.dumps(records, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
         == '17dbdb5a1b238f9835d94816576ba1ed1797f1e6069a8e8357afcfe179aae621',
         'Installed member byte manifest differs from exact pinned wheel')
    return proof


def validate_runtime(runtime, wheel, overlay, python_real):
    need(runtime.get('pillow_version') == '13.0.0' and runtime.get('pillow_wheel') == wheel
         and runtime.get('real_executable') == python_real and runtime.get('python_version') == '3.11.14'
         and runtime.get('python_implementation') == 'cpython' and runtime.get('system') == 'Linux'
         and runtime.get('machine') == 'x86_64', 'Actual Pillow interpreter/platform binding differs')
    for key, relative in [('pillow_origin', 'PIL/__init__.py'), ('image_origin', 'PIL/Image.py'),
                          ('native_decoder_origin', 'PIL/_imaging.cpython-311-x86_64-linux-gnu.so')]:
        need(runtime.get(key) == str(Path(overlay)/relative), 'Actual Pillow import origin differs')
    validate_installation(runtime.get('pillow_installation', {}), wheel['sha256'], overlay)
    return runtime
