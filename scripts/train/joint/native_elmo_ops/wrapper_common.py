"""Small stdlib-only helpers; never imports scientific/model modules."""
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

SERVICE_COMMIT = "abbb00499096faabf7035fde23bf4b1eba984377"  # Verified upstream service publication.
SERVICE_SHA = "adf45524de3c2e6dba186572e7151d48bfca0b3adc8e6cf9defd686b47744c87"
SERVICE_FILE_COUNT = 35
MODES = {'capture': 'native_interruption', 'inspect': 'checkpoint_inspection',
         'diagnostic': 'production_recovery', 'finalize': 'interruption_admission', 'full': 'run'}

_LOADED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()



def need(value, message):
    if not value:
        raise ValueError(message)


def physical(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and p.resolve(strict=True) == p,
         'Absolute physical existing path required: ' + str(p))
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        need(len(items) == len(dict(items)), 'Duplicate JSON field')
        return dict(items)
    need(physical(path).stat().st_size <= 4 * 1024 * 1024, 'Operations JSON exceeds 4 MiB')
    return json.loads(physical(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite operations JSON')))


def bound(binding, parsed=True):
    need(set(binding) == {'path', 'sha256'} and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
         'Exact path/hash binding required')
    path = physical(binding['path'])
    need(path.is_file() and path.stat().st_size <= 4 * 1024 * 1024 and sha(path) == binding['sha256'], 'Changed bound evidence: ' + str(path))
    return read_json(path) if parsed else path.read_text()


def write_once(path, value):
    path = Path(path)
    physical(path.parent)
    payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def command(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    need(len(p.stdout.encode()) <= 1048576 and len(p.stderr.encode()) <= 65536, 'Native query exceeded bounds')
    need(p.returncode == 0 and not p.stderr.strip(), 'Actual bounded command failed: ' + str(argv[0]))
    return p.stdout


def fields(raw):
    pairs = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)', raw)
    need(len(pairs) == len(dict(pairs)), 'Ambiguous native record')
    return dict(pairs)


def service(spec, name):
    need(isinstance(SERVICE_COMMIT, str) and re.fullmatch('[0-9a-f]{40}', SERVICE_COMMIT),
         'Adapter service is unpublished; explicit reviewed publication binding required')
    need(name in {'native_interruption', 'interruption_admission', 'checkpoint_inspection', 'original_observer', 'phase_observer_proof'},
         'Unapproved pre-Torch service import')
    binding = spec['operational_service']
    need(binding['manifest_sha256'] == SERVICE_SHA and sha(binding['manifest']) == SERVICE_SHA,
         'Exact published recovery service required')
    root = physical(binding['root'])
    need(Path(binding['manifest']) == root / 'FILES.sha256', 'Service manifest escaped root')
    names = set()
    lines = Path(binding['manifest']).read_text().splitlines()
    need(0 < len(lines) <= 128, 'Unbounded service manifest')
    for line in lines:
        digest, name_in_manifest = line.split('  ', 1)
        relative = Path(name_in_manifest)
        need(not relative.is_absolute() and '..' not in relative.parts and name_in_manifest not in names,
             'Unsafe or duplicate operational service member')
        names.add(name_in_manifest)
        need(sha(physical(root / relative)) == digest, 'Operational service member changed')
    need(len(names) == SERVICE_FILE_COUNT, 'Incomplete exact adapter service')
    loader = importlib.util.spec_from_file_location('_elmo_wrapper_service', root / 'scripts/train/joint/service_bootstrap.py')
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    module.activate({'operational_service': binding})
    return importlib.import_module('scripts.train.joint.' + name)


def now():
    return datetime.now(timezone.utc).isoformat()
