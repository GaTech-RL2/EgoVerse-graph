import gzip
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import boto3
import yaml
from botocore.config import Config
from egomimic.utils.aws.aws_data_utils import load_env

ROOT = Path(__file__).parent
DEST = ROOT / 'fallback'
DEST.mkdir(exist_ok=True)
plan = json.loads((ROOT / 'recovery-plan.json').read_text())
load_env(required=True)
client = boto3.client('s3', region_name='auto', endpoint_url=os.environ['R2_ENDPOINT_URL'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    config=Config(connect_timeout=8, read_timeout=20, retries={'max_attempts': 2}, proxies=urllib.request.getproxies()))


def query(name):
    for attempt in range(3):
        result = subprocess.run(['osmo', 'workflow', 'query', name, '--format-type', 'json'], capture_output=True, text=True, timeout=50)
        if result.returncode == 0:
            return json.loads(result.stdout)
    raise RuntimeError('Cannot confirm workflow: ' + name)


fallback = []
for spec in plan['specs']:
    if spec['pool'] != 'groot-l40s-03':
        continue
    old_name = spec['name']
    new_name = old_name.replace('arc-ss-', 'arc-st-')
    previous = query(old_name + '-1')
    assert previous['status'] == 'FAILED_CANCELED'
    statuses = {t['name']: t['status'] for g in previous['groups'] for t in g['tasks']}
    assert all(statuses[t] in {'FAILED_CANCELED', 'FAILED_UPSTREAM'} for t in statuses if t.startswith(('train-', 'evaluate-')))
    # Every new container was canceled during bootstrap. Reuse the ORIGINAL
    # pinned checkpoint requests only after proving no newer artifacts exist.
    prefix = 'experiments/arc-oat-20260919/' + old_name
    response = client.list_objects_v2(Bucket='rldb', Prefix=prefix, MaxKeys=1)
    assert not response.get('KeyCount'), 'A new artifact exists: inspect it before selecting an older checkpoint'
    document = yaml.safe_load(Path(spec['file']).read_text())
    document = json.loads(json.dumps(document).replace(old_name, new_name))
    body = document['workflow']
    training = [t for t in body['tasks'] if t['name'].startswith('train-')]
    training.sort(key=lambda t: (not any(f['path'] == '/tmp/resume-request.json' for f in t['files']), t['name']))
    lanes = [None] * 6
    for index, task in enumerate(training):
        task.pop('inputs', None)
        lane = index % 6
        if lanes[lane]:
            task['inputs'] = [{'task': lanes[lane]}]
        lanes[lane] = task['name']
    assert sum(any(f['path'] == '/tmp/resume-request.json' for f in t['files']) for t in training[:6]) == 6
    names = {t['name'] for t in body['tasks']}
    remaining = {t['name']: {i['task'] for i in t.get('inputs', [])} for t in body['tasks']}
    available = set()
    assert all(deps <= names for deps in remaining.values())
    while remaining:
        ready = {n for n, deps in remaining.items() if deps <= available}
        assert ready
        available |= ready
        remaining = {n: deps for n, deps in remaining.items() if n not in ready}
    path = DEST / (new_name + '.yaml')
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    result = subprocess.run(['osmo', 'workflow', 'submit', str(path), '--pool', 'groot-l40s-01', '--priority', 'NORMAL', '--dry-run'], capture_output=True, text=True, timeout=50)
    (DEST / (new_name + '-validation.txt')).write_text(result.stdout + result.stderr)
    result.check_returncode()
    fallback.append({**spec, 'old_name': old_name, 'name': new_name, 'pool': 'groot-l40s-01', 'file': str(path),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'previous_status': previous['status'],
                     'previous_end_time': previous['end_time'], 'previous_artifact_prefix_empty': True,
                     'validation_exit_code': result.returncode})
    print(json.dumps({'validated': new_name, 'pool': 'groot-l40s-01', 'bootstrap_artifacts': 0}), flush=True)
(DEST / 'plan.json').write_text(json.dumps(fallback, indent=2) + '\n')

for spec in fallback:
    receipt = DEST / (spec['name'] + '-submission.json')
    assert not receipt.exists(), 'Inspect prior accepted submission rather than submitting twice'
    result = subprocess.run(['osmo', 'workflow', 'submit', spec['file'], '--pool', spec['pool'], '--priority', 'NORMAL', '--format-type', 'json'], capture_output=True, text=True, timeout=50)
    (DEST / (spec['name'] + '-submission.stderr')).write_text(result.stderr)
    result.check_returncode()
    data = json.loads(result.stdout)
    receipt.write_text(json.dumps(data, indent=2) + '\n')
    assert data['name'] == spec['name'] + '-1'
    accepted = query(data['name'])
    assert accepted['status'] in {'RUNNING', 'PENDING', 'WAITING'}
    (DEST / (data['name'] + '-accepted.json')).write_text(json.dumps(accepted, indent=2) + '\n')
    print(json.dumps({'accepted': data['name'], 'pool': accepted['pool'], 'status': accepted['status']}), flush=True)
