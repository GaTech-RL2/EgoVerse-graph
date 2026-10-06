import concurrent.futures
import datetime
import hashlib
import json
import os
import multiprocessing
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.utils.aws.aws_data_utils import load_env

PREVIOUS_FINAL_RUNS = ['arc-sc-20261004-object-reference-dur-eval', 'arc-se-20261003-goal-reference-stk-eval', 'arc-se-20261003-spatial-reference-stk-eval', 'arc-sr-20261003-10-component-time-dur-eval', 'arc-sr-20261003-10-gripper-stk-eval', 'arc-sr-20261003-10-reference-dur-eval', 'arc-sr-20261003-10-reference-stk-eval', 'arc-sr-20261003-goal-gripper-stk-eval', 'arc-sr-20261003-goal-reference-dur-eval', 'arc-sr-20261003-object-reference-stk-eval', 'arc-sr-20261003-spatial-reference-dur-eval', 'arc-ss-20261005-object-all-scalar-dur-eval', 'arc-ss-20261005-object-all-scalar-stk-eval', 'arc-ss-20261005-object-angular-driver-dur-eval', 'arc-ss-20261005-object-angular-driver-stk-eval', 'arc-ss-20261005-object-angular-scalar-dur-eval', 'arc-ss-20261005-object-angular-scalar-stk-eval', 'arc-ss-20261005-object-component-time-dur-eval', 'arc-ss-20261005-object-component-time-stk-eval', 'arc-ss-20261005-object-component-velocity-dur-eval', 'arc-ss-20261005-object-component-velocity-stk-eval', 'arc-ss-20261005-object-gripper-dur-eval', 'arc-ss-20261005-object-gripper-stk-eval', 'arc-ss-20261005-object-x-yz-dur-eval', 'arc-ss-20261005-object-x-yz-stk-eval', 'arc-ss-20261005-object-xyz-scalar-dur-eval', 'arc-ss-20261005-object-xyz-scalar-stk-eval', 'arc-ss-20261005-object-y-xz-dur-eval', 'arc-ss-20261005-object-y-xz-stk-eval', 'arc-ss-20261005-object-z-xy-dur-eval', 'arc-ss-20261005-object-z-xy-stk-eval', 'arc-st-20261005-10-component-velocity-stk-eval', 'arc-st-20261005-10-gripper-dur-eval', 'arc-st-20261005-10-xyz-scalar-dur-eval', 'arc-st-20261005-10-xyz-scalar-stk-eval', 'arc-st-20261005-goal-component-time-dur-eval', 'arc-st-20261005-goal-component-time-stk-eval', 'arc-st-20261005-goal-component-velocity-dur-eval', 'arc-st-20261005-goal-component-velocity-stk-eval', 'arc-st-20261005-goal-gripper-dur-eval', 'arc-st-20261005-goal-x-yz-dur-eval', 'arc-st-20261005-goal-x-yz-stk-eval', 'arc-st-20261005-goal-xyz-scalar-dur-eval', 'arc-st-20261005-goal-xyz-scalar-stk-eval', 'arc-st-20261005-goal-y-xz-dur-eval', 'arc-st-20261005-goal-y-xz-stk-eval', 'arc-st-20261005-goal-z-xy-dur-eval', 'arc-st-20261005-goal-z-xy-stk-eval', 'arc-st-20261005-spatial-component-time-dur-eval', 'arc-st-20261005-spatial-component-time-stk-eval', 'arc-st-20261005-spatial-component-velocity-dur-eval', 'arc-st-20261005-spatial-component-velocity-stk-eval', 'arc-st-20261005-spatial-gripper-dur-eval', 'arc-st-20261005-spatial-gripper-stk-eval', 'arc-st-20261005-spatial-x-yz-dur-eval', 'arc-st-20261005-spatial-x-yz-stk-eval', 'arc-st-20261005-spatial-xyz-scalar-dur-eval', 'arc-st-20261005-spatial-xyz-scalar-stk-eval', 'arc-st-20261005-spatial-y-xz-dur-eval', 'arc-st-20261005-spatial-z-xy-dur-eval', 'arc-st-20261005-spatial-z-xy-stk-eval']
ROOT = Path(__file__).parent
MANIFEST = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001/results/libero_arc_streams_20261006/parallel_tail_170427/launch.json')
manifest = json.loads(MANIFEST.read_text())
load_env(required=True)
client = boto3.client('s3', region_name='auto', endpoint_url=os.environ['R2_ENDPOINT_URL'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    config=Config(connect_timeout=8, read_timeout=20, retries={'max_attempts': 2}, max_pool_connections=6, proxies=urllib.request.getproxies()))


def expected_plan(suite, repetition_index=None):
    return [dict(task=task, repetition=rep, trial=trial, seed=1000 + (rep * 50 + trial) * len(TASKS[suite]) + i)
            for rep in range(5) for trial in range(50) for i, task in enumerate(TASKS[suite])
            if repetition_index is None or rep == repetition_index]


def read_records(directory, complete):
    protocol = json.loads((directory / 'protocol.json').read_text())
    values = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines() if line]
    def key(row):
        return tuple(row[k] for k in ('task', 'repetition', 'trial', 'seed'))
    records = {key(r): r for r in values}
    expected = {key(r) for r in protocol['plan']}
    assert len(records) == len(values)
    assert records.keys() <= expected
    if complete:
        assert records.keys() == expected
    for row in records.values():
        assert isinstance(row['success'], bool) and row['initial_state_sha256']
        assert 1 <= row['steps'] <= 550
    return protocol, records


def read(run, file):
    try:
        obj = client.get_object(Bucket='rldb', Key=f'experiments/arc-oat-20260919/{run}/{file}')
    except ClientError as error:
        if error.response['Error']['Code'] in ('NoSuchKey', '404', 'NotFound'):
            return None, None
        raise
    raw = obj['Body'].read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == obj['Metadata']['sha256'], (run, file)
    path = ROOT / 'artifacts' / run / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw, {'sha256': sha, 'bytes': len(raw), 'last_modified': obj['LastModified'].isoformat()}


def document(run, file):
    raw, receipt = read(run, file)
    return (json.loads(raw) if raw is not None else None), receipt


def collect(row):
    result = {k: row[k] for k in ('suite', 'variant', 'mode', 'training_run_id', 'evaluation_run_id', 'training_workflow_id', 'evaluation_workflow_id', 'training_task', 'evaluation_task')}
    result['training_status'], result['training_status_receipt'] = document(row['training_run_id'], 'status.json')
    run = row['evaluation_run_id']
    result['evaluation_status'], result['evaluation_status_receipt'] = document(run, 'status.json')
    if result['evaluation_status'] is None:
        return result
    receipts = {}
    for name, key in [('runtime.json', 'runtime'), ('evaluation-request.json', 'request'),
                      ('evaluation-progress.json', 'evaluation_progress'), ('scores.json', 'scores')]:
        result[key], receipts[name] = document(run, name)
    assert result['runtime']['source_commit'] == manifest['source_commit']
    assert result['runtime']['evaluate_from_run'] == row['training_run_id']
    assert result['runtime']['suite'] == row['suite']
    request = result['request']
    assert request['source_run'] == row['training_run_id'] and request['source_commit'] == manifest['source_commit']
    assert request['epochs'] == 5001 and request['total_optimizer_steps'] == manifest['protocol']['optimizer_steps'][row['suite']]
    records, protocol = {}, None
    complete = result['evaluation_status']['state'] == 'POLICY_EVALUATION_COMPLETE'
    relative = f"arc_{row['mode']}/{row['suite']}"
    directories = [relative] if complete else [f'repetitions/{i}/{relative}' for i in range(5)]
    for directory in directories:
        raw, receipt = read(run, directory + '/episodes.jsonl')
        if raw is None:
            continue
        assert not raw or raw.endswith(b'\n'), 'Incomplete episode line'
        receipts[directory + '/episodes.jsonl'] = receipt
        _, receipts[directory + '/protocol.json'] = read(run, directory + '/protocol.json')
        current, entries = read_records(ROOT / 'artifacts' / run / directory, complete)
        assert current['checkpoint_sha256'] == request['checkpoint']['sha256']
        assert current['arc_stream_variant'] == row['variant']
        assert current['representation']['mode'] == row['mode']
        assert current['method'] == 'arc' and current['arc_backbone'] == 'unet'
        if not complete:
            index = int(directory.split('/')[1])
            assert current['plan'] == expected_plan(row['suite'], repetition_index=index)
            assert current['evaluation_repetition'] == index
        full = {**current, 'plan': expected_plan(row['suite'])}
        full.pop('evaluation_repetition', None)
        if protocol is not None:
            assert protocol == full
        protocol = full
        assert not records.keys() & entries.keys()
        records.update(entries)
    if complete:
        assert len(records) == result['scores']['episodes'] == 2500
        assert abs(sum(r['success'] for r in records.values()) / 2500 - result['scores']['mean_success_rate']) < 1e-12
    if records:
        result['protocol'] = protocol
        result['records'] = [{k: r[k] for k in ('task', 'repetition', 'trial', 'seed', 'success', 'steps', 'initial_state_sha256')} for r in records.values()]
        result['success_rate_percent'] = 100 * sum(r['success'] for r in records.values()) / len(records)
        result['episodes'] = len(records)
        result['tasks_seen'] = len({r['task'] for r in records.values()})
        if run not in PREVIOUS_FINAL_RUNS:
            print(json.dumps({k: result[k] for k in ('suite', 'variant', 'mode', 'episodes', 'tasks_seen', 'success_rate_percent')}), flush=True)
    result['artifact_receipts'] = receipts
    return result


def workflow(name):
    for attempt in range(3):
        result = subprocess.run(['osmo', 'workflow', 'query', name, '--format-type', 'json'], capture_output=True, text=True, timeout=50)
        if result.returncode == 0:
            break
        if attempt == 2:
            result.check_returncode()
        time.sleep(2 * (attempt + 1))
    data = json.loads(result.stdout)
    return {**{k: data[k] for k in ('name', 'status', 'pool', 'priority')},
            'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'tasks': [{k: t.get(k) for k in ('name', 'status', 'start_time', 'end_time')} for g in data['groups'] for t in g['tasks']]}


if __name__ == '__main__':
    print('Reading all 88 canonical policies and evaluations', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(collect, manifest['runs']))
    result = {'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'manifest': str(MANIFEST), 'source_commit': manifest['source_commit'], 'runs': rows}
    (ROOT / 'data.json').write_text(json.dumps(result, indent=2) + '\n')
    print('TRAINING', dict(Counter((r['training_status'] or {}).get('state', 'NO_ARTIFACTS') for r in rows)), flush=True)
    print('EVALUATION', dict(Counter((r['evaluation_status'] or {}).get('state', 'NO_ARTIFACTS') for r in rows)), flush=True)
    live_names = sorted({r[k] for r in manifest['runs'] for k in ('training_workflow_id', 'evaluation_workflow_id') if r[k].startswith(('arc-ss-20261005-', 'arc-st-20261005-', 'arc-su-20261006-'))})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(workflow, live_names))
    (ROOT / 'workflows.json').write_text(json.dumps(statuses, indent=2) + '\n')
    tasks = {(w['name'], t['name']): t['status'] for w in statuses for t in w['tasks']}
    for kind in ('training', 'evaluation'):
        print(kind, dict(Counter(tasks.get((r[kind + '_workflow_id'], r[kind + '_task']), 'HISTORICAL') for r in rows)), flush=True)
