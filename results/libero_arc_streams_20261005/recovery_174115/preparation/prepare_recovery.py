"""Recover only preempted cells, preserving full policies and rollout episodes."""
import copy
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import boto3
import yaml
from botocore.config import Config

from egomimic.benchmarks.libero.arc_streams import campaign
from egomimic.benchmarks.libero.evaluate import validate_request
from egomimic.utils.aws.aws_data_utils import load_env
from scripts.benchmarks.launch_libero_osmo import evaluation_workflow
from scripts.benchmarks.launch_libero_stream_expansion import expanded_workflow
from scripts.benchmarks.launch_libero_stream_parallel import pending_cells
from scripts.benchmarks.launch_libero_stream_recovery import attach_resume, resume_request
from scripts.benchmarks.repair_libero_stream_tasks import retry_downloads
from scripts.benchmarks.wait_libero_stream_checkpoint import checkpoint_ready

ROOT = Path(__file__).parent
VAL = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001')
MANIFEST = VAL / 'results/libero_arc_streams_20261004/recovery_211651/launch.json'
SOURCE = '69ff3fab69c2d322b590809461c80d70878ba71f'
HOST = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert HOST == '9c36037a5da54d5a84cd753fe57353be91e436a8'
manifest = json.loads(MANIFEST.read_text())
data = json.loads((ROOT / 'data.json').read_text())
observed = {(r['suite'], r['variant'], r['mode']): r for r in data['runs']}
by_cell = {(r['suite'], r['variant'], r['mode']): r for r in manifest['runs']}
workflows = json.loads((ROOT / 'workflows.json').read_text())
states = {(q['name'], t['name']): t['status'] for q in workflows for t in q['tasks']}
load_env(required=True)
client = boto3.client('s3', region_name='auto', endpoint_url=os.environ['R2_ENDPOINT_URL'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    config=Config(connect_timeout=8, read_timeout=20, retries={'max_attempts': 2}, proxies=urllib.request.getproxies()))
plan = {'source_commit': SOURCE, 'host_commit': HOST, 'previous_manifest': str(MANIFEST),
        'assembly_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'specs': [], 'training_replacements': [], 'evaluation_replacements': []}
(ROOT / 'specs').mkdir(exist_ok=True)
indices = {cell: i for i, cell in enumerate(pending_cells())}


def read_training(run, file):
    obj = client.get_object(Bucket='rldb', Key=f'experiments/arc-oat-20260919/{run}/{file}')
    raw = obj['Body'].read()
    assert hashlib.sha256(raw).hexdigest() == obj['Metadata']['sha256']
    path = ROOT / 'artifacts' / run / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return json.loads(raw)


for suffix in ('object', 'spatial', 'goal', '10'):
    suite = 'libero_' + suffix
    name = 'arc-ss-20261005-' + suffix
    pool = 'groot-l40s-01' if suffix == 'object' else 'groot-l40s-03'
    eval_lanes = {'object': 8, 'spatial': 6, 'goal': 6, '10': 4}[suffix]
    references = {m: by_cell[(suite, 'reference', m)] for m in ('stk', 'dur')}
    assert all(observed[(suite, 'reference', m)]['evaluation_status']['state'] == 'POLICY_EVALUATION_COMPLETE' for m in references)
    proof = json.loads(Path('/tmp/libero-streams-20261002/parallel/replay', suffix + '.json').read_text())
    spec = expanded_workflow(SOURCE, name, suite, replay_proof=proof, references=references)
    body = spec['workflow']
    templates = {t['name']: t for t in body['tasks']}
    body['tasks'] = [templates['reference-stk-ready'], templates['reference-dur-ready']]
    suite_rows = [r for r in manifest['runs'] if r['suite'] == suite]
    training_rows = [r for r in suite_rows if (observed[(suite, r['variant'], r['mode'])]['training_status'] or {}).get('state') != 'TRAINING_COMPLETE']
    training_lanes = [None] * 6
    new_training = {}
    for slot, row in enumerate(training_rows):
        cell = (suite, row['variant'], row['mode'])
        obs = observed[cell]
        assert states[(row['training_workflow_id'], row['training_task'])] in {'FAILED_CANCELED', 'FAILED_UPSTREAM'}
        index = indices[(row['variant'], row['mode'])]
        task = copy.deepcopy(templates[f'train-{index:02d}'])
        task.pop('inputs', None)
        lane = slot % 6
        if training_lanes[lane]:
            task['inputs'] = [{'task': training_lanes[lane]}]
        training_lanes[lane] = task['name']
        request = None
        saved_epochs = None
        if obs['training_status']:
            assert obs['training_status']['state'] == 'TRAINING'
            runtime = read_training(row['training_run_id'], 'runtime.json')
            checkpoints = read_training(row['training_run_id'], 'checkpoint-receipts.json')
            request = resume_request(row, ROOT / 'artifacts', SOURCE)
            attach_resume(task, request)
            epochs = [int(m[1]) + 1 for k in checkpoints if (m := re.fullmatch(r'training/arc_(?:stk|dur)/checkpoints/epoch_epoch=(\d+)\.ckpt', k))]
            saved_epochs = max(epochs, default=0)
        else:
            prefix = f"experiments/arc-oat-20260919/{row['training_run_id']}/"
            assert not client.list_objects_v2(Bucket='rldb', Prefix=prefix, MaxKeys=1).get('KeyCount'), 'Unexpected artifacts require explicit inspection'
        retry_downloads(task)
        body['tasks'].append(task)
        new_training[cell] = task
        plan['training_replacements'].append({'cell': cell, 'workflow': name, 'pool': pool,
            'old_run': row['training_run_id'], 'new_run': task['environment']['RUN_ID'], 'training_task': task['name'],
            'resume_request': request, 'source_saved_epochs_at_least': saved_epochs,
            'source_scheduler_state': states[(row['training_workflow_id'], row['training_task'])]})

    evaluation_rows = [r for r in suite_rows if (observed[(suite, r['variant'], r['mode'])]['evaluation_status'] or {}).get('state') != 'POLICY_EVALUATION_COMPLETE']
    # Ready policies and resumed episodes take the first evaluation slots.
    evaluation_rows.sort(key=lambda r: (bool((suite, r['variant'], r['mode']) in new_training),
        not bool(observed[(suite, r['variant'], r['mode'])].get('records')), indices[(r['variant'], r['mode'])]))
    evaluation_lanes = [None] * eval_lanes
    for slot, row in enumerate(evaluation_rows):
        cell = (suite, row['variant'], row['mode'])
        obs = observed[cell]
        index = indices[(row['variant'], row['mode'])]
        assert states[(row['evaluation_workflow_id'], row['evaluation_task'])] in {'FAILED_CANCELED', 'FAILED_UPSTREAM'}
        run = f"{name}-{row['variant'].replace('_', '-')}-{row['mode']}-eval"
        request, artifact_hashes, resume_from = None, None, None
        if cell in new_training:
            assert not obs.get('records') and not obs['evaluation_status']
            task = copy.deepcopy(templates[f'evaluate-{index:02d}'])
            task['environment']['RUN_ID'] = run
            task['inputs'] = [{'task': new_training[cell]['name']}]
            source_run = new_training[cell]['environment']['RUN_ID']
        else:
            source_run = row['training_run_id']
            identity = {'source_run': source_run, 'source_commit': SOURCE, 'suite': suite,
                'method': 'arc_' + row['mode'], 'arc_stream_variant': row['variant'],
                'total_optimizer_steps': campaign()['optimizer_steps'][suite],
                'stream_reference_run': references[row['mode']]['evaluation_run_id']}
            ready = checkpoint_ready(client, identity)
            assert ready is not None
            request, artifact_hashes = ready
            validate_request(request)
            if obs.get('records'):
                assert request == obs['request'], 'Resumed episodes must retain the identical request'
                resume_from = row['evaluation_run_id']
            ready_spec = evaluation_workflow(SOURCE, run, request, workers=5, resume_evaluation_from=resume_from)
            task = ready_spec['workflow']['tasks'][0]
            task['name'] = f'evaluate-{index:02d}'
            task['resource'] = 'evaluation'
            task['inputs'] = []
        task['inputs'].append({'task': f"reference-{row['mode']}-ready"})
        lane = slot % eval_lanes
        if evaluation_lanes[lane]:
            task['inputs'].append({'task': evaluation_lanes[lane]})
        evaluation_lanes[lane] = task['name']
        retry_downloads(task)
        body['tasks'].append(task)
        plan['evaluation_replacements'].append({'cell': cell, 'workflow': name, 'pool': pool,
            'old_run': row['evaluation_run_id'], 'new_run': run, 'evaluation_task': task['name'],
            'source_run': source_run, 'reference_run': references[row['mode']]['evaluation_run_id'],
            'resume_from': resume_from, 'saved_episodes': len(obs.get('records', [])),
            'request': request, 'artifact_hashes': artifact_hashes,
            'source_scheduler_state': states[(row['evaluation_workflow_id'], row['evaluation_task'])]})

    task_names = {t['name'] for t in body['tasks']}
    assert len(task_names) == len(body['tasks'])
    available = set()
    remaining = {t['name']: {x['task'] for x in t.get('inputs', [])} for t in body['tasks']}
    assert all(deps <= task_names for deps in remaining.values())
    while remaining:
        ready = {n for n, deps in remaining.items() if deps <= available}
        assert ready, 'Dependency cycle'
        available |= ready
        remaining = {n: deps for n, deps in remaining.items() if n not in ready}
    for task in body['tasks']:
        resource = body['resources'][task.get('resource', 'default')]
        assert resource['platform'] == 'ovx-l40s'
        if task['name'].startswith('train-'):
            assert resource['gpu'] == 4
        if task['name'].startswith('evaluate-'):
            assert resource['gpu'] == 1 and resource['memory'] == '120Gi'
            assert task['environment']['EVALUATION_WORKERS'] == '5'
    path = ROOT / 'specs' / (name + '.yaml')
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    plan['specs'].append({'name': name, 'pool': pool, 'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'training_cells': len(training_rows), 'evaluation_cells': len(evaluation_rows),
        'training_lanes': min(6, len(training_rows)), 'evaluation_lanes': eval_lanes})
    print(json.dumps(plan['specs'][-1]), flush=True)

assert len(plan['training_replacements']) == 43
assert sum(bool(r['resume_request']) for r in plan['training_replacements']) == 18
assert len(plan['evaluation_replacements']) == 77
assert sum(bool(r['resume_from']) for r in plan['evaluation_replacements']) == 10
(ROOT / 'recovery-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps({'training_resumes': 18, 'fresh_training': 25, 'evaluation_resumes': 10,
                  'completed_policies_preserved': 45, 'completed_evaluations_preserved': 11}), flush=True)
