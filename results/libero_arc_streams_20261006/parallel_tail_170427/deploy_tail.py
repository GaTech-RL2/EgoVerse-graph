"""Submit the validated six-cell tail, retire empty old prefixes, and record ownership."""
import copy
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from collect import client
from prepare_tail import OUT, compact, query

ROOT = Path(__file__).parent
VAL = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001')
DEST = VAL / 'results/libero_arc_streams_20261006/parallel_tail_170427'
plan = json.loads((OUT / 'plan.json').read_text())
previous_path = Path(plan['previous_manifest'])
assert hashlib.sha256(previous_path.read_bytes()).hexdigest() == plan['previous_manifest_sha256']
assert hashlib.sha256(Path(plan['spec']).read_bytes()).hexdigest() == plan['spec_sha256']
assert plan['validation_exit_code'] == 0 and len(plan['replacements']) == 6
old_manifest = json.loads(previous_path.read_text())
old = query(plan['old_workflow'])
states = {t['name']: t['status'] for g in old['groups'] for t in g['tasks']}
assert all(states[n] == 'WAITING' for n in plan['retired_tasks'])

receipt_path = OUT / 'submission.json'
if receipt_path.exists():
    submitted = json.loads(receipt_path.read_text())
else:
    result = subprocess.run(['osmo', 'workflow', 'submit', plan['spec'], '--pool', plan['pool'], '--priority', plan['priority'], '--format-type', 'json'], capture_output=True, text=True, timeout=50)
    (OUT / 'submission.stderr').write_text(result.stderr)
    result.check_returncode()
    submitted = json.loads(result.stdout)
    receipt_path.write_text(json.dumps(submitted, indent=2) + '\n')
new_id = submitted['name']
assert new_id == plan['name'] + '-1'
accepted = query(new_id)
assert accepted['status'] in {'RUNNING', 'PENDING', 'WAITING', 'QUEUED', 'INITIALIZING', 'SCHEDULING', 'PROCESSING'}
assert accepted['pool'] == 'groot-l40s-01' and accepted['priority'] == 'NORMAL'
print(json.dumps({'accepted': new_id, 'status': accepted['status']}), flush=True)
(OUT / 'accepted.json').write_text(json.dumps(compact(accepted), indent=2) + '\n')
old = query(plan['old_workflow'])
states = {t['name']: t['status'] for g in old['groups'] for t in g['tasks']}
assert all(states[n] == 'WAITING' for n in plan['retired_tasks']), 'Old task started; replacement ownership must be rechecked'
new_names = {t['name'] for g in accepted['groups'] for t in g['tasks']}
assert new_names == set(plan['retired_tasks']) | {'reference-stk-ready', 'reference-dur-ready'}

# Empty-prefix markers invoke the existing overwrite guard before old training
# starts. Only the exact twelve retired descendants can fail; active work stays live.
handoff_path = OUT / 'handoff.json'
if handoff_path.exists():
    handoff = json.loads(handoff_path.read_text())
else:
    handoff = {'operation': 'parallelize_six_unstarted_libero10_cells',
               'old_workflow': plan['old_workflow'], 'new_workflow': new_id,
               'source_commit': plan['source_commit'], 'recorded_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'preserved_tasks': plan['preserved_tasks'], 'retired_tasks': plan['retired_tasks'],
               'retired_training_heads': plan['retired_training_heads'],
               'expected_retirement': 'Existing ArtifactUploader overwrite guards reject reserved old prefixes before training; only the moved evaluation descendants are then FAILED_UPSTREAM.'}
    handoff_path.write_text(json.dumps(handoff, indent=2) + '\n')
body = handoff_path.read_bytes()
sha = hashlib.sha256(body).hexdigest()
prefixes = []
for item in plan['replacements']:
    for kind in ('training', 'evaluation'):
        prefix = 'experiments/arc-oat-20260919/' + item['old_' + kind + '_run'] + '/'
        objects = client.list_objects_v2(Bucket='rldb', Prefix=prefix, MaxKeys=2).get('Contents', [])
        if objects:
            assert kind == 'training' and [o['Key'] for o in objects] == [prefix + 'orchestration-handoff.json']
            saved = client.get_object(Bucket='rldb', Key=objects[0]['Key'])
            assert saved['Body'].read() == body and saved['Metadata']['sha256'] == sha
        elif kind == 'training':
            prefixes.append(prefix)
for prefix in prefixes:
    client.put_object(Bucket='rldb', Key=prefix + 'orchestration-handoff.json', Body=body,
                      ContentType='application/json', Metadata={'sha256': sha}, IfNoneMatch='*')
markers = []
for item in plan['replacements']:
    key = 'experiments/arc-oat-20260919/' + item['old_training_run'] + '/orchestration-handoff.json'
    saved = client.get_object(Bucket='rldb', Key=key)
    assert saved['Body'].read() == body and saved['Metadata']['sha256'] == sha
    markers.append({'uri': 's3://rldb/' + key, 'sha256': sha})
(OUT / 'markers.json').write_text(json.dumps(markers, indent=2) + '\n')
print('Verified all six old-prefix guards; running tasks are preserved.', flush=True)

DEST.mkdir(parents=True, exist_ok=True)
manifest = copy.deepcopy(old_manifest)
by_cell = {(r['suite'], r['variant'], r['mode']): r for r in manifest['runs']}
for item in plan['replacements']:
    row = by_cell[tuple(item['cell'])]
    row.setdefault('assignment_history', []).append({k: copy.deepcopy(v) for k, v in row.items() if k != 'assignment_history'})
    for kind in ('training', 'evaluation'):
        assert row[kind + '_run_id'] == item['old_' + kind + '_run']
        row[kind + '_run_id'] = item['new_' + kind + '_run']
        row[kind + '_workflow_id'] = new_id
        row[kind + '_artifact_prefix'] = 's3://rldb/experiments/arc-oat-20260919/' + row[kind + '_run_id'] + '/'
    row.update(workflow_id=new_id, scheduling_change='unstarted_cell_moved_to_independent_lane', launcher_commit=plan['launcher_commit'])
for workflow in manifest['workflows']:
    path = (previous_path.parent / workflow['spec']['file']).resolve()
    assert path.is_file()
    workflow['spec']['file'] = os.path.relpath(path, DEST)
    for kind in ('training', 'evaluation'):
        canonical = {r[kind + '_task'] for r in manifest['runs'] if r[kind + '_workflow_id'] == workflow['workflow_id']}
        previous = {r[kind + '_task'] for r in old_manifest['runs'] if r[kind + '_workflow_id'] == workflow['workflow_id']}
        workflow['canonical_' + kind + '_tasks'] = sorted(canonical)
        workflow['superseded_' + kind + '_tasks'] = sorted(set(workflow.get('superseded_' + kind + '_tasks', [])) | (previous - canonical))
        if workflow['workflow_id'] == plan['old_workflow']:
            workflow['role'] = 'preserved_running_libero10_jobs_with_unstarted_tail_moved'
raw = Path(plan['spec']).read_bytes()
spec_path = DEST / (plan['name'] + '.yaml.gz')
spec_path.write_bytes(gzip.compress(raw, mtime=0))
manifest['workflows'].append({'suite': 'libero_10', 'workflow_id': new_id, 'url': accepted['overview'],
    'role': 'six_unstarted_cells_with_independent_training_and_evaluation', 'pool': plan['pool'],
    'submitted_at_utc': accepted['submit_time'], 'resources': yaml.safe_load(raw)['workflow']['resources'],
    'spec': {'file': spec_path.name, 'sha256': plan['spec_sha256'], 'uncompressed_bytes': len(raw)},
    'launcher_commit': plan['launcher_commit'], 'canonical_training_tasks': sorted(plan['retired_training_heads']),
    'canonical_evaluation_tasks': sorted(r['evaluation_task'] for r in plan['replacements']),
    'superseded_training_tasks': [], 'superseded_evaluation_tasks': []})
manifest.update(schema=9, recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    supersedes=os.path.relpath(previous_path, DEST), max_concurrent_training_gpus=88,
    max_concurrent_evaluation_gpus=22, scheduling_change='Remove six remaining LIBERO-10 training dependencies without altering any experiment recipe',
    score_snapshot='../status_170427/summary.json')
manifest['notes'] += ['Only six WAITING LIBERO-10 angular-driver, angular-scalar and all-scalar cells are reassigned. All twelve old training/evaluation prefixes were checked empty.',
    'After accepted replacement submission, six conditional old-prefix handoff markers prevent duplicate training through the existing overwrite guard. Existing running tasks and scores are preserved.',
    'All six replacements retain source 69ff3fab, original reference runs, replay proof, seed, model, optimizer, full training budget, GPU preflight and evaluation protocol. Only run identity and scheduling dependencies change.',
    'The old workflow may eventually be FAILED after its deliberately retired tail hits the prefix guards. Canonical task assignments, not the wrapper state, identify active work.']
assert len(manifest['runs']) == 88
for kind in ('training', 'evaluation'):
    assert len({r[kind + '_run_id'] for r in manifest['runs']}) == 88
    assert sum(r[kind + '_workflow_id'] == new_id for r in manifest['runs']) == 6
for before, after in zip(old_manifest['runs'], manifest['runs']):
    if tuple(after[k] for k in ('suite', 'variant', 'mode')) not in {tuple(i['cell']) for i in plan['replacements']}:
        assert before == after
    assert after['reference_evaluation_run'] == by_cell[(after['suite'], 'reference', after['mode'])]['evaluation_run_id']
(DEST / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
for name in ('plan.json', 'validation.txt', 'submission.json', 'submission.stderr', 'old-before.json', 'accepted.json', 'handoff.json', 'markers.json'):
    shutil.copyfile(OUT / name, DEST / name)
for name in ('prepare_tail.py', 'deploy_tail.py'):
    shutil.copyfile(ROOT / name, DEST / name)
new_status = compact(query(new_id))
old_status = compact(query(plan['old_workflow']))
(DEST / 'post-launch-workflows.json').write_text(json.dumps([old_status, new_status], indent=2) + '\n')
print(json.dumps({'canonical_manifest': str(DEST / 'launch.json'), 'workflow': new_id,
                  'new_tasks': {t['name']: t['status'] for t in new_status['tasks']}, 'preserved_cells': 82}), flush=True)
