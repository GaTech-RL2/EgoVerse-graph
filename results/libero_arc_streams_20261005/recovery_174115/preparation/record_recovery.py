import copy
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
VAL = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001')
DEST = VAL / 'results/libero_arc_streams_20261005/recovery_174115'
plan = json.loads((ROOT / 'recovery-plan.json').read_text())
previous_path = Path(plan['previous_manifest'])
old = json.loads(previous_path.read_text())
manifest = copy.deepcopy(old)
rows = {(r['suite'], r['variant'], r['mode']): r for r in manifest['runs']}
old_rows = {(r['suite'], r['variant'], r['mode']): r for r in old['runs']}
progress = json.loads((ROOT / 'data.json').read_text())
observed = {(r['suite'], r['variant'], r['mode']): r for r in progress['runs']}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, data):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


for replacement in plan['training_replacements']:
    row = rows[tuple(replacement['cell'])]
    row.setdefault('assignment_history', []).append({k: copy.deepcopy(v) for k, v in row.items() if k != 'assignment_history'})
    assert row['training_run_id'] == replacement['old_run']
    row.update(training_run_id=replacement['new_run'], training_workflow_id=replacement['workflow'] + '-1',
               workflow_id=replacement['workflow'] + '-1', training_task=replacement['training_task'],
               launcher_commit=plan['host_commit'], recovery='checkpoint_resume' if replacement['resume_request'] else 'unstarted_cell')
    for key in ('resume_request', 'resume_helper_sha256', 'bootstrap_repair', 'training_state', 'source_saved_progress_percent_at_least'):
        row.pop(key, None)
    if replacement['resume_request']:
        row['resume_request'] = replacement['resume_request']
        row['source_saved_epochs_at_least'] = replacement['source_saved_epochs_at_least']
        row['source_saved_progress_percent_at_least'] = round(100 * replacement['source_saved_epochs_at_least'] / 5001, 2)

for replacement in plan['evaluation_replacements']:
    row = rows[tuple(replacement['cell'])]
    assert row['evaluation_run_id'] == replacement['old_run']
    if row == old_rows[tuple(replacement['cell'])]:
        row.setdefault('assignment_history', []).append({k: copy.deepcopy(v) for k, v in row.items() if k != 'assignment_history'})
    row.update(evaluation_run_id=replacement['new_run'], evaluation_workflow_id=replacement['workflow'] + '-1',
               evaluation_task=replacement['evaluation_task'], reference_evaluation_run=replacement['reference_run'])
    for key in ('evaluation_resume_from', 'evaluation_request', 'evaluation_repair', 'checkpoint_readiness_task'):
        row.pop(key, None)
    if replacement['request']:
        row['evaluation_request'] = replacement['request']
    if replacement['resume_from']:
        row['evaluation_resume_from'] = replacement['resume_from']
        row['saved_evaluation_episodes_before_resume'] = replacement['saved_episodes']
    assert row['training_run_id'] == replacement['source_run']

for row in rows.values():
    for kind in ('training', 'evaluation'):
        row[kind + '_artifact_prefix'] = 's3://rldb/experiments/arc-oat-20260919/' + row[kind + '_run_id'] + '/'
    assert row['reference_evaluation_run'] == rows[(row['suite'], 'reference', row['mode'])]['evaluation_run_id']

for workflow in manifest['workflows']:
    workflow['spec']['file'] = os.path.relpath((previous_path.parent / workflow['spec']['file']).resolve(), DEST)
    assert (DEST / workflow['spec']['file']).exists()
    for kind in ('training', 'evaluation'):
        canonical = {r[kind + '_task'] for r in rows.values() if r[kind + '_workflow_id'] == workflow['workflow_id']}
        previous = {r[kind + '_task'] for r in old['runs'] if r[kind + '_workflow_id'] == workflow['workflow_id']}
        workflow['canonical_' + kind + '_tasks'] = sorted(canonical)
        workflow['superseded_' + kind + '_tasks'] = sorted(set(workflow.get('superseded_' + kind + '_tasks', [])) | (previous - canonical))

live = []
for spec in plan['specs']:
    name = spec['name'] + '-1'
    receipt = json.loads((ROOT / 'submissions' / (spec['name'] + '.json')).read_text())
    assert receipt['name'] == name
    for attempt in range(3):
        result = subprocess.run(['osmo', 'workflow', 'query', name, '--format-type', 'json'], capture_output=True, text=True, timeout=50)
        if result.returncode == 0:
            break
    result.check_returncode()
    query = json.loads(result.stdout)
    (ROOT / (name + '-live.json')).write_text(json.dumps(query, indent=2) + '\n')
    tasks = [{k: t.get(k) for k in ('name', 'status', 'start_time', 'end_time')} for g in query['groups'] for t in g['tasks']]
    live.append({**{k: query.get(k) for k in ('name', 'status', 'pool', 'priority', 'submit_time', 'start_time')},
                 'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tasks': tasks})
    document = yaml.safe_load(Path(spec['file']).read_text())['workflow']
    names = {t['name'] for t in document['tasks']}
    assigned = [r for r in rows.values() if name in (r['training_workflow_id'], r['evaluation_workflow_id'])]
    for r in assigned:
        for kind in ('training', 'evaluation'):
            if r[kind + '_workflow_id'] == name:
                assert r[kind + '_task'] in names
    path = DEST / 'preparation' / (spec['name'] + '.yaml.gz')
    assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == spec['sha256']
    manifest['workflows'].append({'suite': assigned[0]['suite'], 'workflow_id': name,
        'url': query['overview'], 'role': 'quota_preemption_recovery', 'pool': spec['pool'], 'priority': 'NORMAL',
        'submitted_at_utc': query['submit_time'], 'resources': document['resources'], 'launcher_commit': plan['host_commit'],
        'training_lanes': spec['training_lanes'], 'evaluation_lanes': spec['evaluation_lanes'],
        **{'canonical_' + kind + '_tasks': [r[kind + '_task'] for r in assigned if r[kind + '_workflow_id'] == name] for kind in ('training', 'evaluation')},
        'spec': {'file': str(path.relative_to(DEST)), 'sha256': spec['sha256'], 'uncompressed_bytes': len(Path(spec['file']).read_bytes())}})
    write('submissions/' + spec['name'] + '.json', receipt)

manifest.update(schema=7, recorded_at_utc=now, supersedes='../../libero_arc_streams_20261004/recovery_211651/launch.json',
                pool='Object evaluation on groot-l40s-01; Spatial, Goal and LIBERO-10 on groot-l40s-03',
                training_completed_before_recovery=45, checkpoint_resumes=18, training_cells_reassigned=43,
                evaluation_cells_reassigned=77, max_concurrent_training_gpus=72, max_concurrent_evaluation_gpus=24,
                notes=[
                    'Pool quota enforcement canceled the replacement workflows around 2026-10-04 23:52 UTC.',
                    'All 22 Object policies completed training in the surviving workflow; its old reference waiter is obsolete.',
                    'Preserves 45 completed policies, 11 final evaluations, 18 partial training checkpoints and ten partial evaluations.',
                    'Four validated workflows recover the 43 pending policies and 77 pending evaluations at NORMAL priority on L40S.',
                    'Completed-policy evaluations and saved partial episodes are placed first in the evaluation lanes.',
                    'Eighteen training lanes request 72 GPUs; 24 evaluation lanes request 24 GPUs. These are ceilings, not guarantees.',
                    'Old canceled workflows and intentional superseded tails are historical; use canonical task assignments.',
                    'No automatic requeue or protected capacity is implied. Model, data, codec, training budgets and rollout protocol are unchanged.',
                ])
assert len(rows) == len(manifest['runs']) == 88
assert sum(a['training_run_id'] != b['training_run_id'] for a, b in zip(old['runs'], manifest['runs'])) == 43
assert sum(a['evaluation_run_id'] != b['evaluation_run_id'] for a, b in zip(old['runs'], manifest['runs'])) == 77
for kind in ('training', 'evaluation'):
    assert len({r[kind + '_run_id'] for r in rows.values()}) == 88
write('launch.json', manifest)
write('workflows.json', live)
task_states = {(w['name'], t['name']): t['status'] for w in live for t in w['tasks']}
report = []
for cell, row in rows.items():
    obs = observed[cell]
    if (row['training_workflow_id'], row['training_task']) not in task_states:
        assert obs['training_status']['state'] == 'TRAINING_COMPLETE'
    if (row['evaluation_workflow_id'], row['evaluation_task']) not in task_states:
        assert obs['evaluation_status']['state'] == 'POLICY_EVALUATION_COMPLETE'
    report.append({**{k: row[k] for k in ('suite', 'variant', 'mode', 'training_run_id', 'evaluation_run_id', 'training_workflow_id', 'evaluation_workflow_id')},
        'training_scheduler_state': task_states.get((row['training_workflow_id'], row['training_task']), 'COMPLETED'),
        'evaluation_scheduler_state': task_states.get((row['evaluation_workflow_id'], row['evaluation_task']), 'COMPLETED'),
        'saved_evaluation_run': obs['evaluation_run_id'], 'saved_evaluation_episodes': len(obs.get('records', [])),
        'saved_success_rate_percent': obs.get('success_rate_percent'),
        'saved_score_is_final': (obs['evaluation_status'] or {}).get('state') == 'POLICY_EVALUATION_COMPLETE'})
write('status.json', {'recorded_at_utc': now, 'result_snapshot': '../status_174115/summary.json',
    'training_complete': 45, 'final_evaluations': 11, 'partial_evaluations_preserved': 10,
    'note': 'Scheduler RUNNING includes container setup. Saved partial scores belong to the recorded old evaluation run until restoration.', 'runs': report})
shutil.copyfile(Path(__file__), DEST / 'preparation/record_recovery.py')
print(json.dumps({'canonical_cells': 88, 'training_reassigned': 43, 'evaluation_reassigned': 77,
    'training_scheduler': dict(Counter(r['training_scheduler_state'] for r in report)),
    'evaluation_scheduler': dict(Counter(r['evaluation_scheduler_state'] for r in report))}), flush=True)
