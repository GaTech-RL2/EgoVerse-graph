import copy
import csv
import datetime
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
VAL = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001')
DEST = VAL / 'results/libero_arc_streams_20261004/recovery_211651'
PREVIOUS = VAL / 'results/libero_arc_streams_20261004/status_005629/launch.json'
plan = json.loads((ROOT / 'recovery-plan.json').read_text())
old = json.loads(PREVIOUS.read_text())
manifest = copy.deepcopy(old)
progress = json.loads((ROOT / 'progress.json').read_text())
progress_by_cell = {(r['suite'], r['variant'], r['mode']): r for r in progress['runs']}
rows = {(r['suite'], r['variant'], r['mode']): r for r in manifest['runs']}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, data):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def preserve(path, name):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)


def history(row):
    previous = {k: copy.deepcopy(v) for k, v in row.items() if k != 'assignment_history'}
    row.setdefault('assignment_history', []).append(previous)


for replacement in plan['training_replacements']:
    row = rows[tuple(replacement['cell'])]
    assert row['training_run_id'] == replacement['old_run']
    history(row)
    row.update(training_run_id=replacement['new_run'], evaluation_run_id=replacement['new_run'] + '-eval',
               training_workflow_id=replacement['workflow'] + '-1', evaluation_workflow_id=replacement['workflow'] + '-1',
               workflow_id=replacement['workflow'] + '-1', training_task=replacement['training_task'],
               evaluation_task=replacement['evaluation_task'], launcher_commit=plan['host_commit'],
               recovery='checkpoint_resume' if replacement['resume_request'] else 'unstarted_cell')
    for key in ('resume_request', 'resume_helper_sha256', 'bootstrap_repair', 'evaluation_repair'):
        row.pop(key, None)
    if replacement['resume_request']:
        row['resume_request'] = replacement['resume_request']
        row['source_saved_progress_percent_at_least'] = progress_by_cell[tuple(replacement['cell'])]['saved_progress_percent_at_least']

for replacement in plan['evaluation_replacements']:
    row = rows[tuple(replacement['cell'])]
    assert row['evaluation_run_id'] == replacement['old_run']
    history(row)
    row.update(evaluation_run_id=replacement['new_run'], evaluation_workflow_id=replacement['new_run'] + '-1',
               evaluation_task='libero', evaluation_request=replacement['request'])
    if replacement['resume_from']:
        row['evaluation_resume_from'] = replacement['resume_from']
    if replacement['request'].get('stream_reference_run'):
        row['reference_evaluation_run'] = replacement['request']['stream_reference_run']

for replacement in plan['object_evaluation_replacements']:
    row = rows[tuple(replacement['cell'])]
    assert row['evaluation_run_id'] == replacement['old_run']
    history(row)
    row.update(evaluation_run_id=replacement['new_run'], evaluation_workflow_id='arc-sv-20261004-object-1',
               evaluation_task=replacement['evaluation_task'], reference_evaluation_run=replacement['reference_run'],
               checkpoint_readiness_task=replacement['evaluation_task'].replace('evaluate-', 'checkpoint-'))

for row in rows.values():
    row['training_artifact_prefix'] = f"s3://rldb/experiments/arc-oat-20260919/{row['training_run_id']}/"
    row['evaluation_artifact_prefix'] = f"s3://rldb/experiments/arc-oat-20260919/{row['evaluation_run_id']}/"
    if row['variant'] == 'reference':
        row['reference_evaluation_run'] = row['evaluation_run_id']

for workflow in manifest['workflows']:
    if 'spec' in workflow:
        workflow['spec']['file'] = os.path.relpath((PREVIOUS.parent / workflow['spec']['file']).resolve(), DEST)
        assert (DEST / workflow['spec']['file']).exists()
    canonical_training = {r['training_task'] for r in rows.values() if r['training_workflow_id'] == workflow['workflow_id']}
    canonical_evaluation = {r['evaluation_task'] for r in rows.values() if r['evaluation_workflow_id'] == workflow['workflow_id']}
    workflow['canonical_training_tasks'] = sorted(canonical_training)
    workflow['canonical_evaluation_tasks'] = sorted(canonical_evaluation)
    retired_training = {r['training_task'] for r in old['runs'] if r['training_workflow_id'] == workflow['workflow_id']} - canonical_training
    retired_evaluation = {r['evaluation_task'] for r in old['runs'] if r['evaluation_workflow_id'] == workflow['workflow_id']} - canonical_evaluation
    workflow['superseded_training_tasks'] = sorted(set(workflow.get('superseded_training_tasks', [])) | retired_training)
    workflow['superseded_evaluation_tasks'] = sorted(set(workflow.get('superseded_evaluation_tasks', [])) | retired_evaluation)
    if workflow['workflow_id'] == 'arc-sx-20261003-object-1':
        workflow['superseded_waiter_tasks'] = ['reference-dur-ready']

for spec in plan['specs']:
    accepted = json.loads((ROOT / 'accepted' / (spec['name'] + '-1.json')).read_text())
    document = yaml.safe_load(Path(spec['file']).read_text())['workflow']
    compressed = DEST / 'preparation' / (spec['name'] + '.yaml.gz')
    if not compressed.exists():
        compressed.parent.mkdir(parents=True, exist_ok=True)
        compressed.write_bytes(gzip.compress(Path(spec['file']).read_bytes(), mtime=0))
    assert hashlib.sha256(gzip.decompress(compressed.read_bytes())).hexdigest() == spec['sha256']
    assigned = [r for r in rows.values() if accepted['name'] in (r['training_workflow_id'], r['evaluation_workflow_id'])]
    tasks = {t['name'] for t in document['tasks']}
    for row in assigned:
        if row.get('checkpoint_readiness_task'):
            assert row['checkpoint_readiness_task'] in tasks
        for kind in ('training', 'evaluation'):
            if row[kind + '_workflow_id'] == accepted['name']:
                assert row[kind + '_task'] in tasks
    manifest['workflows'].append({'suite': assigned[0]['suite'], 'workflow_id': accepted['name'],
        'url': accepted['overview'], 'role': spec['role'], 'pool': accepted['pool'], 'priority': accepted['priority'],
        'submitted_at_utc': accepted['submit_time'], 'resources': document['resources'], 'launcher_commit': plan['host_commit'],
        'canonical_training_tasks': [r['training_task'] for r in assigned if r['training_workflow_id'] == accepted['name']],
        'canonical_evaluation_tasks': [r['evaluation_task'] for r in assigned if r['evaluation_workflow_id'] == accepted['name']],
        'spec': {'file': os.path.relpath(compressed, DEST), 'sha256': spec['sha256'], 'uncompressed_bytes': len(Path(spec['file']).read_bytes())}})
    preserve(ROOT / 'submissions' / (spec['name'] + '.json'), 'submissions/' + spec['name'] + '.json')

for key in ('additional_training_slots', 'additional_training_gpus_requested', 'retained_checkpoint_recovery_slots',
            'fresh_training_slots_after_recovery', 'fresh_training_cells_at_expansion'):
    manifest.pop(key, None)
manifest.update(schema=6, recorded_at_utc=now, supersedes='../status_005629/launch.json',
                pool='mixed; recovered work on groot-l40s-03; surviving work on groot-l40s-01',
                pools=['groot-l40s-01', 'groot-l40s-03'], training_completed_before_recovery=22, checkpoint_resumes=18,
                training_cells_reassigned=54, evaluation_cells_reassigned=78,
                max_concurrent_training_gpus=96, max_concurrent_evaluation_gpus=14,
                previous_notes=manifest['notes'], notes=[
                    'Shared-pool quota enforcement preempted seven workflows at approximately 2026-10-04 08:27 UTC.',
                    'Eighteen interrupted policies resume immutable saved checkpoints; thirty-six unstarted policies move with them.',
                    'Twenty-two completed policies and six surviving Object training lanes are preserved.',
                    'Six completed-policy evaluations are isolated; three recover saved episodes.',
                    'Eighteen Object candidate evaluations use a CPU readiness relay and recovered DUR reference; healthy training is unchanged.',
                    'GPU ceilings are configured concurrency, not current allocations or protected capacity.',
                    'Use explicit training/evaluation workflow IDs and canonical task lists; superseded task failures are historical.',
                    'OSMO RUNNING can include dependency setup; optimizer activity requires runtime/artifact evidence.',
                ])
assert len(rows) == len(manifest['runs']) == 88
assert len({r['training_run_id'] for r in rows.values()}) == 88
assert len({r['evaluation_run_id'] for r in rows.values()}) == 88
assert sum(a['training_run_id'] != b['training_run_id'] for a, b in zip(old['runs'], manifest['runs'])) == 54
assert sum(a['evaluation_run_id'] != b['evaluation_run_id'] for a, b in zip(old['runs'], manifest['runs'])) == 78
for row in rows.values():
    ref = rows[(row['suite'], 'reference', row['mode'])]
    assert row['reference_evaluation_run'] == ref['evaluation_run_id']
write('launch.json', manifest)
preserve(ROOT / 'deploy_recovery.py', 'preparation/deploy_recovery.py')
preserve(Path(__file__), 'preparation/record_recovery.py')
preserve(ROOT / 'object-evaluation-handoff.json', 'object-evaluation-handoff.json')

# Keep raw workflow responses local: some historical scheduler messages name
# unrelated workloads. The durable report contains only this sweep's states.
workflow_reports = []
task_states = {}
for path in sorted((ROOT / 'live').glob('*.json')):
    query = json.loads(path.read_text())
    tasks = [{k: t.get(k) for k in ('name', 'status', 'start_time', 'end_time')} for g in query['groups'] for t in g['tasks']]
    task_states.update({(query['name'], t['name']): t['status'] for t in tasks})
    workflow_reports.append({**{k: query.get(k) for k in ('name', 'status', 'pool', 'priority', 'submit_time', 'start_time')}, 'tasks': tasks})
write('workflows.json', {'observed_at_utc': now, 'workflows': workflow_reports})
report_rows = []
startup = json.loads((ROOT / 'startup-artifacts.json').read_text())
startup_by_run = {r['run']: r for r in startup['runs']}
preserve(ROOT / 'startup-artifacts.json', 'startup-artifacts.json')
for row in manifest['runs']:
    p = progress_by_cell[(row['suite'], row['variant'], row['mode'])]
    changed_training = row['training_run_id'] != p['training_run_id']
    changed_evaluation = row['evaluation_run_id'] != p['evaluation_run_id']
    report = {k: row[k] for k in ('suite', 'variant', 'mode', 'training_run_id', 'evaluation_run_id', 'training_workflow_id', 'evaluation_workflow_id')}
    report.update(training_scheduler_state=task_states.get((row['training_workflow_id'], row['training_task']), 'COMPLETED' if (p['training_status'] or {}).get('state') == 'TRAINING_COMPLETE' else 'HISTORICAL'),
                  evaluation_scheduler_state=task_states.get((row['evaluation_workflow_id'], row['evaluation_task']), 'COMPLETED' if p.get('scores') else 'HISTORICAL'),
                  saved_training_percent=p.get('saved_progress_percent_at_least', 0),
                  saved_training_artifact_run=p['training_run_id'],
                  training_artifact_state=(p['training_status'] or {}).get('state', 'NO_ARTIFACTS') if not changed_training else 'RESUME_SUBMITTED' if row.get('resume_request') else 'UNSTARTED_REQUEUED',
                  saved_evaluation_episodes=(p.get('evaluation_progress') or {}).get('episodes', 0),
                  saved_evaluation_artifact_run=p['evaluation_run_id'],
                  evaluation_artifact_state=(p['evaluation_status'] or {}).get('state', 'NO_ARTIFACTS') if not changed_evaluation else 'RECOVERY_SUBMITTED',
                  success_rate_percent=round(p['scores']['mean_success_rate'] * 100, 2) if p.get('scores') else None)
    for kind in ('training', 'evaluation'):
        latest = startup_by_run.get(row[kind + '_run_id'], {})
        if latest.get('status.json'):
            report[kind + '_artifact_state'] = latest['status.json']['data']['state']
        if kind == 'evaluation' and latest.get('evaluation-progress.json'):
            report['saved_evaluation_episodes'] = latest['evaluation-progress.json']['data']['episodes']
            report['saved_evaluation_artifact_run'] = row['evaluation_run_id']
    if p.get('scores'):
        assert p['scores']['episodes'] == 2500
        assert not changed_evaluation
        report.update(scores=p['scores'], scores_receipt=p['scores_receipt'], evaluation_runtime_receipt=p['evaluation_runtime_receipt'])
        for file in ('scores.json', 'status.json', 'runtime.json'):
            preserve(ROOT / 'artifacts' / p['evaluation_run_id'] / file, f"scores/{row['suite']}/{row['variant']}/{row['mode']}/{file}")
    report_rows.append(report)
status = {'artifact_observed_at_utc': progress['observed_at_utc'], 'scheduler_observed_at_utc': now,
          'startup_artifact_observed_at_utc': startup['observed_at_utc'],
          'completed_training_policies': 22, 'completed_evaluations': sum(r['success_rate_percent'] is not None for r in report_rows),
          'new_workflows_accepted': 10, 'source_commit': manifest['source_commit'],
          'note': 'Saved progress belongs to the listed artifact run; replacement containers may still be setting up.', 'runs': report_rows}
write('status.json', status)
fields = [k for k in report_rows[0] if k not in ('scores', 'scores_receipt', 'evaluation_runtime_receipt')]
for file, selected in [('run-status.csv', report_rows), ('reference-results.csv', [r for r in report_rows if r['variant'] == 'reference']),
                       ('completed-results.csv', [r for r in report_rows if r['success_rate_percent'] is not None])]:
    with (DEST / file).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(selected)
print(json.dumps({'destination': str(DEST), 'cells': 88, 'training_reassigned': 54, 'evaluation_reassigned': 78,
                  'completed_evaluations': status['completed_evaluations'], 'training_scheduler': dict(Counter(r['training_scheduler_state'] for r in report_rows)),
                  'evaluation_scheduler': dict(Counter(r['evaluation_scheduler_state'] for r in report_rows))}))
