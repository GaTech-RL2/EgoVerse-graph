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
fallback = json.loads((ROOT / 'fallback/plan.json').read_text())
original = json.loads((DEST / 'launch.json').read_text())
assert original['schema'] == 7
manifest = copy.deepcopy(original)
data = json.loads((ROOT / 'data.json').read_text())
observed = {(r['suite'], r['variant'], r['mode']): r for r in data['runs']}


def write(name, value):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


write('launch-l40s03-attempt.json', original)
shutil.copyfile(DEST / 'workflows.json', DEST / 'workflows-l40s03-attempt.json')
shutil.copyfile(DEST / 'status.json', DEST / 'status-l40s03-attempt.json')
training_count = evaluation_count = 0
for spec in fallback:
    old_id, new_id = spec['old_name'] + '-1', spec['name'] + '-1'
    receipt = json.loads((ROOT / 'fallback' / (spec['name'] + '-submission.json')).read_text())
    assert receipt['name'] == new_id
    for row in manifest['runs']:
        if old_id not in (row['training_workflow_id'], row['evaluation_workflow_id']):
            continue
        row.setdefault('assignment_history', []).append({k: copy.deepcopy(v) for k, v in row.items() if k != 'assignment_history'})
        for kind in ('training', 'evaluation'):
            if row[kind + '_workflow_id'] == old_id:
                row[kind + '_workflow_id'] = new_id
                row[kind + '_run_id'] = row[kind + '_run_id'].replace(spec['old_name'], spec['name'])
                row[kind + '_artifact_prefix'] = 's3://rldb/experiments/arc-oat-20260919/' + row[kind + '_run_id'] + '/'
                if kind == 'training':
                    row['workflow_id'] = new_id
                    training_count += 1
                else:
                    evaluation_count += 1
    old_workflow = next(w for w in manifest['workflows'] if w['workflow_id'] == old_id)
    workflow = copy.deepcopy(old_workflow)
    for kind in ('training', 'evaluation'):
        old_workflow['superseded_' + kind + '_tasks'] = sorted(set(old_workflow.get('superseded_' + kind + '_tasks', [])) | set(old_workflow['canonical_' + kind + '_tasks']))
        old_workflow['canonical_' + kind + '_tasks'] = []
    old_workflow['role'] = 'quota_preempted_during_bootstrap'
    accepted = json.loads((ROOT / 'fallback' / (new_id + '-accepted.json')).read_text())
    path = DEST / 'preparation' / (spec['name'] + '.yaml.gz')
    raw = Path(spec['file']).read_bytes()
    path.write_bytes(gzip.compress(raw, mtime=0))
    assert hashlib.sha256(raw).hexdigest() == spec['sha256']
    workflow.update(workflow_id=new_id, pool='groot-l40s-01', url=accepted['overview'],
                    submitted_at_utc=accepted['submit_time'], role='quota_recovery_on_l40s01',
                    spec={'file': str(path.relative_to(DEST)), 'sha256': spec['sha256'], 'uncompressed_bytes': len(raw)})
    manifest['workflows'].append(workflow)
    write('submissions/' + spec['name'] + '.json', receipt)
assert (training_count, evaluation_count) == (43, 57)
manifest.update(schema=8, recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                supersedes='launch-l40s03-attempt.json', pools=['groot-l40s-01'], pool='groot-l40s-01')
manifest['notes'] += [
    'L40S-03 quota enforcement canceled all three new workflows within a minute of bootstrap on October 5.',
    'All three new artifact prefixes were verified empty before preserving the original checkpoint and episode requests on L40S-01.',
    'The accepted Object evaluation workflow stays in place; Spatial, Goal and LIBERO-10 move to arc-st-20261005 workflows.',
    'All six partial training checkpoints per suite precede fresh training in the lane order.',
]
for kind in ('training', 'evaluation'):
    assert len({r[kind + '_run_id'] for r in manifest['runs']}) == 88
names = [s['name'] + '-1' for s in fallback] + ['arc-ss-20261005-object-1']
live = []
for name in names:
    for attempt in range(3):
        r = subprocess.run(['osmo', 'workflow', 'query', name, '--format-type', 'json'], capture_output=True, text=True, timeout=50)
        if r.returncode == 0:
            break
    r.check_returncode()
    q = json.loads(r.stdout)
    (ROOT / 'fallback' / (name + '-live.json')).write_text(json.dumps(q, indent=2) + '\n')
    tasks = [{k: t.get(k) for k in ('name', 'status', 'start_time', 'end_time')} for g in q['groups'] for t in g['tasks']]
    live.append({**{k: q[k] for k in ('name', 'status', 'pool', 'priority', 'submit_time')},
                 'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tasks': tasks})
    print(json.dumps({'workflow': name, 'status': q['status'], 'tasks': dict(Counter(t['status'] for t in tasks))}), flush=True)
states = {(w['name'], t['name']): t['status'] for w in live for t in w['tasks']}
reports = []
for row in manifest['runs']:
    obs = observed[(row['suite'], row['variant'], row['mode'])]
    for kind, completed in (('training', 'TRAINING_COMPLETE'), ('evaluation', 'POLICY_EVALUATION_COMPLETE')):
        if (row[kind + '_workflow_id'], row[kind + '_task']) not in states:
            assert obs[kind + '_status']['state'] == completed
    reports.append({**{k: row[k] for k in ('suite', 'variant', 'mode', 'training_run_id', 'evaluation_run_id', 'training_workflow_id', 'evaluation_workflow_id')},
        'training_scheduler_state': states.get((row['training_workflow_id'], row['training_task']), 'COMPLETED'),
        'evaluation_scheduler_state': states.get((row['evaluation_workflow_id'], row['evaluation_task']), 'COMPLETED'),
        'saved_evaluation_run': obs['evaluation_run_id'], 'saved_evaluation_episodes': len(obs.get('records', [])),
        'saved_success_rate_percent': obs.get('success_rate_percent'),
        'saved_score_is_final': (obs['evaluation_status'] or {}).get('state') == 'POLICY_EVALUATION_COMPLETE'})
write('launch.json', manifest)
write('workflows.json', live)
write('status.json', {'recorded_at_utc': manifest['recorded_at_utc'], 'result_snapshot': '../status_174115/summary.json',
    'training_complete': 45, 'final_evaluations': 11, 'partial_evaluations_preserved': 10,
    'note': 'RUNNING includes bootstrap; saved partial scores belong to the recorded source evaluation.', 'runs': reports})
write('preparation/fallback-plan.json', fallback)
for source in ('fallback_l40s01.py', 'record_fallback.py'):
    shutil.copyfile(ROOT / source, DEST / 'preparation' / source)
print(json.dumps({'training_scheduler': dict(Counter(r['training_scheduler_state'] for r in reports)),
                  'evaluation_scheduler': dict(Counter(r['evaluation_scheduler_state'] for r in reports))}), flush=True)
