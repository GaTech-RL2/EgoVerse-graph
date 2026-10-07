"""Export validated per-task scores and checkpoint-backed progress for this audit."""
import csv
import datetime
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
DEST = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001/results/libero_arc_streams_20261007/status_041459')
data = json.loads((ROOT / 'data.json').read_text())
progress = json.loads((ROOT / 'training-progress.json').read_text())
previous = json.loads(Path('/tmp/libero-streams-analysis/20261006T204959Z/training-progress.json').read_text())
previous_runs = {r['training_run_id']: r for r in previous['runs']}
by_run = {r['training_run_id']: r for r in data['runs']}


def write_csv(name, rows):
    with (DEST / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def parsed_time(value):
    return datetime.datetime.fromisoformat(value)


task_scores = []
for row in data['runs']:
    by_task = defaultdict(list)
    for record in row.get('records', []):
        by_task[record['task']].append(record)
    for task, records in sorted(by_task.items()):
        n = len(records)
        assert n <= 250
        task_scores.append({
            **{k: row[k] for k in ('suite', 'variant', 'mode', 'evaluation_run_id')},
            'task': task, 'episodes': n, 'planned_episodes': 250,
            'successes': sum(r['success'] for r in records),
            'success_rate_percent': 100 * sum(r['success'] for r in records) / n,
            'task_complete': n == 250,
            'suite_complete': row['evaluation_status']['state'] == 'POLICY_EVALUATION_COMPLETE',
            'observed_at_utc': data['observed_at_utc'],
        })
write_csv('per-task-policy-scores.csv', task_scores)

training_rows, estimates = [], []
for row in progress['runs']:
    receipt = row['checkpoint_manifest_receipt']
    raw = (ROOT / 'artifacts' / row['training_run_id'] / 'checkpoint-receipts.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['sha256']
    target = DEST / 'training-evidence' / row['training_run_id'] / 'checkpoint-receipts.json.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(raw, mtime=0))
    epochs = row['saved_epochs_at_least']
    training_rows.append({
        **{k: v for k, v in row.items() if k != 'checkpoint_manifest_receipt'},
        'saved_percent_at_least': 100 * epochs / row['planned_epochs'],
        'checkpoint_manifest_uploaded_at': receipt['last_modified'],
    })
    old = previous_runs.get(row['training_run_id'])
    if old and epochs > old['saved_epochs_at_least']:
        elapsed_hours = (parsed_time(receipt['last_modified']) - parsed_time(old['checkpoint_manifest_receipt']['last_modified'])).total_seconds() / 3600
        rate = (epochs - old['saved_epochs_at_least']) / elapsed_hours
        estimates.append({
            **{k: row[k] for k in ('suite', 'variant', 'mode', 'training_run_id', 'saved_epochs_at_least', 'planned_epochs')},
            'previous_saved_epochs_at_least': old['saved_epochs_at_least'],
            'previous_checkpoint_manifest_uploaded_at': old['checkpoint_manifest_receipt']['last_modified'],
            'checkpoint_manifest_uploaded_at': receipt['last_modified'],
            'observed_elapsed_hours': elapsed_hours,
            'observed_epochs_per_hour': rate,
            'estimated_training_hours_remaining': (row['planned_epochs'] - epochs) / rate,
        })
write_csv('training-progress.csv', training_rows)
shutil.copyfile(ROOT / 'training-progress.json', DEST / 'training-progress.json')
shutil.copyfile(ROOT / 'training_progress.py', DEST / 'training_progress.py')
shutil.copyfile(Path(__file__), DEST / 'export_progress.py')
(DEST / 'training-estimates.json').write_text(json.dumps({
    'observed_at_utc': progress['observed_at_utc'],
    'method': 'Extrapolate recent uploaded-epoch throughput between two checksummed checkpoint-manifest observations. Assumes uninterrupted training at the same throughput; evaluation time is additional.',
    'previous_progress_source': '../../libero_arc_streams_20261006/status_204959/training-progress.json',
    'runs': estimates,
}, indent=2) + '\n')
for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
    rows = [r for r in data['runs'] if r['suite'] == suite]
    print(suite, dict(Counter((r['evaluation_status'] or {}).get('state', 'PENDING') for r in rows)))
for group in ('axis_pair', 'angular_scalar'):
    selected = [r for r in estimates if (r['variant'] in ('x_yz', 'y_xz', 'z_xy')) == (group == 'axis_pair')]
    print(group, 'training_hours_remaining', round(min(r['estimated_training_hours_remaining'] for r in selected), 1), round(max(r['estimated_training_hours_remaining'] for r in selected), 1))
print('Per-task rows', len(task_scores))
