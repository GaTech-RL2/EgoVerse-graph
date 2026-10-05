import csv
import datetime
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from egomimic.benchmarks.libero.report import read_run, validate_full_protocol

ROOT = Path(__file__).parent
VAL = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001')
DEST = VAL / 'results/libero_arc_streams_20261005/preliminary_220338'
DEST.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / 'data.json').read_text())
rows = {(r['suite'], r['variant'], r['mode']): r for r in data['runs']}


def write(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2) + '\n')


def csv_file(name, values):
    with (DEST / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def indexed(row):
    return {tuple(r[k] for k in ('task', 'repetition', 'trial', 'seed')): r for r in row.get('records', [])}


def pair(candidate, control, kind):
    a, b = indexed(candidate), indexed(control)
    keys = a.keys() & b.keys()
    if not keys:
        return None
    for field in ('suite', 'max_episode_steps', 'n_obs_steps', 'n_action_steps', 'horizon', 'data_context',
                  'observations_sha256', 'use_ema', 'oat_commit', 'libero_commit', 'plan'):
        assert candidate['protocol'][field] == control['protocol'][field], field
    for field in ('waypoints', 'dt', 'max_translation', 'max_rotation_degrees'):
        assert candidate['protocol']['representation'][field] == control['protocol']['representation'][field], field
    if kind != 'duration_vs_stk':
        assert candidate['mode'] == control['mode']
    assert all(a[k]['initial_state_sha256'] == b[k]['initial_state_sha256'] for k in keys)
    n = len(keys)
    task_rows = []
    for task in sorted({k[0] for k in keys}):
        subset = [k for k in keys if k[0] == task]
        x, y = [sum(records[k]['success'] for k in subset) / len(subset) for records in (a, b)]
        task_rows.append({'task': task, 'episodes': len(subset), 'candidate_sr_percent': 100 * x,
                          'control_sr_percent': 100 * y, 'delta_pp': 100 * (x - y)})
    x, y = [sum(records[k]['success'] for k in keys) / n for records in (a, b)]
    full_tasks = [t for t in task_rows if t['episodes'] == 250]
    return {'suite': candidate['suite'], 'contrast': kind,
            'candidate_variant': candidate['variant'], 'candidate_mode': candidate['mode'],
            'control_variant': control['variant'], 'control_mode': control['mode'],
            'paired_episodes': n, 'tasks_seen': len(task_rows), 'fully_evaluated_tasks': len(full_tasks),
            'complete_suite_comparison': n == 2500,
            'candidate_sr_percent': 100 * x, 'control_sr_percent': 100 * y, 'delta_pp': 100 * (x - y),
            'candidate_only_successes': sum(a[k]['success'] and not b[k]['success'] for k in keys),
            'control_only_successes': sum(b[k]['success'] and not a[k]['success'] for k in keys),
            'fully_evaluated_task_delta_pp': sum(t['delta_pp'] for t in full_tasks) / len(full_tasks) if full_tasks else None,
            'initial_state_mismatches': 0, 'candidate_run': candidate['evaluation_run_id'],
            'control_run': control['evaluation_run_id'], 'per_task': task_rows}


# Reuse native validation after network threads have exited. Importing torch
# in the concurrent TLS fetch process can crash the shared macOS environment.
validated_directories = 0
for row in data['runs']:
    if not row.get('records'):
        continue
    validate_full_protocol(row['protocol'])
    complete = row['evaluation_status']['state'] == 'POLICY_EVALUATION_COMPLETE'
    reconstructed = {}
    for file, receipt in row['artifact_receipts'].items():
        if not file.endswith('/episodes.jsonl') or receipt is None:
            continue
        path = ROOT / 'artifacts' / row['evaluation_run_id'] / file
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
        _, records = read_run(path.parent, require_complete=complete)
        assert not reconstructed.keys() & records.keys()
        reconstructed.update(records)
        validated_directories += 1
    actual = indexed(row)
    assert reconstructed.keys() == actual.keys()
    assert all(reconstructed[k]['success'] == actual[k]['success'] for k in actual)

pairs = []
prespecified = {'gripper': 'reference', 'xyz_scalar': 'gripper', 'component_velocity': 'gripper',
               'component_time': 'component_velocity', 'x_yz': 'gripper', 'y_xz': 'gripper', 'z_xy': 'gripper',
               'angular_driver': 'gripper', 'angular_scalar': 'angular_driver', 'all_scalar': 'angular_scalar'}
for row in data['runs']:
    if not row.get('records'):
        continue
    if row['variant'] != 'reference':
        pairs.append(pair(row, rows[(row['suite'], 'reference', row['mode'])], 'versus_reference'))
        if prespecified[row['variant']] != 'reference':
            pairs.append(pair(row, rows[(row['suite'], prespecified[row['variant']], row['mode'])], 'design_contrast'))
        if row['variant'] == 'component_time':
            pairs.append(pair(row, rows[(row['suite'], 'gripper', row['mode'])], 'combined_timing_change'))
    if row['mode'] == 'dur':
        pairs.append(pair(row, rows[(row['suite'], row['variant'], 'stk')], 'duration_vs_stk'))
pairs = [p for p in pairs if p]
write('paired-comparisons.json', pairs)
csv_file('paired-comparisons.csv', [{k: v for k, v in p.items() if k != 'per_task'} for p in pairs])
task_rows = []
for p in pairs:
    for task in p['per_task']:
        task_rows.append({**{k: p[k] for k in ('suite', 'contrast', 'candidate_variant', 'candidate_mode', 'control_variant', 'control_mode')}, **task})
csv_file('paired-task-results.csv', task_rows)

workflows = json.loads((ROOT / 'workflows.json').read_text())
tasks = {(w['name'], t['name']): t['status'] for w in workflows for t in w['tasks']}
table = []
for row in data['runs']:
    records = row.get('records', [])
    complete = (row['evaluation_status'] or {}).get('state') == 'POLICY_EVALUATION_COMPLETE'
    table.append({**{k: row[k] for k in ('suite', 'variant', 'mode', 'training_run_id', 'evaluation_run_id')},
                  'training_artifact_state': (row['training_status'] or {}).get('state', 'NO_ARTIFACTS'),
                  'training_scheduler_state': tasks.get((row['training_workflow_id'], row['training_task']), 'HISTORICAL'),
                  'evaluation_artifact_state': (row['evaluation_status'] or {}).get('state', 'NO_ARTIFACTS'),
                  'evaluation_scheduler_state': tasks.get((row['evaluation_workflow_id'], row['evaluation_task']), 'HISTORICAL'),
                  'score_is_final': complete, 'episodes': len(records), 'planned_episodes': 2500,
                  'successes': sum(r['success'] for r in records), 'success_rate_percent': row.get('success_rate_percent'),
                  'tasks_seen': len({r['task'] for r in records}), 'observed_at_utc': data['observed_at_utc']})
csv_file('all-88-cells.csv', table)
csv_file('available-policy-scores.csv', [r for r in table if r['episodes']])

score_index = {(r['variant'], r['mode'], r['suite']): r for r in table}
compact_scores = []
for variant in dict.fromkeys(r['variant'] for r in table):
    for mode in ('stk', 'dur'):
        compact = {'variant': variant, 'mode': mode.upper()}
        for suite, label in (('libero_spatial', 'Spatial'), ('libero_object', 'Object'),
                             ('libero_goal', 'Goal'), ('libero_10', 'LIBERO-10')):
            row = score_index[(variant, mode, suite)]
            compact[label] = (
                f"{row['success_rate_percent']:.2f}% | {row['episodes']}/2500 episodes | "
                f"{row['tasks_seen']}/10 tasks | {'FINAL' if row['score_is_final'] else 'PARTIAL'}"
                if row['episodes'] else 'pending'
            )
        compact_scores.append(compact)
csv_file('score-table.csv', compact_scores)

replay_root = VAL / 'results/libero_arc_streams_20261002/recovery2/completed_replays'
replays = {s: json.loads((replay_root / s / 'replay-results.json').read_text()) for s in ('spatial', 'object', 'goal', '10')}
replay_table = []
for variant in dict.fromkeys(r['variant'] for r in data['runs']):
    for mode in ('stk', 'dur'):
        source = [d[variant + '-' + mode] for d in replays.values()]
        successes, n = sum(r['successes'] for r in source), sum(r['episodes'] for r in source)
        replay_table.append({'variant': variant, 'mode': mode, 'successes': successes, 'episodes': n,
            'success_rate_percent': 100 * successes / n, 'raw_successes': sum(r['raw_successes'] for r in source),
            'lost_raw_successes': sum(r['lost_raw_successes'] for r in source),
            'gained_successes': sum(r['gained_successes'] for r in source),
            'mean_action_mse': sum(r['action_mse'] for r in source) / 4,
            'mean_gripper_mismatch': sum(r['gripper_mismatch'] for r in source) / 4,
            'mean_execution_coverage': sum(r['execution_coverage'] for r in source) / 4,
            **{s + '_successes': replays[s][variant + '-' + mode]['successes'] for s in replays}})
csv_file('all-replay-results.csv', replay_table)

write('summary.json', {'observed_at_utc': data['observed_at_utc'], 'source_commit': data['source_commit'],
    'canonical_manifest': '../recovery_174115/launch.json', 'training_complete': sum(r['training_artifact_state'] == 'TRAINING_COMPLETE' for r in table),
    'final_evaluations': sum(r['score_is_final'] for r in table), 'partial_evaluations': sum(bool(r['episodes']) and not r['score_is_final'] for r in table),
    'native_episode_directories_validated': validated_directories, 'paired_initial_state_mismatches': 0,
    'method': 'Partial success is pooled over completed episodes. Each comparison uses the intersection of exact episode IDs and identical initial-state hashes.',
    'limits': ['Partial evaluations follow fixed task order and do not yet represent every task equally.',
               'One training seed; five evaluation repetitions are not independent trained policies.',
               'All evaluated policies reached the full training budget. Partial refers to rollout coverage.',
               'Component-time versus reference also changes gripper grouping and timing labels; use the component-time versus component-velocity contrast to isolate clock sharing.',
               'Replay uses 30 demonstrations per suite, previously used during development. It measures reconstruction, not policy learning.',
               'Representation widths change parameter count by less than 0.03% and timing loss weight is not separately normalized.'],
    'replay_sources': {s: {'file': str((replay_root / s / 'replay-results.json').relative_to(VAL)),
                           'sha256': hashlib.sha256((replay_root / s / 'replay-results.json').read_bytes()).hexdigest()} for s in replays}})
shutil.copyfile(ROOT / 'workflows.json', DEST / 'workflows.json')
for source in ('collect.py', 'analyze.py'):
    shutil.copyfile(ROOT / source, DEST / source)
(DEST / 'verified-records.json.gz').write_bytes(gzip.compress((ROOT / 'data.json').read_bytes(), mtime=0))
for row in data['runs']:
    for file, receipt in row.get('artifact_receipts', {}).items():
        if receipt is None:
            continue
        original = ROOT / 'artifacts' / row['evaluation_run_id'] / file
        target = DEST / 'evidence' / row['evaluation_run_id'] / (file + '.gz')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
print(json.dumps({'destination': str(DEST), 'native_directories_checked': validated_directories,
                  'completed': sum(r['score_is_final'] for r in table), 'partial': sum(bool(r['episodes']) and not r['score_is_final'] for r in table),
                  'paired_contrasts': len(pairs)}))
