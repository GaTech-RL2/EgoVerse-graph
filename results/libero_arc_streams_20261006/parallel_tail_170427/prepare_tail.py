"""Prepare six unstarted LIBERO-10 cells for independent scheduling."""
import copy
import datetime
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from collect import MANIFEST, client

ROOT = Path(__file__).parent
OUT = ROOT / 'parallel_tail'
OUT.mkdir(exist_ok=True)
OLD_ID = 'arc-st-20261005-10-1'
NEW_NAME = 'arc-su-20261006-10-tail'
SOURCE = '69ff3fab69c2d322b590809461c80d70878ba71f'
HOST = '9c36037a5da54d5a84cd753fe57353be91e436a8'


def query(name):
    for attempt in range(3):
        result = subprocess.run(['osmo', 'workflow', 'query', name, '--format-type', 'json'], capture_output=True, text=True, timeout=50)
        if result.returncode == 0:
            return json.loads(result.stdout)
    raise RuntimeError('Cannot query workflow ' + name)


def compact(query_result):
    return {**{k: query_result.get(k) for k in ('name', 'status', 'pool', 'priority', 'submit_time', 'overview')},
            'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'tasks': [{k: t.get(k) for k in ('name', 'status', 'start_time', 'end_time')} for g in query_result['groups'] for t in g['tasks']]}


if __name__ == '__main__':
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == HOST
    subprocess.run(['git', 'diff', '--exit-code', SOURCE, 'HEAD', '--', 'egomimic'], check=True)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['schema'] == 8 and manifest['source_commit'] == SOURCE
    old_workflow = next(w for w in manifest['workflows'] if w['workflow_id'] == OLD_ID)
    raw = gzip.decompress((MANIFEST.parent / old_workflow['spec']['file']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == old_workflow['spec']['sha256']
    old = yaml.safe_load(raw)
    assert old['workflow']['name'] + '-1' == OLD_ID
    tasks = {t['name']: t for t in old['workflow']['tasks']}
    selected = [r for r in manifest['runs'] if r['suite'] == 'libero_10' and r['variant'] in {'angular_driver', 'angular_scalar', 'all_scalar'}]
    assert len(selected) == 6
    selected_names = {r[k + '_task'] for r in selected for k in ('training', 'evaluation')}
    heads = {r['training_task'] for r in selected}
    descendants = set(heads)
    while True:
        expanded = descendants | {name for name, task in tasks.items() if any(i['task'] in descendants for i in task.get('inputs', []))}
        if expanded == descendants:
            break
        descendants = expanded
    assert descendants == selected_names, 'Handoff would affect another policy or evaluation'
    live = query(OLD_ID)
    statuses = {t['name']: t['status'] for g in live['groups'] for t in g['tasks']}
    assert live['status'] == 'RUNNING' and statuses.keys() == tasks.keys()
    assert all(statuses[name] == 'WAITING' for name in selected_names)
    for row in selected:
        for kind in ('training', 'evaluation'):
            assert row[kind + '_workflow_id'] == OLD_ID
            run = tasks[row[kind + '_task']]['environment']['RUN_ID']
            assert run == row[kind + '_run_id']
            assert not client.list_objects_v2(Bucket='rldb', Prefix=f'experiments/arc-oat-20260919/{run}/', MaxKeys=1).get('KeyCount')
    new = copy.deepcopy(old)
    new['workflow']['name'] = NEW_NAME
    new['workflow']['tasks'] = [copy.deepcopy(tasks['reference-' + mode + '-ready']) for mode in ('stk', 'dur')]
    replacements = []
    for row in selected:
        train = copy.deepcopy(tasks[row['training_task']])
        evaluate = copy.deepcopy(tasks[row['evaluation_task']])
        assert not any(f['path'] == '/tmp/resume-request.json' for f in train.get('files', []))
        assert train['environment']['SOURCE_COMMIT'] == SOURCE
        assert train['environment']['SUITE'] == 'libero_10'
        assert train['environment']['ARC_STREAM_VARIANT'] == row['variant']
        assert train['environment']['ARC_STREAM_MODE'] == row['mode']
        assert train['environment']['ARC_STREAM_REFERENCE_RUN'] == row['reference_evaluation_run']
        train.pop('inputs', None)
        run = NEW_NAME + '-' + row['variant'].replace('_', '-') + '-' + row['mode']
        assert len(run + '-eval') <= 63
        train['environment']['RUN_ID'] = run
        evaluate['environment']['RUN_ID'] = run + '-eval'
        evaluate['inputs'] = [{'task': train['name']}, {'task': 'reference-' + row['mode'] + '-ready'}]
        for before, after in ((tasks[row['training_task']], train), (tasks[row['evaluation_task']], evaluate)):
            normalized = copy.deepcopy(after)
            normalized['environment']['RUN_ID'] = before['environment']['RUN_ID']
            assert {k: v for k, v in normalized.items() if k != 'inputs'} == {k: v for k, v in before.items() if k != 'inputs'}
        new['workflow']['tasks'].extend((train, evaluate))
        replacements.append({'cell': [row['suite'], row['variant'], row['mode']],
                             'training_task': train['name'], 'evaluation_task': evaluate['name'],
                             'old_training_run': row['training_run_id'], 'old_evaluation_run': row['evaluation_run_id'],
                             'new_training_run': run, 'new_evaluation_run': run + '-eval'})
    names = {t['name'] for t in new['workflow']['tasks']}
    assert names == selected_names | {'reference-stk-ready', 'reference-dur-ready'}
    assert all(i['task'] in names for t in new['workflow']['tasks'] for i in t.get('inputs', []))
    path = OUT / (NEW_NAME + '.yaml')
    path.write_text(yaml.safe_dump(new, sort_keys=False))
    result = subprocess.run(['osmo', 'workflow', 'submit', str(path), '--pool', 'groot-l40s-01', '--priority', 'NORMAL', '--dry-run'], capture_output=True, text=True, timeout=50)
    (OUT / 'validation.txt').write_text(result.stdout + result.stderr)
    result.check_returncode()
    plan = {'old_workflow': OLD_ID, 'name': NEW_NAME, 'pool': 'groot-l40s-01', 'priority': 'NORMAL',
            'previous_manifest': str(MANIFEST), 'previous_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'source_commit': SOURCE, 'launcher_commit': HOST, 'spec': str(path),
            'spec_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'validation_exit_code': result.returncode,
            'training_gpus': 24, 'evaluation_gpus': 6, 'replacements': replacements,
            'retired_tasks': sorted(selected_names), 'retired_training_heads': sorted(heads),
            'preserved_tasks': sorted(tasks.keys() - selected_names),
            'recipe_changes': ['RUN_ID and workflow identity', 'Remove previous-policy scheduling dependencies'],
            'model_codec_training_evaluation_changes': []}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (OUT / 'old-before.json').write_text(json.dumps(compact(live), indent=2) + '\n')
    print(json.dumps({'validated': NEW_NAME, 'cells': len(selected), 'unchanged_recipes': True, 'training_gpus': 24, 'old_tasks_preserved': len(plan['preserved_tasks'])}))
