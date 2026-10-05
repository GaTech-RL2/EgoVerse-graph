import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
plan = json.loads((ROOT / 'recovery-plan.json').read_text())
manifest = json.loads(Path(plan['previous_manifest']).read_text())
rows = {(r['suite'], r['variant'], r['mode']): r for r in manifest['runs']}
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == plan['host_commit']
assert all(r['exit_code'] == 0 for r in json.loads((ROOT / 'service-validation.json').read_text()))
for directory in ('pre-submit', 'submissions', 'accepted'):
    (ROOT / directory).mkdir(exist_ok=True)


def query(name, destination):
    for _ in range(3):
        result = subprocess.run(['osmo', 'workflow', 'query', name, '--format-type', 'json'], capture_output=True, text=True, timeout=50)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            destination.write_text(json.dumps(data, indent=2) + '\n')
            return data
    raise RuntimeError('Could not confirm workflow state: ' + name)


affected = {}
for kind in ('training', 'evaluation'):
    for replacement in plan[kind + '_replacements']:
        row = rows[tuple(replacement['cell'])]
        affected.setdefault(row[kind + '_workflow_id'], set()).add(row[kind + '_task'])
for name, tasks in affected.items():
    data = query(name, ROOT / 'pre-submit' / (name + '.json'))
    current = {t['name']: t['status'] for g in data['groups'] for t in g['tasks']}
    assert data['status'] == 'FAILED_CANCELED'
    assert all(current[n] in {'FAILED_CANCELED', 'FAILED_UPSTREAM'} for n in tasks)

for spec in plan['specs']:
    assert hashlib.sha256(Path(spec['file']).read_bytes()).hexdigest() == spec['sha256']
    receipt = ROOT / 'submissions' / (spec['name'] + '.json')
    assert not receipt.exists(), 'Inspect the accepted receipt; never submit twice'
    result = subprocess.run(['osmo', 'workflow', 'submit', spec['file'], '--pool', spec['pool'], '--priority', 'NORMAL', '--format-type', 'json'], capture_output=True, text=True, timeout=50)
    (ROOT / 'submissions' / (spec['name'] + '.stderr')).write_text(result.stderr)
    result.check_returncode()
    submitted = json.loads(result.stdout)
    receipt.write_text(json.dumps(submitted, indent=2) + '\n')
    assert submitted['name'] == spec['name'] + '-1'
    current = query(submitted['name'], ROOT / 'accepted' / (submitted['name'] + '.json'))
    assert current['status'] in {'PENDING', 'RUNNING', 'WAITING'}
    print(json.dumps({'accepted': submitted['name'], 'pool': current['pool'], 'status': current['status'],
                      'training_cells': spec['training_cells'], 'evaluation_cells': spec['evaluation_cells']}), flush=True)
