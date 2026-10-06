"""Read checkpoint upload receipts without downloading or mutating checkpoints."""
import datetime
import json
import re
from pathlib import Path

from collect import document

ROOT = Path(__file__).parent
data = json.loads((ROOT / 'data.json').read_text())
manifest = json.loads(Path(data['manifest']).read_text())
rows = []
for run in data['runs']:
    status = run['training_status'] or {}
    if status.get('state') != 'TRAINING':
        continue
    checkpoints, receipt = document(run['training_run_id'], 'checkpoint-receipts.json')
    saved_epochs = []
    for name in checkpoints or {}:
        match = re.fullmatch(r'training/arc_(?:stk|dur)/checkpoints/epoch_epoch=(\d+)\.ckpt', name)
        if match:
            saved_epochs.append(int(match[1]) + 1)
    epochs = max(saved_epochs, default=None)
    total_steps = manifest['protocol']['optimizer_steps'][run['suite']]
    row = {k: run[k] for k in ('suite', 'variant', 'mode', 'training_run_id')}
    row.update({
        'saved_epochs_at_least': epochs,
        'planned_epochs': 5001,
        'saved_optimizer_steps_at_least': epochs * (total_steps // 5001) if epochs else None,
        'planned_optimizer_steps': total_steps,
        'resumed_from_step': status.get('resumed_from_step'),
        'checkpoint_manifest_receipt': receipt,
    })
    rows.append(row)
    print({k: v for k, v in row.items() if k not in ('training_run_id', 'checkpoint_manifest_receipt')}, flush=True)
(ROOT / 'training-progress.json').write_text(json.dumps({
    'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'method': 'Conservative progress from successfully uploaded periodic checkpoint filenames; active training can be ahead of the last upload.',
    'runs': rows,
}, indent=2) + '\n')
