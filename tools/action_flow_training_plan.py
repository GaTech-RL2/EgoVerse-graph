"""Plan-driven task adapter; model, training, diagnostics and reload stay in pinned source."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import hydra
from omegaconf import OmegaConf
import torch
from lightning.pytorch.callbacks import Callback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


class ProofTelemetry(Callback):
    """Retain actual train callback metrics before standalone validation replaces them."""
    def __init__(self, output, cadence=1):
        self.output = output
        self.records = []
        self.cadence = int(cadence)
        assert self.cadence > 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.cadence:
            return
        values = {}
        for key, value in trainer.callback_metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                values[key] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                values[key] = float(value)
        assert all(math.isfinite(v) for v in values.values())
        self.records.append({'optimizer_step': trainer.global_step, 'metrics': values})
        self.records = self.records[-500:]
        save(self.output, self.records)


def prepared_root(request):
    return Path(request.get('data_receipt_root', request['remote_root']))


def config(request, row, phase):
    cfg = OmegaConf.to_container(OmegaConf.load(row['config_path']), resolve=True)
    from libero_oat_pair_config import validate
    check = copy.deepcopy(cfg)
    assert cfg['model']['pipeline']['stages'][-1]['action_velocity_weight'] == row['action_velocity_weight']
    check['model']['pipeline']['stages'][-1]['action_velocity_weight'] = 1.0
    validate(check, 'action_flow', row['suite'])
    root = Path(request['remote_root'])
    out = root / phase / row['key']
    # Composed configs contain resolved literals. Rebind the entire output subtree,
    # including normalization caches and inference-config publication, once.
    def rebind(value):
        if isinstance(value, dict):
            return {key: rebind(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebind(item) for item in value]
        if isinstance(value, str) and (value == row['full_output'] or value.startswith(row['full_output'] + '/')):
            return str(out) + value[len(row['full_output']):]
        return value
    cfg = rebind(cfg)
    if phase != 'full':
        assert row['full_output'] not in json.dumps(cfg), 'unbound full output remains in proof'
    cfg['paths']['output_dir'] = str(out)
    cfg['trainer']['default_root_dir'] = str(out)
    cfg['callbacks']['batch_budget']['report_path'] = str(out / 'training-budget.json')
    cfg['callbacks']['model_checkpoint']['dirpath'] = str(out / 'checkpoints')
    cfg['runtime']['slurm_signal_checkpoint_dir'] = str(out / 'checkpoints')
    cfg['logger']['wandb']['save_dir'] = str(out)
    cfg['logger']['wandb']['id'] = row['full_wandb_id'] if phase == 'full' else row['full_wandb_id'] + '-proof'
    cfg['logger']['wandb']['name'] = cfg['logger']['wandb']['id']
    cfg['callbacks']['ema']['final_checkpoint_path'] = str(out / ('checkpoints/terminal-step000000002.ckpt' if phase == 'smoke' else 'checkpoints/terminal-epoch-{epoch:04d}-step-{step:09d}.ckpt'))
    if phase == 'smoke':
        cfg['model']['gradient_telemetry_cadence'] = 1
        cfg['trainer'].update(max_steps=2, max_epochs=-1, val_check_interval=2, check_val_every_n_epoch=None, log_every_n_steps=1, limit_val_batches=1, num_sanity_val_steps=0)
        cfg['callbacks']['model_checkpoint'].update(every_n_epochs=None, every_n_train_steps=2, save_on_train_epoch_end=False)
        cfg['callbacks']['proof_telemetry'] = {'_target_': 'action_flow_training_plan.ProofTelemetry', 'output': str(out / 'TRAIN_METRICS.json')}
    # Scheduled terminal validation is already required at the exact step budget.
    # Avoid an extra validation against immutable artifacts at the same step.
    assert cfg['trainer']['max_steps'] % cfg['trainer']['val_check_interval'] == 0
    cfg['val_at_end'] = False
    cfg['evaluator'] = {'_target_': 'egomimic.eval.libero_comparison_eval.LiberoComparisonEvaluator', 'artifact_root': str(out / 'validation-artifacts'), 'source_commit': request['source_commit'], 'config_sha256': hashlib.sha256(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True).encode()).hexdigest()}
    if phase == "full":
        cfg["callbacks"]["proof_telemetry"] = {"_target_": "action_flow_training_plan.ProofTelemetry", "output": str(out / "TRAIN_METRICS.json"), "cadence": 100}
    if request.get("diagnostics", "flow-fixed-bank") == "native-only":
        cfg["evaluator"]["diagnostic_config"] = None
        return OmegaConf.create(cfg), out
    diagnostic = json.loads((root / 'DIAGNOSTIC_TEMPLATE.json').read_text())
    split = prepared_root(request) / 'data' / (row['suite'] + '-READY.json')
    diagnostic.update(artifact_root=str(out / 'diagnostics'), noise_seed_bank_path=str(root / 'SEEDS.json'), noise_seed_bank_sha256=sha(root / 'SEEDS.json'))
    diagnostic['validation_view']['split_manifest_sha256'] = sha(split)
    diagnostic['provenance'].update(source_commit=request['source_commit'], split_sha256=sha(split))
    cfg['evaluator']['diagnostic_config'] = diagnostic
    return OmegaConf.create(cfg), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('envelope', 'smoke', 'full'))
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    assert os.environ.get('SLURM_STEP_ID'), 'scheduled srun only'
    request = json.loads(args.request.read_text())
    root, source = Path(request['remote_root']), Path(request['source_lambda'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == request['source_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
    data = json.loads((prepared_root(request) / 'DATA_READY.json').read_text())
    assert data['status'] == 'PASS_ALL_FOUR_EXACT_COWORKER_DP_DATA_AND_SPLITS'
    assert data['source_commit'] == request.get('data_source_commit', request['source_commit'])
    rows = request['rows'] if args.phase == 'envelope' else [request['rows'][args.index]]
    from egomimic.pl_utils.pl_model import ModelWrapper
    from libero_oat_training_proof import validate_optimizer_coverage
    for row in rows:
        cfg, out = config(request, row, 'smoke' if args.phase == 'envelope' else args.phase)
        config_identity = sha(row['config_path'])
        split_identity = sha(prepared_root(request) / 'data' / (row['suite'] + '-READY.json'))
        if args.phase == 'envelope':
            full_cfg, full_out = config(request, row, 'full')
            assert full_cfg.trainer.max_steps == request['optimizer_updates'] == 120000
            assert str(full_out) == row['full_output']
            model = ModelWrapper(config_tree=cfg, enable_grad_norm=False)
            optimizer = model.configure_optimizers()
            optimizer = optimizer['optimizer'] if isinstance(optimizer, dict) else optimizer
            params = [p for group in optimizer.param_groups for p in group['params']]
            assert len(params) == len({id(p) for p in params})
            coverage = validate_optimizer_coverage(model.nets, params, 'action_flow')
            from egomimic.utils.libero_dit_half import resolve_backbones
            assert len(resolve_backbones(model)) == 2
            receipt = {'status': 'PASS', 'source_commit': request['source_commit'], 'config_sha256': config_identity, 'split_sha256': split_identity, 'total_parameters': sum(p.numel() for p in model.parameters()), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'initial_group_lrs': sorted({float(g['lr']) for g in optimizer.param_groups}), 'optimizer_coverage': coverage}
            save(root / 'envelopes' / (row['key'] + '.json'), receipt)
            print('CPU_ENVELOPE_PASS', row['key'], receipt['total_parameters'], flush=True)
            del model, optimizer, params
            gc.collect()
            continue
        assert torch.cuda.is_available()
        envelope = json.loads((root / 'envelopes' / (row['key'] + '.json')).read_text())
        assert envelope['status'] == 'PASS' and envelope['source_commit'] == request['source_commit']
        assert envelope['config_sha256'] == config_identity and envelope['split_sha256'] == split_identity
        if args.phase == 'full':
            proof = json.loads((root / 'smoke' / row['key'] / 'RESULT.json').read_text())
            assert proof['status'] == 'PASS_REAL_DATA_OPTIMIZER_VALIDATION_EMA_RELOAD'
            assert proof['source_commit'] == request['source_commit'] and proof['config_sha256'] == config_identity and proof['split_sha256'] == split_identity
            assert proof['wandb_health'] == 'PASS'
            assert sha(proof['checkpoint']) == proof['checkpoint_sha256']
        assert not out.exists(), 'preserve previous attempts; reconcile before retry'
        out.mkdir(parents=True)
        OmegaConf.save(cfg, out / 'RESOLVED_CONFIG.yaml', resolve=True)
        from egomimic.trainHydra import train
        metrics, objects = train(cfg)
        trainer, model, datamodule = objects['trainer'], objects['model'], objects['datamodule']
        assert trainer.global_step == (2 if args.phase == 'smoke' else 120000)
        expected = data['suites'][row['suite']]
        assert sorted(datamodule.train_datasets['libero_panda'].datasets) == expected['train_episode_ids']
        assert sorted(datamodule.valid_datasets['libero_panda'].datasets) == expected['valid_episode_ids']
        if args.phase == 'full':
            save(out / 'TRAINING_COMPLETED.json', {'status': 'TRAINER_RETURNED', 'global_step': trainer.global_step, 'source_commit': request['source_commit']})
            continue
        values = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in metrics.items() if (torch.is_tensor(v) and v.numel() == 1) or isinstance(v, (int, float))}
        for metric in ('Valid/normalized_reconst_mse', 'Valid/reconst_mse', 'Valid/energy_score32', 'Valid/energy_accuracy32', 'Valid/energy_diversity32'):
            assert metric in values and math.isfinite(values[metric]), metric
        for prefix in (() if request.get('diagnostics') == 'native-only' else ('Valid/ActionFlow/DenoisingTrajectory/', 'Valid/ActionFlow/Alignment/CKA/', 'Valid/ActionFlow/Alignment/CKNNA/', 'Valid/ActionFlow/Alignment/FinalLatentCosine/')):
            assert any(k.startswith(prefix) and math.isfinite(v) for k, v in values.items()), prefix
        observed = json.loads((out / 'TRAIN_METRICS.json').read_text())
        merged = {k: v for record in observed for k, v in record['metrics'].items()}
        for candidates in (('Train/MSE', 'Train/MSE_step'), ('Optimizer/LR/Muon',), ('Optimizer/LR/AdamW',)):
            assert any(k in merged and math.isfinite(merged[k]) for k in candidates), candidates
        for component in ('FM', 'Reconstruction', 'ActionVelocity'):
            metric = 'Train/ActionFlow/GradientNorm/' + component
            assert metric in merged and math.isfinite(merged[metric]) and merged[metric] > 0, metric
        assert merged['Train/ActionFlow/Schedule/EffectiveActionVelocityWeight'] == row['action_velocity_weight']
        checkpoint = out / 'checkpoints/terminal-step000000002.ckpt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert payload['global_step'] == payload['ema_num_updates'] == 2
        assert all(torch.isfinite(v).all() for v in payload['state_dict'].values() if torch.is_tensor(v))
        from egomimic.pl_utils.data_context import state_fingerprint
        saved_context = payload['data_context']
        assert saved_context['normalizer_state']['benchmark_context'] == expected['normalizer_context']
        assert state_fingerprint(saved_context) == model.data_context.fingerprint()
        assert saved_context['sha256'] == state_fingerprint(saved_context['normalizer_state'])
        routes = payload['action_flow_gradient_route_manifest']
        assert set(routes['routes']) == {'FM', 'Reconstruction', 'ActionVelocity'}
        assert all(routes['routes'][name] for name in routes['routes'])
        save(out / 'GRADIENT_ROUTES.json', routes)
        for pair, intersection in routes['intersections'].items():
            prefix = 'Train/ActionFlow/'
            defined = merged[prefix + 'GradientCosineDefined/' + pair]
            cosine = merged[prefix + 'GradientCosine/' + pair]
            assert math.isfinite(cosine) and -1.0 <= cosine <= 1.0
            assert defined == float(bool(intersection)), pair
        from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
        strict_load_pipeline_checkpoint(model.model, payload, use_ema=False)
        strict_load_pipeline_checkpoint(model.model, payload, use_ema=True)
        # The source checker verifies actual bounded online train telemetry after logger flush.
        import wandb
        wandb.finish()
        checker = source / 'scripts/train/check_wandb_health.py'
        command = [os.sys.executable, str(checker), 'rl2-group/pushshapes-action-flow/' + cfg.logger.wandb.id, '--metric', 'Train/MSE_step', '--min-optimizer-step', '1', '--window', '500']
        for metric in ('Optimizer/LR/Muon', 'Optimizer/LR/AdamW', 'Valid/normalized_reconst_mse', 'Valid/energy_score32'):
            command.extend(['--metric', metric])
        for component in ('FM', 'Reconstruction', 'ActionVelocity'):
            command.extend(['--metric', 'Train/ActionFlow/GradientNorm/' + component])
        for prefix in (() if request.get('diagnostics') == 'native-only' else ('Valid/ActionFlow/DenoisingTrajectory/', 'Valid/ActionFlow/Alignment/CKA/', 'Valid/ActionFlow/Alignment/CKNNA/', 'Valid/ActionFlow/Alignment/FinalLatentCosine/')):
            command.extend(['--metric', sorted(k for k in values if k.startswith(prefix))[0]])
        health = subprocess.run(command, capture_output=True, text=True)
        (out / 'WANDB_HEALTH.log').write_text(health.stdout + '\n' + health.stderr)
        assert health.returncode == 0, 'real W&B proof telemetry must pass before full launch'
        save(out / 'RESULT.json', {'status': 'PASS_REAL_DATA_OPTIMIZER_VALIDATION_EMA_RELOAD', 'source_commit': request['source_commit'], 'config_sha256': config_identity, 'split_sha256': split_identity, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'optimizer_steps': 2, 'action_velocity_weight': row['action_velocity_weight'], 'metrics': values, 'actual_training_metrics': merged, 'wandb_health': 'PASS', 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
        print('REAL_PROOF_PASS', row['key'], flush=True)


if __name__ == '__main__':
    main()
