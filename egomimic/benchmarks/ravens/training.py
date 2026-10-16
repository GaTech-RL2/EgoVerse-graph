"""Bind Ravens profiles to the maintained trainHydra runner; no separate training loop."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

# Zero-initialized UNITE heads and a zero-start LR need several updates before
# the action-velocity gradient is observable. Match the existing eight-update gate.
SMOKE_UPDATES = 8


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def compose(family, dataset, output, run_id):
    source = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    with hydra.initialize_config_dir(version_base="1.2", config_dir=str(source / "egomimic/hydra_configs")):
        cfg = hydra.compose(config_name="train_zarr_cartesian", return_hydra_config=True, overrides=[
            "hydra/launcher=basic", f"+experiment=ravens/{family}",
            f"benchmark.dataset={dataset}", f"paths.output_dir={output}", f"paths.work_dir={source}",
            "runtime.slurm_requeue_owner=none", f"evaluator.source_commit={commit}",
            "evaluator.config_sha256=UNBOUND", f"logger.wandb.id={run_id}",
            "logger.wandb.entity=rl2-group", "logger.wandb.project=pushshapes-action-flow",
            "+logger.wandb.resume=never"])
    cfg.hydra.runtime.output_dir = str(output)
    HydraConfig.instance().set_config(cfg)
    job = OmegaConf.to_container(OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True)
    # Output paths and logger IDs do not change the smoke-relevant scientific identity.
    selected = {key: job[key] for key in ("model", "data", "normalizer", "norm_stats", "trainer", "callbacks", "benchmark", "seed")}
    # A node-local copy has the same data identity, verified separately by its
    # immutable manifest and complete normalizer context. Mount paths are not
    # scientific settings, and differ between smoke/full allocations.
    encoded = (json.dumps(selected, sort_keys=True)
               .replace(str(output), "<RUN_OUTPUT>")
               .replace(str(dataset), "<DATASET_ROOT>"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    cfg.evaluator.config_sha256 = digest
    return cfg, commit, digest


def inspect(cfg):
    import torch
    from egomimic.pl_utils.pl_model import ModelWrapper

    torch.set_num_threads(4)
    # Exact module/optimizer inventory without expensive numerical initialization.
    # Real initialized parameters are constructed and checked by train_runner.
    with torch.device("meta"):
        model = ModelWrapper(config_tree=cfg, enable_grad_norm=False)
    optimizer = model.configure_optimizers()
    optimizer = optimizer["optimizer"] if isinstance(optimizer, dict) else optimizer
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    active = {id(p) for p in model.parameters() if p.requires_grad}
    if not active.issubset({id(p) for p in parameters}):
        raise ValueError("Optimizer misses trainable parameters")
    if len(parameters) != len({id(p) for p in parameters}):
        raise ValueError("Optimizer repeats a parameter")
    result = {"construction_device": "meta", "total": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "initial_optimizer_group_lrs": sorted({float(g["lr"]) for g in optimizer.param_groups}),
              "precision": cfg.trainer.precision,
              "action_shape": [32, 7], "effective_batch": cfg.benchmark.batch_size,
              "latent_shape": [16, 16] if "latent_dim" in cfg.model else None,
              "denoising_shape": [16, 16] if "latent_dim" in cfg.model else [32, 7]}
    del model, optimizer, parameters
    gc.collect()
    return result


def make_run_guard(expected_total, expected_trainable, report_path, expected_context=None):
    from lightning import Callback

    class RavensRunGuard(Callback):
        def on_fit_start(self, trainer, model):
            actual = (sum(p.numel() for p in model.parameters()),
                      sum(p.numel() for p in model.parameters() if p.requires_grad))
            if actual != (expected_total, expected_trainable):
                raise ValueError("Real initialized model differs from parameter inventory")
            context = model.model.pipeline.stages[0].normalizer_state["benchmark_context"]
            if expected_context is not None and context != dict(expected_context):
                raise ValueError("Dataset/observation/normalizer context differs from smoke")
            result = {"status": "PASS_REAL_INITIALIZED_MODEL", "total": actual[0], "trainable": actual[1],
                      "initial_optimizer_group_lrs": [float(group["lr"]) for optimizer in trainer.optimizers
                                                      for group in optimizer.param_groups],
                      "precision": str(trainer.precision)}
            write(report_path, result)
            print(json.dumps({"stage": "REAL_CONSTRUCTOR_PASS", **result}), flush=True)
    return RavensRunGuard()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("action_flow", "dp"), required=True)
    parser.add_argument("--phase", choices=("inspect", "smoke", "full"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-trainable", type=int)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--smoke-sha256")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Construction and training both require a scheduled allocation")
    source = Path(__file__).resolve().parents[3]
    actual_source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual_source != args.source_commit:
        raise ValueError("Source differs from the exact approved preflight binding")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise RuntimeError("Construction and training require a clean committed source")
    if args.phase != "inspect" and (args.expected_total is None or args.expected_trainable is None):
        raise ValueError("Training requires the verified parameter counts from preflight")
    output, dataset = args.output.resolve(), args.dataset.resolve()
    if output.is_relative_to(source) or output.exists():
        raise ValueError("Use a fresh output directory outside the exact source checkout")
    output.mkdir(parents=True)
    cfg, commit, identity = compose(args.family, dataset, output, args.wandb_run_id)
    OmegaConf.save(OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), output / "FULL_CONFIG.yaml", resolve=True)
    if args.phase == "inspect":
        counts = inspect(cfg)
        write(output / "CONSTRUCTOR.json", counts)
        print(json.dumps({"stage": "SHAPE_INVENTORY_PASS", **counts}), flush=True)
        return
    # Reuse the verified blueprint count receipt. The actual model is built once
    # by trainHydra, then checked by the callback before any optimizer update.
    counts = {"total": args.expected_total, "trainable": args.expected_trainable}
    if not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Training requires a scheduled GPU-bound step")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if dirty:
        raise RuntimeError("Training source must be committed and clean")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated visible CUDA GPU required")
    manifest = dataset / "benchmark_manifest.json"
    dataset_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    from egomimic.rldb.zarr.libero_dataset import LiberoDataset
    from egomimic.benchmarks.ravens.replay import RavensReplayResolver
    resolver = hydra.utils.instantiate(cfg.data.train_datasets.ravens_ur5.resolver)
    assert isinstance(resolver, RavensReplayResolver)
    train = LiberoDataset._from_resolver(resolver, mode="train", valid_ratio=.1, split_seed=42)
    valid = LiberoDataset._from_resolver(resolver, mode="valid", valid_ratio=.1, split_seed=42)
    train_ids, valid_ids = sorted(train.datasets), sorted(valid.datasets)
    if len(train_ids) != 450 or len(valid_ids) != 50 or set(train_ids) & set(valid_ids):
        raise ValueError("Require the exact disjoint 450/50 episode split")
    split = {"train": train_ids, "valid": valid_ids, "seed": 42, "dataset_manifest_sha256": dataset_sha}
    write(output / "SPLIT.json", split)
    del train, valid, resolver
    gc.collect()
    proof = None
    if args.phase == "full":
        if args.smoke_result is None or args.smoke_sha256 is None:
            raise ValueError("Full training requires a hash-bound passing real-data smoke")
        if hashlib.sha256(args.smoke_result.read_bytes()).hexdigest() != args.smoke_sha256:
            raise ValueError("Smoke receipt hash mismatch")
        proof = json.loads(args.smoke_result.read_text())
        expected = {"status": "PASS_REAL_OPTIMIZER_VALIDATION_RELOAD", "source_commit": commit,
                    "scientific_config_sha256": identity, "dataset_manifest_sha256": dataset_sha,
                    "family": args.family, "optimizer_updates": SMOKE_UPDATES}
        if any(proof.get(key) != value for key, value in expected.items()):
            raise ValueError("Smoke does not match the full scientific identity")
        checkpoint = Path(proof["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != proof["checkpoint_sha256"]:
            raise ValueError("Smoke checkpoint bytes changed")
    else:
        with open_dict(cfg):
            cfg.trainer.max_steps = SMOKE_UPDATES
            cfg.trainer.val_check_interval = SMOKE_UPDATES
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.log_every_n_steps = 1
            cfg.callbacks.model_checkpoint.every_n_train_steps = SMOKE_UPDATES
            if args.family == "action_flow":
                cfg.model.gradient_telemetry_cadence = 1
    with open_dict(cfg):
        cfg.callbacks.ravens_guard = {"_target_": "egomimic.benchmarks.ravens.training.make_run_guard",
                                      "expected_total": counts["total"],
                                      "expected_trainable": counts["trainable"],
                                      "report_path": str(output / "CONSTRUCTOR.json"),
                                      "expected_context": proof["normalizer_context"] if proof else None}
    OmegaConf.save(OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), output / "RUN_CONFIG.yaml", resolve=True)
    from egomimic.trainHydra import train as train_runner
    metrics, objects = train_runner(cfg)
    trainer = objects["trainer"]
    if args.phase == "full":
        write(output / "TRAINER_RETURNED.json", {"global_step": trainer.global_step, "source_commit": commit})
        return
    if trainer.global_step != SMOKE_UPDATES:
        raise ValueError("Smoke did not perform the required real optimizer updates")
    counts = json.loads((output / "CONSTRUCTOR.json").read_text())
    if counts["status"] != "PASS_REAL_INITIALIZED_MODEL":
        raise ValueError("Missing actual initialized-model count proof")
    values = {key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
              for key, value in metrics.items() if torch.is_tensor(value) or isinstance(value, (int, float))}
    if not all(np_isfinite(value) for value in values.values()):
        raise ValueError("Non-finite train/validation smoke telemetry")
    for key in ("Valid/normalized_reconst_mse", "Valid/reconst_mse", "Valid/energy_score32"):
        if key not in values or not np_isfinite(values[key]):
            raise ValueError(f"Missing/non-finite smoke metric: {key}")
    if args.family == "action_flow":
        for component in ("FM", "Reconstruction", "ActionVelocity"):
            matches = [value for key, value in values.items() if key.endswith(f"GradientNorm/{component}")]
            if not matches or not all(np_isfinite(value) and value > 0 for value in matches):
                raise ValueError(f"Missing/nonpositive objective gradients: {component}")
    paths = sorted((output / "checkpoints").glob("*.ckpt"))
    checkpoints = [path for path in paths if path.name != "last.ckpt"]
    if not checkpoints:
        raise ValueError("No immutable checkpoint")
    checkpoint = checkpoints[-1]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload["global_step"] != SMOKE_UPDATES or payload.get("ema_num_updates") != SMOKE_UPDATES
            or not payload.get("optimizer_states") or "ema_state_dict" not in payload):
        raise ValueError("Incomplete model/EMA/optimizer smoke checkpoint")
    def finite_tensors(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tensors(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tensors(item) for item in value)
        return True
    if not finite_tensors(payload):
        raise ValueError("Non-finite tensor in full-state smoke checkpoint")
    from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
    strict_load_pipeline_checkpoint(objects["model"].model, payload, use_ema=False)
    strict_load_pipeline_checkpoint(objects["model"].model, payload, use_ema=True)
    optimizer = trainer.optimizers[0]
    optimizer.load_state_dict(payload["optimizer_states"][0])
    if len(trainer.lr_scheduler_configs) != len(payload.get("lr_schedulers", [])):
        raise ValueError("Checkpoint scheduler count differs from the configured optimizer")
    for configured, saved in zip(trainer.lr_scheduler_configs, payload.get("lr_schedulers", [])):
        configured.scheduler.load_state_dict(saved)
    write(output / "RESULT.json", {"status": "PASS_REAL_OPTIMIZER_VALIDATION_RELOAD",
          "source_commit": commit, "family": args.family, "scientific_config_sha256": identity,
          "dataset_manifest_sha256": dataset_sha, "parameters": counts,
          "optimizer_updates": trainer.global_step, "metrics": values,
          "normalizer_context": payload["normalizer_state"]["benchmark_context"],
          "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})


def np_isfinite(value):
    import math
    return math.isfinite(value)


if __name__ == "__main__":
    main()
