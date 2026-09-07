#!/usr/bin/env python3
"""Verify a real Planar optimizer/validation smoke and write its gate record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

from egomimic.pl_utils.pl_model import ModelWrapper


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wandb_history(path: Path) -> tuple[dict[int, dict[str, float]], int]:
    store = DataStore()
    store.open_for_scan(str(path))
    rows: dict[int, dict[str, float]] = {}
    exits = []
    try:
        while (payload := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            kind = record.WhichOneof("record_type")
            if kind == "exit":
                exits.append(int(record.exit.exit_code))
            if kind != "history":
                continue
            row = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            if "trainer/global_step" not in row:
                continue
            step = int(row["trainer/global_step"])
            for key, value in row.items():
                if key.startswith(("Train/", "Valid/", "Optimizer/", "Timing/")):
                    try:
                        rows.setdefault(step, {})[key] = float(value)
                    except (TypeError, ValueError):
                        pass
    finally:
        store.close()
    assert exits and exits[-1] == 0, exits
    return rows, exits[-1]


def _artifact_order(path: Path, root: Path) -> tuple[int, ...]:
    parts = path.relative_to(root).parts
    job_id, restart = -1, 0
    if len(parts) == 3:
        attempt = re.fullmatch(r"job-(\d+)-restart-(\d+)", parts[0])
        if attempt is None:
            raise ValueError(f"unrecognized EnergyScore execution path: {path}")
        job_id, restart = map(int, attempt.groups())
        parts = parts[1:]
    if len(parts) != 2:
        raise ValueError(f"unrecognized EnergyScore artifact path: {path}")
    progress = re.fullmatch(r"epoch-(\d+)-step-(\d+)", parts[0])
    unit = re.fullmatch(r"rank-(\d+)-batch-(\d+)\.pt", parts[1])
    if progress is None or unit is None:
        raise ValueError(f"unrecognized EnergyScore artifact identity: {path}")
    epoch, step = map(int, progress.groups())
    rank, batch = map(int, unit.groups())
    return job_id, restart, step, epoch, rank, batch


def _verified_energy_artifact(run_dir, *, expected_step, seed_bank_sha256, domains):
    root = run_dir / "validation_predictions/energy_score"
    artifacts = list(root.glob("**/*.pt"))
    if not artifacts:
        raise ValueError("smoke has no EnergyScore artifact")
    path = max(artifacts, key=lambda item: _artifact_order(item, root))
    job_id, restart, step, epoch, rank, batch = _artifact_order(path, root)
    if step != expected_step:
        raise ValueError(f"latest execution artifact step {step} != checkpoint {expected_step}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != 1 or artifact.get("metric") != "EnergyScore@32":
        raise ValueError("unsupported EnergyScore artifact schema or metric")
    for key, expected in (("global_step", step), ("epoch", epoch), ("rank", rank), ("batch_idx", batch)):
        if artifact.get(key) != expected:
            raise ValueError(f"EnergyScore {key} differs from artifact path")
    expected_execution = (
        None if job_id == -1
        else {"slurm_job_id": str(job_id), "slurm_restart_count": restart}
    )
    if artifact.get("execution") != expected_execution:
        raise ValueError("EnergyScore execution differs from artifact path")
    seeds = artifact.get("seed_bank")
    if not isinstance(seeds, list) or len(seeds) != 32 or len(set(seeds)) != 32:
        raise ValueError("EnergyScore artifact requires a seed_bank of 32 unique seeds")
    if artifact.get("seed_bank_sha256") != seed_bank_sha256:
        raise ValueError("EnergyScore seed-bank identity differs from resolved config")
    if set(artifact.get("domains", {})) != set(domains):
        raise ValueError("EnergyScore artifact domains differ from smoke contract")
    return path, len(artifacts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--parameter-count", required=True, type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = run_dir / ".hydra/config.yaml"
    checkpoint_path = run_dir / "checkpoints/last.ckpt"
    assert config_path.is_file() and checkpoint_path.is_file()
    cfg = OmegaConf.load(config_path)
    assert cfg.name == "planar_v2_cotrain_clean_dp_standard_h16"
    assert cfg.run_provenance.obstacle_data is False
    assert int(cfg.trainer.max_steps) == 2
    assert int(cfg.trainer.val_check_interval) == 1
    assert int(cfg.trainer.limit_val_batches) == 1
    assert int(cfg.trainer.devices) == args.world_size
    assert int(cfg.trainer.num_nodes) == 1
    assert str(cfg.trainer.precision) == "bf16"
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert cfg.callbacks.model_checkpoint.filename == "epoch-{epoch}-step-{step}"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert int(checkpoint["global_step"]) == 2
    assert checkpoint.get("optimizer_states")
    assert len(checkpoint.get("lr_schedulers", [])) == 1
    del checkpoint

    restored = ModelWrapper.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=True, weights_only=False
    )
    restored_count = sum(parameter.numel() for parameter in restored.parameters())
    assert restored_count == args.parameter_count
    del restored

    streams = [
        *run_dir.glob("wandb/run-*/run-*.wandb"),
        *run_dir.glob("wandb/offline-run-*/run-*.wandb"),
    ]
    assert len(streams) == 1, streams
    history, exit_code = _wandb_history(streams[0])
    labels = ("pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper")
    train_required = {"Train/MSE", *(f"Train/MSE/{label}" for label in labels)}
    valid_required = {
        "Valid/MSE",
        "Valid/Native_MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    }
    for label in labels:
        valid_required.update(
            {
                f"Valid/MSE/{label}",
                f"Valid/Native_MSE/{label}",
                f"Valid/EnergyScore@32/{label}",
                f"Valid/EnergyScoreAccuracy@32/{label}",
                f"Valid/EnergyScoreDiversity@32/{label}",
            }
        )
    train_rows = [row for row in history.values() if train_required <= row.keys()]
    valid_rows = [
        row
        for step, row in history.items()
        if step >= 1 and valid_required <= row.keys()
    ]
    assert train_rows and valid_rows, history
    checked = {
        key: value
        for row in (train_rows[-1], valid_rows[-1])
        for key, value in row.items()
        if key in train_required or key in valid_required
    }
    assert all(math.isfinite(value) for value in checked.values()), checked

    artifact_path, artifact_count = _verified_energy_artifact(
        run_dir, expected_step=2,
        seed_bank_sha256=str(cfg.evaluator.seed_bank_sha256), domains=labels,
    )

    result = {
        "status": "passed",
        "repo_head": args.expected_head,
        "run_dir": str(run_dir),
        "world_size": args.world_size,
        "global_step": 2,
        "strict_checkpoint_reload": "passed",
        "parameter_count": restored_count,
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "wandb_stream_sha256": _sha256(streams[0]),
        "wandb_exit_code": exit_code,
        "energy_artifact_count": artifact_count,
        "energy_artifact_path": str(artifact_path),
        "energy_artifact_sha256": _sha256(artifact_path),
        "metrics": checked,
    }
    destination = run_dir / "SMOKE_RESULT.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(f"[smoke] PASS {destination}")


if __name__ == "__main__":
    main()
