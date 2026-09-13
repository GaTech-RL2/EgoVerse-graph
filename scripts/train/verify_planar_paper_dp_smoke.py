#!/usr/bin/env python3
"""Verify a real current-main Paper-DP optimizer and validation smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

from egomimic.pl_utils.pl_model import ModelWrapper


DOMAINS = ("pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history(path: Path) -> dict[int, dict[str, float]]:
    store = DataStore()
    store.open_for_scan(str(path))
    rows: dict[int, dict[str, float]] = {}
    exits: list[int] = []
    try:
        while (payload := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            kind = record.WhichOneof("record_type")
            if kind == "exit":
                exits.append(int(record.exit.exit_code))
            if kind != "history":
                continue
            row: dict[str, float] = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    row[key] = float(json.loads(item.value_json))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            if "trainer/global_step" in row:
                rows.setdefault(int(row["trainer/global_step"]), {}).update(row)
    finally:
        store.close()
    assert exits and exits[-1] == 0, exits
    return rows


def finite_state_dict(wrapper: ModelWrapper) -> int:
    count = 0
    for name, value in wrapper.state_dict().items():
        assert torch.isfinite(value).all(), name
        count += value.numel()
    assert count > 0
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-experiment", required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    assert (run_dir / "provenance/source_head.txt").read_text().strip() == args.expected_head
    assert (run_dir / "provenance/experiment.txt").read_text().strip() == args.expected_experiment
    config_path = run_dir / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    assert cfg.mode == "train" and cfg.ckpt_path is None
    # Hydra experiment files provide the semantic run name (for example
    # ``cotrain_obstacle_arc_duration_...``), not their full config path.
    # Gate the resolved contract rather than an unreachable filename prefix.
    assert cfg.name.startswith("cotrain_obstacle_")
    assert bool(cfg.run_provenance.obstacle_data)
    assert int(cfg.run_provenance.dataset_count) == 7920
    assert float(cfg.run_provenance.valid_ratio) == 0.01
    assert int(cfg.planar.batch_size) == 64
    assert bool(cfg.model.train_log_on_step)
    assert int(cfg.trainer.devices) == 1 and int(cfg.trainer.num_nodes) == 1
    assert str(cfg.trainer.precision) == "bf16"
    assert int(cfg.trainer.max_steps) == 2
    assert int(cfg.trainer.val_check_interval) == 1
    assert int(cfg.trainer.limit_val_batches) == 1
    assert int(cfg.trainer.log_every_n_steps) == 1
    assert int(cfg.callbacks.model_checkpoint.every_n_train_steps) == 2
    assert bool(cfg.callbacks.model_checkpoint.save_weights_only)
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert not bool(cfg.callbacks.model_checkpoint.save_last)

    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 1, checkpoints
    checkpoint_path = checkpoints[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert int(checkpoint["global_step"]) == 2
    assert not checkpoint.get("optimizer_states")
    assert not checkpoint.get("lr_schedulers")
    assert checkpoint["state_dict"]
    del checkpoint

    restored = ModelWrapper.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=True, weights_only=False
    )
    parameter_count = sum(parameter.numel() for parameter in restored.parameters())
    assert parameter_count > 0
    tensor_count = finite_state_dict(restored)
    del restored

    streams = [
        *run_dir.glob("wandb/run-*/run-*.wandb"),
        *run_dir.glob("wandb/offline-run-*/run-*.wandb"),
    ]
    assert len(streams) == 1, streams
    rows = history(streams[0])
    train_required = {"Train/MSE", *(f"Train/MSE/{domain}" for domain in DOMAINS)}
    valid_required = {
        "Valid/MSE",
        "Valid/Native_MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    }
    for domain in DOMAINS:
        valid_required.update(
            {
                f"Valid/MSE/{domain}",
                f"Valid/Native_MSE/{domain}",
                f"Valid/EnergyScore@32/{domain}",
                f"Valid/EnergyScoreAccuracy@32/{domain}",
                f"Valid/EnergyScoreDiversity@32/{domain}",
            }
        )
    train_rows = [row for row in rows.values() if train_required <= row.keys()]
    valid_rows = [row for step, row in rows.items() if step >= 1 and valid_required <= row.keys()]
    assert train_rows and valid_rows, rows
    metrics = {
        key: value
        for row in (train_rows[-1], valid_rows[-1])
        for key, value in row.items()
        if key in train_required or key in valid_required
    }
    assert all(math.isfinite(value) for value in metrics.values()), metrics

    artifacts = sorted(run_dir.glob("validation_predictions/energy_score/**/*.pt"))
    assert artifacts
    artifact = torch.load(artifacts[-1], map_location="cpu", weights_only=False)
    assert artifact["metric"] == "EnergyScore@32"
    assert len(artifact["seed_bank"]) == 32
    assert len(set(artifact["seed_bank"])) == 32
    assert set(artifact["domains"]) == set(DOMAINS)

    result = {
        "status": "passed",
        "repo_head": args.expected_head,
        "experiment": args.expected_experiment,
        "run_dir": str(run_dir),
        "global_step": 2,
        "strict_checkpoint_reload": "passed",
        "weights_only_checkpoint": True,
        "parameter_count": parameter_count,
        "state_dict_numel": tensor_count,
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "wandb_stream_sha256": sha256(streams[0]),
        "energy_artifact_count": len(artifacts),
        "energy_artifact_sha256": sha256(artifacts[-1]),
        "metrics": metrics,
    }
    destination = run_dir / "SMOKE_RESULT.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(f"[smoke] PASS {destination}")


if __name__ == "__main__":
    main()
