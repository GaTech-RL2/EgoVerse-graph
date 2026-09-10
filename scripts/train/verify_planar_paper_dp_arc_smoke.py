#!/usr/bin/env python3
"""Verify a two-step Paper-DP arc-tokenizer smoke and write its gate record."""

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

ROWS = {
    "pusht/planar_v2_usocket_arc_paper_uniform_D40_M16_R24deg": (
        "planar_v2_usocket_arc_paper_uniform_D40_M16_R24deg",
        "uniform",
    ),
    "pusht/planar_v2_usocket_arc_paper_curvature_D40_M16_R24deg": (
        "planar_v2_usocket_arc_paper_curvature_D40_M16_R24deg",
        "curvature",
    ),
}
SOURCE = "pushshapes_sim_u_socket"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history(path: Path) -> tuple[dict[int, dict[str, float]], int]:
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
            row = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    continue
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


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict]:
    candidates = []
    for path in run_dir.glob("checkpoints/*.ckpt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        candidates.append((int(checkpoint["global_step"]), path, checkpoint))
    assert candidates, "no immutable smoke checkpoint"
    _, path, checkpoint = max(candidates, key=lambda item: (item[0], item[1].name))
    return path, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--experiment", choices=sorted(ROWS), required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--dataset-names-sha256", required=True)
    parser.add_argument("--parameter-count", required=True, type=int)
    parser.add_argument("--train-batch-size", required=True, type=int)
    parser.add_argument("--valid-batch-size", required=True, type=int)
    parser.add_argument("--wandb-health", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve(strict=True)
    config_path = run_dir / ".hydra/config.yaml"
    assert config_path.is_file()
    cfg = OmegaConf.load(config_path)
    expected_name, expected_sampling = ROWS[args.experiment]
    assert cfg.name == expected_name
    assert cfg.model._target_.endswith("pl_model.ModelWrapper")
    assert cfg.model.pipeline.stages[3].policy.model._target_.endswith(
        "PaperConditionalUnet1D"
    )
    assert int(cfg.planar.raw_action_horizon) == 40
    assert int(cfg.planar.action_horizon) == 17
    assert int(cfg.planar.arc_waypoints) == 16
    assert cfg.planar.arc_waypoint_sampling == expected_sampling
    assert int(cfg.planar.observation_horizon) == 2
    assert int(cfg.planar.action_target_offset) == 1
    assert int(cfg.run_provenance.action_contract.replan_every) == 1
    assert int(cfg.trainer.max_steps) == 2
    assert int(cfg.trainer.val_check_interval) == 1
    assert int(cfg.trainer.limit_val_batches) == 1
    assert int(cfg.trainer.devices) == 1 and int(cfg.trainer.num_nodes) == 1
    assert str(cfg.trainer.precision) == "bf16-mixed"
    assert int(cfg.callbacks.model_checkpoint.every_n_train_steps) == 1
    assert int(cfg.callbacks.model_checkpoint.save_top_k) == -1
    assert cfg.callbacks.model_checkpoint.save_last is False
    assert int(cfg.data.train_dataloader_params[SOURCE].batch_size) == args.train_batch_size
    assert int(cfg.data.valid_dataloader_params[SOURCE].batch_size) == args.valid_batch_size
    assert str(cfg.data.train_datasets[SOURCE].resolver.folder_path) == args.dataset
    assert cfg.run_provenance.split_manifest_sha256 == args.split_manifest_sha256

    checkpoint_path, checkpoint = _latest_checkpoint(run_dir)
    optimizer_step = int(checkpoint["global_step"])
    assert optimizer_step == 2
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
    history, exit_code = _history(streams[0])
    required = {
        "Train/MSE",
        f"Train/MSE/{SOURCE}",
        "Valid/MSE",
        f"Valid/MSE/{SOURCE}",
        "Valid/Native_MSE",
        f"Valid/Native_MSE/{SOURCE}",
        "Valid/EnergyScore@32",
        f"Valid/EnergyScore@32/{SOURCE}",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    }
    observed = {key: value for row in history.values() for key, value in row.items()}
    assert required <= observed.keys(), (required - observed.keys(), history)
    checked = {key: observed[key] for key in required}
    assert all(math.isfinite(value) for value in checked.values()), checked

    artifacts = sorted(run_dir.glob("validation_predictions/energy_score/**/*.pt"))
    assert artifacts
    artifact = torch.load(artifacts[-1], map_location="cpu", weights_only=False)
    assert artifact["metric"] == "EnergyScore@32"
    assert artifact["sample_count"] == 32
    assert len(artifact["seed_bank"]) == len(set(artifact["seed_bank"])) == 32
    assert set(artifact["domains"]) == {SOURCE}

    health = json.loads(args.wandb_health.read_text())
    expected_run_path = f"{cfg.logger.wandb.entity}/{cfg.logger.wandb.project}/{cfg.logger.wandb.id}"
    assert health["status"] in {"PASS", "WARN"}, health
    assert health["run_path"] == expected_run_path
    assert int(health["optimizer_step"]) >= 1

    norm_path = run_dir / "norm_stats/norm_stats.json"
    assert norm_path.is_file()
    result = {
        "status": "passed",
        "source_head": args.expected_head,
        "experiment": args.experiment,
        "dataset": args.dataset,
        "split_manifest_sha256": args.split_manifest_sha256,
        "dataset_names_sha256": args.dataset_names_sha256,
        "optimizer_step": optimizer_step,
        "scheduled_validation": "passed",
        "strict_checkpoint_reload": "passed",
        "wandb_run_visible": True,
        "parameter_count": restored_count,
        "config_sha256": _sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "normalization_path": str(norm_path),
        "normalization_sha256": _sha256(norm_path),
        "wandb_stream_sha256": _sha256(streams[0]),
        "wandb_health_sha256": _sha256(args.wandb_health),
        "wandb_exit_code": exit_code,
        "energy_artifact_sha256": _sha256(artifacts[-1]),
        "metrics": checked,
    }
    destination = run_dir / "SMOKE_RESULT.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(f"[smoke] PASS {destination}")


if __name__ == "__main__":
    main()
