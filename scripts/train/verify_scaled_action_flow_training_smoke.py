#!/usr/bin/env python3
"""Verify two-update H512/D14/H16 Action Flow smokes, including routed co-training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

SCHEMA_VERSION = 1
EXPERIMENTS = {
    "pusht/action_flow_usocket_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42": {
        "config_name": "action_flow_usocket_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42",
        "parameter_count": 190_208_924,
        "sources": {"pushshapes_sim_u_socket": 4},
    },
    "pusht/action_flow_chain_points6_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42": {
        "config_name": "action_flow_chain_points6_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42",
        "parameter_count": 190_210_206,
        "sources": {"pushshapes_sim_chain_gripper": 6},
    },
    "pusht/action_flow_cotrain_uc_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42": {
        "config_name": "action_flow_cotrain_uc_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42",
        "parameter_count": 301_792_030,
        "sources": {
            "pushshapes_sim_u_socket": 4,
            "pushshapes_sim_chain_gripper": 6,
        },
    },
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def finite_tree(value: Any, label: str) -> tuple[int, int]:
    tensors = scalars = 0
    if torch.is_tensor(value):
        require(bool(torch.isfinite(value).all()), f"non-finite tensor in {label}")
        return 1, 0
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_tensors, child_scalars = finite_tree(child, f"{label}.{key}")
            tensors += child_tensors
            scalars += child_scalars
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_tensors, child_scalars = finite_tree(child, f"{label}.{index}")
            tensors += child_tensors
            scalars += child_scalars
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(float(value)), f"non-finite scalar in {label}")
        scalars += 1
    return tensors, scalars


def wandb_history(path: Path) -> tuple[dict[int, dict[str, float]], int]:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

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
            decoded: dict[str, Any] = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    decoded[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    continue
            if "trainer/global_step" not in decoded:
                continue
            step = int(decoded["trainer/global_step"])
            for key, value in decoded.items():
                if not key.startswith(("Train/", "Valid/", "Optimizer/", "Timing/")):
                    continue
                if isinstance(value, bool):
                    continue
                try:
                    rows.setdefault(step, {})[key] = float(value)
                except (TypeError, ValueError):
                    pass
    finally:
        store.close()
    require(exits and exits[-1] == 0, f"W&B exit record is not zero: {exits}")
    return rows, exits[-1]


def metric(row: Mapping[str, float], name: str) -> float | None:
    for candidate in (name, f"{name}_step", f"{name}_epoch"):
        if candidate in row:
            return float(row[candidate])
    return None


def complete_row(
    rows: Mapping[int, Mapping[str, float]], names: tuple[str, ...], label: str
) -> tuple[int, dict[str, float]]:
    for step in sorted(rows, reverse=True):
        if step < 1:
            continue
        values = {name: metric(rows[step], name) for name in names}
        if all(value is not None for value in values.values()):
            concrete = {name: float(value) for name, value in values.items()}
            require(all(math.isfinite(value) for value in concrete.values()), f"non-finite {label}")
            return step, concrete
    raise VerificationError(f"no complete {label} metric row")


def validate_gpu_probes(run_dir: Path) -> list[dict[str, Any]]:
    """Accept the canonical ICE probe receipt, whose terminal state is PASSED."""
    probes = []
    for path in sorted(run_dir.glob("provenance/restart-*/gpu_probe.json")):
        payload = json.loads(path.read_text())
        require(payload.get("status") == "PASSED", f"GPU probe failed: {path}")
        probes.append(payload)
    require(probes, "GPU probe record missing")
    return probes


def step_two_artifact(config_value: Any, *, run_dir: Path, label: str) -> dict[str, Any]:
    root = Path(str(config_value))
    if not root.is_absolute():
        root = run_dir / root
    require(root.is_dir(), f"{label} artifact root is missing: {root}")
    leftovers = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".tmp", ".temporary", ".partial"}
    ]
    require(not leftovers, f"unfinished {label} artifacts: {leftovers}")
    paths = sorted(
        [
            *root.glob("epoch-*-step-2/rank-0-batch-*.pt"),
            *root.glob("job-*-restart-*/epoch-*-step-2/rank-0-batch-*.pt"),
        ]
    )
    require(len(paths) == 1, f"expected one step-two {label} artifact: {paths}")
    path = paths[0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, Mapping), f"{label} artifact is not a mapping")
    require(payload.get("global_step") == 2, f"{label} artifact is not step two")
    finite_tree(payload, f"{label} artifact")
    return {"path": str(path), "sha256": sha256(path), "payload": payload}


def require_sources_in_artifact(
    artifact: Mapping[str, Any], sources: tuple[str, ...], *, label: str
) -> None:
    strings: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                strings.append(str(key))
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, str):
            strings.append(value)

    collect(artifact)
    rendered = "\n".join(strings)
    for source in sources:
        require(source in rendered, f"{label} artifact lacks source {source}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve(strict=True)
    config_path = run_dir / ".hydra/config.yaml"
    require(config_path.is_file(), "resolved Hydra config is missing")
    config = OmegaConf.load(config_path)
    require(args.experiment in EXPERIMENTS, "unexpected experiment")
    row = EXPERIMENTS[args.experiment]
    sources = tuple(row["sources"])
    routed = len(sources) == 2
    require(args.expected_parameter_count > 0, "expected parameter count must be positive")
    require(
        args.expected_parameter_count == row["parameter_count"],
        "requested parameter count is not the pinned row identity",
    )
    secondary_values = (
        args.expected_second_split_sha256,
        args.expected_second_content_manifest_sha256,
        args.expected_second_dataset_content_aggregate_sha256,
    )
    require(
        all(secondary_values) if routed else not any(secondary_values),
        "secondary identities must be supplied exactly for routed co-training",
    )
    require(config.name == row["config_name"], "config name mismatch")
    require(config.run_provenance.source_commit == args.expected_head, "source mismatch")
    require(config.trainer.max_steps == 2, "smoke must use two optimizer steps")
    require(config.trainer.val_check_interval == 2, "validation must follow step two")
    require(config.trainer.limit_val_batches == 1, "smoke must run one real validation batch")
    require(config.model.flow_samples_per_content == 14, "FM sample count mismatch")
    require(config.model.flow_mini_batch == 14, "FM mini-batch mismatch")
    require(config.model.hidden_dim == 512, "model width mismatch")
    require(config.model.cfg_scale == 4.0, "CFG scale mismatch")
    require(config.model.flow_loss_aggregation == "sum_samples", "FM aggregation mismatch")
    require(
        config.model.pipeline.stages[6].flow_clean_gradient_mode == "all_stopgrad",
        "FM clean-endpoint detachment mismatch",
    )
    require(
        config.model.pipeline.stages[8].action_velocity_weight == 1.0,
        "action-velocity weight mismatch",
    )
    require(
        config.model.pipeline.stages[8].flow_aggregation == "sum_samples",
        "objective FM aggregation mismatch",
    )
    require(config.model.pipeline.stages[4]._target_.endswith(
        "RoutedContentEncoderStage" if routed else "ContentEncoderStage"
    ), "content encoder mismatch")
    require(config.model.pipeline.stages[7]._target_.endswith(
        "RoutedContentDecoderStage" if routed else "ContentDecoderStage"
    ), "content decoder mismatch")
    require(set(config.data.train_datasets) == set(sources), "training sources mismatch")
    require(set(config.data.valid_datasets) == set(sources), "validation sources mismatch")
    if routed:
        require(
            {
                source: int(config.model.pipeline.stages[4].encoders[source].action_dim)
                for source in sources
            }
            == row["sources"],
            "configured routed encoder action dimensions mismatch",
        )
    else:
        require(config.model.action_dim == row["sources"][sources[0]], "action dimension mismatch")
    require(config.run_provenance.split_manifest_sha256 == args.expected_split_sha256, "split mismatch")
    require(config.run_provenance.normalization_sha256 == args.expected_normalization_sha256, "normalization mismatch")
    require(config.run_provenance.content_manifest_sha256 == args.expected_content_manifest_sha256, "content manifest mismatch")
    require(config.run_provenance.dataset_content_aggregate_sha256 == args.expected_dataset_content_aggregate_sha256, "content aggregate mismatch")
    require(sha256(config_path) == args.expected_config_sha256, "resolved config hash mismatch")
    if routed:
        secondary = config.run_provenance.content_manifests[sources[1]]
        require(
            config.run_provenance.chain_split_manifest_sha256
            == args.expected_second_split_sha256,
            "secondary split mismatch",
        )
        require(
            secondary.manifest_sha256
            == args.expected_second_content_manifest_sha256,
            "secondary content manifest mismatch",
        )
        require(
            secondary.aggregate_sha256
            == args.expected_second_dataset_content_aggregate_sha256,
            "secondary content aggregate mismatch",
        )

    preflight_path = run_dir / "provenance/restart-0/PREFLIGHT_RESULT.json"
    require(preflight_path.is_file(), "preflight receipt is missing")
    require(sha256(preflight_path) == args.expected_preflight_sha256, "preflight hash mismatch")
    preflight = json.loads(preflight_path.read_text())
    require(preflight["status"] == "PASS", "preflight did not pass")
    require(preflight["source"]["head"] == args.expected_head, "preflight source mismatch")
    if routed:
        require("secondary" in preflight["datasets"], "secondary dataset proof missing")
        secondary_preflight = preflight["datasets"]["secondary"]
        require(
            secondary_preflight["split_manifest_sha256"]
            == args.expected_second_split_sha256,
            "secondary preflight split mismatch",
        )
        require(
            secondary_preflight["content_manifest_sha256"]
            == args.expected_second_content_manifest_sha256,
            "secondary preflight content manifest mismatch",
        )
        require(
            secondary_preflight["content_aggregate_sha256"]
            == args.expected_second_dataset_content_aggregate_sha256,
            "secondary preflight content aggregate mismatch",
        )

    probes = validate_gpu_probes(run_dir)

    last = run_dir / "checkpoints/last.ckpt"
    require(last.is_symlink() and last.is_file(), "last.ckpt is not an intact symlink")
    immutable = last.resolve(strict=True)
    require(immutable.parent == last.parent.resolve(), "checkpoint symlink escapes run")
    checkpoint = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
    require(checkpoint.get("global_step") == 2, "checkpoint is not global step 2")
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops"):
        require(checkpoint.get(key), f"checkpoint lacks {key}")
    require(
        isinstance(checkpoint["optimizer_states"], list)
        and len(checkpoint["optimizer_states"]) == 1,
        "checkpoint must contain one optimizer state",
    )
    optimizer_state = checkpoint["optimizer_states"][0]
    require(
        set(optimizer_state) == {"adamw", "muon", "group_manifest"},
        "checkpoint does not contain the released composite optimizer state",
    )
    for optimizer_name in ("adamw", "muon"):
        require(
            optimizer_state[optimizer_name].get("state"),
            f"{optimizer_name} optimizer state is empty",
        )
    finite_counts = {}
    for key in ("state_dict", "optimizer_states", "lr_schedulers"):
        finite_counts[key] = finite_tree(checkpoint[key], f"checkpoint.{key}")
    route_manifest = checkpoint.get("action_flow_gradient_route_manifest")
    require(isinstance(route_manifest, Mapping) and route_manifest, "gradient route manifest missing")
    require(
        set(route_manifest.get("routes", ()))
        == {"FM", "Reconstruction", "ActionVelocity"},
        "gradient route components mismatch",
    )
    manifest_core = {
        key: route_manifest[key]
        for key in ("routes", "route_sha256", "intersections", "schema_version")
    }
    require(
        route_manifest.get("manifest_sha256") == json_sha256(manifest_core),
        "gradient route manifest hash mismatch",
    )

    from egomimic.pl_utils.pl_model_action_flow import ActionFlowModelWrapper

    try:
        restored = ActionFlowModelWrapper.load_from_checkpoint(
            last, map_location="cpu", strict=True, weights_only=False
        )
    except Exception as error:
        raise VerificationError("strict ActionFlowModelWrapper reload failed") from error
    stages = tuple(restored.model.pipeline.stages)
    if routed:
        require(tuple(stages[4].encoder) == sources, "restored encoder routes mismatch")
        require(tuple(stages[7].decoder) == sources, "restored decoder routes mismatch")
        require(stages[4].encoder[sources[0]] is not stages[4].encoder[sources[1]], "restored encoders alias")
        require(stages[7].decoder[sources[0]] is not stages[7].decoder[sources[1]], "restored decoders alias")
    parameter_count = sum(parameter.numel() for parameter in restored.parameters())
    require(
        parameter_count == args.expected_parameter_count,
        f"restored parameter count mismatch: {parameter_count}",
    )
    del restored, checkpoint

    streams = [*run_dir.glob("wandb/run-*/run-*.wandb"), *run_dir.glob("wandb/offline-run-*/run-*.wandb")]
    require(len(streams) == 1, f"expected one W&B stream, found {len(streams)}")
    rows, exit_code = wandb_history(streams[0])
    components = ("TotalLoss", "FlowMatchingLoss", "ReconstructionLoss", "ActionVelocityLoss")
    train_names = tuple(
        f"Train/ActionFlow/{component}{suffix}"
        for suffix in ("", *(f"/{source}" for source in sources))
        for component in components
    ) + (
        "Train/ActionFlow/Compute/FieldForwardCallsPerStep",
        "Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep",
        "Train/ActionFlow/Compute/FlowMiniBatchPerStep",
        "Train/ActionFlow/Compute/DecoderJVPCallsPerStep",
    )
    valid_names = (
        "Valid/MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
        *(f"Valid/MSE/{source}" for source in sources),
        *(f"Valid/Native_MSE/{source}" for source in sources),
        *(f"Valid/EnergyScore@32/{source}" for source in sources),
        *(f"Valid/EnergyScoreAccuracy@32/{source}" for source in sources),
        *(f"Valid/EnergyScoreDiversity@32/{source}" for source in sources),
        *(f"Valid/ActionFlow/{component}" for component in components),
    )
    train_step, train = complete_row(rows, train_names, "training")
    valid_step, valid = complete_row(rows, valid_names, "validation")
    for suffix in ("", *(f"/{source}" for source in sources)):
        total = train[f"Train/ActionFlow/TotalLoss{suffix}"]
        expected = sum(train[f"Train/ActionFlow/{name}{suffix}"] for name in components[1:])
        require(math.isclose(total, expected, rel_tol=1e-5, abs_tol=1e-7), f"weighted total mismatch: {suffix}")
    require(
        train["Train/ActionFlow/Compute/FieldForwardCallsPerStep"] == 2,
        "field-forward count mismatch",
    )
    require(
        train["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"] == 28,
        "field sample-equivalent count mismatch",
    )
    require(
        train["Train/ActionFlow/Compute/FlowMiniBatchPerStep"] == 14,
        "parallel flow mini-batch mismatch",
    )
    require(
        train["Train/ActionFlow/Compute/DecoderJVPCallsPerStep"] == 1,
        "decoder-JVP count mismatch",
    )
    energy_artifact = step_two_artifact(
        config.evaluator.artifact_root,
        run_dir=run_dir,
        label="EnergyScore@32",
    )
    diagnostic_artifact = step_two_artifact(
        config.evaluator.action_flow_diagnostics.artifact_root,
        run_dir=run_dir,
        label="Action Flow diagnostics",
    )
    require_sources_in_artifact(energy_artifact["payload"], sources, label="EnergyScore")
    require_sources_in_artifact(
        diagnostic_artifact["payload"], sources, label="Action Flow diagnostics"
    )

    result = {
        "checkpoint": {
            "checkpoint_sha256": sha256(last),
            "file_size_bytes": last.stat().st_size,
            "finite_scan": finite_counts,
            "global_step": 2,
            "gradient_routes": {
                "manifest_sha256": route_manifest["manifest_sha256"],
                "route_sha256": route_manifest["route_sha256"],
            },
            "immutable_checkpoint_path": str(immutable),
            "parameter_count": parameter_count,
            "strict_checkpoint_reload": "passed",
        },
        "experiment": args.experiment,
        "gpu_probes": probes,
        "identities": {
            "repo_head": args.expected_head,
            "split_manifest_sha256": args.expected_split_sha256,
            "normalization_sha256": args.expected_normalization_sha256,
            "content_manifest_sha256": args.expected_content_manifest_sha256,
            "dataset_content_aggregate_sha256": args.expected_dataset_content_aggregate_sha256,
            "datasets": {
                sources[0]: {
                    "split_manifest_sha256": args.expected_split_sha256,
                    "content_manifest_sha256": args.expected_content_manifest_sha256,
                    "dataset_content_aggregate_sha256": args.expected_dataset_content_aggregate_sha256,
                },
                **(
                    {
                        sources[1]: {
                            "split_manifest_sha256": args.expected_second_split_sha256,
                            "content_manifest_sha256": args.expected_second_content_manifest_sha256,
                            "dataset_content_aggregate_sha256": args.expected_second_dataset_content_aggregate_sha256,
                        }
                    }
                    if routed
                    else {}
                ),
            },
        },
        "metrics": {"train_step": train_step, "valid_step": valid_step, "train": train, "valid": valid},
        "artifacts": {
            "energy_score": {
                "path": energy_artifact["path"],
                "sha256": energy_artifact["sha256"],
            },
            "action_flow_diagnostics": {
                "path": diagnostic_artifact["path"],
                "sha256": diagnostic_artifact["sha256"],
            },
        },
        "preflight": {"path": str(preflight_path), "sha256": args.expected_preflight_sha256},
        "run_dir": str(run_dir),
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "wandb": {"exit_code": exit_code, "stream_path": str(streams[0]), "stream_sha256": sha256(streams[0])},
    }
    destination = run_dir / "SMOKE_RESULT.json"
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    require(not destination.exists() or destination.read_text() == rendered, "differing smoke result exists")
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered)
        os.link(temporary, destination)
        temporary.unlink()
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("run_dir", type=Path)
    result.add_argument("--expected-experiment", dest="experiment", required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--expected-config-sha256", required=True)
    result.add_argument("--expected-split-sha256", required=True)
    result.add_argument("--expected-normalization-sha256", required=True)
    result.add_argument("--expected-content-manifest-sha256", required=True)
    result.add_argument("--expected-dataset-content-aggregate-sha256", required=True)
    result.add_argument("--expected-second-split-sha256")
    result.add_argument("--expected-second-content-manifest-sha256")
    result.add_argument("--expected-second-dataset-content-aggregate-sha256")
    result.add_argument("--expected-parameter-count", type=int, required=True)
    result.add_argument("--expected-preflight-sha256", required=True)
    result.add_argument("--expected-reconstruction-weight", type=float)
    result.add_argument("--expected-flow-weight", type=float)
    return result


def main() -> int:
    args = parser().parse_args()
    require(args.expected_reconstruction_weight == 1.0, "reconstruction weight mismatch")
    require(args.expected_flow_weight == 1.0, "flow weight mismatch")
    payload = verify(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
