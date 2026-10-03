"""Fail-closed identity and resolved-config checks for the YAM Action Flow launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_hashes(path: Path, expected_sha: str) -> tuple[str, str]:
    if digest(path) != expected_sha:
        raise ValueError("split manifest SHA mismatch")
    obj = json.loads(path.read_text())
    if obj.get("counts") != {"train": 227, "valid": 2, "union": 229}:
        raise ValueError("split episode counts changed")
    groups = {"train": [], "valid": []}
    paths = set()
    for item in obj["episodes"]:
        group, name, source = item["split"], item["episode_hash"], item["zarr_processed_path"]
        if group not in groups or source != f"s3://rldb/processed_v3/yam/{name}.zarr":
            raise ValueError("invalid YAM split row")
        if source in paths or item["num_frames"] <= 0:
            raise ValueError("duplicate path or nonpositive frames")
        paths.add(source)
        groups[group].append(name)
    if len(paths) != 229 or any(len(groups[key]) != obj["counts"][key] for key in groups):
        raise ValueError("split union changed")
    return tuple(
        hashlib.sha256("".join(f"{name}\n" for name in sorted(groups[key])).encode()).hexdigest()
        for key in ("train", "valid")
    )


def content_hash(path: Path, expected_sha: str) -> str:
    if digest(path) != expected_sha:
        raise ValueError("corpus receipt SHA mismatch")
    record = json.loads(path.read_text())
    if record.get("objects") != 7328 or record.get("bytes") != 34631699584:
        raise ValueError("corpus receipt totals changed")
    value = record.get("relative_size_sha256_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("corpus content digest missing")
    return value


def verify_config(cfg_path: Path, args, train_hash: str, valid_hash: str, content: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    def require(path, value):
        actual = OmegaConf.select(cfg, path)
        if actual != value:
            raise ValueError(f"resolved {path} mismatch: {actual!r} != {value!r}")
    require("name", "yam_stationary_action_flow_h816_s42")
    require("mode", "train")
    require("seed", 42)
    require("e1.data_root", str(args.data_root))
    require("e1.chunk_length", 100)
    require("e1.action_dim", 14)
    require("launch_params.gpus_per_node", 1)
    require("launch_params.nodes", 1)
    require("trainer.max_steps", 2 if args.phase == "smoke" else 80000)
    require("trainer.precision", "bf16-mixed")
    require("trainer.accumulate_grad_batches", 16)
    require("data.train_dataloader_params.yam_bimanual.batch_size", 8)
    require("data.valid_dataloader_params.yam_bimanual.batch_size", 4)
    require("evaluator.energy_score_validation_view.per_rank_batch_size", 4)
    require("evaluator.action_flow_diagnostics.validation_view.per_rank_batch_size", 4)
    require("evaluator.action_flow_diagnostics.cknna_k", 2)
    diagnostic_rows = int(
        OmegaConf.select(cfg, "evaluator.action_flow_diagnostics.validation_view.per_rank_batch_size")
    )
    cknna_k = int(OmegaConf.select(cfg, "evaluator.action_flow_diagnostics.cknna_k"))
    if not 2 <= cknna_k < diagnostic_rows:
        raise ValueError("Action Flow CKNNA k must be smaller than diagnostic batch size")
    require("evaluator.action_flow_diagnostics.jacobian_method", "forward_math_chunk4")
    require("paths.output_dir", str(args.run_dir))
    require("evaluator.artifact_root", str(args.run_dir / "validation_predictions/energy_score"))
    require(
        "evaluator.action_flow_diagnostics.artifact_root",
        str(args.run_dir / "validation_predictions/action_flow_diagnostics"),
    )
    require("trainer.limit_val_batches", 1 if args.phase == "smoke" else 256)
    require("model.hidden_dim", 816)
    require("model.flow_samples_per_content", 14)
    require("model.jvp_activation_checkpointing", True)
    train_microbatch = int(
        OmegaConf.select(cfg, "data.train_dataloader_params.yam_bimanual.batch_size")
    )
    bridge_samples = int(OmegaConf.select(cfg, "model.flow_samples_per_content"))
    expanded_jvp_batch = train_microbatch * bridge_samples
    if expanded_jvp_batch != 112:
        raise ValueError(f"unexpected expanded decoder JVP batch: {expanded_jvp_batch}")
    require("model.action_dim", 14)
    require("model.action_horizon", 100)
    require("data.train_datasets.yam_bimanual.mode", "train")
    require("data.valid_datasets.yam_bimanual.mode", "valid")
    require("data.train_datasets.yam_bimanual.expected_train_episode_names_sha256", train_hash)
    require("data.valid_datasets.yam_bimanual.expected_valid_episode_names_sha256", valid_hash)
    require("norm_stats.precomputed_norm_path", str(args.norm_json))
    require("run_provenance.source_commit", args.source_commit)
    require("run_provenance.normalization_sha256", args.norm_sha)
    require("run_provenance.split_manifest_sha256", args.split_sha)
    require("run_provenance.dataset_content_aggregate_sha256", content)
    require("logger.wandb.entity", args.wandb_entity)
    require("logger.wandb.project", args.wandb_project)
    require("logger.wandb.id", args.wandb_id)
    if args.phase == "smoke":
        require("trainer.val_check_interval", 32)
        warmup_steps = int(OmegaConf.select(cfg, "model.scheduler.warmup_steps"))
        if warmup_steps > 0 and cfg.trainer.max_steps < 2:
            raise ValueError("smoke must include a positive-LR optimizer update")
        if cfg.trainer.val_check_interval < 2 * cfg.trainer.accumulate_grad_batches:
            raise ValueError("smoke validation must follow a positive-LR optimizer update")
        require("callbacks.model_checkpoint.every_n_train_steps", 1)
    else:
        require("trainer.val_check_interval", 5000)
        require("callbacks.model_checkpoint.every_n_train_steps", 5000)
    if set(cfg.model.pipeline.stages[0].stems) != {"observations.images.front_img_1"}:
        raise ValueError("front-camera observation contract changed")
    if set(cfg.model.pipeline.stages[0].domain_stems.yam_bimanual) != {"observations.state.ee_pose"}:
        raise ValueError("robot-state observation contract changed")
    if cfg.model.pipeline.stages[2].action_key != "actions_cartesian":
        raise ValueError("native Cartesian action contract changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha", required=True)
    parser.add_argument("--content-summary", type=Path, required=True)
    parser.add_argument("--content-summary-sha", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--phase", choices=("smoke", "full"))
    parser.add_argument("--source-commit")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--norm-json", type=Path)
    parser.add_argument("--norm-sha")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-id")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    train, valid = split_hashes(args.split, args.split_sha)
    content = content_hash(args.content_summary, args.content_summary_sha)
    if args.config is None:
        print(f"{train} {valid} {content}")
        return
    if any(getattr(args, name) is None for name in (
        "receipt", "phase", "source_commit", "data_root", "norm_json", "norm_sha",
        "wandb_entity", "wandb_project", "wandb_id", "run_dir"
    )):
        raise ValueError("resolved-config verification arguments incomplete")
    if digest(args.norm_json) != args.norm_sha:
        raise ValueError("normalization artifact SHA mismatch")
    verify_config(args.config, args, train, valid, content)
    record = {
        "phase": args.phase, "source_commit": args.source_commit,
        "split_sha256": args.split_sha, "normalization_sha256": args.norm_sha,
        "content_summary_sha256": args.content_summary_sha,
        "resolved_config_sha256": digest(args.config),
        "wandb_run_id": args.wandb_id,
    }
    if args.receipt.exists():
        raise FileExistsError("preflight receipt exists")
    args.receipt.write_text(json.dumps(record, sort_keys=True) + "\n")
    print("YAM_ACTION_FLOW_LAUNCH_PREFLIGHT_PASS " + json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
