"""Fail-closed official launch contract for YAM + human Action Flow co-training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.train.verify_yam_action_flow_launch import content_hash, digest


DOMAINS = {
    "yam": ("yam_bimanual", 227, 2, 249565, 2171, "s3://rldb/processed_v3/yam/"),
    "human": ("human_bimanual", 36, 1, 126699, 3307, "s3://rldb/processed_v3/aria/"),
}


def split_identities(path: Path, expected_sha: str) -> dict[str, str]:
    if digest(path) != expected_sha:
        raise ValueError("co-train split SHA mismatch")
    split = json.loads(path.read_text())
    if split.get("split_seed") != 42 or split.get("valid_ratio") != 0.01:
        raise ValueError("co-train split seed/ratio changed")
    if split.get("train_frame_windows_total") != 376264:
        raise ValueError("co-train train-window total changed")
    probabilities = split.get("proportional_train_window_probabilities")
    if probabilities != {"yam": 249565 / 376264, "human": 126699 / 376264}:
        raise ValueError("co-train proportional draw changed")
    identities: dict[str, str] = {}
    all_ids: set[tuple[str, str]] = set()
    all_paths: set[str] = set()
    for domain, (_, train_count, valid_count, train_windows, valid_windows, prefix) in DOMAINS.items():
        record = split[domain]
        summary = record["summary"]
        if (summary["train_episodes"], summary["valid_episodes"], summary["episodes"]) != (
            train_count, valid_count, train_count + valid_count
        ):
            raise ValueError(f"{domain} episode counts changed")
        if (summary["train_frame_windows"], summary["valid_frame_windows"]) != (
            train_windows, valid_windows
        ):
            raise ValueError(f"{domain} window counts changed")
        rows = {"train": [], "valid": []}
        for episode in record["episodes"]:
            name = episode["episode_hash"]
            path_value = episode["zarr_processed_path"]
            key = (domain, name)
            if (
                episode["split"] not in rows
                or path_value != f"{prefix}{name}.zarr"
                or not isinstance(episode["num_frames"], int)
                or episode["num_frames"] <= 0
                or key in all_ids or path_value in all_paths
            ):
                raise ValueError(f"{domain} invalid or overlapping split row")
            all_ids.add(key)
            all_paths.add(path_value)
            rows[episode["split"]].append(episode)
        for group, expected_count, expected_windows in (
            ("train", train_count, train_windows), ("valid", valid_count, valid_windows)
        ):
            if len(rows[group]) != expected_count or sum(row["num_frames"] for row in rows[group]) != expected_windows:
                raise ValueError(f"{domain} {group} rows do not cover frozen corpus")
            names = sorted(row["episode_hash"] for row in rows[group])
            actual = hashlib.sha256("".join(f"{name}\n" for name in names).encode()).hexdigest()
            if actual != summary[f"{group}_ids_sha256"]:
                raise ValueError(f"{domain} {group} ID hash mismatch")
            identities[f"{domain}_{group}"] = actual
    if len(all_ids) != 266 or len(all_paths) != 266:
        raise ValueError("co-train corpus union changed")
    return identities


def aggregate_content(yam_summary: Path, yam_sha: str, human_manifest: Path, human_sha: str) -> str:
    yam_content = content_hash(yam_summary, yam_sha)
    if digest(human_manifest) != human_sha:
        raise ValueError("human content manifest SHA mismatch")
    human = json.loads(human_manifest.read_text())
    if (human.get("episode_count"), human.get("object_count"), human.get("content_bytes")) != (
        37, 822, 5653818935
    ):
        raise ValueError("human content manifest totals changed")
    return hashlib.sha256(f"yam:{yam_content}\nhuman:{human_sha}\n".encode()).hexdigest()


def verify_config(cfg_path: Path, args: argparse.Namespace, ids: dict[str, str], content: str) -> None:
    cfg = OmegaConf.load(cfg_path)

    def require(path: str, expected) -> None:
        actual = OmegaConf.select(cfg, path)
        if actual != expected:
            raise ValueError(f"resolved {path} mismatch: {actual!r} != {expected!r}")

    require("name", "yam_human_keypoints_action_flow_h816_private512_s42")
    require("mode", "train")
    require("seed", 42)
    require("e1.data_root", str(args.yam_root))
    require("launch_params.gpus_per_node", 1)
    require("launch_params.nodes", 1)
    require("trainer.max_steps", 4 if args.phase == "smoke" else 80000)
    require("trainer.precision", "bf16-mixed")
    require("trainer.accumulate_grad_batches", 1)
    require("run_provenance.grad_accumulation", 1)
    require("data._target_", "egomimic.pl_utils.pl_data_utils.ProportionalMultiDataModuleWrapper")
    require("data.proportional_train_batch_size", 8)
    require("data.proportional_train_num_workers", 12)
    require("data.proportional_train_seed", 42)
    require("data.train_datasets.yam_bimanual.resolver.folder_path", str(args.yam_root))
    require("data.train_datasets.human_bimanual.resolver.folder_path", str(args.human_root))
    for domain, (name, train_count, valid_count, train_windows, _, _) in DOMAINS.items():
        require(f"data.train_datasets.{name}.mode", "train")
        require(f"data.train_datasets.{name}.valid_ratio", 0.01)
        require(f"data.train_datasets.{name}.split_seed", 42)
        require(f"data.train_datasets.{name}.expected_train_episode_count", train_count)
        require(f"data.train_datasets.{name}.expected_valid_episode_count", valid_count)
        require(f"data.train_datasets.{name}.expected_train_episode_names_sha256", ids[f"{domain}_train"])
        require(f"data.train_datasets.{name}.expected_valid_episode_names_sha256", ids[f"{domain}_valid"])
        require(f"run_provenance.train_windows_per_domain.{name}", train_windows)
        require(f"data.valid_datasets.{domain}.{name}.mode", "valid")
        require(f"data.valid_datasets.{domain}.{name}.expected_valid_episode_names_sha256", ids[f"{domain}_valid"])
        require(f"data.valid_dataloader_params.{domain}.{name}.batch_size", 4)
    require("model.hidden_dim", 816)
    require("model.private_hidden_dim", 512)
    require("model.private_depth", 6)
    require("model.action_horizon", 100)
    require("model.flow_samples_per_content", 14)
    require("model.num_inference_steps", 50)
    require("model.jvp_activation_checkpointing", True)
    require("model.gradient_telemetry_cadence", 3 if args.phase == "smoke" else 100)
    require("model.pipeline.stages.2._target_", "egomimic.pipeline.stages_io.EmbodimentActionTargetBuilder")
    require("model.pipeline.stages.2.action_keys.yam_bimanual", "actions_cartesian")
    require("model.pipeline.stages.2.action_keys.human_bimanual", "actions_keypoints")
    require("model.pipeline.stages.4.encoders.yam_bimanual.action_dim", 14)
    require("model.pipeline.stages.4.encoders.human_bimanual.action_dim", 138)
    require("model.pipeline.stages.7.decoders.yam_bimanual.action_dim", 14)
    require("model.pipeline.stages.7.decoders.human_bimanual.action_dim", 138)
    if set(cfg.model.pipeline.stages[0].stems) != {"observations.images.front_img_1"}:
        raise ValueError("co-train front-camera observation contract changed")
    if set(cfg.model.pipeline.stages[0].domain_stems.yam_bimanual) != {"observations.state.ee_pose"}:
        raise ValueError("YAM robot-state observation contract changed")
    if set(cfg.model.pipeline.stages[0].domain_stems.human_bimanual) != {"observations.state.keypoints"}:
        raise ValueError("human keypoint-state observation contract changed")
    if set(cfg.model.pipeline.stages[1].domains) != {"yam_bimanual", "human_bimanual"}:
        raise ValueError("co-train shared conditioning domains changed")
    require("evaluator._target_", "egomimic.eval.yam_human_keypoints_action_flow_eval.YamHumanKeypointsActionFlowEval")
    require("evaluator.energy_score_enabled", True)
    require("evaluator.seed_bank_sha256", "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6")
    require("evaluator.action_keys_by_embodiment.yam_bimanual", "actions_cartesian")
    require("evaluator.action_keys_by_embodiment.human_bimanual", "actions_keypoints")
    require("evaluator.energy_score_validation_view.per_rank_batch_size", 4)
    require("evaluator.action_flow_diagnostics.validation_view.per_rank_batch_size", 4)
    require("evaluator.action_flow_diagnostics.cknna_k", 2)
    require("evaluator.action_flow_diagnostics.jacobian_method", "forward_math_chunk4")
    require("evaluator.artifact_root", str(args.run_dir / "validation_predictions/energy_score"))
    require("evaluator.action_flow_diagnostics.artifact_root", str(args.run_dir / "validation_predictions/action_flow_diagnostics"))
    diagnostic_rows = int(OmegaConf.select(cfg, "evaluator.action_flow_diagnostics.validation_view.per_rank_batch_size"))
    cknna_k = int(OmegaConf.select(cfg, "evaluator.action_flow_diagnostics.cknna_k"))
    if not 2 <= cknna_k < diagnostic_rows:
        raise ValueError("Action Flow CKNNA k must be smaller than diagnostic batch size")
    if 8 * int(OmegaConf.select(cfg, "model.flow_samples_per_content")) != 112:
        raise ValueError("co-train expanded decoder JVP batch changed")
    require("norm_stats.precomputed_norm_path", str(args.norm_json))
    require("run_provenance.source_commit", args.source_commit)
    require("run_provenance.launch_recipe", "yam_human_keypoints")
    require("run_provenance.normalization_sha256", args.norm_sha)
    require("run_provenance.split_manifest_sha256", args.split_sha)
    require("run_provenance.dataset_content_aggregate_sha256", content)
    require("paths.output_dir", str(args.run_dir))
    require("logger.wandb.entity", args.wandb_entity)
    require("logger.wandb.project", args.wandb_project)
    require("logger.wandb.id", args.wandb_id)
    require("callbacks.model_checkpoint.save_top_k", -1)
    if args.phase == "smoke":
        require("trainer.val_check_interval", 4)
        require("trainer.limit_val_batches", 1)
        require("trainer.log_every_n_steps", 1)
        if int(OmegaConf.select(cfg, "model.scheduler.warmup_steps")) > 0 and cfg.trainer.max_steps < 2:
            raise ValueError("smoke must include a positive-LR optimizer update")
        if cfg.trainer.val_check_interval < 2 * cfg.trainer.accumulate_grad_batches:
            raise ValueError("smoke validation must follow a positive-LR optimizer update")
        if cfg.trainer.max_steps <= cfg.model.gradient_telemetry_cadence:
            raise ValueError("smoke must continue past gradient telemetry cadence")
        require("callbacks.model_checkpoint.every_n_train_steps", 4)
    else:
        require("trainer.val_check_interval", 5000)
        require("trainer.limit_val_batches", 256)
        require("callbacks.model_checkpoint.every_n_train_steps", 5000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha", required=True)
    parser.add_argument("--yam-content-summary", type=Path, required=True)
    parser.add_argument("--yam-content-sha", required=True)
    parser.add_argument("--human-content-manifest", type=Path, required=True)
    parser.add_argument("--human-content-sha", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--phase", choices=("smoke", "full"))
    parser.add_argument("--source-commit")
    parser.add_argument("--yam-root", type=Path)
    parser.add_argument("--human-root", type=Path)
    parser.add_argument("--norm-json", type=Path)
    parser.add_argument("--norm-sha")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-id")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    ids = split_identities(args.split, args.split_sha)
    content = aggregate_content(
        args.yam_content_summary, args.yam_content_sha,
        args.human_content_manifest, args.human_content_sha,
    )
    if args.config is None:
        print(" ".join((*ids.values(), content)))
        return
    required = (
        "receipt", "phase", "source_commit", "yam_root", "human_root",
        "norm_json", "norm_sha", "wandb_entity", "wandb_project", "wandb_id", "run_dir",
    )
    if any(getattr(args, name) is None for name in required):
        raise ValueError("co-train resolved-config verification arguments incomplete")
    if digest(args.norm_json) != args.norm_sha:
        raise ValueError("co-train normalizer SHA mismatch")
    verify_config(args.config, args, ids, content)
    if args.receipt.exists():
        raise FileExistsError("co-train preflight receipt already exists")
    record = {
        "phase": args.phase, "source_commit": args.source_commit,
        "split_sha256": args.split_sha, "normalization_sha256": args.norm_sha,
        "yam_content_summary_sha256": args.yam_content_sha,
        "human_content_manifest_sha256": args.human_content_sha,
        "dataset_content_aggregate_sha256": content,
        "resolved_config_sha256": digest(args.config),
        "wandb_run_id": args.wandb_id,
    }
    args.receipt.write_text(json.dumps(record, sort_keys=True) + "\n")
    print("YAM_HUMAN_ACTION_FLOW_LAUNCH_PREFLIGHT_PASS " + json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
