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


def corpus_domains(contract_path: Path | None = None, contract_sha: str | None = None):
    """A fresh corpus requires an independently SHA-pinned complete contract.

    Omitting the contract preserves the historical frozen source boundary.
    Sparse numeric/camera audits cannot certify a complete training corpus.
    """
    if contract_path is None and contract_sha is None:
        return DOMAINS
    if contract_path is None or contract_sha is None or digest(contract_path) != contract_sha:
        raise ValueError("refreshed corpus contract SHA mismatch")
    contract = json.loads(contract_path.read_text())
    if contract.get("schema_version") != 1 or contract.get("complete_training_corpus") is not True:
        raise ValueError("refreshed corpus is not complete training data")
    if contract.get("split_seed") != 42 or contract.get("valid_ratio") != 0.01:
        raise ValueError("refreshed corpus split seed/ratio changed")
    if set(contract.get("domains", {})) != set(DOMAINS):
        raise ValueError("refreshed corpus domain set changed")
    if contract.get("window_contract", "native_padded_windows_v1") not in {"native_padded_windows_v1", "physical_complete_native_virtual_views_v1"}:
        raise ValueError("unknown corpus window contract")
    if contract.get("window_contract") == "physical_complete_native_virtual_views_v1":
        if contract.get("human_rates") != [0.2, 0.4, 0.6, 0.8, 1.0] or contract.get("yam_rates") != [1.0] or contract.get("sample_views_per_source") != 5:
            raise ValueError("selected physical rate grid/views changed")
    domains = {}
    for domain, (name, _, _, _, _, prefix) in DOMAINS.items():
        item = contract["domains"][domain]
        train, valid = item["train_episodes"], item["valid_episodes"]
        windows, valid_windows = item["train_frame_windows"], item["valid_frame_windows"]
        for value in (train, valid, windows, valid_windows, item["object_count"], item["content_bytes"]):
            if type(value) is not int or value <= 0:
                raise ValueError("invalid refreshed corpus totals")
        if valid != max(1, int((train + valid) * 0.01)):
            raise ValueError("refreshed corpus does not use native 1percent holdout")
        for field in ("episode_inventory_sha256", "complete_content_manifest_sha256", "aggregate_content_sha256"):
            value = item.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid refreshed corpus {field}")
        domains[domain] = (name, train, valid, windows, valid_windows, prefix)
    return domains


def window_count(episode, domain, contract_path):
    if contract_path is None:
        return episode["num_frames"]
    contract = json.loads(contract_path.read_text())
    kind = contract.get("window_contract", "native_padded_windows_v1")
    if kind == "native_padded_windows_v1":
        return episode["num_frames"]
    if kind != "physical_complete_native_virtual_views_v1":
        raise ValueError("unknown physical window contract")
    from egomimic.rldb.zarr.episode_split import complete_window_count
    return complete_window_count(episode["num_frames"], 100 if domain == "yam" else 30, 5)


def split_identities(path: Path, expected_sha: str, contract_path: Path | None = None, contract_sha: str | None = None) -> dict[str, str]:
    domains = corpus_domains(contract_path, contract_sha)
    total_windows = sum(values[3] for values in domains.values())
    if digest(path) != expected_sha:
        raise ValueError("co-train split SHA mismatch")
    split = json.loads(path.read_text())
    if split.get("split_seed") != 42 or split.get("valid_ratio") != 0.01:
        raise ValueError("co-train split seed/ratio changed")
    if split.get("train_frame_windows_total") != total_windows:
        raise ValueError("co-train train-window total changed")
    probabilities = split.get("proportional_train_window_probabilities")
    if probabilities != {domain: values[3] / total_windows for domain, values in domains.items()}:
        raise ValueError("co-train proportional draw changed")
    identities: dict[str, str] = {}
    all_ids: set[tuple[str, str]] = set()
    all_paths: set[str] = set()
    for domain, (_, train_count, valid_count, train_windows, valid_windows, prefix) in domains.items():
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
            if len(rows[group]) != expected_count or sum(window_count(row, domain, contract_path) for row in rows[group]) != expected_windows:
                raise ValueError(f"{domain} {group} rows do not cover frozen corpus")
            names = sorted(row["episode_hash"] for row in rows[group])
            actual = hashlib.sha256("".join(f"{name}\n" for name in names).encode()).hexdigest()
            if actual != summary[f"{group}_ids_sha256"]:
                raise ValueError(f"{domain} {group} ID hash mismatch")
            identities[f"{domain}_{group}"] = actual
    if contract_path is not None:
        from egomimic.rldb.zarr.episode_split import split_dataset_names
        contract = json.loads(contract_path.read_text())
        for domain in domains:
            episodes = split[domain]["episodes"]
            inventory = sorted((row["episode_hash"], row["zarr_processed_path"], row["num_frames"]) for row in episodes)
            inventory_sha = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
            if inventory_sha != contract["domains"][domain]["episode_inventory_sha256"]:
                raise ValueError("refreshed split does not cover frozen inventory")
            train, valid = split_dataset_names([row["episode_hash"] for row in episodes], valid_ratio=0.01, seed=42)
            if {row["episode_hash"] for row in episodes if row["split"] == "train"} != train or {row["episode_hash"] for row in episodes if row["split"] == "valid"} != valid:
                raise ValueError("refreshed split differs from native deterministic membership")
    union_count = sum(values[1] + values[2] for values in domains.values())
    if len(all_ids) != union_count or len(all_paths) != union_count:
        raise ValueError("co-train corpus union changed")
    return identities


def aggregate_content(yam_summary: Path, yam_sha: str, human_manifest: Path, human_sha: str, contract_path: Path | None = None, contract_sha: str | None = None) -> str:
    if contract_path is not None:
        domains = corpus_domains(contract_path, contract_sha)
        contract = json.loads(contract_path.read_text())
        values = []
        for domain, path, expected_sha in (("yam", yam_summary, yam_sha), ("human", human_manifest, human_sha)):
            pinned = contract["domains"][domain]
            if digest(path) != expected_sha or expected_sha != pinned["complete_content_manifest_sha256"]:
                raise ValueError("complete content manifest SHA mismatch")
            content = json.loads(path.read_text())
            if content.get("complete_training_corpus") is not True or content.get("sparse_audit_cache_not_training_data"):
                raise ValueError("sparse audit is not a complete training corpus")
            expected = {"episode_count": domains[domain][1] + domains[domain][2], "object_count": pinned["object_count"], "content_bytes": pinned["content_bytes"], "episode_inventory_sha256": pinned["episode_inventory_sha256"], "aggregate_content_sha256": pinned["aggregate_content_sha256"]}
            if any(content.get(key) != value for key, value in expected.items()):
                raise ValueError("complete corpus content identity mismatch")
            values.append(f"{domain}:{pinned['aggregate_content_sha256']}\n")
        return hashlib.sha256("".join(values).encode()).hexdigest()
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
    domains = corpus_domains(getattr(args, "corpus_contract", None), getattr(args, "corpus_contract_sha", None))
    augmentation = getattr(args, "augmentation", "none")

    def require(path: str, expected) -> None:
        actual = OmegaConf.select(cfg, path)
        if actual != expected:
            raise ValueError(f"resolved {path} mismatch: {actual!r} != {expected!r}")

    experiment_names = {
        "none": "yam_human_keypoints_action_flow_h816_private512_s42",
        "human_speed_v1": "yam_human_keypoints_speed_h816_private512_s42",
    }
    if augmentation not in experiment_names:
        raise ValueError("unknown augmentation")
    require("name", experiment_names[augmentation])
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
    for domain, (name, train_count, valid_count, train_windows, _, _) in domains.items():
        require(f"data.train_datasets.{name}.mode", "train")
        require(f"data.train_datasets.{name}.valid_ratio", 0.01)
        require(f"data.train_datasets.{name}.split_seed", 42)
        require(f"data.train_datasets.{name}.expected_train_episode_count", train_count)
        require(f"data.train_datasets.{name}.expected_valid_episode_count", valid_count)
        require(f"data.train_datasets.{name}.expected_train_episode_names_sha256", ids[f"{domain}_train"])
        require(f"data.train_datasets.{name}.expected_valid_episode_names_sha256", ids[f"{domain}_valid"])
        require(f"run_provenance.train_windows_per_domain.{name}", train_windows)
        require(f"run_provenance.train_episode_count_per_domain.{name}", train_count)
        require(f"run_provenance.valid_episode_count_per_domain.{name}", valid_count)
        require(f"run_provenance.union_episode_count_per_domain.{name}", train_count + valid_count)
        require(f"data.valid_datasets.{domain}.{name}.expected_train_episode_count", train_count)
        require(f"data.valid_datasets.{domain}.{name}.expected_valid_episode_count", valid_count)
        require(f"data.valid_datasets.{domain}.{name}.mode", "valid")
        require(f"data.valid_datasets.{domain}.{name}.expected_valid_episode_names_sha256", ids[f"{domain}_valid"])
        require(f"data.valid_dataloader_params.{domain}.{name}.batch_size", 4)
    require("model.hidden_dim", 816)
    require("model.private_hidden_dim", 512)
    require("model.private_depth", 6)
    require("model.action_horizon", 100)
    require("model.flow_samples_per_content", 14)
    require("model.num_inference_steps", 50)
    inference_method = getattr(args, "inference_method", "dopri5")
    policy = getattr(args, "checkpoint_policy", "all")
    require("model.pipeline.stages.6.inference_method", inference_method)
    require("run_provenance.inference.sampler", inference_method)
    require("model.pipeline.stages.6.num_inference_steps", 50)
    require("model.pipeline.stages.6.timestep_shift_alpha", 0.5)
    for provenance in ("evaluator.energy_score_provenance", "evaluator.action_flow_diagnostics.provenance"):
        require(provenance + ".sampler", inference_method)
        require(provenance + ".sampler_steps", 50)
    if policy == "dit_half":
        require("run_provenance.execution.checkpoint_policy", "dit_half")
        for path in ("model.pipeline.stages.4.encoders.yam_bimanual.backbone", "model.pipeline.stages.4.encoders.human_bimanual.backbone", "model.pipeline.stages.6.field.backbone"):
            require(path + ".checkpoint_policy", "dit_half")
            require(path + ".gradient_checkpointing", True)
        require("callbacks.yam_dit_half._target_", "egomimic.utils.yam_dit_half.YamHumanDiTHalf")
    if augmentation == "human_speed_v1":
        require("model.pipeline._target_", "egomimic.pipeline.stages_speed.build_speed_conditioned_pipeline")
        require("model.pipeline._recursive_", False)
        require("model.pipeline.encoding", "scalar")
        require("model.pipeline.condition_dim", 256)
        require("stationary_speed.human_rates", [0.2, 0.4, 0.6, 0.8, 1.0])
        require("stationary_speed.yam_rates", [1.0])
        require("stationary_speed.probabilities", [0.2] * 5)
        require("stationary_speed.sample_views_per_source", 5)
        require("stationary_speed.tail_contract", "complete_native_raw_windows_no_padding")
        require("stationary_speed.timestamp_contract", "human_recorded_rgb_nanoseconds_robot_metadata_fps")
        require("run_provenance.speed_augmentation.inference_contract", "external_requested_speed_no_future_action_oracle")
        reference = getattr(args, "speed_reference", None)
        if reference is None or not 0 < reference < float("inf"):
            raise ValueError("positive train-only speed reference required")
        require("stationary_speed.reference", reference)
        reference_path = getattr(args, "speed_reference_receipt", None)
        reference_sha = getattr(args, "speed_reference_receipt_sha", None)
        if reference_path is None or digest(reference_path) != reference_sha:
            raise ValueError("train-only speed reference receipt SHA mismatch")
        reference_record = json.loads(reference_path.read_text())
        if (reference_record.get("status") != "TRAIN_ONLY_PHYSICAL_SPEED_REFERENCE_V1"
                or reference_record.get("reference") != reference
                or reference_record.get("train_ids_sha256") != ids["yam_train"]
                or reference_record.get("statistic") != "refreshed_train_robot_command_both_arm_native_window_median_v1"):
            raise ValueError("train-only speed reference identity mismatch")
        contract = json.loads(args.corpus_contract.read_text())
        if contract.get("window_contract") != "physical_complete_native_virtual_views_v1":
            raise ValueError("speed augmentation requires complete-native-window counts")
        for domain, horizon in (("yam", 100), ("human", 30)):
            name = domains[domain][0]
            path = "data.train_datasets." + name + ".resolver.transform_list.window_transform"
            require(path + "._target_", "egomimic.rldb.zarr.physical_retiming.PhysicalWindowRetiming")
            require(path + ".rates", [1.0] if domain == "yam" else [0.2, 0.4, 0.6, 0.8, 1.0])
            require(path + ".horizon", horizon)
            require(path + ".stride", 1 if domain == "yam" else 3)
            require(path + ".sample_views", 5)
            require(path + ".timestamp_key", None if domain == "yam" else "_physical_timestamps_ns")
        require("data.train_datasets.human_bimanual.resolver.key_map.extra_key_map._physical_timestamps_ns.zarr_key", "obs_rgb_timestamps_ns")
        require("data.train_datasets.human_bimanual.resolver.key_map.extra_key_map._physical_timestamps_ns.horizon", 30)
    elif augmentation != "none":
        raise ValueError("unknown augmentation")
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
    if args.phase == "norm":
        require("norm_stats_only", True)
        require("norm_stats.precomputed_norm_path", None)
        require("norm_stats.save_cache_dir", str(args.norm_cache_dir))
        require("norm_stats.sample_frac", 1.0)
        require("norm_stats.norm_mode", "minmax")
    else:
        require("norm_stats_only", False)
        require("norm_stats.precomputed_norm_path", str(args.norm_json))
    require("run_provenance.source_commit", args.source_commit)
    require("run_provenance.launch_recipe", "yam_human_keypoints")
    require("run_provenance.normalization_sha256", None if args.phase == "norm" else args.norm_sha)
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
    parser.add_argument("--corpus-contract", type=Path)
    parser.add_argument("--corpus-contract-sha")
    parser.add_argument("--emit-corpus-counts", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--phase", choices=("norm", "smoke", "full"))
    parser.add_argument("--source-commit")
    parser.add_argument("--inference-method", choices=("euler", "dopri5"), default="dopri5")
    parser.add_argument("--checkpoint-policy", choices=("all", "dit_half"), default="all")
    parser.add_argument("--augmentation", choices=("none", "human_speed_v1"), default="none")
    parser.add_argument("--speed-reference", type=float)
    parser.add_argument("--speed-reference-receipt", type=Path)
    parser.add_argument("--speed-reference-receipt-sha")
    parser.add_argument("--yam-root", type=Path)
    parser.add_argument("--human-root", type=Path)
    parser.add_argument("--norm-json", type=Path)
    parser.add_argument("--norm-sha")
    parser.add_argument("--norm-cache-dir", type=Path)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-id")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    ids = split_identities(args.split, args.split_sha, args.corpus_contract, args.corpus_contract_sha)
    content = aggregate_content(
        args.yam_content_summary, args.yam_content_sha,
        args.human_content_manifest, args.human_content_sha, args.corpus_contract, args.corpus_contract_sha,
    )
    if args.emit_corpus_counts:
        domains = corpus_domains(args.corpus_contract, args.corpus_contract_sha)
        print(" ".join(str(value) for domain in ("yam", "human") for value in domains[domain][1:5]))
        return
    if args.config is None:
        print(" ".join((*ids.values(), content)))
        return
    required = (
        "receipt", "phase", "source_commit", "yam_root", "human_root",
        "norm_json", "norm_sha", "wandb_entity", "wandb_project", "wandb_id", "run_dir",
    )
    if any(getattr(args, name) is None for name in required):
        raise ValueError("co-train resolved-config verification arguments incomplete")
    if args.phase == "norm" and args.norm_cache_dir is None:
        raise ValueError("normalization phase cache directory missing")
    if args.phase != "norm" and digest(args.norm_json) != args.norm_sha:
        raise ValueError("co-train normalizer SHA mismatch")
    verify_config(args.config, args, ids, content)
    if args.receipt.exists():
        raise FileExistsError("co-train preflight receipt already exists")
    record = {
        "phase": args.phase, "source_commit": args.source_commit,
        "corpus_contract_sha256": args.corpus_contract_sha,
        "inference_method": args.inference_method, "checkpoint_policy": args.checkpoint_policy,
        "augmentation": args.augmentation,
        "speed_reference": args.speed_reference,
        "speed_reference_receipt_sha256": args.speed_reference_receipt_sha,
        "split_sha256": args.split_sha, "normalization_sha256": None if args.phase == "norm" else args.norm_sha,
        "normalization_cache_dir": str(args.norm_cache_dir) if args.phase == "norm" else None,
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
