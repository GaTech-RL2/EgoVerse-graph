"""Maintained profile integration candidate. No launcher or training entry point.

The canonical launcher must call these typed functions instead of its planar
branches. Every model construction/checkpoint/data operation is scheduled-only.
"""
import hashlib
import math
import os
from pathlib import Path

from egomimic.benchmarks.libero.native_launch_profiles import profile_for_config, profile_for_experiment

PROFILE = "libero/action_flow_libero10_h240_euler50_dithalf_80k_s42"
SOURCE = "libero_panda"
EXPECTED_PARAMETER_COUNT = 39750391
SEED_BANK_SHA = "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
METRICS = (
    "reconst_mse", "reconst_mae", "translation_mse", "rotation_command_mse",
    "gripper_mse", "gripper_sign_accuracy", "command_endpoint_error_m",
    "normalized_reconst_mse", "energy_score32_native_equal_components",
    "energy_accuracy32_native_equal_components", "energy_diversity32_native_equal_components",
    "diagnostic_clean_latent_rms", "diagnostic_clean_decoded_action_normalized_rms",
)

def scheduled_only():
    if not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("native LIBERO model/data/checkpoint checks require scheduled srun")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def flatten(value, stem=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten(child, (stem+"." if stem else "")+str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten(child, stem+"."+str(index)))
    else:
        result[stem] = value
    return result

def evaluator_overrides(profile):
    profile_for_experiment(profile)
    # Native LiberoActionFlowEvaluator has no energy_score_validation_view.
    return ("evaluator.energy_sample_count=32",
            "evaluator.energy_seed_bank_sha256="+SEED_BANK_SHA)

def reject_planar_leaks(config):
    flat = flatten(config)
    if any(key.startswith(("planar.", "evaluator.energy_score_validation_view"))
           for key in flat):
        raise ValueError("planar dataset or U-Socket evaluator fields leaked into LIBERO")
    if config.get("norm_stats", {}).get("precomputed_norm_path") is not None:
        raise ValueError("native LIBERO rejects planar precomputed normalization path")

def validate_resolved(config, phase):
    reject_planar_leaks(config)
    # Reuse the exact typed path guard after moving it into the maintained tool.
    from typed_libero_profile_v3 import validate_flat_config
    flat = flatten(config)
    flat.update({"data.train_source_names": sorted(config["data"]["train_datasets"]),
                 "data.valid_source_names": sorted(config["data"]["valid_datasets"]),
                 "typed_profile.velocity_augmentation": False,
                 "typed_profile.homogeneous": "not_applicable_single_source",
                 "typed_profile.checkpoint_policy": "dit-half"})
    profile = profile_for_config(config)
    seed = config.get("seed")
    if type(seed) is not int or seed not in ({42, 43} if profile.suite in {"libero_goal", "libero_object"} else {42}):
        raise ValueError("unsupported native training seed")
    validate_flat_config(flat, phase, training_seed=seed, action_velocity_weight=profile.action_velocity_weight)
    if config["callbacks"]["dit_half"]["_target_"] != "egomimic.utils.libero_dit_half.LiberoDiTHalf":
        raise ValueError("DiT-half callback unreachable")
    if config["callbacks"]["ema"]["validate_with_ema"] is not True:
        raise ValueError("native EMA validation required")
    for mode in ("train", "valid"):
        dataset = config["data"][mode+"_datasets"][SOURCE]
        if dataset["_target_"] != "egomimic.rldb.zarr.libero_action_flow.LiberoActionFlowDataset._from_resolver":
            raise ValueError("native dataset target changed")
        if dataset["valid_ratio"] != .01 or dataset["split_seed"] != 42:
            raise ValueError("native episode split changed")
        if dataset["mode"] != mode:
            raise ValueError("split mode changed")
    if config["normalizer"]["_target_"] != "egomimic.rldb.zarr.libero_action_flow.LiberoActionFlowNormalizer":
        raise ValueError("must use train-only Action Flow normalizer, never base OAT all-frame limits")
    return True

def validate_data_receipt(receipt, *, replay_path, expected_logical_sha, profile=PROFILE):
    """Cheap immutable cached identity; cache miss must run real native validator."""
    selected = profile_for_experiment(profile)
    if receipt["suite"] != selected.suite or receipt["replay_path"] != str(replay_path):
        raise ValueError("receipt suite/path mismatch")
    if receipt["dataset_logical_sha256"] != expected_logical_sha:
        raise ValueError("logical replay identity mismatch")
    if receipt["episodes"] != 500 or receipt["valid_ratio"] != .01 or receipt["split_seed"] != 42:
        raise ValueError("corpus/split contract mismatch")
    train, valid = receipt["train_episode_indices"], receipt["valid_episode_indices"]
    if len(train) != 495 or len(valid) != 5 or len(set(train)) != 495 or len(set(valid)) != 5:
        raise ValueError("duplicate or incorrect episode counts")
    if set(train)&set(valid) or set(train)|set(valid) != set(range(500)):
        raise ValueError("episode overlap/incomplete corpus")
    if "task_uids" in receipt and tuple(receipt["task_uids"]) != selected.task_uids:
        raise ValueError("native receipt task inventory mismatch")
    if receipt["action_shape"] != [receipt["frames"], 7]:
        raise ValueError("native delta-OSC action shape mismatch")
    # These aren't distinct directories: each virtual episode path is rooted
    # in the same real Zarr. Exact indices create non-overlapping frame ranges.
    return {"train_episodes": 495, "valid_episodes": 5, "id_overlap": 0,
            "physical_frame_ranges_require_native_validation": True}

def require_finite_native_metrics(rows):
    for prefix in ("Train", "Valid"):
        row = rows.get(prefix, {})
        wanted = ("normalized_reconst_mse",) if prefix == "Train" else METRICS
        for key in wanted:
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(("missing/nonfinite native metric", prefix, key))
    return True

def construct_native_pipeline(config):
    scheduled_only()
    from hydra.utils import instantiate
    pipeline = instantiate(config.model.pipeline)
    count = sum(p.numel() for p in pipeline.nets.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(("full H240 native constructor count changed", count))
    return pipeline

def strict_native_checkpoint(config, checkpoint_path):
    scheduled_only()
    import torch
    from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
    pipeline = construct_native_pipeline(config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    def finite(value):
        if torch.is_tensor(value):
            return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value, dict): return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    for key in ("state_dict", "ema_state_dict", "optimizer_states", "lr_schedulers", "loops"):
        if not payload.get(key) or not finite(payload[key]):
            raise ValueError(("missing/nonfinite native checkpoint state", key))
    if payload["global_step"] < 2:
        raise ValueError("real optimizer smoke absent")
    strict_load_pipeline_checkpoint(pipeline, payload, use_ema=False)
    strict_load_pipeline_checkpoint(pipeline, payload, use_ema=True)
    return {"checkpoint_sha256": digest(checkpoint_path),
            "global_step": payload["global_step"], "total_parameters": EXPECTED_PARAMETER_COUNT}
