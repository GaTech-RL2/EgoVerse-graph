"""Opt-in native subtask/world-model admission; historical requests stay exact.

The generic training service still owns execution. This module validates the
additional immutable data, conditional expert and hardware contracts only.
"""
import copy
import math
from collections import Counter
from pathlib import Path

KIND = "native-subtask-world-model-v1"
STREAMS = ["image", "proprio", "action", "camera"]
POLICY = dict(implementation="native_subtask_joint", attention_backend="flash4",
              token_execution="packed_valid", compile_mode="default",
              compile_dynamic=True, compile_backend="inductor",
              recompile_limit=64, accumulated_recompile_limit=256)


def selected(request):
    kind = request.get("kind")
    if kind is not None and kind != KIND:
        raise ValueError("Unknown optional joint training request kind")
    return kind == KIND


def _need(value, message):
    if not value:
        raise ValueError(message)


def verify_model_execution(request, cfg):
    from scripts.train.joint.lr_policy import from_config, validate
    _need(selected(request), "Native subtask request kind required")
    policy = request.get("model_execution")
    _need(policy == POLICY, "Production requires exact Flash4/packed/compiled policy")
    stages, ids = cfg["model"]["pipeline"]["stages"], cfg["model"]["pipeline"]["stage_ids"]
    net = stages[ids["joint_denoiser"]]["model"]
    core = net["image_model"]
    interfaces = request["interfaces"]
    _need(net["_target_"] == "egomimic.models.joint_image_proprio.JointImageProprioModel"
          and core["_target_"] == "egomimic.models.image_proprio.ImageProprioModel",
          "Native multimodal constructor required")
    _need(request["active_streams"] == net["active_streams"] == STREAMS
          and net.get("separate_proprio_experts") is True,
          "Private embodiment proprio banks and camera stream required")
    for name in ("pose_dims", "action_dims", "camera_dims"):
        _need(net[name] == interfaces[name] == cfg["joint"][name], "Changed native interfaces: " + name)
    embodiments = set(interfaces["pose_dims"])
    _need(len(embodiments) >= 2 and set(interfaces["camera_dims"]) == embodiments
          and set(interfaces["action_dims"]) == set(cfg["data"]["action_embodiments"])
          and set(interfaces["action_dims"]) < embodiments,
          "Human/robot interfaces must not fabricate missing human commands")
    _need(net["camera_max_count"] == core["max_cameras"] == interfaces["max_cameras"]
          and net["language_feature_dim"] == interfaces["language_feature_dim"] > 0
          and core.get("require_observed_context") is False,
          "Camera identity/language/per-frame noise contract changed")
    for key in ("attention_backend", "token_execution", "compile_mode", "compile_dynamic", "compile_backend"):
        _need(core[key] == policy[key], "Changed effective execution: " + key)
    for key in ("recompile_limit", "accumulated_recompile_limit"):
        _need(core["compile_" + key] == policy[key], "Changed bounded compilation budget")
    from scripts.train.joint import boundary_uncertainty
    boundary_uncertainty.request_config(request,cfg)
    data = cfg["data"]
    _need(data["training_conditioning"] == "independent_frame_noise"
          and data["camera_pose"] is True and data["task"] == "visual_state_action"
          and data["image_only"] is False and data["couple_state_observations"] is True,
          "Subtask multimodal training/noising path changed")
    _need(data["validation_anchors_per_episode"] is None
          and data["validation_batches_per_embodiment"] == 50
          and data["validation_batch_size"] == 1
          and cfg["trainer"]["limit_val_batches"] == (100 // request["world_size"] if request["phase"] == "smoke" else 1.0),
          "Balanced100 held-out validation must include50 robot and50 human batches")
    _need(cfg["trainer"]["strategy"] == "ddp_find_unused_parameters_true"
          and cfg["trainer"]["devices"] == request["world_size"] in (2, 4)
          and data["distributed_world_size"] == request["world_size"]
          and data["batch_size"] == request["per_rank_batch"]
          and data["batch_size"] * request["world_size"] == request["global_batch"] == 32,
          "Conditional DDP/global32 contract changed")
    if request["arm"] == "discrete":
        _need(data["mode"] == "discrete" and data["discrete_frames"] == 16 and data["stride"] == 4,
              "Discrete production must use native16/stride4")
    else:
        _need(request["arm"] == data["mode"] == "continuous" and data["min_frames"] == 2
              and data["max_frames"] == 64 and data["sampling_strategy"] == "full_episode_uniform",
              "Continuous production must use native N2..64")
    for name, field in (("subtask_manifest", "subtask_manifest"), ("language_features", "language_features")):
        binding = request["subtask_contract"][name]
        _need(data[field + "_path"] == binding["path"]
              and data[field + "_sha256"] == binding["sha256"], "Unbound " + name)
    _need(from_config(cfg["model"]) == validate(request["lr_policy"]), "Unbound optimizer policy")
    _need(cfg["evaluator"].get("active_streams_from_data") is True,
          "Native evaluator must respect actual per-embodiment streams")
    if request["phase"] == "full":
        _need(any(v.get("_target_") == "egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage"
                  for v in cfg["callbacks"].values() if isinstance(v, dict)),
              "Production conditional optimizer usage callback required")
    if request["phase"] == "smoke":
        callback = cfg["callbacks"]["joint_smoke"]
        _need(callback["_target_"] == "egomimic.pl_utils.joint_smoke.JointSmokeProof"
              and callback.get("multimodal") is True
              and type(cfg["trainer"]["limit_val_batches"]) is int
              and cfg["trainer"]["limit_val_batches"] == 100 // request["world_size"],
              "Missing actual multimodal smoke instrumentation/100 global validation batches")
    return "visual_state_action"


def verify_dataset(request):
    from scripts.train.joint.contract import bound, sha
    policy = request["dataset_contract"]
    _need(policy.get("schema") == "native-subtask-dataset-v1", "Explicit native subtask dataset contract required")
    manifest = bound(dict(path=request["data_manifest"], sha256=request["data_manifest_sha256"]))
    _need(manifest["schema"] == "pixel-proprio-prepared-v1", "Wrong native pixel manifest")
    norm = manifest["normalization"]
    _need(norm["scope"] == "train_only" and norm["sha256"] == policy["normalization_sha256"]
          and sha(Path(request["data_manifest"]).parent / norm["path"]) == norm["sha256"],
          "Train-only normalization changed")
    contract = request["subtask_contract"]
    selection, index = bound(contract["selection"]), bound(contract["subtask_manifest"])
    _need(manifest.get("preparation", {}).get("selection_sha256") == contract["selection"]["sha256"],
          "Prepared corpus must bind the exact selected original Scale episodes")
    language = bound(contract["language_features"])
    _need(language.get("schema") == "pixel-task-language-features-v1"
          and language.get("prepared_manifest_sha256") == request["data_manifest_sha256"]
          and language.get("subtask_index_sha256") == contract["subtask_manifest"]["sha256"],
          "Only final prepared/index-bound language features admit data")
    expected = {(e["embodiment"], e["episode_id"]): e for e in selection["episodes"]}
    physical = {(e["embodiment"], e["episode_id"]): e for e in manifest["episodes"]}
    _need(len(expected) == len(selection["episodes"]) == policy["episodes"]
          and len(physical) == len(manifest["episodes"]) and set(expected) == set(physical),
          "Physical corpus coverage/identity changed")
    counts = dict(Counter(e["split"] for e in manifest["episodes"]))
    _need(counts == policy["splits"] and set(counts) == {"train", "val"}
          and min(counts.values()) > 0, "Physical train/validation split changed")
    _need(index["schema"] == "pixel-subtask-index-v1"
          and index["prepared_manifest_sha256"] == request["data_manifest_sha256"]
          and index["interval_semantics"] == "half_open"
          and index["concurrent_policy"] == "combine_active_hands_atomic_intervals",
          "Subtask clock/concurrency contract changed")
    seen = set()
    for record in index["episodes"]:
        key = record["embodiment"], record["episode_id"]
        _need(key in expected and key not in seen, "Unknown/duplicate subtask episode")
        seen.add(key)
        source, ep = expected[key], physical[key]
        _need(record["split"] == ep["split"] == source["split"]
              and record["frames"] == ep["frames"] == source["frames"],
              "Subtask expansion changed physical split or native length")
        previous = 0
        _need(bool(record["subtasks"]), "Selected episode has no admitted subtasks")
        for unit in record["subtasks"]:
            lo, hi = unit["start_idx"], unit["end_idx"]
            _need(type(lo) is int and type(hi) is int and previous <= lo < hi <= ep["frames"],
                  "Subtask must be disjoint, half-open and in-bounds")
            _need(any(a <= lo < hi <= b for a, b in ep.get("continuity_segments", [[0, ep["frames"]]])),
                  "Subtask crosses a native recording discontinuity")
            previous = hi
            _need(unit.get("quality_and_names_verified") is True
                  and unit.get("source_clock_alignment_verified") is True
                  and unit.get("instruction", "").strip(), "Unreviewed names/source clock")
            evidence = unit["annotation_provenance"]["source"]
            _need(evidence.get("annotation_source_kind") == "scale_existing"
                  and evidence.get("task_id") == source["annotation_task_id"]
                  and evidence.get("raw_sha256") == source["annotation_raw_sha256"]
                  and unit["annotation_provenance"]["events_sha256"] == index["annotation_events_sha256"]
                  and unit["annotation_provenance"].get("event_ids"),
                  "Only exact original selected Scale annotations are admitted; no generated labels")
    _need(seen == set(expected), "Subtask index does not cover physical corpus")
    tasks = {u["task_id"]: u["instruction"] for e in index["episodes"] for u in e["subtasks"]}
    _need(set(language.get("tasks", {})) == set(tasks)
          and all(language["tasks"][k]["instruction"] == text for k, text in tasks.items()),
          "Language vectors must exactly cover final subtask instructions")
    batches = policy["validation_batches"]
    _need(type(batches) is int and batches == 100
          and policy["validation_batches_per_embodiment"] == 50,
          "Need exactly50 robot plus50 human held-out global batches")
    from scripts.train.joint import boundary_uncertainty
    boundary_uncertainty.eligibility(request,manifest,index,bound)
    return counts, batches


def _rank_capability(row):
    capability = row.get("gpu_capability", row.get("compute_capability"))
    _need(isinstance(capability, (list, tuple)) and len(capability) == 2
          and all(type(value) is int and value >= 0 for value in capability),
          "Missing typed native GPU capability")
    if "gpu_capability" in row and "compute_capability" in row:
        _need(row["gpu_capability"] == row["compute_capability"], "Conflicting GPU capability aliases")
    return capability


def verify_completion(request, smoke, completion, preflight, capacity):
    from scripts.train.joint.contract import bound, canonical_uuid
    from scripts.train.joint.lr_policy import update_lr
    for key in ("kind", "interfaces", "subtask_contract", "dataset_contract", "boundary_sampling_contract"):
        _need(request.get(key) == smoke.get(key), "Production differs from multimodal smoke: " + key)
    _need(completion.get("schema") == "joint-multimodal-smoke-v1"
          and completion.get("status") == "PASS"
          and completion["job_id"] == str(request["smoke"]["job_id"])
          and completion["global_step"] == 50 and completion["validation_batches"] == 100
          and completion["world_size"] == request["world_size"], "Missing actual production topology smoke")
    for key in ("strict_full_wrapper_reload", "optimizer_sampler_all_rank_rng_verified",
                "lr_scheduler_strictly_restored", "all_rank_weights_agree", "sample_finite",
                "private_bank_optimizer_steps_verified", "human_action_absent",
                "language_conditioning_verified", "camera_targets_verified"):
        _need(completion.get(key) is True, "Missing actual multimodal proof: " + key)
    from scripts.train.joint import boundary_uncertainty
    boundary_uncertainty.completion(request,smoke,completion,preflight)
    _need(completion["active_streams"] == request["active_streams"]
          and completion["expert_parameters"] == request["expert_parameters"], "Smoke expert identity changed")
    ranks = completion["ranks"]
    _need(len(ranks) == request["world_size"]
          and len({canonical_uuid(r["gpu_uuid"]) for r in ranks}) == len(ranks), "Distinct owned GPU ranks required")
    for rank, row in enumerate(ranks):
        _need(row["rank"] == rank and row["world_size"] == len(ranks)
              and row["global_step"] == row["bf16_training_outputs"] == row["fp32_losses"] == 50
              and row.get("fp32_gradients_verified") is True
              and "H100" in row["gpu_name"] and _rank_capability(row)[0] >= 9,
              "Actual H100/BF16/FP32 smoke rank proof missing")
        _need(row["metrics"] and all(math.isfinite(float(v)) for v in row["metrics"].values()),
              "Missing finite native held-out metrics")
        _need(len(row["optimizer_update_lrs"]) == 50
              and all(math.isclose(lr, update_lr(request["lr_policy"], i), rel_tol=1e-12)
                      for i, lr in enumerate(row["optimizer_update_lrs"])),
              "Actual optimizer warmup differs from bound policy")
    _need(capacity.get("schema") == "native-multimodal-capacity-v1"
          and capacity.get("status") == "PASS"
          and capacity["config_sha256"] == request["smoke"]["resolved"]["sha256"]
          and capacity["source_manifest_sha256"] == request["source_manifest_sha256"]
          and capacity["data_manifest_sha256"] == request["data_manifest_sha256"]
          and capacity["per_rank_batch"] == request["per_rank_batch"]
          and capacity["world_size"] == request["world_size"]
          and capacity["frames"] == (16 if request["arm"] == "discrete" else 64) and capacity["cameras"] == request["interfaces"]["max_cameras"]
          and capacity["image_size"] == [160, 160]
          and capacity["attention_backend"] == "flash4" and capacity["token_execution"] == "packed_valid"
          and capacity.get("compiled_bf16_forward_backward") is True
          and capacity["peak_memory_bytes"] < capacity["device_memory_bytes"],
          "Actual maximum-shape production memory/backend proof required")
    verify_dataset(request)
    return completion, preflight


def compare_configs(production, smoke, request, legacy):
    from scripts.train.joint import boundary_uncertainty
    boundary_uncertainty.request_config(request,production)
    boundary_uncertainty.request_config(request,smoke)
    _need(production["trainer"].get("limit_val_batches", 1.0) == 1.0
          and type(smoke["trainer"]["limit_val_batches"]) is int
          and smoke["trainer"]["limit_val_batches"] == 100 // request["world_size"],
          "Only smoke diagnostic validation may be capped")
    adjusted = copy.deepcopy(smoke)
    adjusted["trainer"]["limit_val_batches"] = production["trainer"].get("limit_val_batches", 1.0)
    if "limit_val_batches" not in production["trainer"]:
        adjusted["trainer"].pop("limit_val_batches")
    legacy(production, adjusted, request)
    for field in ("world_model_execution", "pixel_video"):
        _need(production.get(field) == smoke.get(field), "Scientific execution config changed: " + field)
