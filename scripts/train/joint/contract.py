"""Exact smoke-to-production admission. Standard-library evidence checks are read-only."""

import copy
import hashlib
import json
import math
import uuid
from scripts.train.joint.lr_policy import validate, update_lr
from pathlib import Path
from datetime import datetime


def need(condition, message):
    if not condition:
        raise ValueError(message)



def verify_model_execution(request, cfg):
    """Bind native historical image/state execution without changing legacy admission."""
    from scripts.train.joint import imagenet, native_subtasks
    if imagenet.selected(request):
        return imagenet.verify_model_execution(request, cfg)
    if native_subtasks.selected(request):
        return native_subtasks.verify_model_execution(request, cfg)
    stages = cfg["model"]["pipeline"]["stages"]
    ids = cfg["model"]["pipeline"]["stage_ids"]
    model = stages[ids["joint_denoiser"]]["model"]
    native = model["_target_"] == "egomimic.models.joint_image_proprio.JointImageProprioModel"
    policy = request.get("model_execution")
    if not native:
        need(policy is None, "Native execution policy requires the native image wrapper")
        return "visual_state_action"
    track = isinstance(policy, dict) and policy.get("implementation") == "native_track_joint"
    primary = "track" if track else "image"
    need(policy == {"implementation": "native_track_joint" if track else "native_image_joint", "compile_mode": "none" if track else "default",
                    "compile_dynamic": True, "compile_backend": "inductor",
                    **({"activation_checkpointing": True} if track else {})},
         "Native image execution requires an explicit historical request binding")
    image = model["image_model"]
    need(image.get("_target_") == ("egomimic.models.track_proprio.TrackTokenCore" if track else "egomimic.models.image_proprio.ImageProprioModel")
         and image["compile_mode"] == policy["compile_mode"]
         and image["compile_dynamic"] == policy["compile_dynamic"]
         and image["compile_backend"] == policy["compile_backend"],
         "Native image compilation differs from bound historical policy")
    independent = cfg["data"].get("training_conditioning", "mixed_context") == "independent_frame_noise"
    need(image.get("require_observed_context", True) is (not independent),
         "Independent noise and clean-context model guard disagree")
    streams = request["active_streams"]
    action = streams == [primary, "proprio", "action"]
    need(streams in ((["track", "proprio", "action"],) if track else (["image"], ["image", "proprio"], ["image", "proprio", "action"]))
         and model["active_streams"] == streams,
         "Native request has unbound streams")
    need([stage["stream"] for stage in stages if "stream" in stage] == streams,
         "Native loss graph differs from active experts")
    if track:
        need(cfg["data"].get("spatial_stream") == "track"
             and image["image_dim"] == 3 and image["max_points"] == 1120
             and image["spatial_grid_size"] == [28, 40]
             and image["use_camera_embedding"] is False
             and image.get("activation_checkpointing") is True
             and cfg["evaluator"].get("spatial_stream") == "track"
             and cfg["evaluator"].get("pixel_diagnostics") is None
             and cfg["evaluator"].get("prediction_key") == "track_proprio_action_prediction",
             "XYZ grid/no-RGB contract changed")
    if not action:
        need(request.get("action_contract") is None
             and not any("action" in name for name in ids), "Action stage in native state graph")
    need(stages[ids["joint_noising"]]["noising"]["_target_"]
         == "egomimic.models.joint_image_proprio.CoupledImageProprioNoising",
         "Native image corruption path changed")
    need(stages[ids["joint_noising"]]["noising"].get("active_streams") == streams,
         "Noising must exclude inactive vector draws")
    need(cfg["data"].get("image_only") == (streams == ["image"]),
         "Image-only data must use the historical image-only path")
    sampler = stages[ids["joint_sampler"]]["sampler"]
    need(sampler["_target_"] == "egomimic.models.joint_image_proprio.JointImageProprioSampler"
         and sampler["state_steps"] == 20
         and (sampler.get("action_steps") == 20 if action else "action_steps" not in sampler),
         "Native sampling path changed")
    if action:
        contract = request.get("action_contract")
        need(isinstance(contract, dict) and contract.get("schema") in (
                 "native-joint-command-v1", "native-joint-command-v2"),
             "Missing explicit native action contract")
        native, encoded = contract.get("native_dimensions"), contract.get("encoded_dimensions")
        mode = cfg["data"]["mode"]
        if contract["schema"] == "native-joint-command-v1":
            need(mode == "discrete", "Historical command v1 is discrete-only")
        else:
            need(mode in ("discrete", "continuous")
                 and contract.get("temporal_mode") == mode,
                 "Command temporal mode is not explicitly bound")
        group = cfg["data"]["stride"] if mode == "discrete" else 1
        need(isinstance(native, dict) and native and all(type(v) is int and v > 0 for v in native.values())
             and type(group) is int and group > 0
             and contract.get("group_size") == group
             and encoded == {k: v * group for k, v in native.items()}
             and model.get("action_dims") == encoded
             and model.get("action_width") == contract.get("hidden_width") == 256
             and contract.get("noise") == "independent_command"
             and contract.get("unknown_past_queries") == "pure_noise_no_loss",
             "Action dimensions/window/noising contract changed")
        need(cfg["data"].get("couple_state_observations") is True
             and cfg["data"]["task"] == "visual_state_action"
             and cfg["evaluator"]["objective"] == "visual_state_action"
             and all(s.get("weight") == 1.0 for s in stages if "stream" in s),
             "Three-stream data/objective/loss contract changed")
        return "visual_state_action"
    need(cfg["data"]["task"] == "visual_diffusion", "Native state data must exclude action streams")
    return "visual_diffusion"


def verify_capacity_profile(request, cfg):
    """Admit the explicit wider image expert without weakening legacy counts."""
    from scripts.train.joint import imagenet, native_subtasks
    if imagenet.selected(request):
        return request["expert_parameters"]["image"]
    if native_subtasks.selected(request):
        return request["expert_parameters"]["image"]
    profile = request.get("capacity_profile")
    if (request.get("model_execution") or {}).get("implementation") == "native_track_joint":
        need(isinstance(profile, dict) and profile.get("schema") == "native-track-capacity-v1",
             "Track constructor count binding required")
        model = cfg["model"]["pipeline"]["stages"][cfg["model"]["pipeline"]["stage_ids"]["joint_denoiser"]]["model"]
        core = model["image_model"]
        counts = profile.get("expert_parameters", {})
        need(set(counts) == {"track", "proprio", "action"}
             and all(type(v) is int and v > 0 for v in counts.values())
             and counts == request["expert_parameters"]
             and sum(counts.values()) == profile["parameters"] == request["parameters"]
             and request["expert_widths"] == [768, 256, 256]
             and (core["width"], core["heads"], core["depth"], core.get("mlp_ratio", 4)) == (768, 6, 8, 4)
             and core["image_dim"] == 3 and core["max_points"] == 1120
             and core["rope_axis_dims"] == [64, 32, 32]
             and model["proprio_width"] == model["action_width"] == 256,
             "Track capacity differs from bound constructor")
        return counts["track"]
    if profile is None:
        return 88178688
    need(isinstance(profile, dict)
         and set(profile) in ({"schema", "image_width", "image_heads", "image_depth",
                               "parameters", "expert_parameters"},
                              {"schema", "image_width", "image_heads", "image_depth",
                               "image_mlp_ratio", "parameters", "expert_parameters"})
         and profile["schema"] == "native-image-capacity-v1"
         and all(type(profile[k]) is int and profile[k] > 0 for k in (
             "image_width", "image_heads", "image_depth", "parameters"))
         and profile["image_width"] > 768 and profile["image_depth"] == 8
         and profile["image_width"] == 128 * profile["image_heads"]
         and type(profile.get("image_mlp_ratio", 4)) is int
         and profile.get("image_mlp_ratio", 4) >= 4
         and isinstance(profile["expert_parameters"], dict)
         and set(profile["expert_parameters"]) == {"image", "proprio", "action"}
         and all(type(v) is int and v > 0 for v in profile["expert_parameters"].values())
         and sum(profile["expert_parameters"].values()) == profile["parameters"],
         "Unknown native capacity profile")
    model = cfg["model"]["pipeline"]["stages"][
        cfg["model"]["pipeline"]["stage_ids"]["joint_denoiser"]]["model"]
    image = model["image_model"]
    need(cfg["data"]["mode"] == request["arm"] == "discrete"
         and request["active_streams"] == model["active_streams"] == ["image", "proprio", "action"]
         and request["expert_widths"] == [profile["image_width"], 256, 256]
         and request["parameters"] == profile["parameters"]
         and request["expert_parameters"] == profile["expert_parameters"]
         and (image["width"], image["heads"], image["depth"]) == (
             profile["image_width"], profile["image_heads"], profile["image_depth"])
         and image["image_dim"] == 768 and image["rope_axis_dims"] == [64, 32, 32]
         and image.get("mlp_ratio", 4) == profile.get("image_mlp_ratio", 4)
         and model["proprio_width"] == model["action_width"] == 256,
         "Capacity profile differs from actual resolved native model")
    return profile["expert_parameters"]["image"]


def verify_sampling_proof(request, completion):
    policy = validate(request.get("lr_policy"))
    native = request.get("model_execution") is not None
    expected_action = (20,) if "action" in request["active_streams"] else ((0,) if native else (0, 20))
    need(completion["sample_state_steps"] == 20
         and completion["sample_action_steps"] in expected_action
         and completion["lr_warmup_steps"] == policy["warmup_optimizer_steps"],
         "Wrong sampling/scheduler proof")
    if native:
        need(completion.get("model_execution") == request["model_execution"],
             "Missing native historical execution proof")

def canonical_uuid(value):
    """Compare complete CUDA/NVML identities without accepting GPU ordinals."""
    text = str(value)
    if text[:4].lower() == "gpu-":
        text = text[4:]
    return str(uuid.UUID(text))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def bound(binding, json_file=True):
    need(
        sha(binding["path"]) == binding["sha256"],
        "Changed evidence: " + binding["path"],
    )
    content = Path(binding["path"]).read_text()
    return json.loads(content) if json_file else content


def verify_allocation_relocation(request, smoke, smoke_request_sha256):
    """Allow explicit account/storage relocation on the same admitted cluster.

    Runtime, topology and hardware policy remain exact; the destination still
    undergoes its own native allocation/UUID admission. Historical evidence is
    always validated against the original smoke owner.
    """
    old, new = smoke.get("allocation"), request.get("allocation")
    if old == new:
        need(request.get("allocation_relocation") is None,
             "Unexpected relocation for an unchanged allocation policy")
        return
    need(isinstance(old, dict) and isinstance(new, dict)
         and old.get("kind") == new.get("kind") == "native_slurm",
         "Relocation requires native Slurm on both sides")
    allowed = {"user", "uid", "account", "qos"}
    need({k: v for k, v in old.items() if k not in allowed}
         == {k: v for k, v in new.items() if k not in allowed}
         and old.get("cluster") and old.get("runtime_binding"),
         "Relocation changed cluster, runtime or hardware policy")
    need(request.get("allocation_relocation") == dict(
        schema="same-cluster-account-relocation-v1",
        smoke_request_sha256=smoke_request_sha256,
        original_allocation=old, destination_allocation=new),
        "Missing exact original/destination relocation binding")
    need(all(isinstance(new.get(k), str) and new[k] for k in ("user", "account", "qos"))
         and type(new.get("uid")) is int and new["uid"] > 0,
         "Invalid destination ownership")
    for key in ("source", "world_size", "active_streams", "parameters"):
        need(request.get(key) == smoke.get(key), "Recovery science changed: " + key)


def validate_submission_job_name(request, argv):
    """Fail before a new claim when a scheduler label differs from its request."""
    need(isinstance(argv, list) and all(isinstance(v, str) for v in argv),
         "Require structured submission argv")
    names = [v.split("=", 1)[1] for v in argv if v.startswith("--job-name=")]
    need(names == [request["job_name"]], "Submitted JobName differs from exact request")
    return names[0]


def smoke_submission_job_name(evidence, smoke):
    """Allow a recorded native-subtask retry name without changing consumed inputs."""
    attempt = evidence.get("submission_attempt")
    if attempt is None:
        return smoke["job_name"]
    from scripts.train.joint import native_subtasks
    need(native_subtasks.selected(smoke), "Retry-name evidence requires native subtask mode")
    need(isinstance(attempt, dict) and set(attempt) == {"plan", "accepted", "allocation"},
         "Incomplete bound submission-attempt evidence")
    plan, accepted, allocation = (bound(attempt[key]) for key in ("plan", "accepted", "allocation"))
    path, digest = evidence["request"]["path"], evidence["request"]["sha256"]
    need(plan.get("request_remote") == accepted.get("request") == path
         and plan.get("request_sha256") == accepted.get("request_sha256") == digest
         and str(accepted.get("job_id")) == str(evidence["job_id"]),
         "Retry submission does not bind the accepted job and exact request")
    argv = plan.get("argv")
    need(isinstance(argv, list) and all(isinstance(v, str) for v in argv)
         and argv and Path(argv[0]).name == "sbatch", "Missing exact submitted sbatch argv")
    names = [v.split("=", 1)[1] for v in argv if v.startswith("--job-name=")]
    need(len(names) == 1 and names[0] == Path(path).stem
         and names[0].startswith(smoke["job_name"] + "-"), "Unbound retry job name")
    name = names[0]
    raw = allocation.get("stdout", "")
    need(isinstance(raw, str), "Missing structured native allocation stdout")
    native = dict(v.split("=", 1) for v in raw.split() if "=" in v)
    owner = smoke.get("quota", {})
    need(allocation.get("rc") == 0 and native.get("JobId") == str(evidence["job_id"])
         and native.get("JobName") == name and native.get("JobState") == "RUNNING"
         and native.get("UserId") == f"{owner.get('user')}({owner.get('uid')})"
         and native.get("NodeList") == smoke.get("lambda_runtime", {}).get("node")
         and native.get("NumCPUs") == "32" and native.get("MinMemoryNode") == "192G"
         and f"GRES=gpu:h100:{smoke['world_size']}(IDX:" in raw,
         "Retry name lacks exact owned allocated CPU/memory/GPU evidence")
    return name


def native_smoke_state(evidence, smoke, request):
    """Read the declared native format; never manufacture a scontrol receipt.

    Accounting is an alternative for a purged Lambda job. Its original bound
    nine-field row remains the evidence; the returned fields are only an
    internal admission view. PACE still requires its association attestation.
    """
    raw = bound(evidence["native"], False)
    kind = evidence.get("native_format", "scontrol-v1")
    if kind == "scontrol-v1":
        return dict(s.split("=", 1) for s in raw.split() if "=" in s)
    need(kind == "sacct-pipe-v1", "Unsupported native completion format")
    need(smoke.get("allocation") is None and request.get("allocation") is None,
         "Accounting-only completion requires Lambda allocation evidence")
    rows = [line for line in raw.splitlines() if line.strip()]
    need(len(rows) == 1, "Exactly one native accounting row required")
    fields = rows[0].split("|")
    need(len(fields) == 9, "Native accounting requires nine fields")
    job, name, user, uid, state, exit_code, node, start, end = fields
    owner = smoke.get("quota", {})
    need(type(owner.get("uid")) is int and owner["uid"] > 0
         and isinstance(owner.get("user"), str) and bool(owner["user"])
         and request.get("quota", {}).get("uid") == owner["uid"]
         and request.get("quota", {}).get("user") == owner["user"],
         "Missing matching native accounting owner binding")
    need(job == str(evidence["job_id"]) and name == smoke_submission_job_name(evidence, smoke)
         and user == owner["user"] and uid == str(owner["uid"])
         and state == "COMPLETED" and exit_code == "0:0"
         and node == smoke.get("lambda_runtime", {}).get("node")
         == request.get("lambda_runtime", {}).get("node"),
         "Wrong native accounting identity or completion")
    try:
        started, ended = datetime.fromisoformat(start), datetime.fromisoformat(end)
        need(ended > started, "Native accounting has no allocated runtime")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid native accounting allocation times") from error
    return dict(JobId=job, JobName=name, UserId=f"{user}({uid})",
                JobState=state, ExitCode=exit_code, NodeList=node)


def verify_evidence(request):
    need(
        request["schema"] == "joint-training-v2" and request["phase"] == "full",
        "Wrong request schema",
    )
    evidence = request["smoke"]
    from scripts.train.joint import imagenet
    if imagenet.selected(request):
        imagenet.precheck_evidence(evidence)
    smoke = bound(evidence["request"])
    need(
        smoke["schema"] == "joint-training-v2" and smoke["phase"] == "smoke",
        "Wrong smoke request",
    )
    completion = bound(evidence["completion"])
    preflight = bound(evidence["preflight"])
    capacity = bound(evidence["capacity"])
    native = native_smoke_state(evidence, smoke, request)
    need(bound(evidence["host_exit"], False).strip() == "0", "Smoke host failed")
    need(
        native["JobId"] == str(evidence["job_id"])
        and native["JobState"] == "COMPLETED"
        and native["ExitCode"] == "0:0"
        and native["UserId"].startswith((smoke.get("allocation") or {}).get("user", "paphiwetsa3")+"(")
        and native["JobName"] == smoke_submission_job_name(evidence, smoke),
        "Wrong native smoke proof",
    )
    if smoke.get("allocation") is None:
        need(
            request.get("allocation") is None
            and smoke.get("lambda_runtime") == request.get("lambda_runtime")
            and native["NodeList"] == request.get("lambda_runtime", {}).get("node", "gpu21"),
            "Wrong Lambda smoke allocation",
        )
    else:
        from scripts.train.joint.allocation import verify_pace

        verify_allocation_relocation(request, smoke, evidence["request"]["sha256"])
        allocation = bound(evidence["allocation"])
        reference = preflight.get("allocation_attestation")
        need(
            reference and reference["sha256"] == evidence["allocation"]["sha256"],
            "Unbound smoke allocation attestation",
        )
        verify_pace(
            smoke,
            allocation,
            dict(
                SLURM_JOB_ID=str(evidence["job_id"]),
                SLURMD_NODENAME=native["NodeList"],
                SLURM_RESTART_COUNT="0",
            ),
        )
        need(
            native["Account"] == smoke["allocation"]["account"]
            and native["QOS"] == smoke["allocation"]["qos"]
            and native["UserId"]
            == f"{smoke['allocation']['user']}({smoke['allocation']['uid']})",
            "Native smoke association differs from ownership proof",
        )
    for key in (
        "arm",
        "source",
        "source_revision",
        "source_manifest",
        "source_manifest_sha256",
        "source_archive",
        "source_archive_sha256",
        "data_manifest",
        "data_manifest_sha256",
        "data_cache",
        "parameters",
        "expert_parameters",
        "expert_widths",
        "world_size",
        "global_batch",
        "per_rank_batch",
        "lr_policy",
        "vector_expert_parameter_budget",
        "active_streams",
        "experiment",
    ):
        need(smoke[key] == request[key], "Production differs from smoke: " + key)
    need(smoke.get("model_execution") == request.get("model_execution"),
         "Production model execution differs from smoke")
    need(
        preflight["status"] == "PREFLIGHT_PASS"
        and preflight["request_sha256"] == evidence["request"]["sha256"],
        "Unbound smoke preflight",
    )
    need(
        preflight["parameters"] == request["parameters"]
        and preflight["expert_parameters"] == request["expert_parameters"]
        and preflight["expert_widths"] == request["expert_widths"]
        and preflight["world_size"] == request["world_size"]
        and preflight["global_batch"] == 32,
        "Smoke preflight model/topology changed",
    )
    for key in ("source_revision", "source_manifest_sha256", "data_manifest_sha256"):
        need(preflight["provenance"][key] == request[key], "Unbound provenance: " + key)
    from scripts.train.joint import imagenet, native_subtasks
    if imagenet.selected(request):
        return imagenet.verify_completion(request, smoke, completion, preflight, capacity)
    if native_subtasks.selected(request):
        return native_subtasks.verify_completion(request, smoke, completion, preflight, capacity)
    need(
        completion["schema"] == "joint-mot-smoke-v1"
        and completion["status"] == "PASS"
        and completion["job_id"] == str(evidence["job_id"])
        and completion["world_size"] == request["world_size"]
        and completion["global_step"] == 50
        and completion["validation_batches"] == 100,
        "Incomplete smoke completion",
    )
    for key in (
        "strict_full_wrapper_reload",
        "optimizer_sampler_all_rank_rng_verified",
        "lr_scheduler_strictly_restored",
        "all_rank_weights_agree",
        "sample_finite",
    ):
        need(completion[key] is True, "Missing strict proof: " + key)
    verify_sampling_proof(request, completion)
    world = request["world_size"]
    metric_names = {
        f"val/libero_gripper/{kind}/{metric}"
        for kind in ("forecast", "completion")
        for metric in (
            {
                "image": "image_normalized_mse",
                "track": "track_normalized_mse",
                "proprio": "proprio_normalized_mse",
                "action": "action_command_mse",
            }[name]
            for name in request["active_streams"]
        )
    }
    need(
        completion["active_streams"] == request["active_streams"]
        and completion["expert_parameters"] == request["expert_parameters"],
        "Wrong active-stream completion",
    )
    ranks = completion["ranks"]
    need(
        len(ranks) == world and len({canonical_uuid(r["gpu_uuid"]) for r in ranks}) == world,
        "Wrong GPU topology",
    )
    need(len({r["weights_sha256"] for r in ranks}) == 1, "Ranks diverged")
    if smoke.get("allocation") is not None:
        need(
            [canonical_uuid(r["gpu_uuid"]) for r in ranks]
            == [canonical_uuid(r["uuid"]) for r in allocation["gpu_probe"]["ranks"]],
            "Training ranks differ from admitted physical GPUs",
        )
    for rank, record in enumerate(ranks):
        need(
            record["status"] == "FIT_PASS"
            and record["rank"] == rank
            and record["world_size"] == world
            and record["job_id"] == str(evidence["job_id"])
            and record["global_step"] == 50
            and record["fp32_losses"] == record["bf16_training_outputs"] == 50
            and record["fp32_gradients_verified"]
            and record["validation_batches"] == 100
            and record["local_validation_batches"] == 100 // world
            and record["data_context_sha256"]
            == preflight["provenance"]["data_context_sha256"],
            "Missing rank optimization/validation proof",
        )
        need(
            record["active_streams"] == request["active_streams"]
            and set(record["metrics"]) == metric_names
            and len(record["paired_track_batch_sha256" if request["active_streams"][0] == "track" else "paired_rgb_batch_sha256"]) == 3,
            "Wrong active metrics or missing paired RGB proof",
        )
        need(
            len(set(record["initial_weights"]))
            == len(set(record["all_rank_weights"]))
            == 1
            and len(record["initial_weights"])
            == len(record["all_rank_weights"])
            == world
            and record["all_rank_weights"][0] == record["weights_sha256"],
            "DDP agreement proof changed",
        )
        need(
            len(record["optimizer_update_lrs"]) == 50
            and all(
                math.isclose(lr, update_lr(request["lr_policy"], i), rel_tol=1e-12)
                for i, lr in enumerate(record["optimizer_update_lrs"])
            ),
            "Wrong actual LR warmup",
        )
        need(
            record["measured_updates"] == 40
            and record["warm_wall_seconds"] > 0
            and len(record["metrics"]) == 2 * len(request["active_streams"])
            and all(math.isfinite(v) for v in record["metrics"].values()),
            "Invalid timing/validation metrics",
        )
    need(
        capacity["status"] == "CAPACITY_PASS"
        and capacity["active_streams"] == request["active_streams"]
        and capacity["sample_all_streams_finite"]
        and capacity["frames_per_example"]
        == (64 if request["arm"] == "continuous" else 16)
        and capacity["per_rank_batch"] == request["per_rank_batch"]
        and math.isclose(capacity["optimizer_update_lr"], update_lr(request["lr_policy"], 0), rel_tol=1e-12),
        "Missing maximum-shape capacity proof",
    )
    bound(evidence["resolved"], False)
    return completion, preflight


def _compare_configs_legacy(production, smoke, request):
    """Allow only the declared budget/cadence, smoke instrumentation, output and logging changes."""

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str):
            return value.replace(
                request["output"], smoke["paths"]["output_dir"]
            ).replace(request["local_logs"], smoke["logger"]["csv"]["save_dir"])
        return value

    production = relocate(production)
    for key in (
        "model",
        "data",
        "seed",
        "runtime",
        "world_model_attention",
        "joint",
        "norm_stats",
        "normalizer",
        "inference_config",
        "val_at_end",
        "mode",
        "ckpt_path",
        "modality_ablation",
        "track_model",
    ):
        need(production.get(key) == smoke.get(key), "Scientific config changed: " + key)
    full_trainer, smoke_trainer = (
        copy.deepcopy(production["trainer"]),
        copy.deepcopy(smoke["trainer"]),
    )
    for key in ("max_steps", "val_check_interval"):
        full_trainer.pop(key)
        smoke_trainer.pop(key)
    need(full_trainer == smoke_trainer, "Training topology or precision changed")
    full_eval, smoke_eval = (
        copy.deepcopy(production["evaluator"]),
        copy.deepcopy(smoke["evaluator"]),
    )
    for item in (full_eval, smoke_eval):
        if item.get("pixel_diagnostics") is not None:
            item["pixel_diagnostics"].pop("artifact_dir")
            item["pixel_diagnostics"].pop("log_wandb")
    need(full_eval == smoke_eval, "Evaluator behavior changed")
    full_cb, smoke_cb = (
        copy.deepcopy(production["callbacks"]),
        copy.deepcopy(smoke["callbacks"]),
    )
    need(
        "joint_smoke" not in full_cb and "joint_smoke" in smoke_cb,
        "Wrong smoke instrumentation",
    )
    smoke_cb.pop("joint_smoke")
    for item in (full_cb, smoke_cb):
        for key in ("dirpath", "every_n_train_steps", "save_last"):
            item["model_checkpoint"].pop(key)
    need(full_cb == smoke_cb, "Callbacks differ beyond smoke/retention changes")
    trainer, ckpt = production["trainer"], production["callbacks"]["model_checkpoint"]
    need(
        trainer["max_steps"] == request["target_steps"]
        and trainer["val_check_interval"] == 20000
        and trainer["check_val_every_n_epoch"] is None
        and trainer["num_sanity_val_steps"] == 0
        and trainer["devices"] == request["world_size"]
        and trainer["precision"] == "bf16-mixed"
        and trainer["accumulate_grad_batches"] == 1,
        "Wrong full training budget/topology",
    )
    need(
        ckpt["every_n_train_steps"] == 20000
        and ckpt["every_n_epochs"] == 0
        and ckpt["save_top_k"] == -1
        and ckpt["save_last"] == "link"
        and ckpt["save_on_train_epoch_end"] is False
        and "epoch" in ckpt["filename"]
        and "step" in ckpt["filename"]
        and not ckpt.get("train_time_interval"),
        "Wrong checkpoint retention",
    )
    need(
        production["logger"]["wandb"]["id"] == request["run_id"]
        and production["logger"]["wandb"]["resume"] == "never"
        and production["logger"]["wandb"]["project"] == "mot-world-model"
        and production["logger"]["wandb"]["entity"] == "rl2-group"
        and production["logger"]["wandb"]["log_model"] is False
        and set(production["logger"]) == {"wandb", "csv"},
        "Wrong fresh W&B identity/logging",
    )
    need(
        production["ckpt_path"] is None
        and production["data"]["batch_size"] == request["per_rank_batch"]
        and production["data"]["batch_size"] * trainer["devices"] == 32,
        "Fresh step0/globalbatch required",
    )


def verify_gap_extension(request):
    """Bind a fresh completion-gap trial to its immutable ungapped parent.

    Parent admission is cached evidence, not a request to reload its model.
    Only the two maintained data modules may change scientific source bytes.
    Native smoke-to-full verification remains mandatory for the new recipe.
    """
    independent = request.get("noise_comparison") is not None
    comparison = request["noise_comparison" if independent else "gap_comparison"]
    need(comparison.get("schema") == ("native-independent-frame-noise-extension-v1" if independent else "native-completion-gap-extension-v1")
         and comparison.get("scope") == ("all_training_frames" if independent else "completion_only"),
         "Sampling comparison requires explicit noise/gap scope")
    parent = bound(comparison["parent_request"])
    proof = bound(comparison["parent_admission"])
    need(proof.get("status") == "FULL_ADMISSION_PASS"
         and proof.get("request_sha256") == comparison["parent_request"]["sha256"]
         and proof.get("provenance", {}).get("source_revision") == parent["source_revision"],
         "Gap parent lacks its exact cached native admission")
    need(request["phase"] == parent["phase"] == "full"
         and request["arm"] == parent["arm"] == "continuous"
         and request["run_id"] != parent["run_id"]
         and request["source_parent"] == parent["source_revision"],
         "Gap comparison must be a fresh direct source child")
    topology_changed = any(request[k] != parent[k] for k in ("world_size", "per_rank_batch"))
    if topology_changed:
        need(comparison.get("topology_control") == "global_batch_matched_rank_rng_unpaired"
             and (parent["world_size"], parent["per_rank_batch"]) == (4, 8)
             and (request["world_size"], request["per_rank_batch"]) == (2, 16)
             and request["global_batch"] == parent["global_batch"] == 32,
             "Gap topology change requires explicit two-rank global32 control")
    else:
        need(comparison.get("topology_control") in (None, "global_batch_matched_rank_rng_unpaired"),
             "Unknown gap topology control")
    for key in ("data_manifest_sha256", "dataset_contract", "active_streams", "parameters",
                "expert_parameters", "expert_widths", "global_batch", "per_rank_batch",
                "world_size", "target_steps", "lr_policy", "model_execution", "action_contract"):
        if topology_changed and key in ("world_size", "per_rank_batch"):
            continue
        need(request[key] == parent[key], "Gap comparison changed control: " + key)
    old_cfg = bound(comparison["parent_resolved"], False)
    new_cfg = bound(request["smoke"]["resolved"], False)
    from omegaconf import OmegaConf
    old_cfg = OmegaConf.to_container(OmegaConf.create(old_cfg), resolve=True)
    new_cfg = OmegaConf.to_container(OmegaConf.create(new_cfg), resolve=True)
    # Hydra derives this operational directory from each run's own output.
    # Check both exact owners before normalizing it; all runtime semantics stay
    # under the strict comparison below.
    derived_outputs = (("runtime", "slurm_signal_checkpoint_dir", "checkpoints"),
                       ("inference_config", "output_path", "checkpoints/inference-config.yaml"))
    for section, key, suffix in derived_outputs:
        if key in old_cfg.get(section, {}) or key in new_cfg.get(section, {}):
            smoke_request = bound(request["smoke"]["request"])
            for cfg, owner in ((old_cfg, parent), (new_cfg, smoke_request)):
                need(cfg.get(section, {}).get(key) == str(Path(owner["output"])/suffix),
                     "Gap derived artifact path is outside its exact output")
                cfg[section][key] = "<output>/"+suffix
    if independent:
        def strip_opt_in_context_guard(value, *, new):
            count = 0
            if isinstance(value, dict):
                if "require_observed_context" in value:
                    need(value.get("_target_") == "egomimic.models.image_proprio.ImageProprioModel"
                         and value["require_observed_context"] is (not new),
                         "Unbound model context-guard change")
                    value.pop("require_observed_context")
                    count += 1
                for child in value.values():
                    count += strip_opt_in_context_guard(child, new=new)
            elif isinstance(value, list):
                for child in value:
                    count += strip_opt_in_context_guard(child, new=new)
            return count
        strip_opt_in_context_guard(old_cfg, new=False)
        need(strip_opt_in_context_guard(new_cfg, new=True) > 0,
             "Independent-noise model did not explicitly allow all-noised context")
    for key in ("model", "seed", "runtime", "world_model_attention", "joint",
                "norm_stats", "normalizer", "inference_config", "modality_ablation"):
        need(old_cfg.get(key) == new_cfg.get(key), "Gap changed scientific config: " + key)
    old_data, new_data = copy.deepcopy(old_cfg["data"]), copy.deepcopy(new_cfg["data"])
    if topology_changed:
        for reference, data in ((parent, old_data), (request, new_data)):
            need(data.pop("batch_size") == reference["per_rank_batch"]
                 and data.pop("distributed_world_size") == reference["world_size"],
                 "Gap resolved topology differs from request")
    previous_schedule = old_data.pop("completion_gap_schedule", None)
    if independent:
        need(previous_schedule == comparison.get("parent_schedule", comparison["schedule"])
             and old_data.pop("training_conditioning", "mixed_context") == "mixed_context"
             and old_data.pop("temporal_gap_schedule", None) is None,
             "Independent-noise parent must be the exact mixed-mask gap trial")
        need(new_data.pop("training_conditioning", None) == "independent_frame_noise"
             and new_data.pop("completion_gap_schedule", None) is None,
             "Independent-noise trial retained context masks or completion-only gap")
        schedule = new_data.pop("temporal_gap_schedule", None)
    else:
        need(previous_schedule is None, "Gap parent must be ungapped")
        schedule = new_data.pop("completion_gap_schedule", None)
    need((schedule is None and independent and comparison["schedule"] is None)
         or (isinstance(schedule, list) and schedule and schedule == comparison["schedule"]),
         "Gap schedule differs from authorized stages")
    last_step, last_gap = -1, 0
    for stage in schedule or []:
        need(isinstance(stage, dict) and set(stage) == {"start_step", "max_gap_seconds"}
             and type(stage["start_step"]) is int and stage["start_step"] > last_step
             and type(stage["max_gap_seconds"]) in (int, float)
             and math.isfinite(stage["max_gap_seconds"])
             and stage["max_gap_seconds"] > 0 and stage["max_gap_seconds"] >= last_gap,
             "Invalid gap curriculum stage")
        last_step, last_gap = stage["start_step"], stage["max_gap_seconds"]
    need(schedule is None or (schedule[0]["start_step"] == 0 and schedule[-1]["start_step"] < request["target_steps"]),
         "Gap curriculum must start at zero and finish inside training budget")
    old_data.pop("manifest_path"); new_data.pop("manifest_path")
    need(old_data == new_data, "Gap changed data beyond completion schedule")
    old_runtime, new_runtime = copy.deepcopy(parent["lambda_runtime"]), copy.deepcopy(request["lambda_runtime"])
    for value in (old_runtime, new_runtime):
        value.pop("node"); value.pop("mounts")
    need(old_runtime == new_runtime, "Gap changed native runtime semantics")
    old_manifest = bound(comparison["parent_source_manifest"], False)
    need(hashlib.sha256(old_manifest.encode()).hexdigest() == parent["source_manifest_sha256"],
         "Gap parent source manifest identity changed")
    def scientific(text):
        return {name: digest for digest, name in (line.split("  ", 1) for line in text.splitlines())
                if name.startswith("egomimic/")}
    old_files = scientific(old_manifest)
    new_files = scientific(Path(request["source_manifest"]).read_text())
    changed = {name for name in set(old_files) | set(new_files)
               if old_files.get(name) != new_files.get(name)}
    allowed = {"egomimic/rldb/pixel_proprio.py", "egomimic/rldb/pixel_proprio_datamodule.py"}
    if independent:
        allowed.add("egomimic/models/image_proprio.py")
    need(changed == allowed, "Sampling source delta includes unrelated scientific changes")
    return dict(status="PASS", scope="all_training_frames" if independent else "completion_only", schedule=schedule,
                fairness="FAIR WITH CAVEATS",
                caveats=(["Independent timestamp strengths replace the context-mask mixture; the same temporal gap stages now apply to every training clip.",
                          "Training targets and contexts differ; compare common held-out metrics and actual rollouts, not raw training loss."] if independent else
                         ["Dedicated node differs; compare optimizer updates and metrics, not raw throughput.",
                          "Completion curriculum intentionally leaves forecast sampling unchanged."])
                        + (["Two ranks with batch16 preserve global32; rank noise draws and throughput are not paired."]
                           if topology_changed else []))


def verify_optimizer_comparison(request, completion):
    """Admit an independent optimizer-recipe trial against an immutable parent.

    Topology differences are explicit confounders, never a claim of paired RNG.
    All source-owned model/data/training/config files must be byte-identical;
    only smoke observation and verification tools may change.
    """
    comparison = request["optimizer_comparison"]
    parent = bound(comparison["parent_request"])
    need(comparison["schema"] == "optimizer-recipe-comparison-v1"
         and comparison["topology_control"] == "global_batch_matched_rank_rng_unpaired"
         and parent["phase"] == "full" and request["run_id"] != parent["run_id"],
         "Invalid fresh optimizer comparison")
    for field in ("data_manifest_sha256", "active_streams", "parameters", "expert_parameters",
                  "expert_widths", "global_batch", "target_steps", "model_execution", "arm", "experiment"):
        need(request[field] == parent[field], "Optimizer comparison changed science: " + field)
    need(request["active_streams"] in (["image"], ["image", "proprio"])
         and request["world_size"] * request["per_rank_batch"] == 32,
         "Optimizer comparison requires native image/state streams and global32")
    need(validate(request["lr_policy"]) != validate(parent["lr_policy"]), "Identical optimizer recipes")
    def scientific_manifest(binding):
        return {line.split("  ", 1)[1]: line.split("  ", 1)[0]
                for line in Path(binding).read_text().splitlines()
                if line.split("  ", 1)[1].startswith("egomimic/")
                and line.split("  ", 1)[1] != "egomimic/pl_utils/joint_smoke.py"}
    need(sha(parent["source_manifest"]) == parent["source_manifest_sha256"], "Changed parent manifest")
    old, new = scientific_manifest(parent["source_manifest"]), scientific_manifest(request["source_manifest"])
    need(old and old == new, "Model/data/learning-loop/config source changed")
    baseline = bound(parent["smoke"]["completion"])
    need(baseline["status"] == "PASS" and completion["status"] == "PASS"
         and baseline["ranks"][0]["initial_image_weights_sha256"]
         == completion["ranks"][0]["initial_image_weights_sha256"],
         "Optimizer trial did not preserve actual initial image weights")
    if "proprio" in request["active_streams"]:
        old_initial = baseline["ranks"][0].get("initial_weights")
        new_initial = completion["ranks"][0].get("initial_weights")
        need(isinstance(old_initial, list) and isinstance(new_initial, list)
             and old_initial and new_initial
             and all(isinstance(x, str) and len(x) == 64 for x in old_initial + new_initial)
             and len(set(old_initial + new_initial)) == 1,
             "Optimizer trial did not preserve actual initial image/proprio weights")
    return parent


def verify_capacity_extension(request, completion):
    """Compare the wider discrete model with its actual three-stream smoke.

    Existing source, labels, optimization and deployment semantics must match.
    Initialization and stochastic draws are recorded, not claimed identical
    between networks of different shapes.
    """
    comparison = request["capacity_comparison"]
    need(comparison.get("schema") == "native-capacity-extension-v1", "Wrong capacity comparison")
    parent = bound(comparison["parent_request"])
    baseline, _ = verify_evidence(parent)
    need(parent.get("capacity_profile") is None and request.get("capacity_profile") is not None,
         "Capacity comparison requires original and wider profiles")
    for key in ("source_revision", "source_manifest_sha256", "source_archive_sha256",
                "data_manifest_sha256", "world_size", "per_rank_batch", "global_batch",
                "target_steps", "arm", "model_execution", "lr_policy", "active_streams",
                "experiment", "action_contract"):
        need(request[key] == parent[key], "Capacity comparison changed control: " + key)
    need(request["arm"] == "discrete" and request["experiment"] == "image_proprio_action"
         and request["run_id"] != parent["run_id"], "Wrong capacity comparison identities")
    roots = [Path(r[k]) for r in (parent, request) for k in ("output", "local_logs")]
    need(all(p.is_absolute() for p in roots)
         and all(a != b and a not in b.parents and b not in a.parents
                 for i, a in enumerate(roots) for b in roots[i+1:]),
         "Overlapping capacity output/log roots")
    need(parent["smoke"]["job_id"] != request["smoke"]["job_id"], "Shared capacity GPU smoke")
    from omegaconf import OmegaConf
    normalized = []
    profile = request["capacity_profile"]
    for reference, width, heads, ratio in ((parent, 768, 6, 4),
            (request, profile["image_width"], profile["image_heads"], profile.get("image_mlp_ratio", 4))):
        cfg = OmegaConf.to_container(OmegaConf.create(bound(reference["smoke"]["resolved"], False)), resolve=True)
        need(verify_model_execution(reference, cfg) == "visual_state_action", "Wrong capacity graph")
        expected_image = verify_capacity_profile(reference, cfg)
        need(reference["expert_parameters"]["image"] == expected_image
             and cfg["ckpt_path"] is None and cfg["seed"] == cfg["data"]["seed"] == 42,
             "Capacity comparison is not a fresh seed42 trial")
        smoke_request = bound(reference["smoke"]["request"])
        roots = ((smoke_request["output"], "<output>"), (smoke_request["local_logs"], "<logs>"))
        image_count = 0
        def normalize(value):
            nonlocal image_count
            if isinstance(value, dict):
                value = {k: normalize(v) for k, v in value.items()}
                if value.get("_target_") == "egomimic.models.image_proprio.ImageProprioModel":
                    need((value["width"], value["heads"], value["depth"]) == (width, heads, 8),
                         "Unexpected image scaling")
                    need(value.get("mlp_ratio", 4) == ratio, "Unexpected image feed-forward scaling")
                    value["width"], value["heads"] = "<width>", "<heads>"
                    if "mlp_ratio" in value:
                        value["mlp_ratio"] = "<mlp_ratio>"
                    image_count += 1
                return value
            if isinstance(value, list):
                return [normalize(v) for v in value]
            if isinstance(value, str):
                for root, token in roots:
                    if value == root or value.startswith(root + "/"):
                        return token + value[len(root):]
            return value
        normalized.append(normalize(cfg))
        need(image_count > 0, "Missing actual image constructor")
    need(normalized[0] == normalized[1], "Capacity comparison changed unrelated resolved configuration")
    need(len(completion["ranks"]) == len(baseline["ranks"]) == request["world_size"],
         "Wrong capacity smoke rank count")
    return parent


def verify_modality_extension(request, completion):
    """Bind the single optional command bank to an actual paired state smoke.

    This is a modality comparison, never an optimizer-only comparison. Existing
    image/state GPU evidence is reused; the new arm must supply its own full
    smoke/restart evidence and actual projected initialization and RGB draws.
    """
    comparison = request["modality_comparison"]
    need(comparison["schema"] == "native-command-extension-v1", "Wrong extension comparison")
    parent = bound(comparison["parent_request"])
    baseline, _ = verify_evidence(parent)
    need(parent["active_streams"] == ["image", "proprio"]
         and request["active_streams"] == ["image", "proprio", "action"]
         and request["experiment"] == "image_proprio_action"
         and parent["run_id"] != request["run_id"], "Wrong command extension parent")
    for key in ("data_manifest_sha256", "world_size", "per_rank_batch", "global_batch",
                "target_steps", "arm", "model_execution", "lr_policy"):
        need(request[key] == parent[key], "Command comparison changed control: " + key)
    for name in parent["active_streams"]:
        need(request["expert_parameters"][name] == parent["expert_parameters"][name],
             "Existing expert parameter count changed")
    need(sha(parent["source_manifest"]) == parent["source_manifest_sha256"], "Changed parent source")
    def scientific(path):
        return {name: digest for digest, name in (line.split("  ", 1)
                for line in Path(path).read_text().splitlines())
                if name.startswith("egomimic/") and name != "egomimic/pl_utils/joint_smoke.py"}
    old, new = scientific(parent["source_manifest"]), scientific(request["source_manifest"])
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    need(changed == {"egomimic/models/joint_image_proprio.py",
                    "egomimic/hydra_configs/experiment/pixel_proprio/native_joint/discrete_image_proprio_action.yaml"},
         "Command extension changed unrelated model/data/config/learning-loop source")
    old_cfg = json.loads(json.dumps(bound(parent["smoke"]["resolved"], False)))
    new_cfg = bound(request["smoke"]["resolved"], False)
    from omegaconf import OmegaConf
    old_cfg = OmegaConf.to_container(OmegaConf.create(old_cfg), resolve=True)
    new_cfg = OmegaConf.to_container(OmegaConf.create(new_cfg), resolve=True)
    for key in ("seed", "world_model_attention"):
        need(old_cfg[key] == new_cfg[key], "Command comparison changed config: " + key)
    runtimes = []
    for cfg, reference in ((old_cfg, parent), (new_cfg, request)):
        runtime = copy.deepcopy(cfg["runtime"])
        smoke_output = bound(reference["smoke"]["request"])["output"]
        need(runtime.pop("slurm_signal_checkpoint_dir", None)
             == str(Path(smoke_output)/"checkpoints"),
             "Command comparison has an unbound signal checkpoint directory")
        runtimes.append(runtime)
    need(runtimes[0] == runtimes[1], "Command comparison changed runtime semantics")
    old_data, new_data = copy.deepcopy(old_cfg["data"]), copy.deepcopy(new_cfg["data"])
    need(old_data.pop("task") == "visual_diffusion"
         and new_data.pop("task") == "visual_state_action" and old_data == new_data,
         "Command comparison changed data beyond command labels")
    need(len(completion["ranks"]) == len(baseline["ranks"]) == request["world_size"], "Wrong paired rank count")
    for old_rank, new_rank in zip(baseline["ranks"], completion["ranks"]):
        need(new_rank.get("initial_state_control_weights_sha256") in old_rank["initial_weights"]
             and new_rank["initial_image_weights_sha256"] == old_rank["initial_image_weights_sha256"]
             and new_rank["paired_rgb_batch_sha256"] == old_rank["paired_rgb_batch_sha256"],
             "Unpaired actual image/state initialization or RGB/mask/noise draws")
    return parent


def compare_configs(production, smoke, request):
    from scripts.train.joint import imagenet, native_subtasks
    if imagenet.selected(request):
        return imagenet.compare_configs(production, smoke, request)
    if native_subtasks.selected(request):
        return native_subtasks.compare_configs(production, smoke, request, _compare_configs_legacy)
    return _compare_configs_legacy(production, smoke, request)
