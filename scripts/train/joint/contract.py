"""Exact smoke-to-production admission. Standard-library evidence checks are read-only."""

import copy
import hashlib
import json
import math
import uuid
from scripts.train.joint.lr_policy import validate, update_lr
from pathlib import Path


def need(condition, message):
    if not condition:
        raise ValueError(message)



def verify_model_execution(request, cfg):
    """Bind native historical image/state execution without changing legacy admission."""
    stages = cfg["model"]["pipeline"]["stages"]
    ids = cfg["model"]["pipeline"]["stage_ids"]
    model = stages[ids["joint_denoiser"]]["model"]
    native = model["_target_"] == "egomimic.models.joint_image_proprio.JointImageProprioModel"
    policy = request.get("model_execution")
    if not native:
        need(policy is None, "Native execution policy requires the native image wrapper")
        return "visual_state_action"
    need(policy == {"implementation": "native_image_joint", "compile_mode": "default",
                    "compile_dynamic": True, "compile_backend": "inductor"},
         "Native image execution requires an explicit historical request binding")
    image = model["image_model"]
    need(image["_target_"] == "egomimic.models.image_proprio.ImageProprioModel"
         and image["compile_mode"] == policy["compile_mode"]
         and image["compile_dynamic"] == policy["compile_dynamic"]
         and image["compile_backend"] == policy["compile_backend"],
         "Native image compilation differs from bound historical policy")
    streams = request["active_streams"]
    action = streams == ["image", "proprio", "action"]
    need(streams in (["image"], ["image", "proprio"], ["image", "proprio", "action"])
         and model["active_streams"] == streams,
         "Native request has unbound streams")
    need([stage["stream"] for stage in stages if "stream" in stage] == streams,
         "Native loss graph differs from active experts")
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
        need(isinstance(contract, dict) and contract.get("schema") == "native-joint-command-v1",
             "Missing explicit native action contract")
        native, encoded = contract.get("native_dimensions"), contract.get("encoded_dimensions")
        group = cfg["data"]["stride"]
        need(isinstance(native, dict) and native and all(type(v) is int and v > 0 for v in native.values())
             and cfg["data"]["mode"] == "discrete" and type(group) is int and group > 0
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


def verify_evidence(request):
    need(
        request["schema"] == "joint-training-v2" and request["phase"] == "full",
        "Wrong request schema",
    )
    evidence = request["smoke"]
    smoke = bound(evidence["request"])
    need(
        smoke["schema"] == "joint-training-v2" and smoke["phase"] == "smoke",
        "Wrong smoke request",
    )
    completion = bound(evidence["completion"])
    preflight = bound(evidence["preflight"])
    capacity = bound(evidence["capacity"])
    native = dict(
        s.split("=", 1) for s in bound(evidence["native"], False).split() if "=" in s
    )
    need(bound(evidence["host_exit"], False).strip() == "0", "Smoke host failed")
    need(
        native["JobId"] == str(evidence["job_id"])
        and native["JobState"] == "COMPLETED"
        and native["ExitCode"] == "0:0"
        and native["UserId"].startswith((smoke.get("allocation") or {}).get("user", "paphiwetsa3")+"(")
        and native["JobName"] == smoke["job_name"],
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
            and len(record["paired_rgb_batch_sha256"]) == 3,
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


def compare_configs(production, smoke, request):
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
