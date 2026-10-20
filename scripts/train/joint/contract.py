"""Exact smoke-to-production admission. Standard-library evidence checks are read-only."""

import copy
import hashlib
import json
import math
from pathlib import Path


def need(condition, message):
    if not condition:
        raise ValueError(message)


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
        and native["UserId"].startswith("paphiwetsa3(")
        and native["JobName"] == smoke["job_name"],
        "Wrong native smoke proof",
    )
    if smoke.get("allocation") is None:
        need(
            request.get("allocation") is None and native["NodeList"] == "gpu21",
            "Wrong Lambda smoke allocation",
        )
    else:
        from scripts.train.joint.allocation import verify_pace

        need(
            request.get("allocation") == smoke["allocation"],
            "Destination policy changed after smoke",
        )
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
    need(
        completion["sample_state_steps"] == completion["sample_action_steps"] == 20
        and completion["lr_warmup_steps"] == 3000,
        "Wrong sampling/scheduler proof",
    )
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
        len(ranks) == world and len({r["gpu_uuid"] for r in ranks}) == world,
        "Wrong GPU topology",
    )
    need(len({r["weights_sha256"] for r in ranks}) == 1, "Ranks diverged")
    if smoke.get("allocation") is not None:
        need(
            [r["gpu_uuid"].lower() for r in ranks]
            == [r["uuid"].lower() for r in allocation["gpu_probe"]["ranks"]],
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
                math.isclose(lr, 5e-6 * (i + 1) / 3000, rel_tol=1e-12)
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
        and math.isclose(capacity["optimizer_update_lr"], 5e-6 / 3000, rel_tol=1e-12),
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
