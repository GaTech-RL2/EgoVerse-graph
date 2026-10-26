"""Strict request chain for scheduled native joint-training allocations.

This only extends the shared full service's allocation lifetime. The scientific
target, optimizer schedule, model and scheduled validation/save cadence stay fixed.
"""

from pathlib import Path

from scripts.train.joint.contract import bound, need, sha


STABLE_FIELDS = (
    "run_id", "arm", "source", "source_revision", "source_manifest",
    "source_manifest_sha256", "data_manifest", "data_manifest_sha256", "data_cache",
    "source_archive", "source_archive_sha256", "parameters", "expert_parameters",
    "expert_widths", "world_size", "global_batch", "per_rank_batch", "lr_policy",
    "vector_expert_parameter_budget", "active_streams", "experiment", "target_steps",
    "output", "allocation", "smoke", "factorial_controls", "factorial_experiments",
    "allocation_boundaries", "archive_policy", "quota", "reserved_checkpoint_bytes",
    "allocation_recovery",
    "operational_service", "temporal_comparison", "model_execution", "action_contract",
    "capacity_profile", "noise_comparison", "track_recipe_comparison",
    "kind", "interfaces", "subtask_contract", "dataset_contract",
)


def allocation_steps(request):
    """No continuation is inferred from a directory or W&B's status."""
    target = request["target_steps"]
    stop = request.get("allocation_stop_step", target)
    start = 0
    continuation = request.get("continuation")
    if continuation is not None:
        parent = bound(continuation["parent_request"])
        receipt = bound(continuation["completion"])
        need(parent["schema"] == "joint-training-v2" and parent["phase"] == "full",
             "Continuation parent is not a full request")
        need(all(request.get(k) == parent.get(k) for k in STABLE_FIELDS),
             "Continuation changed its scientific identity or output")
        need(receipt["schema"] == "joint-mot-allocation-v1" and receipt["status"] == "PASS"
             and receipt["request_sha256"] == continuation["parent_request"]["sha256"]
             and receipt["run_id"] == request["run_id"]
             and receipt["world_size"] == request["world_size"]
             and receipt["global_step"] == parent["allocation_stop_step"]
             and receipt["global_step"] < target
             and receipt["strict_full_wrapper_optimizer_scheduler_reload"]
             and receipt["sampler_all_rank_rng_verified"] and receipt["sample_finite"],
             "Missing strict prior-allocation completion")
        checkpoint = continuation["checkpoint"]
        need(checkpoint["path"] == receipt["checkpoint"]
             and checkpoint["sha256"] == receipt["checkpoint_sha256"]
             and sha(checkpoint["path"]) == checkpoint["sha256"],
             "Continuation recovery checkpoint changed")
        start = receipt["global_step"]
    need(type(start) is int and type(stop) is int
         and 0 <= start < stop <= target
         and start % 20000 == stop % 20000 == target % 20000 == 0,
         "Allocation boundaries must preserve20k cadence and total budget")
    if stop < target or continuation is not None:
        plan = request["allocation_boundaries"]
        need(plan == sorted(set(plan)) and plan[-1] == target
             and stop == next((v for v in plan if v > start), None),
             "Allocation skipped or changed its declared boundaries")
        proof = bound(request["allocation_recovery"])
        need(proof["schema"] == "joint-recovery-check-v1" and proof["status"] == "PASS"
             and proof["source_revision"] == request["source_revision"]
             and proof["data_manifest_sha256"] == request["data_manifest_sha256"]
             and proof["active_streams"] == request["active_streams"]
             and proof["world_size"] == request["world_size"] and proof["global_batch"] == 32
             and proof["smoke_completion_sha256"] == request["smoke"]["completion"]["sha256"]
             and proof["actual_updates"] == 6 and proof["all_rank_full_state_restored"]
             and proof["model_data_unchanged"] and proof["no_extra_validation"],
             "Actual matching BF16/DDP allocation recovery is not proven")
    return start, stop


def configure_allocation(cfg, request, start, stop):
    """Apply operational additions only AFTER the existing science comparison."""
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.logger.csv.save_dir = request["local_logs"]
        cfg.logger.csv.version = f"step{start:09d}"
        if start:
            cfg.ckpt_path = request["continuation"]["checkpoint"]["path"]
            cfg.logger.wandb.resume = "must"
        if stop < request["target_steps"]:
            cfg.callbacks.allocation_boundary = dict(
                _target_="scripts.train.joint.allocation_boundary.ScheduledAllocationBoundary",
                start_step=start, stop_step=stop, target_step=request["target_steps"],
                cadence=20000, output=str(Path(request["output"])/"allocation-evidence"),
            )
    return cfg


def log_segments(request):
    """Keep each allocation's CSV events; reject cycles and repeated log roots."""
    result, seen = [], set()
    for _ in range(16):
        log = request["local_logs"]
        need(log not in seen, "Repeated continuation log root or request cycle")
        seen.add(log)
        previous = request.get("continuation")
        start = 0 if previous is None else bound(previous["completion"])["global_step"]
        relative = Path("csv")/f"step{start:09d}"/"metrics.csv"
        current = Path(log)/relative
        result.append(current if current.is_file() else Path(request["output"])/"log-mirror"/relative)
        if previous is None:
            return result[::-1]
        request = bound(previous["parent_request"])
    raise ValueError("Too many allocation attempts for the retained20k cadence")
