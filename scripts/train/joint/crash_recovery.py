"""Explicit, evidence-bound recovery after failure; scheduled continuations stay separate.

No submission lives here.  A full request cannot enter this path until actual
allocated checkpoint inspection and matching recovery smoke have passed.
"""

import os
import json
from pathlib import Path
import shutil
import subprocess

from scripts.train.joint.contract import bound, need, sha


SOURCE = "535d4fc26c4c85cf57bbd2a776cc79cfec085c4e"
RUN_ID = "mot-shared-libero-image-discrete-lr1e-4-constant-s42-300k-20261021-v1"
TERMINAL_FAILURES = {"CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"}
STABLE = (
    "schema", "phase", "run_id", "arm", "source", "source_revision",
    "source_manifest", "source_manifest_sha256", "source_archive",
    "source_archive_sha256", "source_parent", "data_manifest",
    "data_manifest_sha256", "data_cache", "lambda_runtime", "runtime_localization",
    "allocation", "model_execution", "parameters", "expert_parameters",
    "expert_widths", "world_size", "global_batch", "per_rank_batch", "lr_policy",
    "vector_expert_parameter_budget", "active_streams", "experiment", "target_steps",
    "smoke", "optimizer_comparison", "modality_comparison", "temporal_comparison",
    "factorial_controls", "factorial_experiments", "archive_policy", "quota",
    "reserved_checkpoint_bytes",
)


def parse_native(text):
    return dict(part.split("=", 1) for part in text.split() if "=" in part)


def foundation_request(request):
    """Original immutable fresh admission remains the comparison foundation."""
    run_id, source = request["run_id"], request["source_revision"]
    seen = set()
    for _ in range(32):
        previous = request.get("crash_recovery") or request.get("continuation")
        if previous is None:
            return request
        binding = previous["parent_request"]
        need(binding["sha256"] not in seen, "Cyclic recovery admission foundation")
        seen.add(binding["sha256"])
        request = bound(binding)
        need(request["run_id"] == run_id and request["source_revision"] == source,
             "Recovery foundation belongs to another scientific identity")
    raise ValueError("Too many recovery admission ancestors")


def _separate(selected, previous, label):
    a, b = Path(selected), Path(previous)
    need(a.is_absolute() and b.is_absolute() and ".." not in a.parts,
         "Unsafe " + label)
    need(a != b and a not in b.parents and b not in a.parents,
         "Recovery " + label + " overlaps retained history")


def admission(request):
    """Read genuine failure/inspection/smoke evidence, without inventing boundary PASS."""
    recovery = request["crash_recovery"]
    need(request.get("continuation") is None
         and request.get("allocation_stop_step", request["target_steps"]) == request["target_steps"],
         "Crash and scheduled-boundary recovery cannot be combined")
    need(recovery["schema"] == "joint-crash-recovery-v1", "Wrong crash recovery binding")
    parent = bound(recovery["parent_request"])
    need(parent["phase"] == request["phase"] == "full"
         and request["source_revision"] == SOURCE
         and request["run_id"] == RUN_ID and request["active_streams"] == ["image"]
         and request["arm"] == "discrete",
         "This crash service currently covers only the exact image-only1e-4 run")
    need(all(parent.get(key) == request.get(key) for key in STABLE),
         "Crash recovery changed scientific identity or runtime")
    need(request["lr_policy"] == dict(schedule="constant", peak_lr=1e-4,
                                      warmup_optimizer_steps=0)
         and request["world_size"] == 4 and request["global_batch"] == 32
         and request["per_rank_batch"] == 8 and request["target_steps"] == 300000,
         "Changed authorized constant-LR recovery contract")
    _separate(request["output"], parent["output"], "output")
    _separate(request["local_logs"], parent["local_logs"], "log namespace")
    expected_overrides = [value.replace(parent["output"], request["output"])
                          .replace(parent["local_logs"], request["local_logs"])
                          for value in parent["overrides"]]
    need(request["overrides"] == expected_overrides,
         "Recovery overrides changed beyond output and log relocation")
    failure = bound(recovery["failure"])
    need(failure["schema"] == "joint-crash-failure-v1"
         and failure["parent_request_sha256"] == recovery["parent_request"]["sha256"]
         and failure["run_id"] == request["run_id"]
         and failure["classification"] in ("infrastructure", "worker_crash", "unexpected_scheduler_cancel")
         and failure["recovery_authorized"] is True and failure["human_retired"] is False,
         "Scientific failure or human retirement is not an automatic recovery")
    native = parse_native(bound(failure["native"], False))
    need(native["JobId"] == str(failure["job_id"])
         and native["JobName"] == parent["job_name"]
         and native["UserId"] == "paphiwetsa3(41003)"
         and native["JobState"] in TERMINAL_FAILURES
         and native["Requeue"] == "0" and native["Restarts"] == "0",
         "Parent failure is not a verified terminal owned allocation")
    need(bool(failure["logs"]), "Missing immutable failure logs")
    for item in failure["logs"]:
        need(bool(bound(item, False).strip()), "Empty failure log")
    preflight = bound(failure["parent_preflight"])
    need(preflight["status"] == "FULL_ADMISSION_PASS"
         and preflight["job_id"] == str(failure["job_id"])
         and preflight["request_sha256"] == recovery["parent_request"]["sha256"]
         and preflight["provenance"]["run_id"] == request["run_id"]
         and preflight["provenance"]["source_revision"] == SOURCE,
         "Failed allocation was not admitted with this exact parent request")
    inspection = bound(recovery["inspection"])
    need(inspection["schema"] == "joint-crash-checkpoint-inspection-v1"
         and inspection["status"] == "PASS"
         and inspection["parent_request_sha256"] == recovery["parent_request"]["sha256"]
         and inspection["run_id"] == request["run_id"]
         and inspection["source_revision"] == SOURCE
         and inspection["world_size"] == 4 and inspection["global_batch"] == 32
         and inspection["strict_full_wrapper_optimizer_reload"] is True
         and inspection["sampler_all_rank_rng_verified"] is True,
         "Missing actual strict checkpoint inspection")
    step = inspection["global_step"]
    need(type(step) is int and 0 < step < request["target_steps"] and step % 20000 == 0,
         "Recovery must use a healthy retained production milestone")
    need(inspection["highest_complete_checkpoint_inspected"] is True
         and inspection["all_newer_candidates_reconciled"] is True,
         "Newer checkpoint candidates are unresolved")
    checkpoint = inspection["checkpoint"]
    need(Path(checkpoint["path"]).is_absolute(), "Absolute inspected checkpoint required")
    from scripts.train.joint.crash_inspect import inventory_highest

    selected, inventory = inventory_highest(Path(parent["output"]) / "checkpoints", parent)
    need(selected.resolve() == Path(checkpoint["path"]).resolve()
         and any(Path(row["path"]).resolve() == selected.resolve()
                 and row["global_step"] == step for row in inventory),
         "Inspection is not the current highest retained ancestor checkpoint")
    need(sha(checkpoint["path"]) == checkpoint["sha256"], "Inspected checkpoint changed")
    proof = bound(recovery["recovery_smoke"])
    need(proof["schema"] == "joint-crash-recovery-smoke-v1" and proof["status"] == "PASS"
         and proof["source_revision"] == SOURCE
         and proof["parent_request_sha256"] == recovery["parent_request"]["sha256"]
         and proof["inspection_sha256"] == recovery["inspection"]["sha256"]
         and proof["checkpoint_sha256"] == checkpoint["sha256"]
         and proof["service_manifest_sha256"] == request["operational_service"]["manifest_sha256"]
         and proof["world_size"] == 4 and proof["global_batch"] == 32
         and proof["all_rank_full_state_restored"] is True
         and proof["finite_optimizer_updates"] >= 2
         and proof["scheduled_validation_anchors"] == 100
         and proof["no_replayed_boundary_validation_or_save"] is True
         and proof["production_science_unchanged"] is True
         and proof["sample_finite"] is True and proof["hidden_query_labels_nan"] is True
         and proof["sample_steps"] == 20,
         "Actual matching four-rank crash recovery smoke is missing")
    diagnostic = bound(proof["request"])
    need(proof["request_sha256"] == proof["request"]["sha256"]
         and diagnostic["schema"] == "joint-crash-recovery-diagnostic-v1"
         and diagnostic["parent_request"] == recovery["parent_request"]
         and diagnostic["inspection"] == recovery["inspection"]
         and diagnostic["operational_service"] == request["operational_service"]
         and all(diagnostic.get(key) == request.get(key) for key in (
             "source", "source_revision", "source_manifest_sha256", "data_manifest_sha256",
             "lambda_runtime", "runtime_localization", "data_cache", "world_size",
             "global_batch", "per_rank_batch", "active_streams", "lr_policy", "run_id")),
         "Recovery smoke request is not bound to this exact checkpoint/science/service")
    smoke_native = parse_native(bound(proof["native"], False))
    need(smoke_native["JobId"] == str(proof["job_id"])
         and smoke_native["JobState"] == "COMPLETED" and smoke_native["ExitCode"] == "0:0"
         and smoke_native["JobName"] == proof["job_name"] == diagnostic["job_name"]
         and smoke_native["UserId"] == "paphiwetsa3(41003)"
         and smoke_native["NodeList"] == request["lambda_runtime"]["node"]
         and smoke_native["Requeue"] == "0" and smoke_native["Restarts"] == "0"
         and bound(proof["host_exit"], False).strip() == "0",
         "Recovery smoke host/native did not finish successfully")
    bound(proof["candidate"])
    diagnostic_output = Path(diagnostic["output"])
    expected = {str(diagnostic_output/"restore-proof"/f"restore-step{step:09d}-rank{rank}.json")
                for rank in range(4)}
    expected |= {str(diagnostic_output/"restore-proof"/f"restore-validation-rank{rank}.json")
                 for rank in range(4)}
    expected |= {str(diagnostic_output/"validation-proof"/f"rank{rank}.json") for rank in range(4)}
    need(len(proof["rank_proofs"]) == 12
         and {item["path"] for item in proof["rank_proofs"]} == expected,
         "Recovery smoke is missing its actual rank proof inventory")
    for item in proof["rank_proofs"]:
        row = bound(item)
        need(row["status"] == "PASS" and str(row["job_id"]) == str(proof["job_id"])
             and row["world_size"] == 4 and row["rank"] in range(4)
             and row["start_step"] == step and row["global_step"] == step + 2,
             "Recovery smoke rank proof changed")
    final = proof["final_diagnostic_checkpoint"]
    need(Path(final["path"]).parent == diagnostic_output/"checkpoints"
         and final["strict_full_wrapper_optimizer_reload"] is True
         and sha(final["path"]) == final["sha256"],
         "Recovery diagnostic full-state checkpoint changed")
    return parent, inspection, step


def assert_native_writer_quiescence(request):
    """Repeat parent/writer checks in the new allocation immediately before training."""
    recovery = request["crash_recovery"]
    failure = bound(recovery["failure"])
    current = os.environ.get("SLURM_JOB_ID", "")
    need(current.isdecimal() and current != str(failure["job_id"]),
         "Fresh allocated recovery identity required")
    result = subprocess.run(["/usr/local/bin/scontrol", "show", "job", str(failure["job_id"]), "-o"],
                            capture_output=True, text=True, timeout=15)
    if result.returncode == 0:
        parent = parse_native(result.stdout)
        need(parent.get("JobState") in TERMINAL_FAILURES,
             "Parent allocation is still active or completing")
    else:
        need("Invalid job id" in result.stderr,
             "Parent native state unavailable for an unexplained reason")
        # Its hash-bound terminal native proof was validated by admission().
    result = subprocess.run(["/usr/local/bin/squeue", "-u", "paphiwetsa3", "-h", "-o", "%i|%j|%T"],
                            capture_output=True, text=True, timeout=15)
    need(result.returncode == 0, "Current writer queue inspection failed")
    prefix = {"image_only": "joint-image-lr1e4",
              "image_proprio": "joint-" + request["arm"] + "-image-proprio-lr1e4"}.get(request["experiment"])
    need(prefix is not None and request["job_name"].startswith(prefix),
         "Missing unique campaign writer name prefix")
    for line in result.stdout.splitlines():
        job, name, state = line.split("|", 2)
        need(job == current or not name.startswith(prefix),
             "Another campaign writer is active/pending/completing: " + job)


def prove_destination_storage(request):
    """One retained exclusive fsync/readback probe in the actual destination FS.

    Call only from the admitted allocation after prerequisite infrastructure
    recovery.  The allocation wrapper bounds startup; no retries or cleanup.
    """
    job = os.environ.get("SLURM_JOB_ID", "")
    need(job.isdecimal(), "Allocated storage proof required")
    destination = Path(request["output"])
    parent = destination.parent
    need(parent.is_dir() and not destination.exists(),
         "Fresh recovery output parent missing or destination already exists")
    free = shutil.disk_usage(parent).free
    need(free > max(request["reserved_checkpoint_bytes"], 40_000_000_000),
         "Insufficient actual destination checkpoint space")
    payload = json.dumps(dict(schema="joint-crash-storage-probe-v1", job_id=job,
                              run_id=request["run_id"], output=str(destination),
                              free_bytes=free), sort_keys=True).encode()
    path = parent / (".crash-recovery-storage-" + job + ".json")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    need(path.read_bytes() == payload, "Destination fsync/readback failed")
    return dict(path=str(path), sha256=sha(path), free_bytes=free,
                write_fsync_readback_pass=True)


def configure(cfg, request, inspection, start, *, request_sha256):
    """Only operational config edits, after unchanged science has been compared."""
    from omegaconf import open_dict
    need(isinstance(request_sha256, str) and len(request_sha256) == 64,
         "Actual recovery request content identity required")
    with open_dict(cfg):
        cfg.ckpt_path = inspection["checkpoint"]["path"]
        cfg.logger.wandb.resume = "must"
        cfg.logger.csv.save_dir = request["local_logs"]
        cfg.logger.csv.version = f"step{start:09d}"
        cfg.callbacks.crash_restore_proof = dict(
            _target_="scripts.train.joint.crash_restore.CrashRestoreProof",
            checkpoint=inspection["checkpoint"]["path"],
            inspection_path=request["crash_recovery"]["inspection"]["path"],
            inspection_sha256=request["crash_recovery"]["inspection"]["sha256"],
            output=str(Path(request["output"]) / "restore-proof"),
            world_size=request["world_size"], start_step=start, proof_updates=2,
            request_sha256=request_sha256,
        )
    return cfg
