"""Finalized, current-intent-bound native Elmo production interruption admission.

No submission/cancellation occurs here. Producer completion is checked from raw
native records and host exits. The live launch gate repeats intent, queues and
all named claim directories; a local test or a producer's own PASS is insufficient.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

from scripts.train.joint.contract import bound, need, sha, write_once
from scripts.train.joint.native_interruption import (
    NATIVE_KEYS, command, fields, native_identity, classify_failure, check_steps, process_identity, destination_probe,
)


CHANGING_ATTEMPT_FIELDS = {"job_name", "output", "local_logs", "overrides", "operational_service",
    "interruption", "continuation", "allocation_stop_step", "allocation_recovery", "allocation_boundaries"}


def fingerprint(request):
    """Avoid circular finalizer/request hashes by omitting only the admission bundle."""
    value = copy.deepcopy(request)
    value["interruption"].pop("admission", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compatible(parent, child):
    need(parent["schema"] == child["schema"] == "joint-training-v2"
         and parent["phase"] == child["phase"] == "full"
         and parent["kind"] == child["kind"] == "native-subtask-world-model-v1"
         and child["parameters"] == 314187064 and child["world_size"] == 4,
         "Exact production native Elmo314M identity required")
    need({k: v for k, v in parent.items() if k not in CHANGING_ATTEMPT_FIELDS}
         == {k: v for k, v in child.items() if k not in CHANGING_ATTEMPT_FIELDS},
         "Interrupted allocation changed science/runtime/cache/total budget")
    need(child.get("continuation") is None and child.get("allocation_stop_step", child["target_steps"]) == child["target_steps"],
         "Interruption v1 resumes to the original total target; no fake scheduled boundary")
    need(child["output"] != parent["output"] and child["local_logs"] != parent["local_logs"]
         and child["job_name"] != parent["job_name"], "Fresh immutable output/log/job namespaces required")


def completed_producer(bundle, result_schema, status):
    """Successful new recovery phases need complete service-verified observers."""
    from scripts.train.joint.phase_observer_proof import verify_observer, MODES, bound as proof_bound, read_json, text
    admitted = {
        "native-elmo-interruption-evidence-v1": ("capture", "CAPTURED_REQUIRES_NATIVE_PRODUCER_COMPLETION"),
        "native-elmo-checkpoint-inspection-result-v1": ("inspect", "INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION"),
        "native-elmo-production-recovery-diagnostic-v1": ("diagnostic", "DIAGNOSTIC_COMPLETE_REQUIRES_HOST_AND_NATIVE_SUCCESS"),
        "native-elmo-interruption-finalized-v1": ("finalize", "FINALIZED_REQUIRES_HOST_AND_NATIVE_SUCCESS"),
    }
    need(result_schema in admitted and status == admitted[result_schema][1], "Unknown recovery producer result/status contract")
    mode = admitted[result_schema][0]
    q, result = read_json(proof_bound(bundle["request"])), read_json(proof_bound(bundle["result"]))
    need(q["schema"] == MODES[mode][0], "Recovery producer request/result phase differs")
    filenames = dict(capture="interruption-capture.json", inspect="checkpoint-inspection.json",
                     diagnostic="production-recovery-diagnostic.json", finalize="interruption-finalized.json")
    need(bundle["result"]["path"] == str(Path(q["output"]) / filenames[mode]),
         "Producer result escaped its canonical allocated output")
    expected = q["producer_native"]
    proof = verify_observer(bundle["observer"], producer_request=bundle["request"],
                            producer_native=expected, terminal_native=bundle["native"])
    need(proof["spec"]["mode"] == mode and proof["request"] == q
         and proof["running_admission_status"] == "VALIDATED", "Successful phase lacks actual matching RUNNING provenance")
    native = native_identity(text(proof_bound(bundle["native"])), expected, "COMPLETED")
    host_path = proof["root"] / "status" / (mode + "-" + expected["JobId"] + ".exit")
    need(bundle["host_exit"]["path"] == str(host_path) and native["ExitCode"] == "0:0"
         and text(proof_bound(bundle["host_exit"])).strip() == "0", "Producer did not actually exit canonical host0/nativeCOMPLETED0:0")
    jobs = [result[k] for k in ("job_id", "producer_job_id") if k in result]
    need(jobs and all(job == expected["JobId"] for job in jobs)
         and result["request_sha256"] == bundle["request"]["sha256"]
         and result["schema"] == result_schema and result["status"] == status,
         "Completed native producer does not bind the exact request/result")
    return q, result


def current_intent(path, run_id, expected_sha=None):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), "Current human intent file missing")
    if expected_sha is not None:
        need(sha(path) == expected_sha, "Human intent changed since finalization; re-evaluate before restart")
    intent = json.loads(path.read_text())
    need(intent["schema"] == "native-elmo-current-intent-v1" and intent["authority"] == "human"
         and intent["status"] == "ACTIVE" and run_id in intent["authorized_recovery_run_ids"]
         and run_id not in intent["retired_run_ids"] and not intent["hold_recovery"],
         "Recovery is paused, held, retired, or outside current human authorization")
    need(intent.get("source_reference") and intent.get("updated_at"), "Traceable current human instruction required")
    return sha(path)


def quiescence(spec, run_id, *, current_job=None, current_request_sha=None):
    """Check every registered claim directory and the fresh complete owner queue."""
    need(spec["user"] and spec["claim_roots"] and spec["scientific_job_names"]
         and spec["scientific_name_prefixes"], "Complete campaign claim/queue scope required")
    need(all(len(p) >= 16 for p in spec["scientific_name_prefixes"]), "Scientific writer prefix too broad")
    queue = command(["/usr/local/bin/squeue", "--noheader", "--user", spec["user"], "--format=%i|%j|%T|%k"])
    # Do not target old IDs: Slurm may have purged them after our genuine native
    # observer retained their terminal records. Still query the complete live
    # owner step inventory and reject every surviving step of each prior writer.
    steps = command(["/usr/local/bin/squeue", "--steps", "--noheader", "--user", spec["user"], "--format=%i|%T"])
    from scripts.train.joint.original_observer import strict_step_rows
    strict_step_rows(steps)
    for line in queue.splitlines():
        if not line.strip():
            continue
        row = [v.strip() for v in line.split("|")]
        need(len(row) == 4, "Unparseable actual owner queue")
        job, name, state, comment = row
        owned = name in spec["scientific_job_names"] or any(name.startswith(p) for p in spec["scientific_name_prefixes"]) or comment == run_id
        if owned:
            need(job == current_job and state == "RUNNING", "Other pending/active scientific writer: " + job)
    from scripts.train.joint.phase_observer_proof import read_json, text, physical, check_owned_steps
    seen, evidence = set(), []
    for item in spec["claim_roots"]:
        root = physical(item["path"])
        need(root.is_absolute() and root.is_dir() and not root.is_symlink(), "Registered claim directory unavailable")
        for claim in sorted(root.iterdir()):
            need(claim.is_dir() and not claim.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", claim.name),
                 "Unrecognized claim entry requires exact inspection")
            prepared = read_json(claim / "PREPARED.json")
            need(prepared["request_sha256"] == claim.name, "Claim/request identity mismatch")
            job = text(claim / "PARENT_JOB_ID").strip()
            need(job.isdecimal() and job not in seen, "Unknown/duplicate native submission acceptance")
            seen.add(job)
            if job == current_job:
                need(claim.name == current_request_sha, "Current native writer belongs to another request claim")
                evidence.append(dict(path=str(claim), job_id=job, current=True))
                continue
            from scripts.train.joint.phase_observer_proof import verify_claim_terminal
            terminal_proof = verify_claim_terminal(claim, item["terminal_directory"], run_id)
            need(terminal_proof["job_id"] == job, "Terminal observer belongs to another accepted claim")
            check_owned_steps(steps, job)
            evidence.append(dict(path=str(claim), **terminal_proof))
    if current_job is not None:
        need(current_job in seen, "Current allocation is not registered in the complete claim census")
    return dict(queue=queue, owned_steps=steps, claims=evidence, checked_at=datetime.now(timezone.utc).isoformat())


def validate_failure(bundle, parent_binding):
    q, result = completed_producer(bundle, "native-elmo-interruption-evidence-v1",
                                   "CAPTURED_REQUIRES_NATIVE_PRODUCER_COMPLETION")
    need(q["parent_request"] == result["parent_request"] == parent_binding,
         "Failure evidence belongs to another production parent")
    raw = bound(result["evidence"]["parent-native.txt"], False)
    from scripts.train.joint.original_observer import resolve_terminal, validate_steps
    resolved_raw, provenance = resolve_terminal(q, bound(result["evidence"]["parent-native-query.json"]))
    need(resolved_raw == raw and provenance == bound(result["evidence"]["parent-terminal-provenance.json"])
         and provenance["source"] == result["parent_terminal_source"], "Terminal provenance changed after capture")
    state = fields(raw).get("JobState")
    terminal = native_identity(raw, q["parent_native"], state)
    running = bound(q["parent_running_admission"])
    need(running["job_id"] == q["parent_job_id"] and running["request_sha256"] == parent_binding["sha256"],
         "Parent actual RUNNING claim differs")
    native_identity(running["native"], q["parent_native"], "RUNNING")
    steps = validate_steps(bound(result["evidence"]["parent-step-query.json"]), q["parent_job_id"])
    need(steps == bound(result["evidence"]["parent-live-steps.txt"], False), "Preserved complete step evidence differs")
    tails = "\n".join(bound(value, False) for name, value in result["evidence"].items() if name.endswith(".tail64.txt"))
    need(tails and classify_failure(terminal, tails) == result["reason"], "Actual failure classification differs")
    from scripts.train.joint.native_interruption import parent_csv_paths, newest_loss
    parent = bound(parent_binding)
    need({name: value["path"] for name, value in result["loss_sources"].items()} == parent_csv_paths(parent),
         "Failure loss evidence came from another production log namespace")
    snapshots = {name: {part: bound(result["evidence"]["loss-" + name + "-" + part + ".txt"], False)
                        for part in ("header", "tail")}
                 for name, value in result["loss_sources"].items() if value["exists"]}
    need(newest_loss(snapshots) == result["newest_numeric_loss"], "Preserved newest numeric loss evidence differs")
    return result


def resume_native_contract(expected, parent_native, child):
    """Bind future immutable allocation shape before its job ID exists."""
    need(set(expected) == (set(NATIVE_KEYS) | {"NodeList", "TresPerNode"}) - {"JobId"},
         "Complete future production native contract required")
    stable = set(expected) - {"JobName", "Command"}
    need(all(expected[key] == parent_native[key] for key in stable)
         and expected["JobName"] == child["job_name"]
         and expected["Command"] != parent_native["Command"] and Path(expected["Command"]).is_absolute()
         and expected["NumCPUs"] == "32" and expected["MinMemoryNode"] == "192G"
         and expected["TresPerNode"] == "gres/gpu:h100:4",
         "Resume native resources/owner/node differ or command/job namespace is reused")
    return expected


def recovery_phase_admission(q, request_path, *, scientific_writer=False):
    """Recheck real failure, human intent and all writers before costly phases."""
    parent = bound(q["parent_request"])
    intent = current_intent(q["current_intent_path"], parent["run_id"])
    failure = validate_failure(q["failure"], q["parent_request"])
    options = dict(current_job=os.environ["SLURM_JOB_ID"], current_request_sha=sha(request_path)) if scientific_writer else {}
    live = quiescence(q["quiescence"], parent["run_id"], **options)
    return dict(current_intent_sha256=intent, failure_reason=failure["reason"], quiescence=live)


def finalize(request_path):
    """Separate allocated CPU finalizer; emits no job or production checkpoint."""
    q = json.loads(Path(request_path).read_text())
    need(q["schema"] == "native-elmo-interruption-finalizer-v1", "Wrong native interruption finalizer request")
    producer = q["producer_native"]
    process_identity(producer, cpu=True)
    need(producer["JobId"] == os.environ.get("SLURM_JOB_ID")
         and producer["NodeList"] == os.environ.get("SLURMD_NODENAME")
         and producer["NumCPUs"] == "4" and producer["MinMemoryNode"] == "24G"
         and producer["TimeLimit"] == "00:20:00" and producer["TresPerNode"] == ""
         and not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS"),
         "Separate real4CPU24G20min finalizer required")
    raw = command(["/usr/local/bin/scontrol", "show", "job", producer["JobId"], "-o", "-dd"])
    native_identity(raw, producer, "RUNNING")
    parent, child = bound(q["parent_request"]), bound(q["resume_request_template"])
    from scripts.train.joint.checkpoint_inspection import verify_source_binding
    verify_source_binding(parent, q)
    need(child["interruption"]["parent_request"] == q["parent_request"]
         and child["interruption"].get("admission") is None, "Finalizer requires an unadmitted immutable template")
    compatible(parent, child)
    intent_sha = current_intent(q["current_intent_path"], parent["run_id"])
    failure = validate_failure(q["failure"], q["parent_request"])
    expected_resume = resume_native_contract(q["resume_native_contract"], failure["parent_native"], child)
    inspect_q, inspection = completed_producer(q["inspection"], "native-elmo-checkpoint-inspection-result-v1",
                                               "INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION")
    need(inspect_q["parent_request"] == inspection["parent_request"] == q["parent_request"]
         and inspection["highest_healthy_known"] is True, "Highest healthy exact-parent production inspection missing")
    selected = inspection["selected"]
    need(selected["status"] == "CPU_STRICT_INSPECTION_PASS"
         and selected["global_step"] == max(row["global_step"] for row in inspection["census"]["candidates"])
         and any(all(selected[k] == row[k] for k in ("path", "global_step", "size_bytes", "mtime_ns", "inode", "device"))
                 for row in inspection["census"]["candidates"])
         and all(row["status"] == "CPU_STRICT_INSPECTION_PASS" for row in inspection["results"]),
         "Selected checkpoint is not the highest strictly inspected retained production milestone")
    need(inspect_q["production_output"] == inspection["production_destination"]["output"] == child["output"]
         and inspection["production_destination"]["actual_write_fsync_readback"] is True
         and sha(inspection["production_destination"]["probe_path"]) == inspection["production_destination"]["probe_sha256"],
         "Allocated inspection did not prove the actual production destination")
    need(child["interruption"]["checkpoint"] == {k: selected[k] for k in ("path", "sha256", "global_step", "size_bytes")}
         and sha(selected["path"]) == selected["sha256"], "Resume selected a different or changed production checkpoint")
    from scripts.train.joint.checkpoint_inspection import checkpoint_census
    need(checkpoint_census(parent) == inspection["census"], "Retained production census changed after inspection")
    diagnostic_q, diagnostic = completed_producer(q["diagnostic"], "native-elmo-production-recovery-diagnostic-v1",
                                                  "DIAGNOSTIC_COMPLETE_REQUIRES_HOST_AND_NATIVE_SUCCESS")
    need(diagnostic_q["parent_request"] == q["parent_request"]
         and diagnostic_q["inspection"] == q["inspection"]["result"]
         and diagnostic["production_checkpoint"] == child["interruption"]["checkpoint"],
         "Actual GPU diagnostic used another parent/checkpoint")
    expected_rank_paths = {str(Path(diagnostic_q["output"]) / "proof" / f"resume-{first}-{last}-rank{rank}.json")
                           for first, last in ((selected["global_step"], selected["global_step"] + 2),
                                               (selected["global_step"] + 2, selected["global_step"] + 6))
                           for rank in range(4)}
    need(len(diagnostic["rank_proofs"]) == 8
         and {b["path"] for b in diagnostic["rank_proofs"]} == expected_rank_paths,
         "Completed diagnostic does not bind the exact eight actual rank receipts")
    for binding in diagnostic["rank_proofs"]:
        bound(binding)
    need(child["operational_service"] == q["operational_service"] == diagnostic_q["operational_service"]
         == inspect_q["operational_service"], "Finalizer/inspection/GPU/resume operational implementations differ")
    from scripts.train.joint.production_recovery import read_phase_proofs, validate_phase_proofs
    from scripts.train.joint.lr_policy import update_lr
    from scripts.train.joint.production_history import retained_milestones, scheduled_validation

    start = selected["global_step"]
    rows = read_phase_proofs(diagnostic_q["output"], start, parent["target_steps"], 4, diagnostic["job_id"])
    flags = validate_phase_proofs(rows, selected, diagnostic.get("allocation_attestation"))
    for phase in rows:
        for row in phase:
            need(row["optimizer_update_lrs"] == [update_lr(parent["lr_policy"], i)
                 for i in range(row["start_step"], row["global_step"])], "Actual diagnostic LR sequence differs")
    metric_names = set(rows[-1][0]["validation_metrics"])
    _, checkpoints = retained_milestones(parent, start)
    _, history = scheduled_validation(parent, start, metric_names)
    live = quiescence(q["quiescence"], parent["run_id"])
    need(not Path(child["output"]).exists() and not Path(child["local_logs"]).exists(),
         "Fresh resume output/log namespace already exists")
    destination = destination_probe(child["output"], producer["JobId"], child["reserved_checkpoint_bytes"])
    output = Path(q["output"])
    output.mkdir()
    write_once(output / "interruption-finalized.json", dict(
        schema="native-elmo-interruption-finalized-v1", status="FINALIZED_REQUIRES_HOST_AND_NATIVE_SUCCESS",
        job_id=producer["JobId"], request_sha256=sha(request_path), parent_request=q["parent_request"],
        run_id=parent["run_id"], resume_request_fingerprint=fingerprint(child),
        checkpoint=child["interruption"]["checkpoint"], failure_reason=failure["reason"],
        current_intent_path=q["current_intent_path"], current_intent_sha256=intent_sha,
        retained_checkpoint_history=checkpoints, committed_validation_history=history,
        production_destination=destination,
        resume_native_contract=expected_resume,
        quiescence=live, diagnostic=q["diagnostic"], inspection=q["inspection"], failure=q["failure"],
        actual_updates=6, original_total_target=parent["target_steps"],
        last_verified_production_step=start,
        lost_unsaved_updates=None,
        lost_unsaved_updates_status="Requires actual final worker-update evidence; checkpoint step is not crash progress",
        finalized_at=datetime.now(timezone.utc).isoformat(), **flags))


def admission(request):
    """Static immutable chain check, safe for later full-completion verification."""
    interruption = request["interruption"]
    parent = bound(interruption["parent_request"])
    compatible(parent, request)
    q, result = completed_producer(interruption["admission"], "native-elmo-interruption-finalized-v1",
                                   "FINALIZED_REQUIRES_HOST_AND_NATIVE_SUCCESS")
    need(q["parent_request"] == interruption["parent_request"] == result["parent_request"]
         and result["resume_request_fingerprint"] == fingerprint(request)
         and result["checkpoint"] == interruption["checkpoint"]
         and result["run_id"] == request["run_id"] and result["original_total_target"] == request["target_steps"]
         and result["actual_updates"] == 6 and result["diagnostic_scope"] == "production-checkpoint-only",
         "Finalized recovery admission differs from this exact resume request")
    start = interruption["checkpoint"]["global_step"]
    need(type(start) is int and 0 < start < request["target_steps"] and start % 20000 == 0
         and sha(interruption["checkpoint"]["path"]) == interruption["checkpoint"]["sha256"],
         "Highest inspected production checkpoint changed")
    return start, request["target_steps"]


def live_admission(request, request_path):
    """Called once in actual full RUNNING allocation before any training starts."""
    admission(request)
    q, result = completed_producer(request["interruption"]["admission"], "native-elmo-interruption-finalized-v1",
                                   "FINALIZED_REQUIRES_HOST_AND_NATIVE_SUCCESS")
    need(q["resume_native_contract"] == result["resume_native_contract"], "Finalized production native contract differs")
    expected = dict(result["resume_native_contract"], JobId=os.environ["SLURM_JOB_ID"])
    process_identity(expected)
    native = command(["/usr/local/bin/scontrol", "show", "job", expected["JobId"], "-o", "-dd"])
    native_identity(native, expected, "RUNNING")
    current_intent(result["current_intent_path"], request["run_id"], result["current_intent_sha256"])
    need(not Path(request["output"]).exists() and not Path(request["local_logs"]).exists(),
         "Resume attempt would overwrite an existing output/log namespace")
    need(os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Fresh immutable no-requeue attempt required")
    live = quiescence(q["quiescence"], request["run_id"], current_job=os.environ["SLURM_JOB_ID"],
                      current_request_sha=sha(request_path))
    live["production_destination"] = destination_probe(request["output"], os.environ["SLURM_JOB_ID"],
                                                       request["reserved_checkpoint_bytes"])
    live["actual_production_native"] = native
    return live


if __name__ == "__main__":
    import sys
    finalize(sys.argv[1])
