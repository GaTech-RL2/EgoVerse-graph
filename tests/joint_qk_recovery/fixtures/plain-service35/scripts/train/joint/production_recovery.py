"""Actual2+4-update recovery diagnostic from an inspected production milestone.

The maintained training loader and ResumeOptimizerProof perform all restoration
and updates. This executable diagnostic does not admit a production relaunch.
Its extra checkpoints are diagnostic evidence only, never production inputs.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.train.joint.contract import bound, need, sha, write_once, canonical_uuid


def phases(start, target):
    need(type(start) is type(target) is int and 0 < start < target
         and start % 20000 == target % 20000 == 0 and start + 6 < target,
         "Production diagnostic needs a retained nonterminal20k milestone")
    return ((start, start + 2), (start + 2, start + 6))


def phase_config(original, *, output, checkpoint, origin, start, stop, target, world, proof):
    from omegaconf import OmegaConf, open_dict

    need((start, stop) in phases(origin, target) and world == 4,
         "Only exact4-rank2+4 production diagnostic allowed")
    cfg = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(original, resolve=True)))
    science = {k: OmegaConf.to_container(cfg[k], resolve=True) for k in ("model", "data")}
    need(any(c.get("_target_") == "egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage"
             for c in cfg.callbacks.values()), "Maintained native conditional optimizer callback required")
    with open_dict(cfg):
        cfg.ckpt_path = checkpoint
        cfg.paths.output_dir = str(output)
        cfg.val_at_start = cfg.val_at_end = False
        cfg.trainer.max_steps = origin + 6
        cfg.trainer.val_check_interval = 2
        cfg.trainer.limit_val_batches = 100 // world
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.check_val_every_n_epoch = None
        cfg.callbacks.pop("joint_smoke", None)
        cfg.callbacks.pop("allocation_boundary", None)
        cfg.callbacks.pop("wandb_profiler", None)
        cfg.logger.pop("wandb", None)
        cfg.logger.csv.save_dir = str(output / "logs")
        cfg.logger.csv.version = "resume-" + str(start)
        cb = cfg.callbacks.model_checkpoint
        cb.every_n_train_steps, cb.every_n_epochs = 2, 0
        cb.save_top_k, cb.save_last = -1, "link"
        cb.dirpath = str(output / "checkpoints")
        cb.filename = "diagnostic-epoch{epoch:04d}-step{step:09d}"
        cfg.callbacks.resume_proof = dict(
            _target_="scripts.train.joint.recovery_check.ResumeOptimizerProof",
            output=str(proof), world_size=world, start_step=start, stop_step=stop,
            checkpoint=str(checkpoint), multimodal=True)
        if stop == origin + 2:
            cfg.callbacks.allocation_boundary = dict(
                _target_="scripts.train.joint.allocation_boundary.ScheduledAllocationBoundary",
                start_step=start, stop_step=stop, target_step=origin + 6, cadence=2,
                output=str(proof))
        if cfg.evaluator.get("pixel_diagnostics") is not None:
            cfg.evaluator.pixel_diagnostics.log_wandb = False
            cfg.evaluator.pixel_diagnostics.artifact_dir = str(proof / ("validation-" + str(stop)))
    need(all(OmegaConf.to_container(cfg[k], resolve=True) == v for k, v in science.items()),
         "Production diagnostic changed model/loss/data/sampling")
    return cfg


def read_phase_proofs(output, start, target, world, job_id):
    """Reopen exact callback files; reused by an independent finalizer."""
    output = Path(output)
    proof = output / "proof"
    rows = []
    for first, last in phases(start, target):
        ranks = [json.loads((proof / f"resume-{first}-{last}-rank{rank}.json").read_text())
                 for rank in range(world)]
        need(all(row["status"] == "RESUME_PASS" and row["job_id"] == job_id
                 and row["rank"] == rank and row["world_size"] == world
                 and row["start_step"] == first and row["global_step"] == last
                 and row["optimizer_updates"] == row["fp32_losses"] == row["bf16_outputs"] == last - first
                 and row["fp32_gradients_verified"] is True
                 and row["strict_start_weights_optimizer_scheduler_rng_cursor"] is True
                 and row["validation_steps"] == list(range(first + 2, last + 1, 2))
                 for rank, row in enumerate(ranks)), "Actual all-rank production restore proof incomplete")
        rows.append(ranks)
    return rows


def validate_phase_proofs(rows, selected, admission=None):
    from scripts.train.joint.native_recovery import summarize_native_phases, state_digest

    start = selected["global_step"]
    flags = summarize_native_phases(rows, start_step=start, production=True)
    need(all(row["initial_weights_sha256"] == selected["proof"]["model_state_sha256"]
             for row in rows[0]), "Diagnostic did not initialize from inspected production weights")
    for rank, row in enumerate(rows[0]):
        need(row["actual_rank_rng_restore"]["expected_sha256"] == selected["proof"]["rank_rng_sha256"][rank],
             "Diagnostic restored RNG differs from the strictly inspected production checkpoint")
        restored_usage = dict(schema="native-conditional-optimizer-usage-v1", global_step=start,
                              parameter_names=row["conditional_optimizer_record"]["parameter_names"],
                              parameter_update_counts=row["initial_parameter_usage"])
        need(state_digest(restored_usage) == selected["proof"]["conditional_optimizer_usage_sha256"],
             "Diagnostic conditional counters differ from inspected production state")
    # Both sides use the maintained weights_digest. The launch path separately
    # checks the byte-identical complete input checkpoint SHA.
    need(all(a["weights_sha256"] == b["initial_weights_sha256"] for a, b in zip(*rows)),
         "Diagnostic second process did not restore first-process weights")
    devices = [[canonical_uuid(row["gpu_uuid"]) for row in phase] for phase in rows]
    need(len(set(devices[0])) == 4 and devices[0] == devices[1],
         "Actual four distinct GPUs must be unchanged across both diagnostic processes")
    if admission is not None:
        attestation = bound(admission)
        need(all([canonical_uuid(r["gpu_uuid"]) for r in ranks]
                 == [canonical_uuid(r["uuid"]) for r in attestation["gpu_probe"]["ranks"]]
                 for ranks in rows), "Actual restored ranks differ from admitted GPUs")
    return flags


def summarize(request_path, q, original, admission, selected):
    """Publish diagnostic results, leaving host/native completion to a producer."""
    output, start, world = Path(q["output"]), selected["global_step"], original["world_size"]
    proof = output / "proof"
    rows = read_phase_proofs(output, start, original["target_steps"], world, os.environ["SLURM_JOB_ID"])
    flags = validate_phase_proofs(rows, selected, admission)
    boundary = json.loads((proof / f"allocation-step{start + 2:09d}.json").read_text())
    need(boundary["status"] == "PASS" and boundary["checkpoint_saved_after_validation"] is True
         and boundary["start_step"] == start and boundary["global_step"] == start + 2
         and sha(boundary["checkpoint"]["path"]) == boundary["checkpoint"]["sha256"],
         "Actual diagnostic post-validation process boundary missing")
    write_once(output / "production-recovery-diagnostic.json", dict(
        schema="native-elmo-production-recovery-diagnostic-v1",
        status="DIAGNOSTIC_COMPLETE_REQUIRES_HOST_AND_NATIVE_SUCCESS",
        request_sha256=sha(request_path), job_id=os.environ["SLURM_JOB_ID"],
        parent_request=q["parent_request"], inspection=q["inspection"],
        production_checkpoint={k: selected[k] for k in ("path", "sha256", "global_step", "size_bytes")},
        diagnostic_checkpoint=boundary["checkpoint"], actual_updates=6,
        run_id=original["run_id"], world_size=world, resumed_from=start, resumed_to=start + 6,
        no_extra_validation=True, model_data_unchanged=True,
        source_manifest_sha256=original["source_manifest_sha256"],
        data_manifest_sha256=original["data_manifest_sha256"],
        rank_proofs=[dict(path=str(p), sha256=sha(p)) for p in sorted(proof.glob("resume-*-rank*.json"))],
        allocation_attestation=admission, production_relaunch_authorized=False, **flags))


def main(request_path):
    # The hash-bound allocated launcher must perform current runtime/cache/GPU
    # health/claim admission before dispatching this module. Check that actual
    # RUNNING native record again before any Torch/checkpoint import.
    from scripts.train.joint.native_interruption import command, native_identity, process_identity

    q = json.loads(Path(request_path).read_text())
    need(q["schema"] == "native-elmo-production-recovery-request-v1", "Wrong production diagnostic request")
    native = q["producer_native"]
    process_identity(native)
    need(native["JobId"] == os.environ.get("SLURM_JOB_ID")
         and native["NodeList"] == os.environ.get("SLURMD_NODENAME")
         and native["NumCPUs"] == "32" and native["MinMemoryNode"] == "192G"
         and native["TimeLimit"] == "01:00:00" and native["TresPerNode"] == "gres/gpu:h100:4"
         and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Exact fresh4H10032CPU192G1h diagnostic required")
    raw = command(["/usr/local/bin/scontrol", "show", "job", native["JobId"], "-o", "-dd"])
    native_identity(raw, native, "RUNNING")
    prerequisite_path = Path(q["allocated_prerequisites_path"])
    need(prerequisite_path.is_file() and not prerequisite_path.is_symlink(), "Actual allocated prerequisites missing")
    prerequisite = json.loads(prerequisite_path.read_text())
    need(prerequisite["status"] == "ACTUAL_PREREQUISITES_AND_NATIVE_ALLOCATION_PASS"
         and prerequisite["job_id"] == native["JobId"] and prerequisite["request_sha256"] == sha(request_path),
         "Actual producer prerequisite/claim admission differs")
    native_identity(prerequisite["native"], native, "RUNNING")
    original = bound(q["parent_request"])
    need(original["phase"] == "full" and original["kind"] == "native-subtask-world-model-v1"
         and original["world_size"] == 4 and original["parameters"] == 314187064,
         "Exact native Elmo314M parent required")
    from scripts.train.joint.checkpoint_inspection import verify_source_binding
    from scripts.train.joint.interruption_admission import recovery_phase_admission, completed_producer
    eligibility = recovery_phase_admission(q, request_path, scientific_writer=True)
    verify_source_binding(original, q)
    _, inspection = completed_producer(q["inspection_producer"], "native-elmo-checkpoint-inspection-result-v1",
                                       "INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION")
    need(q["inspection"] == q["inspection_producer"]["result"], "GPU diagnostic bound another inspection result")
    need(inspection["parent_request"] == q["parent_request"]
         and inspection["status"] == "INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION"
         and inspection["highest_healthy_known"] is True, "Actual highest-production inspection missing")
    selected = inspection["selected"]
    need(sha(selected["path"]) == selected["sha256"], "Inspected production checkpoint changed")
    phases(selected["global_step"], original["target_steps"])
    bound(q["resolved"], False)
    for line in Path(original["source_manifest"]).read_text().splitlines():
        digest, relative = line.split("  ", 1)
        need(sha(Path(original["source"]) / relative) == digest, "Checkpoint-bound scientific source changed")
    need(sha(original["source_manifest"]) == original["source_manifest_sha256"]
         and sha(original["data_manifest"]) == original["data_manifest_sha256"], "Source/data binding changed")
    output, root = Path(q["output"]), Path(request_path).parent.parent
    need(not output.exists() and q["reserved_checkpoint_bytes"] >= 4 * selected["size_bytes"] + 1000000000
         and shutil.disk_usage(output.parent).free > q["reserved_checkpoint_bytes"],
         "Fresh diagnostic output and retained full-state reserve required")
    output.mkdir()
    write_once(output / "phase-admission.json", dict(request_sha256=sha(request_path), **eligibility))
    (output / "proof").mkdir()
    (output / "configs").mkdir()
    from omegaconf import OmegaConf
    from scripts.train.joint.allocation import admit
    from scripts.train.joint.service_bootstrap import command as service_command

    execution = copy.deepcopy(original)
    execution["operational_service"] = q["operational_service"]
    execution["allocation"] = q.get("allocation")
    execution["lambda_runtime"] = q.get("lambda_runtime", original.get("lambda_runtime"))
    admission = admit(execution, status_root=root / "status", bootstrap=root / "runtime_bootstrap.py")
    cfg = OmegaConf.load(q["resolved"]["path"])
    checkpoint = selected["path"]
    origin = selected["global_step"]
    for start, stop in phases(origin, original["target_steps"]):
        phase = phase_config(cfg, output=output, checkpoint=checkpoint, origin=origin,
                             start=start, stop=stop, target=original["target_steps"], world=4,
                             proof=output / "proof")
        path = output / "configs" / f"resume-{start}-{stop}.yaml"
        with path.open("x") as stream:
            stream.write(OmegaConf.to_yaml(phase, resolve=True))
        argv = [sys.executable, str(root / "runtime_bootstrap.py"), "-m", "torch.distributed.run",
                "--standalone", "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
                *service_command(q, Path(request_path), root / "runtime_bootstrap.py",
                                 ["-m", "egomimic.trainHydra"])[1:],
                "--config-path", str(path.parent), "--config-name", path.stem,
                "hydra.run.dir=" + str(output / f"hydra-{stop}"), "hydra.job.chdir=false"]
        subprocess.run(argv, check=True, cwd=original["source"], env=dict(os.environ, WANDB_MODE="disabled"))
        if stop == origin + 2:
            boundary = json.loads((output / "proof" / f"allocation-step{stop:09d}.json").read_text())
            need(boundary["status"] == "PASS" and boundary["checkpoint_saved_after_validation"],
                 "Actual first diagnostic process did not complete after validation")
            checkpoint = boundary["checkpoint"]["path"]
            need(sha(checkpoint) == boundary["checkpoint"]["sha256"], "Diagnostic boundary checkpoint changed")
    summarize(request_path, q, original, admission, selected)


if __name__ == "__main__":
    main(sys.argv[1])
