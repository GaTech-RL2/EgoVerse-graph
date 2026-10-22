"""Allocated two-update crash-resume diagnostic and native-proof finalizer.

This driver never submits work and never writes the production run.  Its short
save/validation cadence belongs only to a fresh, W&B-free diagnostic directory.
PASS is withheld until separate host and terminal native evidence is bound.
"""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.train.joint.contract import bound, need, sha, write_once


SCHEMA = "joint-crash-recovery-diagnostic-v1"
SOURCE = "535d4fc26c4c85cf57bbd2a776cc79cfec085c4e"


def separate(path, retained):
    path, retained = Path(path), Path(retained)
    need(path.is_absolute() and ".." not in path.parts
         and path != retained and path not in retained.parents
         and retained not in path.parents, "Diagnostic path overlaps retained production")


def validate_request(q):
    need(q["schema"] == SCHEMA, "Wrong crash diagnostic request")
    parent, inspection = bound(q["parent_request"]), bound(q["inspection"])
    need(parent["schema"] == "joint-training-v2" and parent["phase"] == "full"
         and parent["source_revision"] == SOURCE and parent["active_streams"] == ["image"]
         and parent["arm"] == "discrete" and parent["parameters"] == 88178688
         and parent["world_size"] == 4 and parent["global_batch"] == 32
         and parent["per_rank_batch"] == 8
         and parent["lr_policy"] == dict(schedule="constant", peak_lr=1e-4,
                                          warmup_optimizer_steps=0),
         "Diagnostic differs from the exact image-only1e-4 production parent")
    need(inspection["schema"] == "joint-crash-checkpoint-inspection-v1"
         and inspection["status"] == "PASS"
         and inspection["parent_request_sha256"] == q["parent_request"]["sha256"]
         and inspection["parent_resolved_sha256"] == q["resolved"]["sha256"]
         and inspection["source_revision"] == SOURCE
         and inspection["run_id"] == parent["run_id"]
         and inspection["world_size"] == 4 and inspection["global_batch"] == 32
         and inspection["strict_full_wrapper_optimizer_reload"] is True
         and inspection["sampler_all_rank_rng_verified"] is True,
         "Missing exact allocated production checkpoint inspection")
    start = inspection["global_step"]
    need(type(start) is int and start > 0 and start % 20000 == 0
         and start + 2 < parent["target_steps"], "Invalid diagnostic recovery start")
    for key in ("source", "source_revision", "source_manifest", "source_manifest_sha256",
                "source_archive", "source_archive_sha256", "data_manifest",
                "data_manifest_sha256", "data_cache", "lambda_runtime",
                "runtime_localization", "allocation", "world_size", "global_batch",
                "per_rank_batch", "active_streams", "lr_policy", "run_id",
                "parameters", "expert_parameters", "expert_widths"):
        need(q.get(key) == parent.get(key), "Diagnostic changed bound infrastructure/science: " + key)
    need(q.get("operational_service") is not None, "Hash-bound operational service required")
    separate(q["output"], parent["output"])
    separate(q["local_logs"], parent["local_logs"])
    need(q["reserved_checkpoint_bytes"] >= 4_000_000_000, "Diagnostic checkpoint reserve too small")
    need(q["operational_service"]["manifest_sha256"]
         == sha(q["operational_service"]["manifest"]), "Changed diagnostic service manifest")
    need(sha(inspection["checkpoint"]["path"]) == inspection["checkpoint"]["sha256"],
         "Inspected production checkpoint changed")
    bound(q["resolved"], False)
    return parent, inspection


def assert_parent_terminal(q, parent):
    """A diagnostic must not race an active production writer of this identity."""
    failure = bound(q["failure"])
    need(failure["schema"] == "joint-crash-failure-v1"
         and failure["parent_request_sha256"] == q["parent_request"]["sha256"]
         and failure["run_id"] == parent["run_id"]
         and failure["recovery_authorized"] is True and failure["human_retired"] is False,
         "Missing honest diagnostic parent failure/authorization")
    native = dict(item.split("=", 1) for item in bound(failure["native"], False).split() if "=" in item)
    terminal = {"CANCELLED", "FAILED", "PREEMPTED", "TIMEOUT", "NODE_FAIL"}
    need(native["JobId"] == str(failure["job_id"]) and native["JobName"] == parent["job_name"]
         and native["UserId"] == "paphiwetsa3(41003)" and native["JobState"] in terminal,
         "Diagnostic parent is not a terminal owned failure")
    current = subprocess.run(["/usr/local/bin/scontrol", "show", "job", str(failure["job_id"]), "-o"],
                             capture_output=True, text=True, timeout=15)
    if current.returncode == 0:
        state = dict(item.split("=", 1) for item in current.stdout.split() if "=" in item)
        need(state.get("JobState") in terminal, "Parent is still active or completing")
    else:
        need("Invalid job id" in current.stderr, "Parent scheduler read failed")
    queue = subprocess.run(["/usr/local/bin/squeue", "-u", "paphiwetsa3", "-h", "-o", "%i|%j|%T"],
                           capture_output=True, text=True, timeout=15)
    need(queue.returncode == 0, "Diagnostic duplicate writer inspection failed")
    for row in queue.stdout.splitlines():
        job, name, _ = row.split("|", 2)
        need(job == os.environ["SLURM_JOB_ID"] or not name.startswith("joint-image-lr1e4"),
             "Another image-only1e-4 campaign writer is active/pending/completing")


def phase_config(original, q, parent, inspection, *, request_sha256):
    """The model/data remain exact; only explicit diagnostic operations change."""
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(original, resolve=True)))
    original_values = OmegaConf.to_container(cfg, resolve=True)
    preserved = {key: original_values[key] for key in
                 ("model", "data", "seed", "runtime", "joint", "world_model_attention")
                 if key in cfg}
    start, output = inspection["global_step"], str(q["output"])
    with open_dict(cfg):
        cfg.ckpt_path = inspection["checkpoint"]["path"]
        cfg.paths.output_dir = output
        cfg.val_at_start = cfg.val_at_end = False
        cfg.trainer.max_steps = start + 2
        cfg.trainer.val_check_interval = 2
        cfg.trainer.check_val_every_n_epoch = None
        cfg.trainer.num_sanity_val_steps = 0
        cfg.logger.pop("wandb", None)
        need(set(cfg.logger) == {"csv"}, "Diagnostic must use only local CSV logging")
        cfg.logger.csv.save_dir = q["local_logs"]
        cfg.logger.csv.name = "csv"
        cfg.logger.csv.version = f"step{start:09d}"
        cfg.logger.csv.flush_logs_every_n_steps = 1
        cfg.evaluator.pixel_diagnostics.log_wandb = False
        cfg.evaluator.pixel_diagnostics.artifact_dir = output + "/validation-artifacts"
        cfg.callbacks.pop("allocation_boundary", None)
        cfg.callbacks.pop("joint_smoke", None)
        cb = cfg.callbacks.model_checkpoint
        cb.dirpath = output + "/checkpoints"
        cb.filename = "diagnostic-epoch{epoch:04d}-step{step:09d}"
        cb.every_n_train_steps, cb.every_n_epochs = 2, 0
        cb.save_top_k, cb.save_last, cb.save_on_train_epoch_end = -1, "link", False
        cb.train_time_interval = None
        cfg.callbacks.crash_restore_proof = dict(
            _target_="scripts.train.joint.crash_restore.CrashRestoreProof",
            checkpoint=inspection["checkpoint"]["path"],
            inspection_path=q["inspection"]["path"],
            inspection_sha256=q["inspection"]["sha256"], output=output + "/restore-proof",
            world_size=4, start_step=start, proof_updates=2,
            diagnostic=True, validation_interval=2, request_sha256=request_sha256)
        cfg.callbacks.crash_validation_proof = dict(
            _target_="scripts.train.joint.crash_smoke.diagnostic_validation_callback",
            output=output + "/validation-proof", start_step=start,
            stop_step=start + 2, world_size=4)
    configured_values = OmegaConf.to_container(cfg, resolve=True)
    for key, value in preserved.items():
        need(configured_values[key] == value,
             "Diagnostic changed scientific configuration: " + key)
    need(cfg.trainer.devices == cfg.data.distributed_world_size == 4
         and cfg.data.batch_size == 8 and cfg.trainer.precision == "bf16-mixed"
         and cfg.model.optimizer.lr == 1e-4 and cfg.model.optimizer.weight_decay == .01
         and cfg.model.scheduler is None, "Diagnostic topology/optimizer changed")
    return cfg


def diagnostic_validation_callback(*, output, start_step, stop_step, world_size):
    """Hydra factory keeps ML imports behind actual allocated runtime admission."""
    import torch
    from lightning.pytorch.callbacks import Callback

    class DiagnosticValidationProof(Callback):
        def __init__(self):
            need(world_size == 4 and stop_step == start_step + 2,
                 "Exactly two resumed optimizer updates required")
            self.starts = []

        def on_validation_start(self, trainer, pl_module):
            need(not trainer.sanity_checking and trainer.global_step == stop_step,
                 "Diagnostic repeated boundary validation or ran an extra event")
            self.starts.append(trainer.global_step)

        def on_train_end(self, trainer, pl_module):
            need(trainer.world_size == world_size and trainer.global_step == stop_step
                 and self.starts == [stop_step]
                 and trainer.datamodule.consumed_batches == stop_step
                 and pl_module.evaluator.has_completed_validation(stop_step)
                 and pl_module.evaluator.global_validation_batches == 100,
                 "Diagnostic did not consume exactly100 scheduled validation anchors")
            metrics = {str(k): float(v) for k, v in trainer.callback_metrics.items()
                       if str(k).startswith("val/libero_gripper/")}
            need(set(metrics) == {"val/libero_gripper/forecast/image_normalized_mse",
                                  "val/libero_gripper/completion/image_normalized_mse"}
                 and all(math.isfinite(v) for v in metrics.values()),
                 "Missing finite actual image validation metrics")
            need(torch.distributed.is_initialized()
                 and torch.distributed.get_world_size() == world_size,
                 "Actual four-rank validation required")
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            write_once(directory / f"rank{trainer.global_rank}.json", dict(
                schema="joint-crash-diagnostic-validation-rank-v1", status="PASS",
                job_id=os.environ["SLURM_JOB_ID"], rank=trainer.global_rank,
                world_size=world_size, start_step=start_step, global_step=stop_step,
                scheduled_validation_steps=self.starts, scheduled_validation_anchors=100,
                local_validation_anchors=pl_module.evaluator._validation_batches,
                metrics=metrics, no_replayed_boundary_validation=True))

    return DiagnosticValidationProof()


def verify_validation(q, inspection, job_id):
    output, stop = Path(q["output"]), inspection["global_step"] + 2
    rows = []
    for rank in range(4):
        record = json.loads((output / "validation-proof" / f"rank{rank}.json").read_text())
        need(record["schema"] == "joint-crash-diagnostic-validation-rank-v1"
             and record["status"] == "PASS" and record["job_id"] == job_id
             and record["rank"] == rank and record["world_size"] == 4
             and record["global_step"] == stop
             and record["scheduled_validation_steps"] == [stop]
             and record["scheduled_validation_anchors"] == 100
             and record["local_validation_anchors"] == 25
             and record["no_replayed_boundary_validation"] is True,
             "Missing actual per-rank scheduled validation proof")
        rows.append(record)
    path = Path(q["local_logs"]) / "csv" / f"step{inspection['global_step']:09d}" / "metrics.csv"
    events = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            metrics = {key: float(value) for key, value in row.items()
                       if key.startswith("val/libero_gripper/") and value}
            if metrics:
                raw_step = float(row["step"])
                need(raw_step.is_integer() and all(math.isfinite(v) for v in metrics.values()),
                     "Invalid diagnostic validation history")
                events.setdefault(int(raw_step), {}).update(metrics)
    need(len(events) == 1 and next(iter(events)) in (stop-1, stop)
         and len(next(iter(events.values()))) == 2,
         "Diagnostic CSV contains missing or replayed validation events")
    return rows


def verify_final_checkpoint(q, parent, inspection, cfg):
    """Strictly reload the diagnostic output, preserving every production byte."""
    import torch
    from scripts.train.joint.crash_inspect import strict_cpu_restore

    directory = Path(q["output"]) / "checkpoints"
    files = [path for path in directory.iterdir() if path.name != "last.ckpt"]
    stop = inspection["global_step"] + 2
    need(len(files) == 1 and files[0].is_file() and not files[0].is_symlink()
         and files[0].name.endswith(f"step={stop:09d}.ckpt"),
         "Diagnostic checkpoint cadence repeated or skipped a save")
    checkpoint_path = files[0]
    need((directory / "last.ckpt").is_symlink()
         and (directory / "last.ckpt").resolve() == checkpoint_path.resolve(),
         "Diagnostic last alias changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    need(checkpoint["global_step"] == stop and checkpoint.get("lr_schedulers", []) == []
         and all(t.dtype == torch.float32 and torch.isfinite(t).all()
                 for t in checkpoint["state_dict"].values())
         and checkpoint["PixelProprioDataModule"]["consumed_batches"] == stop,
         "Invalid final diagnostic full state")
    states = checkpoint["optimizer_states"]
    need(len(states) == 1 and states[0]["state"]
         and all(group["lr"] == 1e-4 and group["weight_decay"] == .01
                 for group in states[0]["param_groups"])
         and all(float(value["step"]) == stop and all(
             value[key].dtype == torch.float32 and torch.isfinite(value[key]).all()
             for key in ("exp_avg", "exp_avg_sq")) for value in states[0]["state"].values()),
         "Diagnostic optimizer did not continue in finite FP32")
    rng = checkpoint["mot_rng_state"]
    need(rng["schema"] == "mot_rng_state.distributed.v1" and rng["world_size"] == 4
         and len(rng["ranks"]) == 4, "Diagnostic lost all-rank RNG state")
    restored = strict_cpu_restore(checkpoint, cfg, parent, inspection["data_context_sha256"])
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from egomimic.pipeline.construction import checkpoint_construction
    from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
    from egomimic.pl_utils.pl_model import ModelWrapper
    data = instantiate(cfg.data)
    context = data.prepare_context(mode="eval", normalization="prepared", normalizer=None,
                                   restored_state=checkpoint["data_context"])
    model_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["config_tree"])
    validate_checkpoint_binding(checkpoint, model_cfg, context)
    with checkpoint_construction():
        wrapper = ModelWrapper(config_tree=model_cfg, enable_grad_norm=False).cuda().eval()
        context.bind(wrapper.model)
    wrapper.data_context = context
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    sample = next(iter(data.val_dataloader()))["libero_gripper"]
    for stream in sample["world"].values():
        stream["value"] = torch.where(stream["observed"][..., None], stream["value"], float("nan"))
    groups = wrapper.model.process_batch_for_training({"libero_gripper": sample})
    wrapper.model.device = next(wrapper.parameters()).device
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = wrapper.model.forward_eval(groups)["libero_gripper"]["image_proprio_prediction"]
    need(set(predictions) == {"image"}, "Diagnostic sampler changed active streams")
    for name, value in predictions.items():
        stream = groups["libero_gripper"]["world"][name]
        need(value.dtype == torch.float32 and value.shape == stream["value"].shape
             and torch.isfinite(value).all()
             and torch.equal(value[stream["observed"]], stream["value"][stream["observed"]]),
             "Restored diagnostic hidden-label-NaN sample failed")
    return dict(path=str(checkpoint_path), sha256=sha(checkpoint_path),
                strict_full_wrapper_optimizer_reload=restored["strict_full_wrapper_optimizer_reload"],
                sample_finite=True, hidden_query_labels_nan=True, sample_steps=20)


def run(request_path, q):
    from scripts.train.joint.crash_inspect import verify_allocation

    verify_allocation(os.environ.get("SLURM_JOB_ID", ""), q["lambda_runtime"]["node"])
    parent, inspection = validate_request(q)
    assert_parent_terminal(q, parent)
    from scripts.train.joint.contract import verify_evidence
    verify_evidence(parent)
    for line in Path(parent["source_manifest"]).read_text().splitlines():
        expected, relative = line.split("  ", 1)
        need(sha(Path(parent["source"]) / relative) == expected,
             "Frozen diagnostic source changed: " + relative)
    need(sha(parent["source_manifest"]) == parent["source_manifest_sha256"]
         and sha(parent["data_manifest"]) == parent["data_manifest_sha256"],
         "Frozen diagnostic source/data manifest changed")
    from scripts.train.joint.lambda_runtime import filesystem_recovery_reserve
    filesystem_recovery_reserve(q)
    output = Path(q["output"])
    need(not output.exists() and shutil.disk_usage(output.parent).free > q["reserved_checkpoint_bytes"],
         "Diagnostic output collision or insufficient space")
    root = request_path.parent.parent
    from scripts.train.joint.allocation import admit
    allocation = admit(q, status_root=root / "status", bootstrap=root / "runtime_bootstrap.py")
    if allocation is None:
        from gpu_health_set import admit_gpu_set
        admit_gpu_set(root / "status" / f"crash-diagnostic-{os.environ['SLURM_JOB_ID']}-gpu.json", 4)
    from omegaconf import OmegaConf
    original = OmegaConf.load(q["resolved"]["path"])
    cfg = phase_config(original, q, parent, inspection, request_sha256=sha(request_path))
    output.mkdir()
    probe = output / "diagnostic-storage-probe.json"
    payload = json.dumps(dict(job_id=os.environ["SLURM_JOB_ID"], request_sha256=sha(request_path))).encode()
    with probe.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    need(probe.read_bytes() == payload, "Diagnostic persistent fsync/readback failed")
    directory = output / "config"
    directory.mkdir()
    config_path = directory / "diagnostic.yaml"
    with config_path.open("x") as stream:
        stream.write(OmegaConf.to_yaml(cfg, resolve=True))
    from scripts.train.joint.service_bootstrap import command
    bootstrap = root / "runtime_bootstrap.py"
    subprocess.run([
        sys.executable, str(bootstrap), "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
        *command(q, request_path, bootstrap, ["-m", "egomimic.trainHydra"])[1:],
        "--config-path", str(directory), "--config-name", "diagnostic",
        "hydra.run.dir=" + str(output / "hydra"), "hydra.job.chdir=false",
        "+run_provenance={" + ",".join(key + ":" + json.dumps(value)
            for key, value in dict(source_revision=parent["source_revision"],
                source_manifest_sha256=parent["source_manifest_sha256"],
                data_manifest_sha256=parent["data_manifest_sha256"],
                data_context_sha256=inspection["data_context_sha256"], run_id=parent["run_id"]).items()) + "}",
    ], check=True, cwd=parent["source"])
    from scripts.train.joint.crash_restore import assert_actual_restore
    execution = dict(q, crash_recovery=dict(inspection=q["inspection"]))
    assert_actual_restore(execution, diagnostic=True, expected_request_sha256=sha(request_path))
    validation = verify_validation(q, inspection, os.environ["SLURM_JOB_ID"])
    final = verify_final_checkpoint(q, parent, inspection, original)
    proof_paths = sorted((output / "restore-proof").glob("*.json"))
    proof_paths += sorted((output / "validation-proof").glob("*.json"))
    need(len(proof_paths) == 12, "Expected all startup/update and diagnostic validation rank proofs")
    write_once(output / "crash-recovery-smoke.candidate.json", dict(
        schema="joint-crash-recovery-smoke-candidate-v1", status="ACTUAL_DIAGNOSTIC_PASS_NATIVE_PENDING",
        request_sha256=sha(request_path), job_id=os.environ["SLURM_JOB_ID"],
        request=dict(path=str(request_path), sha256=sha(request_path)), job_name=q["job_name"],
        source_revision=SOURCE, parent_request_sha256=q["parent_request"]["sha256"],
        inspection_sha256=q["inspection"]["sha256"], checkpoint_sha256=inspection["checkpoint"]["sha256"],
        service_manifest_sha256=q["operational_service"]["manifest_sha256"],
        world_size=4, global_batch=32, all_rank_full_state_restored=True,
        finite_optimizer_updates=2, scheduled_validation_anchors=100,
        no_replayed_boundary_validation_or_save=True, production_science_unchanged=True,
        final_diagnostic_checkpoint=final, validation_ranks=validation,
        sample_finite=final["sample_finite"], hidden_query_labels_nan=True, sample_steps=20,
        rank_proofs=[dict(path=str(path), sha256=sha(path)) for path in proof_paths],
        parent_checkpoint_modified=False, production_wandb_written=False))


def finalize(request_path, q, *, native_binding, host_binding, job_id):
    """Only a terminal successful native job and host zero can produce PASS."""
    from scripts.train.joint.crash_inspect import verify_allocation
    # Finalization may rehash a full checkpoint, so it too must run on compute.
    verify_allocation(os.environ.get("SLURM_JOB_ID", ""), q["lambda_runtime"]["node"])
    need(q["schema"] == SCHEMA and str(job_id).isdigit(), "Wrong diagnostic finalization")
    parent, inspection = validate_request(q)
    candidate_path = Path(q["output"]) / "crash-recovery-smoke.candidate.json"
    candidate = json.loads(candidate_path.read_text())
    native = dict(item.split("=", 1) for item in bound(native_binding, False).split() if "=" in item)
    need(candidate["schema"] == "joint-crash-recovery-smoke-candidate-v1"
         and candidate["status"] == "ACTUAL_DIAGNOSTIC_PASS_NATIVE_PENDING"
         and candidate["request_sha256"] == sha(request_path)
         and candidate["job_id"] == str(job_id)
         and native["JobId"] == str(job_id) and native["JobState"] == "COMPLETED"
         and native["ExitCode"] == "0:0" and native["UserId"] == "paphiwetsa3(41003)"
         and native["NodeList"] == q["lambda_runtime"]["node"]
         and native["JobName"] == q["job_name"]
         and native["Requeue"] == "0" and native["Restarts"] == "0"
         and bound(host_binding, False).strip() == "0",
         "Diagnostic host/native success is not proved")
    need(candidate["source_revision"] == parent["source_revision"]
         and candidate["parent_request_sha256"] == q["parent_request"]["sha256"]
         and candidate["inspection_sha256"] == q["inspection"]["sha256"]
         and candidate["checkpoint_sha256"] == inspection["checkpoint"]["sha256"]
         and candidate["service_manifest_sha256"] == q["operational_service"]["manifest_sha256"]
         and candidate["world_size"] == 4 and candidate["global_batch"] == 32
         and candidate["all_rank_full_state_restored"] is True
         and candidate["finite_optimizer_updates"] == 2
         and candidate["scheduled_validation_anchors"] == 100
         and candidate["no_replayed_boundary_validation_or_save"] is True
         and candidate["production_science_unchanged"] is True
         and candidate["sample_finite"] is True and candidate["hidden_query_labels_nan"] is True
         and candidate["sample_steps"] == 20
         and candidate["parent_checkpoint_modified"] is False
         and candidate["production_wandb_written"] is False,
         "Actual diagnostic candidate is incomplete or bound to other inputs")
    output, start = Path(q["output"]), inspection["global_step"]
    expected_proofs = {str(output / "restore-proof" / f"restore-step{start:09d}-rank{rank}.json")
                       for rank in range(4)}
    expected_proofs |= {str(output / "restore-proof" / f"restore-validation-rank{rank}.json")
                        for rank in range(4)}
    expected_proofs |= {str(output / "validation-proof" / f"rank{rank}.json") for rank in range(4)}
    need(len(candidate["rank_proofs"]) == 12
         and {item["path"] for item in candidate["rank_proofs"]} == expected_proofs,
         "Missing exact diagnostic rank proof inventory")
    for item in candidate["rank_proofs"]:
        row = bound(item)
        need(row["status"] == "PASS" and str(row["job_id"]) == str(job_id)
             and row["world_size"] == 4 and row["rank"] in range(4)
             and row["start_step"] == start and row["global_step"] == start + 2,
             "Diagnostic rank proof changed or belongs to another allocation")
    final = candidate["final_diagnostic_checkpoint"]
    need(Path(final["path"]).parent == output / "checkpoints"
         and final["strict_full_wrapper_optimizer_reload"] is True
         and sha(final["path"]) == final["sha256"], "Diagnostic final full checkpoint changed")
    from scripts.train.joint.crash_restore import assert_actual_restore
    execution = dict(q, crash_recovery=dict(inspection=q["inspection"]))
    assert_actual_restore(execution, expected_job_id=str(job_id), diagnostic=True,
                          expected_request_sha256=sha(request_path))
    verify_validation(q, inspection, str(job_id))
    record = {key: value for key, value in candidate.items()
              if key not in ("schema", "status", "request_sha256")}
    record.update(schema="joint-crash-recovery-smoke-v1", status="PASS",
                  request_sha256=sha(request_path), native=native_binding, host_exit=host_binding,
                  request=dict(path=str(request_path), sha256=sha(request_path)), job_name=q["job_name"],
                  candidate=dict(path=str(candidate_path), sha256=sha(candidate_path)))
    write_once(Path(q["output"]) / "crash-recovery-smoke.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--native-proof", type=Path)
    parser.add_argument("--native-sha256")
    parser.add_argument("--host-exit", type=Path)
    parser.add_argument("--host-exit-sha256")
    args = parser.parse_args(argv)
    q = json.loads(args.request.read_text())
    if args.finalize:
        need(all((args.job_id, args.native_proof, args.native_sha256,
                  args.host_exit, args.host_exit_sha256)), "Exact finalization bindings required")
        finalize(args.request, q, job_id=args.job_id,
                 native_binding=dict(path=str(args.native_proof), sha256=args.native_sha256),
                 host_binding=dict(path=str(args.host_exit), sha256=args.host_exit_sha256))
    else:
        need(not any((args.job_id, args.native_proof, args.native_sha256,
                      args.host_exit, args.host_exit_sha256)), "Unexpected finalization flags")
        run(args.request, q)


if __name__ == "__main__":
    main()
