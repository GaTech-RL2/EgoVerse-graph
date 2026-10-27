"""Allocated CPU inspection of retained native Elmo production checkpoints.

The existing ModelWrapper, data module, conditional optimizer checker and
checkpoint binding own semantic restoration. No model or training loop is
reimplemented here. This is not a full-completion verifier or GPU restore proof.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys

from scripts.train.joint.contract import bound, need, sha, write_once


def scheduled_filename(name):
    """Exact production template, with Lightning's configurable metric insertion.

    Census accepts either setting without loading ML. Strict allocated inspection
    then runs the maintained formatter against the actual resolved callback and
    saved epoch; no checkpoint name is inferred solely from this parser.
    """
    match = re.fullmatch(r"(?:epoch_[0-9]{4,}-step_|epoch_epoch=[0-9]{4,}-step_step=)([0-9]{9})\.ckpt", name)
    need(match is not None, "Checkpoint does not use the maintained production filename template")
    return int(match[1])


def verify_source_binding(request, execution):
    """Bind the executing import root and runtime/cache policy to the parent."""
    import importlib.util

    for key in ("source", "source_revision", "source_manifest", "source_manifest_sha256",
                "data_manifest", "data_manifest_sha256", "lambda_runtime", "data_cache"):
        need(execution.get(key) == request.get(key), "Recovery execution changed parent binding: " + key)
    root = Path(request["source"]).resolve(strict=True)
    need(Path(os.environ["PIXEL_SOURCE_ROOT"]).resolve(strict=True) == root,
         "Actual scientific import environment points to another source")
    spec = importlib.util.find_spec("egomimic")
    need(spec is not None and spec.origin is not None and Path(spec.origin).resolve().is_relative_to(root),
         "Actual scientific package escaped hash-bound source")
    for name, module in sys.modules.items():
        if name == "egomimic" or name.startswith("egomimic."):
            filename = getattr(module, "__file__", None)
            need(filename is not None and Path(filename).resolve().is_relative_to(root),
                 "Scientific module imported from another source before admission")


def attempt_chain(request):
    """Enumerate only hash-bound attempts, never guess from run-name globbing."""
    seen, result = set(), []
    identity = {k: request[k] for k in ("run_id", "source_revision", "source_manifest_sha256",
                                       "data_manifest_sha256", "parameters", "target_steps")}
    for _ in range(64):
        need(request["schema"] == "joint-training-v2" and request["phase"] == "full"
             and request["kind"] == "native-subtask-world-model-v1"
             and all(request[k] == v for k, v in identity.items()), "Attempt chain changed scientific identity")
        output = Path(request["output"])
        need(output.is_absolute() and ".." not in output.parts, "Absolute production output required")
        # Scheduled allocations deliberately reuse their output; interruption
        # attempts must use a fresh one. A repeated request hash is a cycle.
        result.append(request)
        previous = request.get("interruption") or request.get("continuation")
        need(not (request.get("interruption") and request.get("continuation")),
             "Ambiguous scheduled and interrupted parent")
        if previous is None:
            return result
        binding = previous["parent_request"]
        need(binding["sha256"] not in seen, "Cycle in immutable production attempt chain")
        seen.add(binding["sha256"])
        request = bound(binding)
    raise ValueError("Production attempt chain exceeds64; explicit audit required")


def checkpoint_census(request):
    """Small stat-only census; unreadable/missing collections fail closed."""
    directories, rows, extras = set(), [], []
    for attempt in attempt_chain(request):
        need(attempt.get("archive_policy") is None,
             "Native interruption v1 requires all checkpoints retained locally; archived recovery is not implemented")
        directory = Path(attempt["output"]) / "checkpoints"
        if str(directory) in directories:
            continue
        directories.add(str(directory))
        need(directory.is_dir() and not directory.is_symlink(), "Retained checkpoint collection missing")
        for path in sorted(directory.iterdir()):
            if path.name == "last.ckpt":
                need(path.is_symlink() and path.resolve(strict=True).parent == directory.resolve(),
                     "last.ckpt is not a retained local milestone link")
                continue
            stat = path.lstat()
            if path.suffix != ".ckpt":
                extras.append(dict(path=str(path), size_bytes=stat.st_size, retained=True))
                continue
            need(path.is_file() and not path.is_symlink(), "Unqualified production checkpoint")
            step = scheduled_filename(path.name)
            need(step > 0 and step % 20000 == 0 and step <= request["target_steps"],
                 "Checkpoint is not an immutable scheduled production milestone")
            rows.append(dict(path=str(path), global_step=step, size_bytes=stat.st_size,
                             mtime_ns=stat.st_mtime_ns, inode=stat.st_ino, device=stat.st_dev))
    need(rows, "No retained production milestone exists; never initialize from smoke/step0")
    return dict(collections=sorted(directories), candidates=sorted(rows, key=lambda r: (-r["global_step"], r["path"])),
                preserved_noncheckpoint_entries=extras)


def check_rng(rng, world):
    """Validate CPU/Python/NumPy states using isolated RNG objects, never global draws."""
    import random
    import numpy as np
    import torch

    need(rng.get("schema") == "mot_rng_state.distributed.v1" and rng.get("world_size") == world
         and len(rng.get("ranks", [])) == world, "Complete all-rank RNG state required")
    for row in rng["ranks"]:
        need(set(row) == {"torch_cpu", "torch_cuda", "numpy", "python", "cuda_scope"}
             and row["cuda_scope"] == "current_device" and len(row["torch_cuda"]) == 1,
             "Rank RNG scope changed")
        need(all(isinstance(x, torch.Tensor) and x.dtype == torch.uint8 and x.ndim == 1
                 and x.numel() > 0 for x in (row["torch_cpu"], row["torch_cuda"][0])),
             "Malformed saved RNG tensor")
        torch.Generator(device="cpu").set_state(row["torch_cpu"])
        np.random.RandomState().set_state(row["numpy"])
        random.Random().setstate(row["python"])


def inspect_checkpoint(path, step, request, cfg, preflight):
    """Use only inside an actual allocated inspection producer."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
    from egomimic.pipeline.construction import checkpoint_construction
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.pl_utils.joint_smoke import weights_digest
    from scripts.train.joint.lr_policy import validate, update_lr, verify_scheduler
    from scripts.train.joint.native_recovery import checkpoint_science, conditional_record, state_digest
    from scripts.train.joint.recovery_check import equal_state

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    need(checkpoint["global_step"] == step, "Filename and actual checkpoint step disagree")
    from lightning.pytorch.callbacks import ModelCheckpoint
    callback = cfg.callbacks.model_checkpoint
    need(callback.filename == "epoch_{epoch:04d}-step_{step:09d}", "Production checkpoint template changed")
    formatter = ModelCheckpoint(filename=callback.filename,
                                auto_insert_metric_name=callback.get("auto_insert_metric_name", True))
    need(Path(formatter.format_checkpoint_name({"epoch": checkpoint["epoch"], "step": step})).name == Path(path).name,
         "Actual maintained checkpoint formatter/epoch does not match the retained filename")
    saved = OmegaConf.create(checkpoint["hyper_parameters"]["config_tree"])
    checkpoint_science(saved, cfg)
    need(saved.run_provenance.run_id == request["run_id"], "Checkpoint W&B identity differs")
    need(checkpoint.get("loops") and checkpoint.get("callbacks"), "Lightning full-state loops/callbacks missing")
    need(all(isinstance(v, torch.Tensor) and v.dtype == torch.float32 and torch.isfinite(v).all()
             for v in checkpoint["state_dict"].values()), "Invalid/nonfinite FP32 model state")
    optimizers, schedulers = checkpoint.get("optimizer_states", []), checkpoint.get("lr_schedulers", [])
    need(len(optimizers) == 1 and optimizers[0]["state"], "Missing actual AdamW state")
    policy = validate(request["lr_policy"])
    need(all(g["lr"] == update_lr(policy, step) and g["weight_decay"] == 0.01
             for g in optimizers[0]["param_groups"]), "Wrong checkpoint LR/weight decay")
    verify_scheduler(policy, schedulers, step)
    sampler = checkpoint["PixelProprioDataModule"]
    need(sampler["schema"] == "mot-sampling-state-v1" and sampler["consumed_batches"] == step
         and sampler["distributed_sampling"]["world_size"] == request["world_size"]
         and sampler["distributed_sampling"]["global_batch_size"] == 32,
         "Wrong production sampling cursor/topology")
    check_rng(checkpoint["mot_rng_state"], request["world_size"])
    data = instantiate(cfg.data)
    data.load_state_dict(sampler)
    need(equal_state(data.state_dict(), sampler), "Complete sampling settings/state did not restore exactly")
    context = data.prepare_context(mode="eval", normalization="prepared", normalizer=None,
                                   restored_state=checkpoint["data_context"])
    need(context.fingerprint() == preflight["provenance"]["data_context_sha256"],
         "Checkpoint context differs from actual CPU/smoke preflight")
    validate_checkpoint_binding(checkpoint, saved, context)
    with checkpoint_construction():
        wrapper = ModelWrapper(config_tree=saved, enable_grad_norm=False).cpu().eval()
        context.bind(wrapper.model)
    wrapper.data_context = context
    need(sum(p.numel() for p in wrapper.model.parameters()) == request["parameters"],
         "Strict reconstructed model size differs")
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    need(equal_state(wrapper.state_dict(), checkpoint["state_dict"]), "Strict weights reload differs")
    optimizer = instantiate(cfg.model.optimizer, params=wrapper.parameters())
    if callable(optimizer):
        optimizer = optimizer()
    optimizer.load_state_dict(optimizers[0])
    scheduler = instantiate(cfg.model.scheduler, optimizer=optimizer)
    if scheduler is not None:
        scheduler.load_state_dict(schedulers[0])
    need(equal_state(optimizer.state_dict(), optimizers[0])
         and equal_state([] if scheduler is None else [scheduler.state_dict()], schedulers),
         "Strict AdamW/scheduler reload differs")
    usage = checkpoint["native_conditional_optimizer_usage"]
    actual = conditional_record(optimizer, wrapper.named_parameters(), usage["parameter_update_counts"], step)
    need(actual == usage, "Conditional parameter-order/count identity differs")
    return dict(global_step=step, strict_full_wrapper_optimizer_scheduler_reload=True,
                conditional_optimizer_usage_sha256=state_digest(usage),
                model_state_sha256=weights_digest(checkpoint["state_dict"]),
                loop_state_sha256=state_digest(checkpoint["loops"]),
                data_context_sha256=context.fingerprint(), sampler_state_sha256=state_digest(sampler),
                rank_rng_sha256=[state_digest(r) for r in checkpoint["mot_rng_state"]["ranks"]],
                all_rank_rng_structure_verified=True, actual_gpu_restore_proven=False,
                scheduled_validation_resume_proven=False)


def inspect_main(request_path):
    """Allocated producer: native authentication/ownership before ML imports/hash loads."""
    from scripts.train.joint.native_interruption import command, native_identity, process_identity, destination_probe

    q = json.loads(Path(request_path).read_text())
    need(q["schema"] == "native-elmo-checkpoint-inspection-v1", "Wrong allocated inspection request")
    need(os.environ.get("SLURM_JOB_ID", "").isdecimal()
         and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
         and not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS"),
         "Checkpoint inspection must run in a fresh owned CPU allocation")
    expected = q["producer_native"]
    process_identity(expected, cpu=True)
    need(expected["JobId"] == os.environ["SLURM_JOB_ID"]
         and expected["NodeList"] == os.environ["SLURMD_NODENAME"]
         and expected["NumCPUs"] == "4" and expected["MinMemoryNode"] == "24G"
         and expected["TimeLimit"] == "00:20:00" and expected["TresPerNode"] == "",
         "Exact4CPU24G20minute bound inspection required")
    native = command(["/usr/local/bin/scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o", "-dd"])
    native_identity(native, expected, "RUNNING")
    request = bound(q["parent_request"])
    need(request["kind"] == "native-subtask-world-model-v1" and request["phase"] == "full",
         "Exact native production parent required")
    from scripts.train.joint.interruption_admission import recovery_phase_admission
    eligibility = recovery_phase_admission(q, request_path)
    verify_source_binding(request, q)
    # The allocated launcher owns current runtime/cache/source closure checks.
    # This producer reopens its actual admitted receipt rather than accepts flags.
    admission_path = Path(q["allocated_prerequisites_path"])
    need(admission_path.is_file() and not admission_path.is_symlink(), "Actual allocated prerequisite path missing")
    admission = json.loads(admission_path.read_text())
    need(admission["job_id"] == os.environ["SLURM_JOB_ID"]
         and admission["request_sha256"] == sha(request_path)
         and admission["status"] == "ACTUAL_PREREQUISITES_AND_NATIVE_ALLOCATION_PASS"
         , "Current allocated prerequisite/claim admission missing")
    native_identity(admission["native"], expected, "RUNNING")
    need(q["required_free_bytes"] >= request["reserved_checkpoint_bytes"],
         "Inspection must reserve the actual complete production checkpoint budget")
    destination = destination_probe(q["production_output"], os.environ["SLURM_JOB_ID"], q["required_free_bytes"])
    output = Path(q["output"])
    need(output.is_absolute() and not output.exists()
         and shutil.disk_usage(output.parent).free >= q["required_free_bytes"] > 0,
         "Fresh inspection output and actual free space required")
    output.mkdir()
    payload = ("native inspection " + os.environ["SLURM_JOB_ID"]).encode()
    probe = output / "write-fsync-readback.probe"
    with probe.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    need(probe.read_bytes() == payload, "Actual write/fsync/readback failed")
    fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    need(sha(request["source_manifest"]) == request["source_manifest_sha256"]
         and sha(request["data_manifest"]) == request["data_manifest_sha256"], "Source/data manifest changed")
    for line in Path(request["source_manifest"]).read_text().splitlines():
        digest, relative = line.split("  ", 1)
        need(sha(Path(request["source"]) / relative) == digest, "Checkpoint-bound source changed")
    from omegaconf import OmegaConf
    from scripts.train.joint.contract import verify_evidence

    _, preflight = verify_evidence(request)
    bound(q["resolved"], False)
    cfg = OmegaConf.load(q["resolved"]["path"])
    census = checkpoint_census(request)
    write_once(output / "census-before.json", census)
    results, selected = [], None
    for row in census["candidates"]:
        if selected is not None and row["global_step"] < selected["global_step"]:
            break
        path = Path(row["path"])
        digest = sha(path)
        try:
            proof = inspect_checkpoint(path, row["global_step"], request, cfg, preflight)
        except (ValueError, KeyError, RuntimeError, EOFError) as error:
            results.append(dict(**row, sha256=digest, status="REJECTED_REQUIRES_DIAGNOSIS",
                                error_type=type(error).__name__, error=str(error)[:4000]))
            continue
        valid = dict(**row, sha256=digest, status="CPU_STRICT_INSPECTION_PASS", proof=proof)
        results.append(valid)
        need(selected is None or selected["sha256"] == digest, "Conflicting healthy checkpoints at the same step")
        selected = selected or valid
    need(checkpoint_census(request) == census, "Checkpoint collection changed during strict inspection")
    # Invalid newer milestones are retained and diagnosed; this tool never
    # silently loops a corrupt/nonfinite scientific failure by falling backward.
    ready = selected is not None and all(r["status"] == "CPU_STRICT_INSPECTION_PASS" for r in results)
    write_once(output / "checkpoint-inspection.json", dict(
        schema="native-elmo-checkpoint-inspection-result-v1",
        status="INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION" if ready else "BLOCKED_CHECKPOINT_DIAGNOSIS",
        job_id=os.environ["SLURM_JOB_ID"], node=os.environ["SLURMD_NODENAME"],
        request_sha256=sha(request_path), parent_request=q["parent_request"], run_id=request["run_id"],
        created_at=datetime.now(timezone.utc).isoformat(), census=census, results=results, selected=selected,
        phase_admission=eligibility,
        source_manifest_sha256=request["source_manifest_sha256"], data_manifest_sha256=request["data_manifest_sha256"],
        highest_healthy_known=ready, actual_gpu_restore_proven=False,
        source_config_context_strictly_checked=ready, write_fsync_readback_sha256=sha(probe),
        production_destination=destination))


if __name__ == "__main__":
    import sys
    inspect_main(sys.argv[1])
