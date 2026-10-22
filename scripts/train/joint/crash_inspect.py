"""Inspect an actual highest production checkpoint on allocated compute only.

This is read-only for source/data/checkpoints. The sole write is an exclusive
inspection receipt. No training, GPU allocation, scheduler mutation, RNG draw,
validation or checkpoint save is performed. A PASS verifies saved/restored CPU
state; the actual four-rank recovery smoke must still certify live restoration.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
from collections.abc import Mapping

from scripts.train.joint.contract import bound, need, sha, write_once


SCHEMA = "joint-crash-checkpoint-inspection-v1"
FROZEN_SOURCE = "535d4fc26c4c85cf57bbd2a776cc79cfec085c4e"
RUN_ID = "mot-shared-libero-image-discrete-lr1e-4-constant-s42-300k-20261021-v1"
MILESTONE = re.compile(r"epoch_epoch=(\d+)-step_step=(\d{9})\.ckpt")


def state_digest(value):
    """Content digest preserving Python container/scalar and tensor identities."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            need(tensor.layout == torch.strided, "Unsupported sparse state tensor")
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray\0" + item.dtype.str.encode() + b"\0")
            digest.update(json.dumps(list(item.shape)).encode() + b"\0")
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                add(key)
                add(item[key])
            digest.update(b"enddict\0")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                add(child)
            digest.update(b"endsequence\0")
        elif item is None or type(item) in (str, bool, int, float):
            if type(item) is float:
                need(math.isfinite(item), "Nonfinite scalar in saved state")
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(json.dumps(item, allow_nan=False).encode() + b"\0")
        else:
            raise ValueError("Unsupported state value: " + type(item).__name__)

    add(value)
    return digest.hexdigest()


def verify_allocation(job_id, node, *, native_command="/usr/local/bin/scontrol", env=None,
                      uid=None, native_reader=None):
    """Check native ownership before importing any ML library or reading a ckpt."""
    env = os.environ if env is None else env
    uid = os.getuid() if uid is None else uid
    need(str(job_id).isdigit() and re.fullmatch(r"gpu\d+", node), "Unsafe allocation identity")
    need(uid == 41003 and env.get("SLURM_JOB_ID") == str(job_id)
         and env.get("SLURMD_NODENAME") == node
         and env.get("SLURM_RESTART_COUNT", "0") == "0",
         "Inspector requires its exact fresh allocated compute job/node/owner")
    if native_reader is None:
        result = subprocess.run([native_command, "show", "job", str(job_id), "-o"],
                                text=True, capture_output=True, timeout=12, check=True)
        text = result.stdout
    else:
        text = native_reader()
    tokens = dict(field.split("=", 1) for field in text.split() if "=" in field)
    need(tokens.get("JobId") == str(job_id) and tokens.get("JobState") == "RUNNING"
         and tokens.get("UserId") == "paphiwetsa3(41003)"
         and tokens.get("NodeList") == node and tokens.get("NumNodes") == "1"
         and tokens.get("Restarts") == "0", "Native inspector allocation identity mismatch")
    return dict(job_id=str(job_id), node=node, uid=uid, native_sha256=hashlib.sha256(text.encode()).hexdigest(),
                native_state="RUNNING", native_command=native_command)


def retained_attempts(parent):
    """Follow only hash-bound crash parents, assigning each its owned step range."""
    identity = ("run_id", "source_revision", "target_steps", "world_size", "global_batch",
                "per_rank_batch", "active_streams", "arm", "parameters", "lr_policy")
    attempts, seen = [], set()
    current, through = parent, parent["target_steps"]
    for _ in range(32):
        output = Path(current["output"])
        need(output.is_absolute() and ".." not in output.parts and str(output) not in seen,
             "Unsafe or repeated crash history output/cycle")
        need(all(current.get(key) == parent.get(key) for key in identity),
             "Crash ancestor has a foreign scientific identity")
        seen.add(str(output))
        recovery = current.get("crash_recovery")
        need(current.get("continuation") is None,
             "Checkpoint inspector does not mix scheduled and crash ancestry")
        if recovery is None:
            attempts.append(dict(output=output, start_step=0, through_step=through,
                                 resume_checkpoint=None))
            return attempts[::-1]
        need(recovery.get("schema") == "joint-crash-recovery-v1", "Unknown crash parent binding")
        prior = bound(recovery["parent_request"])
        receipt = bound(recovery["inspection"])
        need(receipt.get("schema") == SCHEMA and receipt.get("status") == "PASS"
             and receipt.get("parent_request_sha256") == recovery["parent_request"]["sha256"]
             and receipt.get("run_id") == parent["run_id"]
             and receipt.get("source_revision") == parent["source_revision"],
             "Crash history inspection belongs to another parent")
        start = receipt.get("global_step")
        need(type(start) is int and 0 < start <= through and start % 20000 == 0,
             "Invalid crash history step range")
        checkpoint = receipt.get("checkpoint", {})
        need(Path(checkpoint.get("path", "")).is_absolute()
             and re.fullmatch(r"[a-f0-9]{64}", checkpoint.get("sha256", "")),
             "Invalid ancestor checkpoint content binding")
        attempts.append(dict(output=output, start_step=start, through_step=through,
                             resume_checkpoint=dict(checkpoint, global_step=start)))
        current, through = prior, start
    raise ValueError("Crash checkpoint history exceeds32 attempts")


def inventory_highest(directory, parent):
    """Inspect bounded retained lineage; newest corruption never selects older state.

    A failed child may own no checkpoint before its next20k event. Its missing or
    empty directory is valid only if prior retained history contains exactly its
    restored milestone. No ancestor may introduce a file beyond its owned cutoff.
    """
    directory = Path(directory)
    need(directory.is_absolute() and directory == Path(parent["output"])/"checkpoints",
         "Checkpoint inventory must be the exact failed parent's bounded directory")
    rows, steps, indexed = [], set(), {}
    attempts = retained_attempts(parent)
    for attempt in attempts:
        output, start, through = attempt["output"], attempt["start_step"], attempt["through_step"]
        selected_parent = attempt["resume_checkpoint"]
        if selected_parent is not None:
            path = selected_parent["path"]
            need(path in indexed and indexed[path]["global_step"] == start,
                 "Crash ancestor checkpoint is outside prior retained history")
            known = indexed[path].get("expected_sha256")
            need(known is None or known == selected_parent["sha256"],
                 "Conflicting historical checkpoint content bindings")
            indexed[path]["expected_sha256"] = selected_parent["sha256"]
        current = output/"checkpoints"
        need(not output.is_symlink() and not current.is_symlink(),
             "Symlink checkpoint history directory")
        if not current.exists():
            need(start > 0 and steps and max(steps) == start,
                 "Missing checkpoint directory without a retained ancestor milestone")
            continue
        need(current.is_dir(), "Invalid checkpoint history directory")
        local = []
        for path in current.iterdir():
            if path.name == "last.ckpt":
                continue
            need(path.name.endswith(".ckpt"), "Unexpected file in production checkpoint directory")
            match = MILESTONE.fullmatch(path.name)
            need(match and path.is_file() and not path.is_symlink(), "Invalid immutable checkpoint candidate")
            step = int(match[2])
            need(start < step <= through and step < parent["target_steps"]
                 and step % 20000 == 0 and step not in steps,
                 "Invalid, duplicate, unscheduled or newer-than-cutoff production milestone")
            steps.add(step)
            status = path.stat()
            row = dict(path=str(path), global_step=step, epoch=int(match[1]),
                       bytes=status.st_size, mtime_ns=status.st_mtime_ns,
                       ctime_ns=status.st_ctime_ns, inode=status.st_ino, device=status.st_dev,
                       owner_output=str(output), owner_start_step=start, owner_through_step=through)
            rows.append(row); local.append(row); indexed[str(path)] = row
        latest = current/"last.ckpt"
        if not local:
            need(start > 0 and steps and max(steps) == start and not latest.exists()
                 and not latest.is_symlink(), "Empty attempt has no retained ancestor or unexpected last alias")
            continue
        local_highest = max(local, key=lambda row: row["global_step"])
        need(latest.is_symlink()
             and latest.resolve(strict=True) == Path(local_highest["path"]).resolve(strict=True),
             "last.ckpt does not bind the highest immutable milestone")
    need(rows, "No complete production checkpoint is available")
    rows.sort(key=lambda row: row["global_step"])
    selected = rows[-1]
    need(set(steps) == set(range(20000, selected["global_step"]+1, 20000)),
         "Missing retained production milestone in crash history")
    need(all(row["bytes"] > 0 for row in rows),
         "Newest checkpoint is empty or retained checkpoint is empty; never fall back silently")
    return Path(selected["path"]), rows


def verify_inventory_stable(directory, parent, selected, inventory):
    """A new/changed milestone during slow CPU restoration invalidates selection."""
    current_selected, current_inventory = inventory_highest(directory, parent)
    need(current_selected == selected and current_inventory == inventory,
         "Checkpoint inventory changed during inspection; newest candidates are unresolved")


def validate_loop_state(checkpoint, step):
    """Validate saved Lightning counters while refusing to predict replay behavior."""
    loops = checkpoint.get("loops")
    need(isinstance(loops, dict) and isinstance(loops.get("fit_loop"), dict),
         "Missing Lightning fit loop state")
    try:
        raw = json.dumps(loops, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Malformed non-JSON Lightning loop state") from error
    need(len(raw) <= 128_000, "Malformed or oversized Lightning loop state")
    fit = loops["fit_loop"]
    try:
        optimizer = fit["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]
        batch = fit["epoch_loop.batch_progress"]["total"]
        stepped = fit["epoch_loop.state_dict"]["_batches_that_stepped"]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown Lightning loop counter structure") from error
    need(type(optimizer.get("completed")) is int and optimizer["completed"] == step
         and type(optimizer.get("ready")) is int and optimizer["ready"] == step,
         "Saved optimizer loop counter disagrees with checkpoint step")
    need(type(stepped) is int and stepped in (step-1, step)
         and all(type(batch.get(key)) is int for key in ("completed", "processed", "ready", "started"))
         and batch["completed"] in (step-1, step)
         and batch["processed"] == batch["ready"] == batch["started"] == step,
         "Malformed Lightning training batch counters")
    return dict(raw=loops, sha256=hashlib.sha256(raw.encode()).hexdigest(),
                optimizer_steps_completed=optimizer["completed"], batches_that_stepped=stepped,
                batch_completed=batch["completed"],
                resume_loop_policy="reject_replayed_boundary_validation",
                loop_training_resume_supported=False,
                interpretation="Saved counters verified; actual recovery smoke must prove no boundary validation/save replay")


def validate_rng(rng, world):
    import numpy as np
    import torch

    need(isinstance(rng, dict) and rng.get("schema") == "mot_rng_state.distributed.v1"
         and rng.get("world_size") == world and isinstance(rng.get("ranks"), list)
         and len(rng["ranks"]) == world, "Missing exact distributed RNG payload")
    digests = []
    for state in rng["ranks"]:
        need(isinstance(state, dict)
             and set(state) == {"torch_cpu", "torch_cuda", "numpy", "python", "cuda_scope"}
             and state["cuda_scope"] == "current_device", "Malformed rank RNG schema")
        cpu, cuda = state["torch_cpu"], state["torch_cuda"]
        need(isinstance(cpu, torch.Tensor) and cpu.dtype == torch.uint8 and cpu.ndim == 1
             and cpu.numel() > 0 and isinstance(cuda, list) and len(cuda) == 1
             and isinstance(cuda[0], torch.Tensor) and cuda[0].dtype == torch.uint8
             and cuda[0].ndim == 1 and cuda[0].numel() > 0, "Malformed Torch rank RNG states")
        # Validate using private CPU/NumPy/Python generators, preserving global RNG.
        try:
            torch.Generator(device="cpu").set_state(cpu)
            np.random.RandomState().set_state(state["numpy"])
            random.Random().setstate(state["python"])
        except (RuntimeError, ValueError, TypeError) as error:
            raise ValueError("Unrestorable saved CPU/NumPy/Python RNG") from error
        digests.append(state_digest(state))
    return digests


def validate_payload(checkpoint, parent, resolved, *, expected_context_sha256, context_fingerprint):
    """Structural/content inspection shared by the real CLI and real-tensor CPU tests."""
    import torch

    need(isinstance(checkpoint, dict), "Checkpoint must be a full-state mapping")
    step = checkpoint.get("global_step")
    need(type(step) is int and 0 < step < parent["target_steps"] and step % 20000 == 0,
         "Not a recoverable scheduled production checkpoint")
    need(type(checkpoint.get("epoch")) is int and checkpoint["epoch"] >= 0, "Invalid checkpoint epoch")
    model_cfg = checkpoint.get("hyper_parameters", {}).get("config_tree")
    from omegaconf import OmegaConf
    if OmegaConf.is_config(model_cfg):
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True, throw_on_missing=True)
    need(isinstance(model_cfg, Mapping) and model_cfg.get("model") == resolved["model"],
         "Saved model configuration differs from parent")
    provenance = model_cfg.get("run_provenance", {})
    expected = {key: parent[key] for key in ("source_revision", "source_manifest_sha256", "data_manifest_sha256", "run_id")}
    expected["data_context_sha256"] = expected_context_sha256
    need(all(provenance.get(key) == value for key, value in expected.items()),
         "Saved run/source/data provenance differs from parent")
    need(isinstance(checkpoint.get("data_context"), dict)
         and context_fingerprint(checkpoint["data_context"]) == expected_context_sha256
         and checkpoint.get("inference_binding", {}).get("data_context_sha256") == expected_context_sha256,
         "Saved data context differs from its parent binding")
    state = checkpoint.get("state_dict")
    need(isinstance(state, dict) and state and all(isinstance(key, str) and isinstance(value, torch.Tensor)
         and value.dtype == torch.float32 and value.layout == torch.strided
         and bool(torch.isfinite(value).all()) for key, value in state.items()),
         "Nonfinite, non-FP32 or invalid stored wrapper weights")
    optimizers = checkpoint.get("optimizer_states")
    need(isinstance(optimizers, list) and len(optimizers) == 1
         and isinstance(optimizers[0], dict) and optimizers[0].get("state")
         and optimizers[0].get("param_groups"), "Missing full AdamW state")
    need(parent["lr_policy"] == {"schedule": "constant", "peak_lr": 1e-4, "warmup_optimizer_steps": 0}
         and resolved["model"]["scheduler"] is None
         and checkpoint.get("lr_schedulers", []) == [], "Unexpected LR schedule")
    optimizer = optimizers[0]
    ids = []
    for group in optimizer["param_groups"]:
        need(group.get("lr") == 1e-4 and group.get("weight_decay") == .01
             and isinstance(group.get("params"), list) and group["params"], "Wrong saved optimizer groups/LR")
        ids.extend(group["params"])
    need(len(ids) == len(set(ids)) and set(ids) == set(optimizer["state"]),
         "Missing, duplicated or unmatched optimizer parameter state")
    for saved in optimizer["state"].values():
        count = saved.get("step")
        if isinstance(count, torch.Tensor):
            need(count.numel() == 1 and bool(torch.isfinite(count).all()), "Invalid optimizer step tensor")
            count = float(count)
        need(type(count) in (int, float) and count == step, "AdamW step disagrees with global step")
        need(all(isinstance(saved.get(key), torch.Tensor) and saved[key].dtype == torch.float32
                 and saved[key].layout == torch.strided and bool(torch.isfinite(saved[key]).all())
                 for key in ("exp_avg", "exp_avg_sq"))
             and saved["exp_avg"].shape == saved["exp_avg_sq"].shape
             and bool((saved["exp_avg_sq"] >= 0).all()), "Invalid FP32 AdamW moments")
    sampler = checkpoint.get("PixelProprioDataModule", {})
    topology = sampler.get("distributed_sampling", {})
    need(sampler.get("schema") == "mot-sampling-state-v1" and sampler.get("consumed_batches") == step
         and topology.get("schema") == "pixel-distributed-sampling-v1"
         and topology.get("world_size") == parent["world_size"] == 4
         and topology.get("global_batch_size") == parent["global_batch"] == 32
         and topology.get("per_rank_batch_size") == parent["per_rank_batch"] == 8
         and topology.get("training_sharding") == "global_draw_rank_slots"
         and topology.get("validation_sharding") == "contiguous_global_batches_no_padding",
         "Saved sampler cursor/topology differs from parent")
    rank_rng = validate_rng(checkpoint.get("mot_rng_state"), parent["world_size"])
    loops = validate_loop_state(checkpoint, step)
    return dict(global_step=step, epoch=checkpoint["epoch"], data_context_sha256=expected_context_sha256,
                optimizer_state_sha256=state_digest(optimizer), sampler_state_sha256=state_digest(sampler),
                wrapper_state_sha256=state_digest(state), rank_rng_sha256=rank_rng,
                optimizer_steps=[step], sampling_consumed_batches=step, loop_state=loops)


def strict_cpu_restore(checkpoint, cfg, parent, expected_context_sha256):
    """Actually construct/reload the frozen wrapper, optimizer and DataModule."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
    from egomimic.pipeline.construction import checkpoint_construction
    from egomimic.pl_utils.pl_model import ModelWrapper

    model_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["config_tree"])
    data = instantiate(cfg.data)
    context = data.prepare_context(mode="train", normalization="prepared", normalizer=None,
                                   restored_state=checkpoint["data_context"])
    need(context.fingerprint() == expected_context_sha256, "Actual restored data context changed")
    binding = validate_checkpoint_binding(checkpoint, model_cfg, context)
    data.load_state_dict(checkpoint["PixelProprioDataModule"])
    need(state_digest(data.state_dict()) == state_digest(checkpoint["PixelProprioDataModule"]),
         "Actual DataModule did not restore exact sampling state")
    with checkpoint_construction():
        wrapper = ModelWrapper(config_tree=model_cfg, enable_grad_norm=False).cpu().eval()
        context.bind(wrapper.model)
    wrapper.data_context = context
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    need(state_digest(wrapper.state_dict()) == state_digest(checkpoint["state_dict"]),
         "Strict wrapper reload differs from stored state")
    network = wrapper.model.pipeline.stage_by_id("joint_denoiser").model
    counts = network.expert_parameter_counts()
    need(counts == parent["expert_parameters"]
         and sum(parameter.numel() for parameter in network.parameters()) == parent["parameters"]
         and all(parameter.requires_grad for parameter in network.parameters()),
         "Reloaded expert count/trainability differs")
    optimizer = instantiate(cfg.model.optimizer, params=wrapper.parameters())
    if callable(optimizer):
        optimizer = optimizer()
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    need(state_digest(optimizer.state_dict()) == state_digest(checkpoint["optimizer_states"][0]),
         "Actual AdamW groups/moments did not restore exactly")
    need(instantiate(cfg.model.scheduler, optimizer=optimizer) is None, "Unexpected instantiated scheduler")
    for parameter, saved in optimizer.state.items():
        need(saved["exp_avg"].shape == parameter.shape == saved["exp_avg_sq"].shape,
             "Restored moment shape differs from parameter")
    return dict(strict_full_wrapper_optimizer_reload=True, actual_datamodule_restore=True,
                actual_expert_parameters=counts, inference_binding=dict(binding))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-request", required=True, type=Path)
    parser.add_argument("--parent-request-sha256", required=True)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--scontrol", default="/usr/local/bin/scontrol")
    args = parser.parse_args(argv)
    allocation = verify_allocation(args.expected_job_id, args.expected_node, native_command=args.scontrol)
    need(not args.output.exists(), "Refusing inspection receipt overwrite")
    parent = bound(dict(path=str(args.parent_request), sha256=args.parent_request_sha256))
    need(parent["schema"] == "joint-training-v2" and parent["phase"] == "full"
         and parent["source_revision"] == FROZEN_SOURCE and parent["run_id"] == RUN_ID
         and parent["active_streams"] == ["image"] and parent["arm"] == "discrete",
         "Wrong frozen production parent")
    need(sha(args.resolved_config) == args.resolved_config_sha256, "Parent resolved configuration changed")
    need(sha(parent["source_manifest"]) == parent["source_manifest_sha256"], "Source manifest changed")
    for line in Path(parent["source_manifest"]).read_text().splitlines():
        expected, relative = line.split("  ", 1)
        member = Path(relative)
        need(not member.is_absolute() and ".." not in member.parts, "Unsafe source manifest member")
        need(sha(Path(parent["source"])/member) == expected, "Frozen source changed: "+relative)
    need(sha(parent["data_manifest"]) == parent["data_manifest_sha256"], "Data manifest changed")
    selected, inventory = inventory_highest(args.checkpoint_directory, parent)
    before = selected.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    checkpoint_sha = sha(selected)
    selected_inventory = next(row for row in inventory if row["path"] == str(selected))
    need(selected_inventory.get("expected_sha256", checkpoint_sha) == checkpoint_sha,
         "Selected ancestor checkpoint changed from its prior inspection binding")
    # All heavyweight imports and checkpoint reads occur after allocation checks.
    import torch
    from omegaconf import OmegaConf
    from egomimic.pl_utils.data_context import state_fingerprint

    cfg = OmegaConf.load(args.resolved_config)
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    preflight = bound(parent["smoke"]["preflight"])
    expected_context = preflight["provenance"]["data_context_sha256"]
    checkpoint = torch.load(selected, map_location="cpu", weights_only=False, mmap=True)
    checked = validate_payload(checkpoint, parent, resolved, expected_context_sha256=expected_context,
                               context_fingerprint=state_fingerprint)
    need(checked["global_step"] == inventory[-1]["global_step"], "Filename and saved optimizer step differ")
    restored = strict_cpu_restore(checkpoint, cfg, parent, expected_context)
    after = selected.stat()
    need(identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
         "Checkpoint changed during content inspection")
    verify_inventory_stable(args.checkpoint_directory, parent, selected, inventory)
    record = dict(schema=SCHEMA, status="PASS", parent_request_sha256=args.parent_request_sha256,
                  parent_resolved_sha256=args.resolved_config_sha256, job_id=args.expected_job_id,
                  source_revision=parent["source_revision"], run_id=parent["run_id"],
                  checkpoint=dict(path=str(selected), sha256=checkpoint_sha),
                  world_size=parent["world_size"], global_batch=parent["global_batch"],
                  context_sha256=expected_context, lr_policy=parent["lr_policy"], lr_schedulers=[],
                  sampler_all_rank_rng_verified=True, highest_complete_checkpoint_inspected=True,
                  all_newer_candidates_reconciled=True,
                  newer_candidate_resolution="Highest numeric file inspected; no older fallback allowed",
                  inventory=inventory, allocation=allocation,
                  resume_loop_policy="reject_replayed_boundary_validation",
                  loop_training_resume_supported=False, actual_gpu_rng_restoration_proved=False,
                  production_resume_authorized_by_this_receipt=False,
                  checkpoint_modified=False, **checked, **restored)
    write_once(args.output, record)
    print(json.dumps({key: record[key] for key in ("schema", "status", "global_step", "checkpoint",
                                                 "loop_training_resume_supported")}))


if __name__ == "__main__":
    main()
