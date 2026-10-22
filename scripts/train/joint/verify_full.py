"""Allocated postflight after normal request-bound training; never creates another checkpoint."""

import csv
import json
from scripts.train.joint.lr_policy import validate, update_lr, verify_scheduler
import math
import os
import re
import sys
from pathlib import Path

from scripts.train.joint.contract import need, sha, verify_evidence, write_once


def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    _, preflight = verify_evidence(request)
    output = Path(request["output"])
    from scripts.train.joint.continuation import allocation_steps

    start, target = allocation_steps(request)
    partial = target < request["target_steps"]
    need(request["target_steps"] in (100000, 300000), "Unsupported retained training budget")
    streams = request["active_streams"]
    count = target // 20000
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
    from egomimic.pipeline.construction import checkpoint_construction
    from egomimic.pl_utils.pl_model import ModelWrapper

    milestones = {}
    directory = output / "checkpoints"
    for path in directory.glob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        match = re.search(r"([0-9]{9})\.ckpt$", path.name)
        need(
            match is not None and path.is_file() and not path.is_symlink(),
            "Unexpected checkpoint file",
        )
        step = int(match[1])
        need(step not in milestones, "Duplicate checkpoint step")
        milestones[step] = path
    expected = set(range(20000, target + 1, 20000))
    need(set(milestones) <= expected and target in milestones, "Unexpected milestones or missing local recovery")
    from scripts.train.joint.retention import archived_milestones

    archived, archive_proof = archived_milestones(request, expected - set(milestones))
    milestones.update(archived)
    latest = directory / "last.ckpt"
    need(
        latest.is_symlink() and latest.resolve() == milestones[target].resolve(),
        "Last must link to immutable terminal milestone",
    )
    selected = latest.resolve()
    if partial:
        receipt_path = output/"allocation-evidence"/f"allocation-step{target:09d}.json"
        boundary = json.loads(receipt_path.read_text())
        need(boundary["status"] == "PASS" and boundary["start_step"] == start
             and boundary["global_step"] == target
             and boundary["target_step"] == request["target_steps"]
             and boundary["checkpoint_saved_after_validation"], "Missing graceful boundary proof")
        selected = Path(boundary["checkpoint"]["path"])
        need(sha(selected) == boundary["checkpoint"]["sha256"], "Changed post-validation recovery")
    checkpoint = torch.load(selected, map_location="cpu", weights_only=False, mmap=True)
    need(checkpoint["global_step"] == target, "Training ended before requested budget")
    cfg = OmegaConf.load(request_path.with_suffix(".resolved.yaml"))
    model_config = OmegaConf.create(checkpoint["hyper_parameters"]["config_tree"])
    need(
        OmegaConf.to_container(cfg.model, resolve=True)
        == OmegaConf.to_container(model_config.model, resolve=True),
        "Checkpoint model differs from run",
    )
    need(
        model_config.run_provenance.run_id == request["run_id"],
        "Checkpoint belongs to another W&B identity",
    )
    need(
        all(
            t.dtype == torch.float32 and torch.isfinite(t).all()
            for t in checkpoint["state_dict"].values()
        ),
        "Invalid FP32 stored weights",
    )
    optimizers, schedulers = (
        checkpoint.get("optimizer_states", []),
        checkpoint.get("lr_schedulers", []),
    )
    need(
        len(optimizers) == 1 and bool(optimizers[0]["state"]),
        "Missing AdamW/scheduler state",
    )
    policy = validate(request["lr_policy"])
    need(
        all(g["lr"] == update_lr(policy, target) and (not policy["warmup_optimizer_steps"] or g["initial_lr"] == policy["peak_lr"]) for g in optimizers[0]["param_groups"]),
        "Wrong terminal LR",
    )
    verify_scheduler(policy, schedulers, target)
    for state in optimizers[0]["state"].values():
        need(
            float(state["step"]) == target
            and all(
                state[k].dtype == torch.float32 and torch.isfinite(state[k]).all()
                for k in ("exp_avg", "exp_avg_sq")
            ),
            "Invalid AdamW moments/step",
        )
    sampler = checkpoint.get("PixelProprioDataModule", {})
    need(
        sampler.get("schema") == "mot-sampling-state-v1"
        and sampler.get("consumed_batches") == target
        and sampler["distributed_sampling"]["world_size"] == request["world_size"]
        and sampler["distributed_sampling"]["global_batch_size"] == 32,
        "Wrong sampling continuation state",
    )
    rng = checkpoint.get("mot_rng_state", {})
    need(
        rng.get("schema") == "mot_rng_state.distributed.v1"
        and rng.get("world_size") == request["world_size"]
        and len(rng.get("ranks", [])) == request["world_size"]
        and all(
            set(r) == {"torch_cpu", "torch_cuda", "numpy", "python", "cuda_scope"}
            and r["cuda_scope"] == "current_device"
            and len(r["torch_cuda"]) == 1
            for r in rng["ranks"]
        ),
        "Missing all-rank RNG state",
    )
    data = instantiate(cfg.data)
    context = data.prepare_context(
        mode="eval",
        normalization="prepared",
        normalizer=None,
        restored_state=checkpoint["data_context"],
    )
    need(
        context.fingerprint() == preflight["provenance"]["data_context_sha256"],
        "Changed terminal context",
    )
    validate_checkpoint_binding(checkpoint, model_config, context)
    with checkpoint_construction():
        wrapper = (
            ModelWrapper(config_tree=model_config, enable_grad_norm=False).cuda().eval()
        )
        context.bind(wrapper.model)
    wrapper.data_context = context
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer = instantiate(cfg.model.optimizer, params=wrapper.parameters())
    if callable(optimizer):
        optimizer = optimizer()
    scheduler = instantiate(cfg.model.scheduler, optimizer=optimizer)
    optimizer.load_state_dict(optimizers[0])
    if scheduler is not None:
        scheduler.load_state_dict(schedulers[0])
    need(
        (scheduler.state_dict() == schedulers[0] if policy["warmup_optimizer_steps"] else scheduler is None)
        and all(g["lr"] == update_lr(policy, target) for g in optimizer.param_groups),
        "Strict AdamW/scheduler reload failed",
    )
    batch = next(iter(data.val_dataloader()))
    sample = batch["libero_gripper"]
    for stream in sample["world"].values():
        stream["value"] = torch.where(
            stream["observed"][..., None], stream["value"], float("nan")
        )
    groups = wrapper.model.process_batch_for_training({"libero_gripper": sample})
    wrapper.model.device = next(wrapper.parameters()).device
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = wrapper.model.forward_eval(groups)["libero_gripper"][
            "image_proprio_prediction"
        ]
    for name, value in predictions.items():
        stream = groups["libero_gripper"]["world"][name]
        need(
            value.dtype == torch.float32
            and value.shape == stream["value"].shape
            and torch.isfinite(value).all(),
            "Invalid terminal sample",
        )
        need(
            torch.equal(value[stream["observed"]], stream["value"][stream["observed"]]),
            "Observed values changed",
        )
    validation = {}
    from scripts.train.joint.continuation import log_segments

    for csv_path in log_segments(request):
        segment = {}
        with csv_path.open() as stream:
            for row in csv.DictReader(stream):
                metrics = {
                    k: float(v)
                    for k, v in row.items()
                    if k.startswith("val/libero_gripper/") and v
                }
                if metrics:
                    step = int(float(row["step"]))
                    need(all(math.isfinite(v) for v in metrics.values()), "Nonfinite validation history")
                    segment.setdefault(step, {}).update(metrics)
        need(not validation.keys() & segment.keys(), "Repeated validation across allocations")
        validation.update(segment)
    need(
        len(validation) == count
        and all(len(v) == 2 * len(streams) for v in validation.values())
        and all(
            step in validation or step - 1 in validation
            for step in range(20000, target + 1, 20000)
        ),
        "Missing scheduled validation events",
    )
    write_once(
        (output/"allocation-evidence"/f"completion-step{target:09d}.json"
         if partial else output / "full-completion.json"),
        dict(
            schema="joint-mot-allocation-v1" if partial else "joint-mot-full-v1",
            status="PASS",
            request_sha256=sha(request_path),
            job_id=os.environ["SLURM_JOB_ID"],
            run_id=request["run_id"],
            global_step=target,
            world_size=request["world_size"],
            checkpoint_sha256=sha(selected),
            strict_full_wrapper_optimizer_scheduler_reload=True,
            sampler_all_rank_rng_verified=True,
            sample_finite=True,
            retained_milestones=count,
            validation_events=count,
            checkpoint=str(selected),
            archive_inventory=archive_proof,
        ),
    )


if __name__ == "__main__":
    main()
