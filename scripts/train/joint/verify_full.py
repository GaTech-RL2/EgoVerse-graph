"""Allocated postflight after normal request-bound training; never creates another checkpoint."""

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

from scripts.train.joint.contract import need, sha, verify_evidence, write_once


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    _, preflight = verify_evidence(request)
    output = Path(request["output"])
    target = request["target_steps"]
    need(target in (100000, 300000), "Unsupported retained training budget")
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
    need(
        sorted(milestones) == list(range(20000, target + 1, 20000)),
        "Missing or extra production milestones",
    )
    latest = directory / "last.ckpt"
    need(
        latest.is_symlink() and latest.resolve() == milestones[target].resolve(),
        "Last must link to immutable terminal milestone",
    )
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False, mmap=True)
    need(checkpoint["global_step"] == target, "Training ended before requested budget")
    cfg = OmegaConf.load(output / ".hydra/config.yaml")
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
        len(optimizers) == len(schedulers) == 1 and bool(optimizers[0]["state"]),
        "Missing AdamW/scheduler state",
    )
    need(
        all(g["lr"] == g["initial_lr"] == 5e-6 for g in optimizers[0]["param_groups"]),
        "Wrong terminal LR",
    )
    need(
        schedulers[0]["last_epoch"] == target
        and schedulers[0]["base_lrs"] == [5e-6]
        and schedulers[0]["lr_lambdas"] == [{"warmup_steps": 3000}],
        "Wrong scheduler state",
    )
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
    scheduler.load_state_dict(schedulers[0])
    need(
        scheduler.state_dict() == schedulers[0]
        and all(g["lr"] == 5e-6 for g in optimizer.param_groups),
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
    csv_path = Path(request["local_logs"]) / "csv/version_0/metrics.csv"
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            metrics = {
                k: float(v)
                for k, v in row.items()
                if k.startswith("val/libero_gripper/") and v
            }
            if metrics:
                need(
                    all(math.isfinite(v) for v in metrics.values()),
                    "Nonfinite validation history",
                )
                validation[int(float(row["step"]))] = metrics
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
        output / "full-completion.json",
        dict(
            schema="joint-mot-full-v1",
            status="PASS",
            job_id=os.environ["SLURM_JOB_ID"],
            run_id=request["run_id"],
            global_step=target,
            world_size=request["world_size"],
            checkpoint_sha256=sha(latest),
            strict_full_wrapper_optimizer_scheduler_reload=True,
            sampler_all_rank_rng_verified=True,
            sample_finite=True,
            retained_milestones=count,
            validation_events=count,
            checkpoint=str(latest.resolve()),
        ),
    )


if __name__ == "__main__":
    main()
