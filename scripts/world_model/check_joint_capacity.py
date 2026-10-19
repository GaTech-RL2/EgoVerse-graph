"""Allocated-GPU worst-frame real-data capacity check, separate from fit smoke."""

# ruff: noqa: E402
import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from egomimic.pl_utils.joint_smoke import require, write_once
from egomimic.rldb.pixel_proprio import collate_pixel_proprio


def check_capacity(cfg, data, destination):
    dataset = data.datasets["train"]
    desired = (
        cfg.data.discrete_frames if cfg.data.mode == "discrete" else cfg.data.max_frames
    )
    required = desired * cfg.data.stride if cfg.data.mode == "discrete" else desired
    offset, selected = 0, None
    for episode, count in zip(dataset.episodes, dataset.counts):
        if episode["embodiment"] == "libero_gripper" and episode["frames"] >= required:
            selected = offset
            break
        offset += count
    require(selected is not None, "No real maximum-frame example")
    original = dict(dataset.sampling)
    algo = instantiate(cfg.model.pipeline, device="cuda")
    optimizer = instantiate(cfg.model.optimizer, params=algo.nets.parameters())()
    scheduler = instantiate(cfg.model.scheduler, optimizer=optimizer)
    try:
        if cfg.data.mode == "continuous":
            dataset.sampling.update(min_frames=desired, max_frames=desired)
        groups = collate_pixel_proprio(
            [dataset[(selected, 0, i)] for i in range(cfg.data.batch_size)]
        )
        batch = algo.process_batch_for_training(groups)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = algo.forward_training(batch)
            losses = algo.compute_losses(output, batch)
        require(
            losses["loss"].dtype == torch.float32 and torch.isfinite(losses["loss"]),
            "Worst-frame MSE must be finite FP32",
        )
        losses["loss"].backward()
        require(
            all(
                p.grad is not None and p.grad.dtype == torch.float32
                for p in algo.nets.parameters()
            ),
            "Missing worst-frame expert gradient",
        )
        require(
            bool(
                torch.stack(
                    [torch.isfinite(p.grad).all() for p in algo.nets.parameters()]
                ).all()
            ),
            "Nonfinite worst-frame gradient",
        )
        optimization_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        del losses, output
        optimizer.zero_grad(set_to_none=True)
        algo.nets.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = algo.forward_eval(batch)["libero_gripper"][
                "image_proprio_prediction"
            ]
        require(
            all(
                t.dtype == torch.float32 and torch.isfinite(t).all()
                for t in sampled.values()
            ),
            "Worst-frame sampled state/command invalid",
        )
        record = dict(
            status="CAPACITY_PASS",
            frames_per_example=int(desired),
            per_rank_batch=int(cfg.data.batch_size),
            train_seconds=elapsed,
            train_peak_allocated_bytes=peak,
            optimizer_update_lr=optimization_lr,
            sample_all_streams_finite=True,
        )
        write_once(destination, record)
        return record
    finally:
        dataset.sampling = original
        del algo, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    data = instantiate(cfg.data)
    data.prepare_context(mode="train", normalization="prepared", normalizer=None)
    check_capacity(cfg, data, args.output)


if __name__ == "__main__":
    main()
