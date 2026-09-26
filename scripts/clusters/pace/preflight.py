#!/usr/bin/env python3
"""Allocated-node DDP, visual-BC forward/backward, and read-only catalog checks.

The pace counterpart of scripts/clusters/lambda/preflight.py. The lambda script
asserts a world size of 8 and composes against submitit_lambda_h100, because
lambda only hands out whole 8-GPU nodes. Pace runs this group as four
independent 2-GPU jobs, so the same checks are made at that shape instead.

Beyond the lambda checks this also asserts the ARC contract, since this group's
whole point is a three-way chunking-mode comparison: every ARC experiment must
carry the measured D rather than the 0.40 m default it would otherwise inherit,
and the token row count must match 2M for per-waypoint velocity.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# The profile launch.sh actually uses. Composing against a different one
# would leave the launcher wiring unvalidated.
LAUNCHER = "submitit_pace_h100"
GPUS_PER_JOB = 2
# The value measured by scripts/data/measure_arc_distance_rl2_organize.py. Any
# ARC experiment resolving to something else has silently lost its override.
EXPECTED_ARC_D = 0.464146895489191


def model_check(experiment):
    import hydra
    import torch
    from omegaconf import OmegaConf

    from egomimic.trainHydra import _instantiate_model_wrapper

    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment={experiment}",
                f"hydra/launcher={LAUNCHER}",
                f"launch_params.gpus_per_node={GPUS_PER_JOB}",
            ],
        )
    assert cfg.norm_stats.sample_frac == 0.2
    assert cfg.evaluator.distance_dtw_enabled
    assert "qwen" not in OmegaConf.to_yaml(cfg.model, resolve=True).lower()
    assert cfg.evaluator.execute_fraction == 0.30, cfg.evaluator.execute_fraction

    # Batch size is per source under train_dataloader_params, not a top-level
    # data key: MultiDataModuleWrapper builds one loader per training dataset.
    source = next(iter(cfg.data.train_datasets))
    batch_size = cfg.data.train_dataloader_params[source].batch_size
    assert batch_size == 64, batch_size

    # Effective global batch = 64 * 2 ranks * 1 accumulation = 128, the value
    # config/policy.yaml launch_defaults.training requires.
    effective = batch_size * GPUS_PER_JOB * cfg.trainer.accumulate_grad_batches
    assert effective == 128, effective

    is_arc = "arc_tokenizer" in cfg.abc.action_mode
    if is_arc:
        assert cfg.abc.arc_distance == EXPECTED_ARC_D, cfg.abc.arc_distance
        assert cfg.abc.arc_velocity_mode == "per_waypoint"
        # per_waypoint emits one position row and one velocity row per waypoint.
        assert cfg.hpt.action_horizon == 2 * cfg.abc.arc_waypoints
        assert cfg.abc.arc_chunking_mode in ("race", "multistream", "joint_distance")
        assert cfg.evaluator.ground_truth_action_key == "actions_cartesian_untokenized"

    embodiment = 3 if source == "human_bimanual" else 7
    wrapper = _instantiate_model_wrapper(cfg).cuda(0).train()
    wrapper.model.device = torch.device("cuda:0")
    values = {
        "observations.images.front_img_1": torch.rand(1, 3, 480, 640, device="cuda:0"),
        "observations.state.ee_pose": torch.rand(1, 14, device="cuda:0"),
        "actions_cartesian": torch.rand(1, cfg.hpt.action_horizon, 14, device="cuda:0"),
        "embodiment": embodiment,
    }
    if embodiment == 7:
        for side in ("left", "right"):
            values[f"observations.images.{side}_wrist_img"] = torch.rand(
                1, 3, 480, 640, device="cuda:0"
            )
    batch = {source: values}
    prediction = wrapper.model.forward_training(batch)
    loss = wrapper.model.compute_losses(prediction, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in wrapper.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    result = dict(
        experiment=experiment,
        embodiment=source,
        parameters=sum(p.numel() for p in wrapper.parameters()),
        action_rows=cfg.hpt.action_horizon,
        arc_chunking_mode=cfg.abc.arc_chunking_mode if is_arc else None,
        arc_distance=cfg.abc.arc_distance if is_arc else None,
        effective_global_batch_size=effective,
        loss=float(loss.detach()),
        train_step="forward_backward_passed",
        config=cfg,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Pace compute preflight must run inside a Slurm allocation")
    import torch
    import torch.distributed as dist

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    assert world == GPUS_PER_JOB and int(os.environ["SLURM_JOB_NUM_NODES"]) == 1
    torch.cuda.set_device(local)
    port = 15000 + int(os.environ["SLURM_JOB_ID"]) % 30000
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    value = torch.ones(1, device=f"cuda:{local}")
    dist.all_reduce(value)
    assert value.item() == world
    print(
        json.dumps(
            dict(
                job=os.environ["SLURM_JOB_ID"],
                rank=rank,
                device=local,
                gpu=torch.cuda.get_device_name(local),
                all_reduce=float(value),
            )
        ),
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()
    if rank:
        return
    results = []
    dataframe = None
    if args.check_data:
        from egomimic.utils.aws.aws_sql import (
            create_default_engine,
            episode_table_to_df,
        )

        engine = create_default_engine()
        try:
            dataframe = episode_table_to_df(engine)
        finally:
            engine.dispose()
    for experiment in args.experiment:
        result = model_check(experiment)
        cfg = result.pop("config")
        if dataframe is not None:
            import hydra

            from egomimic.rldb.zarr.zarr_dataset_multi import _normalize_filter_row

            dataset = cfg.data.train_datasets[result["embodiment"]]
            predicate = hydra.utils.instantiate(dataset.filters)
            selected = dataframe.loc[
                dataframe.apply(
                    lambda row: predicate.matches(_normalize_filter_row(row.to_dict())),
                    axis=1,
                )
            ]
            if selected.empty:
                raise RuntimeError(f"No catalog episodes for {experiment}")
            result.update(
                catalog_episodes=len(selected),
                catalog_frames=int(selected.num_frames.sum()),
                catalog_hours_at_30hz=float(selected.num_frames.sum()) / 30 / 3600,
                valid_ratio=float(dataset.valid_ratio),
                split_seed=int(dataset.get("split_seed", 42)),
            )
        results.append(result)
        print(json.dumps(result), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            dict(job=os.environ["SLURM_JOB_ID"], world_size=world, experiments=results),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
