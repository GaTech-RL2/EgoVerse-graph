#!/usr/bin/env python3
"""Allocated-node DDP, visual-BC forward/backward, and read-only catalog checks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


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
                "hydra/launcher=submitit_lambda_h100",
                "launch_params.gpus_per_node=8",
            ],
        )
    assert cfg.norm_stats.sample_frac == 0.2
    assert cfg.evaluator.distance_dtw_enabled
    assert "qwen" not in OmegaConf.to_yaml(cfg.model, resolve=True).lower()
    source = next(iter(cfg.data.train_datasets))
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
        raise RuntimeError(
            "Lambda compute preflight must run inside a Slurm allocation"
        )
    import torch
    import torch.distributed as dist

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    assert world == 8 and int(os.environ["SLURM_JOB_NUM_NODES"]) == 1
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
