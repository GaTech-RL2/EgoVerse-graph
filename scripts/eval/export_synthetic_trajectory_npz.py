#!/usr/bin/env python3
"""Export one checkpoint or analytic coupling through SyntheticTrajectoryEval."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.shared_latent_flow import (
    SyntheticDirectFlow,
    SyntheticSharedLatentFlow,
)


def validation_data(path: Path, source_key: str, limit: int):
    archive = np.load(path, allow_pickle=False)
    indices = np.flatnonzero(archive["split"] == 1)[:limit]
    source = torch.from_numpy(archive[source_key][indices]).float()
    target = torch.from_numpy(archive["target_3d"][indices]).float()
    return source, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--ground-truth-source-key")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--particles", type=int, default=128)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.ground_truth_source_key is None):
        raise SystemExit("select exactly one of --checkpoint or --ground-truth-source-key")

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        source, target = validation_data(
            Path(config["dataset"]),
            config.get("source_key", "source_2d"),
            args.particles,
        )
        architecture = config.get("architecture", "shared_latent")
        if architecture == "shared_latent":
            model = SyntheticSharedLatentFlow(**config["model"])
        elif architecture == "direct_flow":
            model = SyntheticDirectFlow(**config["model"])
        else:
            raise SystemExit(f"unknown architecture: {architecture}")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        SyntheticTrajectoryEval.export(
            model, source, target, args.output, steps=args.steps
        )
    else:
        if args.dataset is None:
            raise SystemExit("ground truth export requires --dataset")
        source, target = validation_data(
            args.dataset, args.ground_truth_source_key, args.particles
        )
        SyntheticTrajectoryEval.export_linear_ground_truth(
            source, target, args.output, steps=args.steps
        )


if __name__ == "__main__":
    main()
