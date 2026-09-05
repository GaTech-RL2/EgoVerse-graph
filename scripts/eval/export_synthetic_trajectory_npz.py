#!/usr/bin/env python3
"""Export one checkpoint or analytic coupling through SyntheticTrajectoryEval."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow
from egomimic.synthetic.shared_latent_flow import (
    SyntheticDirectFlow,
    SyntheticSharedLatentFlow,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--ground-truth-source-key")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if (args.checkpoint is None) == (args.ground_truth_source_key is None):
        raise SystemExit(
            "select exactly one of --checkpoint or --ground-truth-source-key"
        )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        dataset = args.dataset if args.dataset is not None else Path(config["dataset"])
        source, target = SyntheticTrajectoryEval.load_validation_data(
            dataset,
            config.get("source_key", "source_2d"),
            args.particles,
        )
        source, target = source.to(device), target.to(device)
        architecture = config.get("architecture", "shared_latent")
        if architecture == "shared_latent":
            model = SyntheticSharedLatentFlow(**config["model"])
        elif architecture == "direct_flow":
            model = SyntheticDirectFlow(**config["model"])
        elif architecture == "action_adapter_flow":
            model = SyntheticActionAdapterFlow(**config["model"])
        else:
            raise SystemExit(f"unknown architecture: {architecture}")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        SyntheticTrajectoryEval.export(
            model, source, target, args.output, steps=args.steps
        )
    else:
        if args.dataset is None:
            raise SystemExit("ground truth export requires --dataset")
        source, target = SyntheticTrajectoryEval.load_validation_data(
            args.dataset, args.ground_truth_source_key, args.particles
        )
        source, target = source.to(device), target.to(device)
        SyntheticTrajectoryEval.export_linear_ground_truth(
            source, target, args.output, steps=args.steps
        )


if __name__ == "__main__":
    main()
