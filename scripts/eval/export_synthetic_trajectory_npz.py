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
from egomimic.synthetic.decoder_inversion_flow import SyntheticDecoderInversionFlow
from egomimic.synthetic.endpoint_lift_flow import SyntheticEndpointLiftFlow
from egomimic.synthetic.gaussian_relift_flow import SyntheticGaussianReliftFlow
from egomimic.synthetic.latent_bridge_likelihood import SyntheticLatentBridgeLikelihood
from egomimic.synthetic.multi_action_adapter_flow import (
    SyntheticMultiActionAdapterFlow,
)
from egomimic.synthetic.noninvertible_endpoint_flow import (
    SyntheticGraphSectionFlow,
    SyntheticMMDEndpointFlow,
)
from egomimic.synthetic.projected_invertible_flow import (
    SyntheticProjectedInvertibleFlow,
)
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


class _EmbodimentTrajectoryView:
    """Expose one private decoder while retaining the checkpoint's shared field."""

    def __init__(
        self, model: SyntheticMultiActionAdapterFlow, embodiment: str
    ) -> None:
        self.model = model
        self.embodiment = embodiment

    def trajectory(self, source: torch.Tensor, *, steps: int) -> torch.Tensor:
        return self.model.trajectory(
            source, embodiment=self.embodiment, steps=steps
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--embodiment")
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
        architecture = config.get("architecture", "shared_latent")
        if architecture == "multi_action_adapter_flow":
            if args.embodiment is None:
                raise SystemExit("multi-action export requires --embodiment")
            configured_datasets = config.get(
                "evaluation_datasets", config["datasets"]
            )
            if args.embodiment not in configured_datasets:
                raise SystemExit(f"unknown embodiment: {args.embodiment}")
            dataset = (
                args.dataset
                if args.dataset is not None
                else Path(configured_datasets[args.embodiment])
            )
        else:
            if args.embodiment is not None:
                raise SystemExit("--embodiment is only valid for multi-action export")
            dataset = (
                args.dataset if args.dataset is not None else Path(config["dataset"])
            )
        source, target = SyntheticTrajectoryEval.load_validation_data(
            dataset,
            config.get("source_key", "source_2d"),
            args.particles,
        )
        source, target = source.to(device), target.to(device)
        if architecture == "shared_latent":
            model = SyntheticSharedLatentFlow(**config["model"])
        elif architecture == "direct_flow":
            model = SyntheticDirectFlow(**config["model"])
        elif architecture == "action_adapter_flow":
            model = SyntheticActionAdapterFlow(**config["model"])
        elif architecture == "decoder_inversion_flow":
            model = SyntheticDecoderInversionFlow(**config["model"])
        elif architecture == "projected_invertible_flow":
            model = SyntheticProjectedInvertibleFlow(**config["model"])
        elif architecture == "endpoint_lift_flow":
            model = SyntheticEndpointLiftFlow(**config["model"])
        elif architecture == "gaussian_relift_flow":
            model = SyntheticGaussianReliftFlow(**config["model"])
        elif architecture == "mmd_endpoint_flow":
            model = SyntheticMMDEndpointFlow(**config["model"])
        elif architecture == "graph_section_flow":
            model = SyntheticGraphSectionFlow(**config["model"])
        elif architecture == "latent_bridge_likelihood":
            model = SyntheticLatentBridgeLikelihood(**config["model"])
        elif architecture == "multi_action_adapter_flow":
            model = SyntheticMultiActionAdapterFlow(**config["model"])
        else:
            raise SystemExit(f"unknown architecture: {architecture}")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        trajectory_model = (
            _EmbodimentTrajectoryView(model, args.embodiment)
            if architecture == "multi_action_adapter_flow"
            else model
        )
        SyntheticTrajectoryEval.export(
            trajectory_model, source, target, args.output, steps=args.steps
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
