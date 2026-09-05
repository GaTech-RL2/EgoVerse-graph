#!/usr/bin/env python3
"""Audit held-out per-embodiment gradients in a shared synthetic field."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from egomimic.synthetic.multi_action_adapter_flow import (
    SyntheticMultiActionAdapterFlow,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_gradients(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter], *, retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=False
    )
    return torch.cat([gradient.detach().flatten() for gradient in gradients])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("architecture") != "multi_action_adapter_flow":
        raise ValueError("gradient audit requires a multi-action checkpoint")
    model = SyntheticMultiActionAdapterFlow(**config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    field_parameters = list(model.field.parameters())
    names = list(config["model"]["embodiments"])
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    dataset_records = {}
    sample_indices = None
    for name in names:
        dataset_path = Path(config["evaluation_datasets"][name])
        with np.load(dataset_path, allow_pickle=False) as archive:
            indices = np.flatnonzero(archive["split"] == 1)[: args.batch_size]
            noise = torch.from_numpy(
                archive[config.get("source_key", "source_gaussian_latent")][indices]
            ).float()
            target = torch.from_numpy(archive["target_3d"][indices]).float()
        if len(indices) != args.batch_size:
            raise ValueError(f"{name} lacks {args.batch_size} validation examples")
        if sample_indices is None:
            sample_indices = indices
        elif not np.array_equal(indices, sample_indices):
            raise ValueError("embodiments must use the same validation index contract")
        noise, target = noise.to(device), target.to(device)
        expanded = args.batch_size * int(config.get("flow_samples", 14))
        time = torch.linspace(
            1.0 / (expanded + 1),
            expanded / (expanded + 1),
            expanded,
            device=device,
        )[:, None]
        losses = model.losses_for_embodiment(
            name,
            target,
            flow_samples=int(config.get("flow_samples", 14)),
            lambda_reconstruction=float(config.get("lambda_reconstruction", 100.0)),
            lambda_scale=float(config.get("lambda_scale", 1.0)),
            lambda_action_velocity=float(config.get("lambda_action_velocity", 1.0)),
            noise=noise,
            time=time,
        )
        gradients[name] = {
            "flow": flatten_gradients(
                losses["flow_loss"], field_parameters, retain_graph=True
            ),
            "action_velocity": flatten_gradients(
                losses["action_velocity_loss"],
                field_parameters,
                retain_graph=True,
            ),
            "total": flatten_gradients(
                losses["loss"], field_parameters, retain_graph=False
            ),
        }
        dataset_records[name] = {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
        }
    if len(names) != 2:
        raise ValueError("this audit currently requires exactly two embodiments")
    first, second = names
    metrics = {}
    for component in ("flow", "action_velocity", "total"):
        left = gradients[first][component]
        right = gradients[second][component]
        left_norm = left.norm()
        right_norm = right.norm()
        cosine = torch.dot(left, right) / (left_norm * right_norm).clamp_min(1e-12)
        combined_norm = (0.5 * (left + right)).norm()
        metrics[component] = {
            f"{first}_norm": float(left_norm),
            f"{second}_norm": float(right_norm),
            "cosine": float(cosine),
            "mean_gradient_norm": float(combined_norm),
            "cancellation_ratio": float(
                combined_norm / (0.5 * (left_norm + right_norm)).clamp_min(1e-12)
            ),
        }
    if not all(
        np.isfinite(value)
        for component in metrics.values()
        for value in component.values()
    ):
        raise RuntimeError("gradient audit produced non-finite metrics")
    result = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "global_step": int(checkpoint["step"]),
        },
        "datasets": dataset_records,
        "batch_size": args.batch_size,
        "flow_samples": int(config.get("flow_samples", 14)),
        "time_grid": "deterministic evenly spaced interior points in (0,1)",
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
