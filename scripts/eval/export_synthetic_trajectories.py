#!/usr/bin/env python3
"""Export fixed validation-particle trajectories for the synthetic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from egomimic.synthetic.shared_latent_flow import SyntheticSharedLatentFlow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--unite-checkpoint", type=Path, required=True)
    parser.add_argument("--vfm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    indices = np.flatnonzero(data["split"] == 1)
    first_checkpoint = torch.load(args.unite_checkpoint, map_location="cpu", weights_only=False)
    source_key = first_checkpoint["config"].get("source_key", "source_2d")
    source_latent = torch.from_numpy(data[source_key][indices]).float()
    source_3d = torch.from_numpy(data["source_3d"][indices]).float()
    target_3d = torch.from_numpy(data["target_3d"][indices]).float()
    times = torch.linspace(0.0, 1.0, args.steps + 1)

    ground_truth = torch.stack(
        [(1.0 - time) * source_3d + time * target_3d for time in times]
    )

    def model_trajectory(checkpoint_path: Path) -> torch.Tensor:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = SyntheticSharedLatentFlow(**config["model"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        checkpoint_source_key = config.get("source_key", "source_2d")
        if checkpoint_source_key != source_key:
            raise ValueError(
                f"checkpoint source keys differ: {source_key} vs {checkpoint_source_key}"
            )
        state = source_latent.clone()
        decoded = [model.decoder(state)]
        dt = 1.0 / args.steps
        with torch.inference_mode():
            for index in range(args.steps):
                time = torch.full((len(state), 1), index * dt)
                state = state + dt * model.velocity(state, time)
                decoded.append(model.decoder(state))
        return torch.stack(decoded)

    unite = model_trajectory(args.unite_checkpoint)
    vfm = model_trajectory(args.vfm_checkpoint)

    coordinate_scale = 1_000

    def quantized(value: torch.Tensor) -> list:
        scaled = np.rint(value.detach().numpy() * coordinate_scale)
        if np.abs(scaled).max() > np.iinfo(np.int32).max:
            raise ValueError("trajectory exceeds int32 quantization range")
        return scaled.astype(np.int32).tolist()

    payload = {
        "times": np.round(times.numpy(), 5).tolist(),
        "target": quantized(target_3d),
        "trajectories": {
            "ground_truth": quantized(ground_truth),
            "unite": quantized(unite),
            "vfm": quantized(vfm),
        },
        "metadata": {
            "particles": int(len(indices)),
            "split": "validation",
            "integration": "Euler",
            "integration_steps": int(args.steps),
            "coordinate_scale": coordinate_scale,
            "unite_checkpoint": str(args.unite_checkpoint),
            "vfm_checkpoint": str(args.vfm_checkpoint),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
