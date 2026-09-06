#!/usr/bin/env python3
"""Measure decoder-inversion contraction on a fixed target cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from egomimic.synthetic.decoder_inversion_flow import SyntheticDecoderInversionFlow


def load_targets(path: Path, particles: int) -> torch.Tensor:
    archive = np.load(path, allow_pickle=False)
    for key in ("target_3d", "target"):
        if key in archive:
            targets = torch.from_numpy(archive[key]).float()
            if targets.ndim == 3:
                targets = targets[-1]
            if len(targets) < particles:
                raise ValueError(f"requested {particles} targets, found {len(targets)}")
            return targets[:particles]
    raise KeyError("input must contain target_3d or target")


def perturb_decoder(model: SyntheticDecoderInversionFlow) -> None:
    generator = torch.Generator().manual_seed(314159)
    with torch.no_grad():
        model.decoder.weight.add_(
            0.10
            * torch.randn(
                model.decoder.weight.shape,
                generator=generator,
                dtype=model.decoder.weight.dtype,
            )
        )
        model.decoder.bias.add_(
            0.05
            * torch.randn(
                model.decoder.bias.shape,
                generator=generator,
                dtype=model.decoder.bias.dtype,
            )
        )
        if model.decoder_family == "nonlinear":
            model.decoder.residual[-1].weight.normal_(
                std=0.05, generator=generator
            )
            model.decoder.residual[-1].bias.normal_(
                std=0.05, generator=generator
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=256)
    args = parser.parse_args()
    targets = load_targets(args.targets, args.particles)
    initialization = torch.randn(
        len(targets), 8, generator=torch.Generator().manual_seed(271828)
    )
    rows = []
    for family in ("joint_affine", "nonlinear"):
        model = SyntheticDecoderInversionFlow(
            latent_dim=8, decoder_family=family
        )
        perturb_decoder(model)
        singular_values = model.decoder_jacobian_singular_values(
            initialization[:128]
        )
        for steps in (1, 2, 4, 8):
            for step_size in (0.25, 0.5, 1.0):
                code, metrics = model.infer_codes(
                    targets,
                    initialization.detach().clone(),
                    steps=steps,
                    step_size=step_size,
                    create_graph=False,
                )
                rows.append(
                    {
                        "decoder_family": family,
                        "steps": steps,
                        "step_size": step_size,
                        "before_mse": float(metrics["inversion_before_mse"]),
                        "after_mse": float(metrics["inversion_after_mse"]),
                        "after_rmse_q99": float(
                            metrics["inversion_after_rmse_q99"]
                        ),
                        "contraction": float(
                            metrics["inversion_after_mse"]
                            / metrics["inversion_before_mse"]
                        ),
                        "finite": bool(torch.isfinite(code).all()),
                        "jacobian_singular_min": float(singular_values.min()),
                        "jacobian_singular_max": float(singular_values.max()),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
