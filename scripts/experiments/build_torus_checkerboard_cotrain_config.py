#!/usr/bin/env python3
"""Build one balanced Variant-G torus/checkerboard co-training contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--torus", type=Path, required=True)
    parser.add_argument("--checkerboard", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--evaluation-particles", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite config: {args.output}")
    for path in (args.torus, args.checkerboard):
        if not path.is_file():
            raise FileNotFoundError(path)
    centers = [
        [-2.25, -2.25],
        [0.75, -2.25],
        [-0.75, -0.75],
        [2.25, -0.75],
        [-2.25, 0.75],
        [0.75, 0.75],
        [-0.75, 2.25],
        [2.25, 2.25],
    ]
    config = {
        "seed": args.seed,
        "output_dir": str(args.experiment_root / "runs" / args.run_id),
        "datasets": {"torus_3d": str(args.torus), "checkerboard_2d": str(args.checkerboard)},
        "evaluation_datasets": {
            "torus_3d": str(args.torus),
            "checkerboard_2d": str(args.checkerboard),
        },
        "target_keys": {"torus_3d": "target_3d", "checkerboard_2d": "target_2d"},
        "surface_specs": {
            "torus_3d": {"kind": "torus", "major_radius": 2.0, "minor_radius": 0.65},
            "checkerboard_2d": {
                "kind": "checkerboard_gaussian_mixture",
                "centers": centers,
            },
        },
        "source_key": "source_gaussian_latent",
        "batch_size_per_embodiment": 512,
        "flow_samples": 14,
        "lambda_reconstruction": 100.0,
        "lambda_scale": 1.0,
        "lambda_action_velocity": 1.0,
        "clean_gradient_mode": "full",
        "learning_rate": 3e-4,
        "max_steps": args.max_steps,
        "log_every": 100,
        "checkpoint_every": 50_000,
        "inference_steps": 32,
        "evaluation_particles": args.evaluation_particles,
        "model": {
            "embodiments": ["torus_3d", "checkerboard_2d"],
            "action_dims": {"torus_3d": 3, "checkerboard_2d": 2},
            "latent_dim": 8,
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
        "wandb": {
            "entity": "rl2-group",
            "project": "synthetic-action-flow-cotrain",
            "id": args.run_id,
            "name": args.run_id,
            "resume": "never",
            "dir": str(args.experiment_root / "wandb"),
            "tags": [
                "variant-g",
                "cotrain",
                "torus-3d",
                "checkerboard-2d",
                "shared-field",
                "private-action-adapters",
                "latent8",
                "seed42",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "config": str(args.output.resolve()),
        "config_sha256": sha256(args.output),
        "datasets": {
            "torus_3d": {"path": str(args.torus.resolve()), "sha256": sha256(args.torus)},
            "checkerboard_2d": {
                "path": str(args.checkerboard.resolve()),
                "sha256": sha256(args.checkerboard),
            },
        },
        "fairness": {
            "balanced_per_optimizer_step": True,
            "examples_per_embodiment_per_step": 512,
            "flow_times_per_example": 14,
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
