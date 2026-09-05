#!/usr/bin/env python3
"""Build the frozen three-seed A--G action-flow comparison configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

VARIANTS = {
    "a-direct": {
        "architecture": "direct_flow",
        "source_key": "source_gaussian_3d",
        "model": {"data_dim": 3, "field_width": 128, "field_depth": 4},
    },
    "b-fixed-lift": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "none",
        "lambda_reconstruction": 0.0,
        "lambda_scale": 0.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 0.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "fixed_affine",
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "c-joint-affine": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "reconstruction",
        "lambda_reconstruction": 1.0,
        "lambda_scale": 0.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 0.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "joint_affine",
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "d-affine-scale": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "reconstruction",
        "lambda_reconstruction": 1.0,
        "lambda_scale": 1.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 0.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "joint_affine",
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "e-nonlinear-reconstruction-rec1": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "reconstruction",
        "lambda_reconstruction": 1.0,
        "lambda_scale": 1.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 0.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "e-nonlinear-reconstruction-rec10": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "reconstruction",
        "lambda_reconstruction": 10.0,
        "lambda_scale": 1.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 0.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "e-nonlinear-reconstruction-rec100": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "reconstruction",
        "lambda_reconstruction": 100.0,
        "lambda_scale": 1.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 0.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "f-nonlinear-path": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "path",
        "lambda_reconstruction": 0.0,
        "lambda_scale": 1.0,
        "lambda_path": 1.0,
        "lambda_action_velocity": 0.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "g-nonlinear-action-velocity-rec1": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "action_velocity",
        "lambda_reconstruction": 1.0,
        "lambda_scale": 1.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 1.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "g-nonlinear-action-velocity-rec10": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "action_velocity",
        "lambda_reconstruction": 10.0,
        "lambda_scale": 1.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 1.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
    },
    "g-nonlinear-action-velocity-rec100": {
        "architecture": "action_adapter_flow",
        "source_key": "source_gaussian_latent",
        "adapter_objective": "action_velocity",
        "lambda_reconstruction": 100.0,
        "lambda_scale": 1.0,
        "lambda_path": 0.0,
        "lambda_action_velocity": 1.0,
        "model": {
            "latent_dim": 8,
            "adapter_family": "nonlinear",
            "residual_width": 32,
            "residual_depth": 2,
            "field_width": 128,
            "field_depth": 4,
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-steps", type=int, default=60_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--evaluation-particles", type=int, default=2048)
    parser.add_argument("--run-suffix", default="")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse config directory: {args.output_dir}")
    if args.evaluation_particles <= 1:
        raise ValueError("evaluation-particles must exceed one")
    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_suffix):
        raise ValueError("run-suffix must be a path-safe identifier")
    for label, path in {
        "training dataset": args.training_dataset,
        "evaluation dataset": args.evaluation_dataset,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    args.output_dir.mkdir(parents=True)
    written = []
    for variant, overrides in VARIANTS.items():
        for seed in args.seeds:
            suffix = f"-{args.run_suffix}" if args.run_suffix else ""
            run_id = f"gaussian-torus-{variant}-seed{seed}{suffix}"
            config = {
                "variant": variant,
                "seed": seed,
                "dataset": str(args.training_dataset),
                "evaluation_dataset": str(args.evaluation_dataset),
                "output_dir": str(args.experiment_root / "runs" / run_id),
                "flow_samples": 14,
                "learning_rate": args.learning_rate,
                "batch_size": 512,
                "max_steps": args.max_steps,
                "inference_steps": 32,
                "evaluation_particles": args.evaluation_particles,
                "diagnostic_noise_samples": 4096,
                "angular_bins": 16,
                "torus_major_radius": 2.0,
                "torus_minor_radius": 0.65,
                "log_every": 100,
                "checkpoint_every": 50_000,
                "wandb": {
                    "entity": "rl2-group",
                    "project": "synthetic-action-flow-affine",
                    "id": run_id,
                    "name": run_id,
                    "resume": "never",
                    "dir": str(args.experiment_root / "wandb"),
                },
                **overrides,
            }
            path = args.output_dir / f"{run_id}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            written.append(str(path))
    manifest = {
        "schema_version": 1,
        "variants": list(VARIANTS),
        "seeds": args.seeds,
        "primary_metric": "validation_generation_symmetric_nn_mse",
        "comparison_contract": (
            "compare identical evaluation target cloud and particle count; "
            "report each seed plus mean and standard deviation"
        ),
        "pass_threshold": None,
        "checkpoint_every": 50_000,
        "run_suffix": args.run_suffix,
        "datasets": {
            "training": {
                "path": str(args.training_dataset.resolve()),
                "sha256": sha256(args.training_dataset),
            },
            "evaluation": {
                "path": str(args.evaluation_dataset.resolve()),
                "sha256": sha256(args.evaluation_dataset),
            },
        },
        "configs": written,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
