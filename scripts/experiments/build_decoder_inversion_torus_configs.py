#!/usr/bin/env python3
"""Build the paired affine/nonlinear torus decoder-inversion trial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--inversion-steps", type=int, required=True)
    parser.add_argument("--inversion-step-size", type=float, required=True)
    parser.add_argument(
        "--training-objective",
        choices=("endpoint_difference", "conditional_relifting"),
        default="endpoint_difference",
    )
    parser.add_argument(
        "--wandb-project", default="synthetic-action-flow-decoder-inversion"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.inversion_steps <= 0 or args.inversion_step_size <= 0:
        raise ValueError("inversion steps and step size must be positive")
    for path in (args.training_dataset, args.evaluation_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True)

    configs = []
    method_label = (
        "decoder-inversion"
        if args.training_objective == "endpoint_difference"
        else "conditional-relifting"
    )
    for family in ("joint_affine", "nonlinear"):
        label = "affine" if family == "joint_affine" else family
        run_id = (
            f"gaussian-torus-{method_label}-{label}-k{args.inversion_steps}"
            f"-h{args.inversion_step_size:g}-seed{args.seed}-v1"
        )
        config = {
            "variant": f"{method_label}-{label}",
            "seed": args.seed,
            "architecture": "decoder_inversion_flow",
            "dataset": str(args.training_dataset),
            "evaluation_dataset": str(args.evaluation_dataset),
            "output_dir": str(args.experiment_root / "runs" / run_id),
            "source_key": "source_gaussian_latent",
            "inversion_steps": args.inversion_steps,
            "inversion_step_size": args.inversion_step_size,
            "training_objective": args.training_objective,
            "inversion_failure_rmse": 0.1,
            "lambda_scale": 1.0,
            "model": {
                "latent_dim": 8,
                "decoder_family": family,
                "residual_width": 32,
                "residual_depth": 2,
                "field_width": 128,
                "field_depth": 4,
            },
            "flow_samples": 14,
            "learning_rate": 3e-4,
            "batch_size": 512,
            "max_steps": args.max_steps,
            "inference_steps": 32,
            "evaluation_particles": 2048,
            "diagnostic_noise_samples": 4096,
            "angular_bins": 16,
            "torus_major_radius": 2.0,
            "torus_minor_radius": 0.65,
            "log_every": 100,
            "checkpoint_every": 50_000,
            "wandb": {
                "entity": "rl2-group",
                "project": args.wandb_project,
                "id": run_id,
                "name": run_id,
                "resume": "never",
                "dir": str(args.experiment_root / "wandb"),
            },
        }
        path = args.output_dir / f"{run_id}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        configs.append(str(path))

    manifest = {
        "schema_version": 1,
        "task": f"torus {method_label} Action Flow trial",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "inversion": {
            "steps": args.inversion_steps,
            "step_size": args.inversion_step_size,
        },
        "training_objective": args.training_objective,
        "variants": ["affine", "nonlinear"],
        "primary_metric": "validation_generation_symmetric_nn_mse",
        "controls": {
            "direct_fm": "gaussian-torus-a-direct-seed42-full-v2",
            "reconstruction_action_flow": (
                "gaussian-torus-g-nonlinear-action-velocity-rec100-seed42-full-v2"
            ),
        },
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
        "validation_command": (
            "scripts/experiments/check_decoder_inversion.sh"
        ),
        "configs": configs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
