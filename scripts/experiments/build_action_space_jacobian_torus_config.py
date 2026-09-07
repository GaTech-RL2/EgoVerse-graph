#!/usr/bin/env python3
"""Build the encoder-free action-space Jacobian CFM torus trial."""

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
    parser.add_argument(
        "--wandb-project", default="synthetic-action-space-jacobian-flow"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.training_dataset, args.evaluation_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True)
    run_id = f"gaussian-torus-action-space-jacobian-seed{args.seed}-v1"
    config = {
        "variant": "action-space-jacobian-cfm",
        "seed": args.seed,
        "architecture": "action_space_jacobian_flow",
        "dataset": str(args.training_dataset),
        "evaluation_dataset": str(args.evaluation_dataset),
        "output_dir": str(args.experiment_root / "runs" / run_id),
        "source_key": "source_gaussian_latent",
        "model": {
            "latent_dim": 8,
            "action_dim": 3,
            "decoder_family": "nonlinear",
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
    config_path = args.output_dir / f"{run_id}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "algorithm": {
            "bridge_space": "action",
            "source": "g(epsilon)",
            "prediction": "J_g(epsilon) v_theta(A_t, t)",
            "target": "g(epsilon) - A",
            "inference": "action-space Euler with epsilon fixed",
            "encoder": False,
            "latent_flow_loss": False,
            "reconstruction_loss": False,
        },
        "seed": args.seed,
        "max_steps": args.max_steps,
        "datasets": {
            "training": {
                "path": str(args.training_dataset),
                "sha256": sha256(args.training_dataset),
            },
            "evaluation": {
                "path": str(args.evaluation_dataset),
                "sha256": sha256(args.evaluation_dataset),
            },
        },
        "configs": [str(config_path)],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
