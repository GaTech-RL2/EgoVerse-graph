#!/usr/bin/env python3
"""Build the frozen four-run Action Flow clean-latent stop-gradient ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VARIANTS = (
    ("target-sg-rec100", "target_stopgrad", 100.0),
    ("target-sg-rec10", "target_stopgrad", 10.0),
    ("target-sg-rec1", "target_stopgrad", 1.0),
    ("all-sg-rec1", "all_stopgrad", 1.0),
)


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
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (args.training_dataset, args.evaluation_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True)
    configs = []
    for label, mode, reconstruction_weight in VARIANTS:
        run_id = f"gaussian-torus-g-{label}-seed{args.seed}-v1"
        config = {
            "variant": f"g-{label}",
            "seed": args.seed,
            "architecture": "action_adapter_flow",
            "dataset": str(args.training_dataset),
            "evaluation_dataset": str(args.evaluation_dataset),
            "output_dir": str(args.experiment_root / "runs" / run_id),
            "source_key": "source_gaussian_latent",
            "adapter_objective": "action_velocity",
            "clean_gradient_mode": mode,
            "lambda_reconstruction": reconstruction_weight,
            "lambda_scale": 1.0,
            "lambda_path": 0.0,
            "lambda_action_velocity": 1.0,
            "model": {"latent_dim": 8, "adapter_family": "nonlinear", "residual_width": 32, "residual_depth": 2, "field_width": 128, "field_depth": 4},
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
            "wandb": {"entity": "rl2-group", "project": "synthetic-action-flow-stopgrad", "id": run_id, "name": run_id, "resume": "never", "dir": str(args.experiment_root / "wandb")},
        }
        path = args.output_dir / f"{run_id}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        configs.append(str(path))
    manifest = {"schema_version": 1, "seed": args.seed, "max_steps": args.max_steps, "variants": [{"label": label, "clean_gradient_mode": mode, "lambda_reconstruction": weight} for label, mode, weight in VARIANTS], "control": "gaussian-torus-g-nonlinear-action-velocity-rec100-seed42-full-v2", "datasets": {"training": {"path": str(args.training_dataset), "sha256": sha256(args.training_dataset)}, "evaluation": {"path": str(args.evaluation_dataset), "sha256": sha256(args.evaluation_dataset)}}, "configs": configs}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
