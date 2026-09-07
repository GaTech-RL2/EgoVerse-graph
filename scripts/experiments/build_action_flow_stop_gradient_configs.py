#!/usr/bin/env python3
"""Build the stop-gradient x action-velocity 2x2 on the action-flow torus suite.

(Run 2026-09-06 as ICE array 5713133 at b3c4264, where the stop-gradient arm was
expressed as detach_flow_target=True -- exactly clean_gradient_mode="all_stopgrad"
here. Results: docs/experiments/action-flow-stop-gradient.md.)

Variants: {E (reconstruction only), G (reconstruction + action velocity)}
        x {attached (Elmo's suite as pushed), sg (detach_flow_target)}
        x reconstruction weight {1, 100}
plus the direct 3-D FM baseline. Every run evaluates on the identical frozen
2,048-particle seed-4242 validation cloud; symmetric NN MSE is primary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

BASE_MODEL = {
    "latent_dim": 8,
    "adapter_family": "nonlinear",
    "residual_width": 32,
    "residual_depth": 2,
    "field_width": 128,
    "field_depth": 4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variants(rec_weights):
    out = {}
    for rec in rec_weights:
        for objective, tag in (("reconstruction", "e"), ("action_velocity", "g")):
            for sg in (False, True):
                name = f"{tag}-rec{rec:g}-{'sg' if sg else 'attached'}"
                out[name] = {
                    "architecture": "action_adapter_flow",
                    "source_key": "source_gaussian_latent",
                    "adapter_objective": objective,
                    "lambda_reconstruction": float(rec),
                    "lambda_scale": 1.0,
                    "lambda_path": 0.0,
                    "lambda_action_velocity": 1.0 if objective == "action_velocity" else 0.0,
                    "clean_gradient_mode": "all_stopgrad" if sg else "full",
                    "model": dict(BASE_MODEL),
                }
    out["a-direct"] = {
        "architecture": "direct_flow",
        "source_key": "source_gaussian_3d",
        "model": {"data_dim": 3, "field_width": 128, "field_depth": 4},
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--evaluation-dataset", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rec-weights", type=float, nargs="+", default=[1.0, 100.0])
    parser.add_argument("--max-steps", type=int, default=60_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--evaluation-particles", type=int, default=2048)
    parser.add_argument("--wandb-project", default="synthetic-action-flow-affine")
    parser.add_argument("--wandb-group", default="aidan-sg-2x2")
    parser.add_argument("--run-suffix", default="")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse config directory: {args.output_dir}")
    if args.run_suffix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_suffix):
        raise ValueError("run-suffix must be a path-safe identifier")
    for label, path in {"training": args.training_dataset, "evaluation": args.evaluation_dataset}.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} dataset does not exist: {path}")
    args.output_dir.mkdir(parents=True)
    suffix = f"-{args.run_suffix}" if args.run_suffix else ""
    written = []
    table = variants(args.rec_weights)
    for variant, overrides in table.items():
        for seed in args.seeds:
            run_id = f"sg2x2-{variant}-seed{seed}{suffix}"
            config = {
                "variant": variant,
                "seed": seed,
                "dataset": str(args.training_dataset),
                "evaluation_dataset": str(args.evaluation_dataset),
                "output_dir": str(args.experiment_root / "runs" / run_id),
                "flow_samples": 14 if overrides["architecture"] != "direct_flow" else 1,
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
                "checkpoint_every": args.max_steps,
                "wandb": {
                    "entity": "rl2-group",
                    "project": args.wandb_project,
                    "group": args.wandb_group,
                    "id": run_id,
                    "name": run_id,
                    "resume": "never",
                    "mode": "offline",
                    "dir": str(args.experiment_root / "wandb"),
                },
                **overrides,
            }
            path = args.output_dir / f"{run_id}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            written.append(str(path))
    manifest = {
        "schema_version": 1,
        "variants": list(table),
        "seeds": args.seeds,
        "primary_metric": "validation_generation_symmetric_nn_mse",
        "comparison_contract": (
            "identical frozen seed-4242 evaluation cloud (first "
            f"{args.evaluation_particles} validation particles) for every run; "
            "report each seed plus mean and sample standard deviation"
        ),
        "pass_threshold": None,
        "checkpoint_every": args.max_steps,
        "run_suffix": args.run_suffix,
        "datasets": {
            "training": {"path": str(args.training_dataset.resolve()), "sha256": sha256(args.training_dataset)},
            "evaluation": {"path": str(args.evaluation_dataset.resolve()), "sha256": sha256(args.evaluation_dataset)},
        },
        "configs": written,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "configs.txt").write_text("\n".join(written) + "\n")
    print(f"wrote {len(written)} configs to {args.output_dir}")


if __name__ == "__main__":
    main()
