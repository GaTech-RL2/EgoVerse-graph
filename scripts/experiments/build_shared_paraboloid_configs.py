#!/usr/bin/env python3
"""Build the paired independent-versus-shared paraboloid experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    for name in ("shallow", "steep"):
        parser.add_argument(f"--{name}-train", type=Path, required=True)
        parser.add_argument(f"--{name}-eval", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-steps", type=int, default=200_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse config directory: {args.output_dir}")
    datasets = {
        "shallow": {"train": args.shallow_train, "eval": args.shallow_eval, "curvature": 0.125},
        "steep": {"train": args.steep_train, "eval": args.steep_eval, "curvature": 0.5},
    }
    for values in datasets.values():
        for key in ("train", "eval"):
            if not values[key].is_file():
                raise FileNotFoundError(values[key])
    args.output_dir.mkdir(parents=True)
    config_paths = []

    def common(run_id: str, seed: int) -> dict:
        return {
            "seed": seed,
            "output_dir": str(args.experiment_root / "runs" / run_id),
            "flow_samples": 14,
            "learning_rate": 3e-4,
            "max_steps": args.max_steps,
            "inference_steps": 32,
            "evaluation_particles": 2048,
            "log_every": 100,
            "checkpoint_every": 50_000,
            "wandb": {
                "entity": "rl2-group",
                "project": "synthetic-shared-paraboloid",
                "id": run_id,
                "name": run_id,
                "resume": "never",
                "dir": str(args.experiment_root / "wandb"),
            },
        }

    for seed in args.seeds:
        for name, values in datasets.items():
            tag = "k0125" if name == "shallow" else "k0500"
            direct_id = f"gaussian-paraboloid-{tag}-direct-seed{seed}-v1"
            direct = {
                **common(direct_id, seed),
                "architecture": "direct_flow",
                "variant": "independent_direct",
                "dataset": str(values["train"]),
                "evaluation_dataset": str(values["eval"]),
                "source_key": "source_gaussian_3d",
                "batch_size": 512,
                "model": {"data_dim": 3, "field_width": 128, "field_depth": 4},
            }
            action_id = f"gaussian-paraboloid-{tag}-g-rec100-seed{seed}-v1"
            action = {
                **common(action_id, seed),
                "architecture": "action_adapter_flow",
                "variant": "independent_action_flow_g_rec100",
                "dataset": str(values["train"]),
                "evaluation_dataset": str(values["eval"]),
                "source_key": "source_gaussian_latent",
                "batch_size": 512,
                "adapter_objective": "action_velocity",
                "lambda_reconstruction": 100.0,
                "lambda_scale": 1.0,
                "lambda_path": 0.0,
                "lambda_action_velocity": 1.0,
                "surface_kind": "paraboloid",
                "paraboloid_curvature": values["curvature"],
                "diagnostic_noise_samples": 4096,
                "model": {
                    "latent_dim": 8,
                    "adapter_family": "nonlinear",
                    "residual_width": 32,
                    "residual_depth": 2,
                    "field_width": 128,
                    "field_depth": 4,
                },
            }
            for run_id, config in ((direct_id, direct), (action_id, action)):
                path = args.output_dir / f"{run_id}.json"
                path.write_text(json.dumps(config, indent=2) + "\n")
                config_paths.append(str(path))
        joint_id = f"gaussian-paraboloid-k0125-k0500-shared-g-rec100-seed{seed}-v1"
        joint = {
            **common(joint_id, seed),
            "architecture": "multi_action_adapter_flow",
            "variant": "shared_action_flow_g_rec100",
            "datasets": {name: str(values["train"]) for name, values in datasets.items()},
            "evaluation_datasets": {name: str(values["eval"]) for name, values in datasets.items()},
            "curvatures": {name: values["curvature"] for name, values in datasets.items()},
            "source_key": "source_gaussian_latent",
            "batch_size_per_embodiment": 512,
            "lambda_reconstruction": 100.0,
            "lambda_scale": 1.0,
            "lambda_action_velocity": 1.0,
            "model": {
                "embodiments": list(datasets),
                "latent_dim": 8,
                "residual_width": 32,
                "residual_depth": 2,
                "field_width": 128,
                "field_depth": 4,
            },
        }
        path = args.output_dir / f"{joint_id}.json"
        path.write_text(json.dumps(joint, indent=2) + "\n")
        config_paths.append(str(path))
    manifest = {
        "schema_version": 1,
        "comparison": "independent direct FM and independent G rec100 versus private-interface/shared-field G rec100",
        "seeds": args.seeds,
        "curvatures": {name: values["curvature"] for name, values in datasets.items()},
        "datasets": {
            name: {
                key: {"path": str(values[key].resolve()), "sha256": sha256(values[key])}
                for key in ("train", "eval")
            }
            for name, values in datasets.items()
        },
        "configs": config_paths,
        "fairness": {
            "per_embodiment_batch": 512,
            "steps": args.max_steps,
            "learning_rate": 3e-4,
            "flow_samples": 14,
            "inference_steps": 32,
            "evaluation_particles": 2048,
            "note": "joint run sees the same examples per embodiment per optimizer step; compare total joint system against the pair of independent systems",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
