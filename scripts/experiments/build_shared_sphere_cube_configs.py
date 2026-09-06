#!/usr/bin/env python3
"""Build the independent-versus-shared sphere/cube experiment matrix."""

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
    for name in ("sphere", "cube"):
        parser.add_argument(f"--{name}-train", type=Path, required=True)
        parser.add_argument(f"--{name}-eval", type=Path, required=True)
    parser.add_argument("--sphere-radius", type=float, required=True)
    parser.add_argument("--cube-half-extent", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-steps", type=int, default=200_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse config directory: {args.output_dir}")
    datasets = {
        "sphere": {
            "train": args.sphere_train,
            "eval": args.sphere_eval,
            "surface_spec": {"kind": "sphere", "radius": args.sphere_radius},
        },
        "cube": {
            "train": args.cube_train,
            "eval": args.cube_eval,
            "surface_spec": {
                "kind": "cube",
                "half_extent": args.cube_half_extent,
            },
        },
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
                "project": "synthetic-shared-sphere-cube",
                "id": run_id,
                "name": run_id,
                "resume": "never",
                "dir": str(args.experiment_root / "wandb"),
            },
        }

    for seed in args.seeds:
        for name, values in datasets.items():
            direct_id = f"gaussian-{name}-surface-direct-seed{seed}-v1"
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
            action_id = f"gaussian-{name}-surface-g-rec100-seed{seed}-v1"
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
                "surface_kind": name,
                **(
                    {"sphere_radius": args.sphere_radius}
                    if name == "sphere"
                    else {"cube_half_extent": args.cube_half_extent}
                ),
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

        joint_id = f"gaussian-sphere-cube-shared-g-rec100-seed{seed}-v1"
        joint = {
            **common(joint_id, seed),
            "architecture": "multi_action_adapter_flow",
            "variant": "shared_action_flow_g_rec100",
            "datasets": {name: str(values["train"]) for name, values in datasets.items()},
            "evaluation_datasets": {
                name: str(values["eval"]) for name, values in datasets.items()
            },
            "surface_specs": {
                name: values["surface_spec"] for name, values in datasets.items()
            },
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
        "comparison": "sphere/cube independent direct FM and independent G rec100 versus private-interface/shared-field G rec100",
        "seeds": args.seeds,
        "surface_specs": {
            name: values["surface_spec"] for name, values in datasets.items()
        },
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
            "scale_contract": "equal expected squared target radius",
            "note": "joint run sees the same examples per embodiment per optimizer step; compare the joint system against each pair of independent systems",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
