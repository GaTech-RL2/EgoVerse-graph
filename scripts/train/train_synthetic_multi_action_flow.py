#!/usr/bin/env python3
"""Train private action interfaces around one shared synthetic latent field."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.multi_action_adapter_flow import (
    SyntheticMultiActionAdapterFlow,
)
from scripts.train.train_synthetic_manifold import energy_distance


def _load_dataset(path: str | Path, source_key: str) -> dict[str, torch.Tensor]:
    archive = np.load(path, allow_pickle=False)
    return {
        "source": torch.from_numpy(archive[source_key]).float(),
        "target": torch.from_numpy(archive["target_3d"]).float(),
        "train_indices": torch.from_numpy(
            np.flatnonzero(archive["split"] == 0)
        ).long(),
    }


def _export_trajectory(
    model: SyntheticMultiActionAdapterFlow,
    embodiment: str,
    source: torch.Tensor,
    target: torch.Tensor,
    output: Path,
    *,
    steps: int,
) -> torch.Tensor:
    trajectory = model.trajectory(source, embodiment=embodiment, steps=steps)
    expected = (steps + 1, len(target), 3)
    if tuple(trajectory.shape) != expected or not bool(torch.isfinite(trajectory).all()):
        raise RuntimeError(f"invalid {embodiment} trajectory: {tuple(trajectory.shape)}")
    np.savez_compressed(
        output,
        times=torch.linspace(0.0, 1.0, steps + 1).numpy(),
        points=trajectory.detach().cpu().numpy(),
        target=target.detach().cpu().numpy(),
    )
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    (output / "checkpoints").mkdir(parents=True)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    source_key = config.get("source_key", "source_gaussian_latent")
    dataset_paths = config["datasets"]
    evaluation_paths = config["evaluation_datasets"]
    names = list(dataset_paths)
    if names != list(config["model"]["embodiments"]):
        raise ValueError("dataset order must exactly match model embodiments")
    if set(evaluation_paths) != set(names):
        raise ValueError("evaluation datasets must match training embodiments")
    datasets = {
        name: _load_dataset(path, source_key) for name, path in dataset_paths.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyntheticMultiActionAdapterFlow(**config["model"]).to(device)
    for name, dataset in datasets.items():
        if dataset["source"].shape[-1] != model.latent_dim:
            raise ValueError(f"{name} source width does not match latent_dim")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(seed + 2)
    wandb_run = None
    if config.get("wandb"):
        import wandb

        wandb_run = wandb.init(config=config, **config["wandb"])

    batch_size = int(config["batch_size_per_embodiment"])
    checkpoint_every = int(config.get("checkpoint_every", 50_000))
    if batch_size <= 1 or checkpoint_every <= 0:
        raise ValueError("batch size must exceed one and checkpoint cadence be positive")
    log_path = output / "metrics.jsonl"
    for step in range(1, int(config["max_steps"]) + 1):
        per_embodiment = {}
        for name, dataset in datasets.items():
            indices = dataset["train_indices"]
            chosen = indices[
                torch.randint(len(indices), (batch_size,), generator=generator)
            ]
            target = dataset["target"][chosen].to(device)
            noise = dataset["source"][chosen].to(device)
            per_embodiment[name] = model.losses_for_embodiment(
                name,
                target,
                flow_samples=int(config.get("flow_samples", 14)),
                lambda_reconstruction=float(config.get("lambda_reconstruction", 100.0)),
                lambda_scale=float(config.get("lambda_scale", 1.0)),
                lambda_action_velocity=float(config.get("lambda_action_velocity", 1.0)),
                noise=noise,
            )
        total = torch.stack([losses["loss"] for losses in per_embodiment.values()]).mean()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step == 1 or step % int(config["log_every"]) == 0:
            row = {"step": step, "loss": float(total.detach())}
            for name, losses in per_embodiment.items():
                row.update(
                    {
                        f"{name}/{key}": float(value.detach())
                        for key, value in losses.items()
                    }
                )
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)
        if step % checkpoint_every == 0 or step == int(config["max_steps"]):
            total_examples = step * batch_size * len(names)
            epoch_equivalent = total_examples // sum(
                len(dataset["train_indices"]) for dataset in datasets.values()
            )
            checkpoint = output / "checkpoints" / (
                f"epoch-equivalent-{epoch_equivalent:06d}-global-step-{step:06d}.pt"
            )
            torch.save(
                {
                    "step": step,
                    "epoch_equivalent": epoch_equivalent,
                    "config": config,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else [],
                        "batch_generator": generator.get_state(),
                    },
                },
                checkpoint,
            )

    model.eval()
    summary = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "shared_field_parameters": sum(
            parameter.numel() for parameter in model.field.parameters()
        ),
        "embodiments": {},
    }
    with torch.inference_mode():
        for name in names:
            source, target = SyntheticTrajectoryEval.load_validation_data(
                evaluation_paths[name], source_key, int(config["evaluation_particles"])
            )
            source, target = source.to(device), target.to(device)
            reconstruction = model.decode(name, model.encode(name, target))
            trajectory = _export_trajectory(
                model,
                name,
                source,
                target,
                output / f"validation_trajectory_{name}.npz",
                steps=int(config["inference_steps"]),
            )
            generated = trajectory[-1]
            curvature = float(config["curvatures"][name])
            singular_values = model.decoder_jacobian_singular_values(
                name, source[:128]
            )
            summary["embodiments"][name] = {
                "curvature": curvature,
                "validation_generation_energy_distance": float(
                    energy_distance(generated, target)
                ),
                "validation_generation_symmetric_nn_mse": float(
                    SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(
                        generated, target
                    )
                ),
                "validation_reconstruction_mse": float(
                    (reconstruction - target).square().mean()
                ),
                "validation_paraboloid_surface_rmse": float(
                    SyntheticTrajectoryEval.paraboloid_surface_rmse(
                        generated, curvature=curvature
                    )
                ),
                "validation_latent_code_rms": float(
                    model.encode(name, target).square().mean().sqrt()
                ),
                "validation_decoder_jacobian_singular_min": float(
                    singular_values.min()
                ),
                "validation_decoder_jacobian_singular_median": float(
                    singular_values.median()
                ),
                "validation_decoder_jacobian_singular_max": float(
                    singular_values.max()
                ),
            }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run is not None:
        scalar_summary = {
            f"{name}/{key}": value
            for name, values in summary["embodiments"].items()
            for key, value in values.items()
        }
        wandb_run.log(scalar_summary, step=int(config["max_steps"]))
        wandb_run.finish()


if __name__ == "__main__":
    main()
