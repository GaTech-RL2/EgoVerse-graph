#!/usr/bin/env python3
"""Train and checkpoint a synthetic shared-latent flow benchmark."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow
from egomimic.synthetic.shared_latent_flow import (
    SyntheticDirectFlow,
    SyntheticSharedLatentFlow,
)


def energy_distance(samples: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Exact empirical energy distance between two validation point clouds."""
    cross = torch.cdist(samples, targets).mean()
    within_samples = torch.cdist(samples, samples).mean()
    within_targets = torch.cdist(targets, targets).mean()
    return 2.0 * cross - within_samples - within_targets


_CHECKPOINT_STEP = re.compile(r"global-step-(\d+)\.pt$")
_RESUME_MUTABLE_CONFIG_KEYS = {"max_steps", "wandb"}


def _validate_resume_config(config: dict, saved_config: dict, step: int) -> None:
    current = {k: v for k, v in config.items() if k not in _RESUME_MUTABLE_CONFIG_KEYS}
    saved = {
        k: v for k, v in saved_config.items() if k not in _RESUME_MUTABLE_CONFIG_KEYS
    }
    if current != saved:
        raise ValueError(
            "resume config differs from checkpoint outside max_steps/wandb"
        )
    if int(config["max_steps"]) <= step:
        raise ValueError(
            f"max_steps {config['max_steps']} must exceed checkpoint step {step}"
        )


def _future_checkpoint_steps(checkpoint_dir: Path, start_step: int) -> list[int]:
    steps = []
    for path in checkpoint_dir.glob("*.pt"):
        match = _CHECKPOINT_STEP.search(path.name)
        if match and int(match.group(1)) > start_step:
            steps.append(int(match.group(1)))
    return sorted(steps)


def _capture_rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "batch_generator": generator.get_state(),
    }


def _restore_rng_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])
    generator.set_state(state["batch_generator"].cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    checkpoint_every = int(config.get("checkpoint_every", 50_000))
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    output = Path(config["output_dir"])
    if args.resume is None:
        if output.exists():
            raise FileExistsError(f"refusing to reuse output directory: {output}")
        (output / "checkpoints").mkdir(parents=True)
    else:
        if not output.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {output}")
        expected_parent = (output / "checkpoints").resolve()
        if args.resume.resolve().parent != expected_parent:
            raise ValueError("resume checkpoint must belong to output_dir/checkpoints")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = np.load(config["dataset"], allow_pickle=False)
    train_indices = np.flatnonzero(data["split"] == 0)
    val_indices = np.flatnonzero(data["split"] == 1)
    source_key = config.get("source_key", "source_2d")
    source = torch.from_numpy(data[source_key]).float()
    target = torch.from_numpy(data["target_3d"]).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = config.get("architecture", "shared_latent")
    if architecture == "shared_latent":
        model = SyntheticSharedLatentFlow(**config["model"]).to(device)
    elif architecture == "direct_flow":
        model = SyntheticDirectFlow(**config["model"]).to(device)
    elif architecture == "action_adapter_flow":
        model = SyntheticActionAdapterFlow(**config["model"]).to(device)
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    if source.shape[-1] != model.latent_dim:
        raise ValueError(
            f"dataset {source_key} width {source.shape[-1]} does not match "
            f"model latent_dim {model.latent_dim}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(seed + 2)
    start_step = 0
    resume_rng_restored = False
    if args.resume is not None:
        checkpoint_state = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        start_step = int(checkpoint_state["step"])
        _validate_resume_config(config, checkpoint_state["config"], start_step)
        future_steps = _future_checkpoint_steps(output / "checkpoints", start_step)
        if future_steps:
            raise FileExistsError(
                f"refusing to overwrite checkpoints after resume step: {future_steps}"
            )
        model.load_state_dict(checkpoint_state["model"], strict=True)
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        if "rng" in checkpoint_state:
            _restore_rng_state(checkpoint_state["rng"], generator)
            resume_rng_restored = True
        else:
            continuation_seed = seed + start_step
            random.seed(continuation_seed)
            np.random.seed(continuation_seed)
            torch.manual_seed(continuation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(continuation_seed)
            generator.manual_seed(seed + 2 + start_step)
    wandb_run = None
    if config.get("wandb"):
        import wandb

        wandb_run = wandb.init(config=config, **config["wandb"])
    log_path = output / "metrics.jsonl"
    for step in range(start_step + 1, config["max_steps"] + 1):
        chosen = train_indices[
            torch.randint(
                len(train_indices), (config["batch_size"],), generator=generator
            ).numpy()
        ]
        batch_source = source[chosen].to(device)
        batch_target = target[chosen].to(device)
        if architecture == "shared_latent":
            losses = model.losses(
                batch_source,
                batch_target,
                method=config["method"],
                flow_samples=config["flow_samples"],
                reconstruction_noise_min=config["reconstruction_noise_range"][0],
                reconstruction_noise_max=config["reconstruction_noise_range"][1],
                reconstruction_updates_field=config.get(
                    "reconstruction_updates_field", True
                ),
            )
        elif architecture == "direct_flow":
            losses = model.losses(
                batch_source,
                batch_target,
                flow_samples=config.get("flow_samples", 1),
            )
        else:
            losses = model.losses(
                batch_target,
                objective=config["adapter_objective"],
                flow_samples=config.get("flow_samples", 1),
                lambda_reconstruction=config.get("lambda_reconstruction", 1.0),
                lambda_scale=config.get("lambda_scale", 1.0),
                lambda_path=config.get("lambda_path", 1.0),
                noise=batch_source,
            )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        if step == 1 or step % config["log_every"] == 0:
            row = {
                "step": step,
                **{key: float(value.detach()) for key, value in losses.items()},
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)
        if step % checkpoint_every == 0 or step == config["max_steps"]:
            epoch_equivalent = (step * config["batch_size"]) // len(train_indices)
            checkpoint = (
                output
                / "checkpoints"
                / f"epoch-equivalent-{epoch_equivalent:06d}-global-step-{step:06d}.pt"
            )
            torch.save(
                {
                    "step": step,
                    "epoch_equivalent": epoch_equivalent,
                    "config": config,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": _capture_rng_state(generator),
                    "resume": {
                        "checkpoint": str(args.resume) if args.resume else None,
                        "rng_restored": resume_rng_restored,
                    },
                },
                checkpoint,
            )
    model.eval()
    with torch.inference_mode():
        if "evaluation_dataset" in config:
            src, tgt = SyntheticTrajectoryEval.load_validation_data(
                config["evaluation_dataset"],
                source_key,
                int(config["evaluation_particles"]),
            )
            src, tgt = src.to(device), tgt.to(device)
        else:
            src = source[val_indices].to(device)
            tgt = target[val_indices].to(device)
        if architecture in {"shared_latent", "action_adapter_flow"}:
            clean_reconstruction = model.decoder(model.encoder(tgt))
            trajectory = SyntheticTrajectoryEval.export(
                model,
                src,
                tgt,
                output / "validation_trajectory.npz",
                steps=config["inference_steps"],
            )
            generated = trajectory[-1]
            particles = {
                "source": src.cpu().numpy(),
                "target": tgt.cpu().numpy(),
                "reconstruction": clean_reconstruction.cpu().numpy(),
                "generation": generated.cpu().numpy(),
            }
        else:
            clean_reconstruction = None
            trajectory = SyntheticTrajectoryEval.export(
                model,
                src,
                tgt,
                output / "validation_trajectory.npz",
                steps=config["inference_steps"],
            )
            generated = trajectory[-1]
            particles = {
                "source": src.cpu().numpy(),
                "target": tgt.cpu().numpy(),
                "generation": generated.cpu().numpy(),
            }
        np.savez_compressed(output / "validation_particles.npz", **particles)
    summary = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    summary["validation_generation_energy_distance"] = float(
        energy_distance(generated, tgt)
    )
    summary["validation_generation_symmetric_nn_mse"] = float(
        SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(generated, tgt)
    )
    if clean_reconstruction is not None:
        summary["validation_reconstruction_mse"] = float(
            (clean_reconstruction - tgt).square().mean()
        )
    if architecture == "action_adapter_flow":
        fixed_noise = torch.randn(
            int(config.get("diagnostic_noise_samples", 4096)),
            model.latent_dim,
            generator=torch.Generator(device="cpu").manual_seed(seed + 30_000),
        ).to(device)
        decoded_noise = model.decoder(fixed_noise)
        radii = decoded_noise.norm(dim=-1)
        singular_values = model.decoder_jacobian_singular_values(fixed_noise[:128])
        diagnostic_time = torch.linspace(0.0, 1.0, len(tgt), device=device)[:, None]
        summary.update(
            {
                "validation_path_consistency_mse": float(
                    model.path_consistency_loss(tgt, src, diagnostic_time)
                ),
                "validation_scale_loss": float(model.scale_loss(fixed_noise)),
                "validation_latent_code_rms": float(
                    model.encoder(tgt).square().mean().sqrt()
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
                "validation_decoded_noise_radius_q50": float(
                    torch.quantile(radii, 0.50)
                ),
                "validation_decoded_noise_radius_q90": float(
                    torch.quantile(radii, 0.90)
                ),
                "validation_decoded_noise_radius_q99": float(
                    torch.quantile(radii, 0.99)
                ),
                "validation_decoded_noise_radius_max": float(radii.max()),
                "validation_torus_surface_rmse": float(
                    SyntheticTrajectoryEval.torus_surface_rmse(
                        generated,
                        major_radius=float(config.get("torus_major_radius", 2.0)),
                        minor_radius=float(config.get("torus_minor_radius", 0.65)),
                    )
                ),
                **{
                    f"validation_{key}": float(value)
                    for key, value in SyntheticTrajectoryEval.torus_angular_coverage(
                        generated,
                        tgt,
                        bins=int(config.get("angular_bins", 16)),
                        major_radius=float(config.get("torus_major_radius", 2.0)),
                    ).items()
                },
            }
        )
        trajectory_radii = trajectory.norm(dim=-1)
        quantiles = torch.tensor([0.5, 0.9, 0.99], device=device)
        summary["validation_decoded_trajectory_radius_quantiles"] = [
            [float(value) for value in torch.quantile(row, quantiles)]
            for row in trajectory_radii
        ]
        summary["validation_decoded_trajectory_times"] = [
            index / (len(trajectory_radii) - 1)
            for index in range(len(trajectory_radii))
        ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run is not None:
        # Keep structured trajectory diagnostics in summary.json. W&B summary
        # logging is deliberately scalar-only so backend coercion cannot turn a
        # successful training run into a late logging failure.
        wandb_run.log(
            {key: value for key, value in summary.items() if np.isscalar(value)},
            step=config["max_steps"],
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
