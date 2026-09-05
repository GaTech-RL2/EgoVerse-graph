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

from egomimic.synthetic.shared_latent_flow import (
    SyntheticDirectFlow,
    SyntheticSharedLatentFlow,
)
from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval


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
        raise ValueError("resume config differs from checkpoint outside max_steps/wandb")
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
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["batch_generator"].cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
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
        checkpoint_state = torch.load(args.resume, map_location=device, weights_only=False)
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
        else:
            losses = model.losses(
                batch_source,
                batch_target,
                flow_samples=config.get("flow_samples", 1),
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
        if step % config["checkpoint_every"] == 0 or step == config["max_steps"]:
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
        src = source[val_indices].to(device)
        tgt = target[val_indices].to(device)
        if architecture == "shared_latent":
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
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.log(summary, step=config["max_steps"])
        wandb_run.finish()


if __name__ == "__main__":
    main()
