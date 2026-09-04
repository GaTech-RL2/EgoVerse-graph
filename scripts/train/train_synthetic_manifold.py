#!/usr/bin/env python3
"""Train and checkpoint a synthetic shared-latent flow benchmark."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from egomimic.synthetic.shared_latent_flow import SyntheticSharedLatentFlow


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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = np.load(config["dataset"], allow_pickle=False)
    train_indices = np.flatnonzero(data["split"] == 0)
    val_indices = np.flatnonzero(data["split"] == 1)
    source_key = config.get("source_key", "source_2d")
    source = torch.from_numpy(data[source_key]).float()
    target = torch.from_numpy(data["target_3d"]).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyntheticSharedLatentFlow(**config["model"]).to(device)
    if source.shape[-1] != model.latent_dim:
        raise ValueError(
            f"dataset {source_key} width {source.shape[-1]} does not match "
            f"model latent_dim {model.latent_dim}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    wandb_run = None
    if config.get("wandb"):
        import wandb

        wandb_run = wandb.init(config=config, **config["wandb"])
    generator = torch.Generator().manual_seed(seed + 2)
    log_path = output / "metrics.jsonl"
    for step in range(1, config["max_steps"] + 1):
        chosen = train_indices[torch.randint(len(train_indices), (config["batch_size"],), generator=generator).numpy()]
        losses = model.losses(
            source[chosen].to(device), target[chosen].to(device),
            method=config["method"], flow_samples=config["flow_samples"],
            reconstruction_noise_min=config["reconstruction_noise_range"][0],
            reconstruction_noise_max=config["reconstruction_noise_range"][1],
        )
        optimizer.zero_grad(set_to_none=True); losses["loss"].backward(); optimizer.step()
        if step == 1 or step % config["log_every"] == 0:
            row = {"step": step, **{key: float(value.detach()) for key, value in losses.items()}}
            with log_path.open("a") as stream: stream.write(json.dumps(row) + "\n")
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
                },
                checkpoint,
            )
    model.eval()
    with torch.inference_mode():
        src = source[val_indices].to(device); tgt = target[val_indices].to(device)
        clean_reconstruction = model.decoder(model.encoder(tgt))
        generated = model.decoder(model.integrate(src, config["inference_steps"]))
        np.savez_compressed(
            output / "validation_particles.npz",
            source=src.cpu().numpy(), target=tgt.cpu().numpy(),
            reconstruction=clean_reconstruction.cpu().numpy(), generation=generated.cpu().numpy(),
        )
    summary = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "validation_reconstruction_mse": float((clean_reconstruction - tgt).square().mean()),
        "validation_generation_paired_mse": float((generated - tgt).square().mean()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.log(summary, step=config["max_steps"])
        wandb_run.finish()


if __name__ == "__main__":
    main()
