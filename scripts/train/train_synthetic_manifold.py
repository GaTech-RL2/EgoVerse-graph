#!/usr/bin/env python3
"""Train and checkpoint a synthetic shared-latent flow benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import sys
import time
from pathlib import Path


class _CheckpointRequests:
    """Latch signals until a completed optimizer step can be saved safely."""

    def __init__(self) -> None:
        self.received = 0
        self.saved = 0

    def request(self, _signum: int, _frame: object) -> None:
        self.received += 1

    def install(self) -> None:
        if hasattr(signal, "SIGUSR2"):
            signal.signal(signal.SIGUSR2, self.request)


_checkpoint_requests = _CheckpointRequests()
if __name__ == "__main__":
    # Imports/model initialization can exceed the scheduler's warning interval.
    # Do not reset this latch in main(), or an early request would be lost.
    _checkpoint_requests.install()

import numpy as np
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from egomimic.eval.noninvertible_diagnostics import noninvertible_diagnostics
from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow
from egomimic.synthetic.decoder_inversion_flow import SyntheticDecoderInversionFlow
from egomimic.synthetic.endpoint_lift_flow import SyntheticEndpointLiftFlow
from egomimic.synthetic.gaussian_relift_flow import SyntheticGaussianReliftFlow
from egomimic.synthetic.gradient_surgery import (
    ENCODER_GRADIENT_SURGERIES,
    action_adapter_gradient_telemetry,
    backward_with_encoder_gradient_surgery,
)
from egomimic.synthetic.latent_bridge_likelihood import SyntheticLatentBridgeLikelihood
from egomimic.synthetic.noninvertible_endpoint_flow import (
    SyntheticGraphSectionFlow,
    SyntheticMMDEndpointFlow,
)
from egomimic.synthetic.projected_invertible_flow import (
    SyntheticProjectedInvertibleFlow,
)
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


def _atomic_torch_save(state: dict, checkpoint: Path) -> None:
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)


def _flow_clean_gradient_kwargs(config: dict) -> dict:
    # Absence must preserve the legacy clean_gradient_mode semantics.
    if "flow_clean_gradient_mode" not in config:
        return {}
    return {"flow_clean_gradient_mode": config["flow_clean_gradient_mode"]}


def _periodic_generation_metrics(model, source, target, *, steps: int) -> dict:
    """Measure current generation without advancing training RNGs or mode state."""
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    training_modes = [(module, module.training) for module in model.modules()]
    devices = [source.device.index] if source.is_cuda else []
    if source.is_cuda:
        torch.cuda.synchronize(source.device)
    started = time.monotonic()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            points = SyntheticTrajectoryEval.evaluate(model, source, target, steps=steps)
            score = float(SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(
                points[-1], target
            ))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, training in training_modes:
            module.training = training
    return {
        "generation_symmetric_nn_mse": score,
        "generation_eval_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    checkpoint_every = int(config.get("checkpoint_every", 50_000))
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    generation_log_every = int(config.get("generation_log_every", 0))
    if generation_log_every < 0:
        raise ValueError("generation_log_every must be nonnegative")
    if generation_log_every and not all(
        key in config for key in ("evaluation_dataset", "evaluation_particles", "inference_steps")
    ):
        raise ValueError("periodic generation requires explicit evaluation data, count and steps")
    gradient_telemetry_every = int(config.get("gradient_telemetry_every", 0))
    if gradient_telemetry_every < 0:
        raise ValueError("gradient_telemetry_every must be nonnegative")
    if gradient_telemetry_every and (
        config.get("architecture") != "action_adapter_flow"
        or config.get("adapter_objective") != "action_velocity"
    ):
        raise ValueError(
            "gradient telemetry requires action_adapter_flow with action_velocity"
        )
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
    encoder_gradient_surgery = config.get("encoder_gradient_surgery", "none")
    if encoder_gradient_surgery not in ENCODER_GRADIENT_SURGERIES:
        raise ValueError(
            f"unknown encoder gradient surgery: {encoder_gradient_surgery}"
        )
    if encoder_gradient_surgery != "none" and architecture != "action_adapter_flow":
        raise ValueError(
            "encoder gradient surgery is only supported for action_adapter_flow"
        )
    if architecture == "shared_latent":
        model = SyntheticSharedLatentFlow(**config["model"]).to(device)
    elif architecture == "direct_flow":
        model = SyntheticDirectFlow(**config["model"]).to(device)
    elif architecture == "action_adapter_flow":
        model = SyntheticActionAdapterFlow(**config["model"]).to(device)
    elif architecture == "decoder_inversion_flow":
        model = SyntheticDecoderInversionFlow(**config["model"]).to(device)
    elif architecture == "projected_invertible_flow":
        model = SyntheticProjectedInvertibleFlow(**config["model"]).to(device)
    elif architecture == "endpoint_lift_flow":
        model = SyntheticEndpointLiftFlow(**config["model"]).to(device)
    elif architecture == "gaussian_relift_flow":
        model = SyntheticGaussianReliftFlow(**config["model"]).to(device)
    elif architecture == "mmd_endpoint_flow":
        model = SyntheticMMDEndpointFlow(**config["model"]).to(device)
    elif architecture == "graph_section_flow":
        model = SyntheticGraphSectionFlow(**config["model"]).to(device)
    elif architecture == "latent_bridge_likelihood":
        model = SyntheticLatentBridgeLikelihood(**config["model"]).to(device)
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
    generation_data = None
    if generation_log_every:
        generation_source, generation_target = SyntheticTrajectoryEval.load_validation_data(
            config["evaluation_dataset"], source_key, int(config["evaluation_particles"])
        )
        generation_data = generation_source.to(device), generation_target.to(device)
    training_started = time.monotonic()
    for step in range(start_step + 1, config["max_steps"] + 1):
        chosen = train_indices[
            torch.randint(
                len(train_indices), (config["batch_size"],), generator=generator
            ).numpy()
        ]
        batch_source = source[chosen].to(device)
        batch_target = target[chosen].to(device)
        if config.get("training_noise", "dataset") == "fresh_gaussian":
            batch_source = torch.randn_like(batch_source)
        elif config.get("training_noise", "dataset") != "dataset":
            raise ValueError("training_noise must be dataset or fresh_gaussian")
        log_step = step == 1 or step % config["log_every"] == 0
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
        elif architecture == "action_adapter_flow":
            losses = model.losses(
                batch_target,
                objective=config["adapter_objective"],
                flow_samples=config.get("flow_samples", 1),
                lambda_reconstruction=config.get("lambda_reconstruction", 1.0),
                lambda_scale=config.get("lambda_scale", 1.0),
                lambda_path=config.get("lambda_path", 1.0),
                lambda_action_velocity=config.get("lambda_action_velocity", 1.0),
                clean_gradient_mode=config.get("clean_gradient_mode", "full"),
                noise=batch_source,
                **_flow_clean_gradient_kwargs(config),
            )
        elif architecture in {"endpoint_lift_flow", "gaussian_relift_flow"}:
            objective_args = (
                {"objective": config["endpoint_objective"]}
                if architecture == "endpoint_lift_flow" else {}
            )
            losses = model.losses(
                batch_target,
                flow_samples=config.get("flow_samples", 1),
                lambda_scale=config.get("lambda_scale", 1.0),
                noise=batch_source,
                return_diagnostics=log_step,
                **objective_args,
            )
        elif architecture == "projected_invertible_flow":
            losses = model.losses(
                batch_target,
                flow_samples=config.get("flow_samples", 1),
                lambda_scale=config.get("lambda_scale", 1.0),
                lambda_latent_flow=config.get("lambda_latent_flow", 0.0),
                noise=batch_source,
            )
        elif architecture in {
            "mmd_endpoint_flow", "graph_section_flow", "latent_bridge_likelihood"
        }:
            objective_args = {}
            if architecture != "latent_bridge_likelihood":
                objective_args["lambda_scale"] = config.get("lambda_scale", 1.0)
            if architecture == "mmd_endpoint_flow":
                objective_args["lambda_endpoint"] = config.get("lambda_endpoint", 10.0)
            losses = model.losses(
                batch_target,
                flow_samples=config.get("flow_samples", 1),
                noise=batch_source,
                return_diagnostics=log_step,
                **objective_args,
            )
        else:
            losses = model.losses(
                batch_target,
                flow_samples=config.get("flow_samples", 1),
                inversion_steps=config["inversion_steps"],
                inversion_step_size=config["inversion_step_size"],
                lambda_scale=config.get("lambda_scale", 1.0),
                noise=batch_source,
                training_objective=config.get(
                    "training_objective", "endpoint_difference"
                ),
            )
        optimizer.zero_grad(set_to_none=True)
        telemetry_step = bool(
            gradient_telemetry_every
            and (step == 1 or step % gradient_telemetry_every == 0)
        )
        gradient_metrics = (
            action_adapter_gradient_telemetry(losses, model)
            if telemetry_step
            else {}
        )
        if architecture == "action_adapter_flow":
            surgery_metrics = backward_with_encoder_gradient_surgery(
                losses, model, encoder_gradient_surgery
            )
            duplicate_metrics = gradient_metrics.keys() & surgery_metrics.keys()
            if duplicate_metrics:
                raise RuntimeError(
                    f"duplicate gradient metrics: {sorted(duplicate_metrics)}"
                )
            gradient_metrics.update(surgery_metrics)
        else:
            losses["loss"].backward()
        optimizer.step()
        generation_step = (
            generation_log_every > 0
            and step % generation_log_every == 0
            and step < config["max_steps"]
        )
        if log_step or generation_step or telemetry_step:
            row = {"step": step}
            if log_step:
                row.update({key: float(value.detach()) for key, value in losses.items()})
            row.update({
                key: float(value.detach())
                for key, value in gradient_metrics.items()
            })
            if generation_step:
                row.update(_periodic_generation_metrics(
                    model, *generation_data, steps=int(config["inference_steps"])
                ))
            elapsed = time.monotonic() - training_started
            row["training_elapsed_seconds"] = elapsed
            row["training_steps_per_second"] = (step - start_step) / elapsed
            if device.type == "cuda":
                row["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            if not all(np.isfinite(value) for value in row.values()):
                raise FloatingPointError(f"non-finite logged training metric at step {step}")
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log(row, step=step)
        requested = _checkpoint_requests.received
        if (
            step % checkpoint_every == 0
            or step == config["max_steps"]
            or requested != _checkpoint_requests.saved
        ):
            epoch_equivalent = (step * config["batch_size"]) // len(train_indices)
            checkpoint = (
                output
                / "checkpoints"
                / f"epoch-equivalent-{epoch_equivalent:06d}-global-step-{step:06d}.pt"
            )
            _atomic_torch_save(
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
            # A signal received during serialization belongs to the next save.
            _checkpoint_requests.saved = requested
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed = time.monotonic() - training_started
    model.eval()
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
    # JVP-defined samplers need forward AD, which inference_mode disables.
    with torch.no_grad():
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
    summary["training_elapsed_seconds"] = training_elapsed
    if device.type == "cuda":
        summary["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    if architecture in {
        "mmd_endpoint_flow", "graph_section_flow", "latent_bridge_likelihood"
    } or config.get("noninvertible_diagnostics", False):
        summary.update(noninvertible_diagnostics(model, src, tgt, trajectory, config))
    if architecture in {"endpoint_lift_flow", "gaussian_relift_flow"}:
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed + 30_000)
            fixed_noise = torch.randn(
                int(config.get("diagnostic_noise_samples", 4096)), model.latent_dim,
                device=device,
            )
            aux = torch.randn(len(tgt), model.latent_dim - model.action_dim, device=device)
            clean = model.exact_lift(tgt, aux)
            decoded_noise = model.decoder(fixed_noise)
            singular = model.decoder_jacobian_singular_values(fixed_noise[:128])
            objective_args = ({"objective": config["endpoint_objective"]}
                              if architecture == "endpoint_lift_flow" else {})
            validation_losses = model.losses(
                tgt, noise=src, aux_noise=aux, flow_samples=1,
                lambda_scale=config.get("lambda_scale", 1.0),
                return_diagnostics=True, **objective_args,
            )
            summary.update({f"validation_{key}": float(value) for key, value in validation_losses.items()})
            summary.update({
                "validation_path_noise_scale_loss": float(validation_losses["scale_loss"]),
                "validation_scale_loss": float(model.scale_loss(fixed_noise)),
                "validation_exact_lift_mse": float((model.decoder(clean) - tgt).square().mean()),
                "validation_decoder_jacobian_singular_min": float(singular.min()),
                "validation_decoder_jacobian_singular_median": float(singular.median()),
                "validation_decoder_jacobian_singular_max": float(singular.max()),
                "validation_decoded_noise_mean_norm": float(decoded_noise.mean(0).norm()),
                "validation_decoded_noise_radius_q50": float(decoded_noise.norm(dim=-1).median()),
                "validation_decoded_noise_radius_q99": float(torch.quantile(decoded_noise.norm(dim=-1), .99)),
                "validation_generated_endpoint_spread": float(generated.var(dim=0).mean()),
                "validation_trajectory_max_displacement": float((trajectory[-1] - trajectory[0]).norm(dim=-1).max()),
                "validation_torus_surface_rmse": float(SyntheticTrajectoryEval.torus_surface_rmse(
                    generated, major_radius=float(config.get("torus_major_radius", 2.0)),
                    minor_radius=float(config.get("torus_minor_radius", .65)))),
                **{f"validation_{key}": float(value) for key, value in SyntheticTrajectoryEval.torus_angular_coverage(
                    generated, tgt, bins=int(config.get("angular_bins", 16)),
                    major_radius=float(config.get("torus_major_radius", 2.0))).items()},
            })
    if clean_reconstruction is not None:
        summary["validation_reconstruction_mse"] = float(
            (clean_reconstruction - tgt).square().mean()
        )
    if architecture in {
        "action_adapter_flow",
        "decoder_inversion_flow",
        "projected_invertible_flow",
    }:
        fixed_noise = torch.randn(
            int(config.get("diagnostic_noise_samples", 4096)),
            model.latent_dim,
            generator=torch.Generator(device="cpu").manual_seed(seed + 30_000),
        ).to(device)
        decoded_noise = model.decoder(fixed_noise)
        radii = decoded_noise.norm(dim=-1)
        singular_values = model.decoder_jacobian_singular_values(fixed_noise[:128])
        diagnostic_time = torch.linspace(0.0, 1.0, len(tgt), device=device)[:, None]
        if architecture == "action_adapter_flow":
            diagnostic_clean = model.encoder(tgt)
            inversion_metrics = {}
        elif architecture == "projected_invertible_flow":
            diagnostic_null_noise = torch.randn(
                len(tgt),
                model.latent_dim,
                generator=torch.Generator(device="cpu").manual_seed(seed + 40_000),
            ).to(device)
            diagnostic_clean = model.exact_lift(tgt, diagnostic_null_noise)
            projection = model.projection()
            projected_velocity = model.velocity(
                diagnostic_clean, diagnostic_time
            ) @ projection.T
            full_velocity = model.velocity(diagnostic_clean, diagnostic_time)
            null_velocity = full_velocity - projected_velocity @ projection
            inversion_metrics = {
                "validation_exact_lift_mse": float(
                    (model.decode(diagnostic_clean) - tgt).square().mean()
                ),
                "validation_projection_orthonormality_max_error": float(
                    (
                        projection @ projection.T
                        - torch.eye(
                            model.action_dim,
                            device=device,
                            dtype=projection.dtype,
                        )
                    )
                    .abs()
                    .max()
                ),
                "validation_null_velocity_rms": float(
                    null_velocity.square().mean().sqrt()
                ),
            }
        else:
            diagnostic_initialization = torch.randn(
                len(tgt),
                model.latent_dim,
                generator=torch.Generator(device="cpu").manual_seed(seed + 40_000),
            ).to(device)
            with torch.enable_grad():
                diagnostic_clean, inversion_metrics = model.infer_codes(
                    tgt,
                    diagnostic_initialization,
                    steps=int(config["inversion_steps"]),
                    step_size=float(config["inversion_step_size"]),
                    create_graph=False,
                )
            diagnostic_clean = diagnostic_clean.detach()
            inversion_metrics = {
                f"validation_{key}": float(value.detach())
                for key, value in inversion_metrics.items()
            }
            per_example_rmse = (
                model.decoder(diagnostic_clean) - tgt
            ).square().mean(dim=-1).sqrt()
            inversion_metrics["validation_inversion_failure_rate"] = float(
                (
                    per_example_rmse
                    > float(config.get("inversion_failure_rmse", 0.1))
                )
                .float()
                .mean()
            )
        diagnostic_state = (
            (1.0 - diagnostic_time) * diagnostic_clean + diagnostic_time * src
        )
        if architecture in {"action_adapter_flow", "projected_invertible_flow"}:
            diagnostic_velocity = src - diagnostic_clean
            diagnostic_residual = (
                model.velocity(diagnostic_state, diagnostic_time)
                - diagnostic_velocity
            )
            diagnostic_action_velocity_mse = model.action_velocity_loss(
                diagnostic_state,
                diagnostic_residual,
            )
        elif config.get("training_objective") == "conditional_relifting":
            reference_action = model.decoder(diagnostic_state)
            reference_action_velocity = model.decoder_jvp(
                diagnostic_state,
                src - diagnostic_clean,
            )
            diagnostic_relift_initialization = torch.randn(
                len(tgt),
                model.latent_dim,
                generator=torch.Generator(device="cpu").manual_seed(seed + 50_000),
            ).to(device)
            with torch.enable_grad():
                diagnostic_relifted, relift_metrics = model.infer_codes(
                    reference_action,
                    diagnostic_relift_initialization,
                    steps=int(config["inversion_steps"]),
                    step_size=float(config["inversion_step_size"]),
                    create_graph=False,
                )
            diagnostic_relifted = diagnostic_relifted.detach()
            relift_metrics = {
                f"validation_relift_{key}": float(value.detach())
                for key, value in relift_metrics.items()
            }
            relift_rmse = (
                model.decoder(diagnostic_relifted) - reference_action
            ).square().mean(dim=-1).sqrt()
            relift_metrics["validation_relift_inversion_failure_rate"] = float(
                (
                    relift_rmse
                    > float(config.get("inversion_failure_rmse", 0.1))
                )
                .float()
                .mean()
            )
            second_relift_initialization = torch.randn(
                len(tgt),
                model.latent_dim,
                generator=torch.Generator(device="cpu").manual_seed(seed + 60_000),
            ).to(device)
            with torch.enable_grad():
                second_relifted, _ = model.infer_codes(
                    reference_action,
                    second_relift_initialization,
                    steps=int(config["inversion_steps"]),
                    step_size=float(config["inversion_step_size"]),
                    create_graph=False,
                )
            second_relifted = second_relifted.detach()
            first_prediction = model.decoder_jvp(
                diagnostic_relifted,
                model.velocity(diagnostic_relifted, diagnostic_time),
            )
            second_prediction = model.decoder_jvp(
                second_relifted,
                model.velocity(second_relifted, diagnostic_time),
            )
            relift_metrics["validation_relift_velocity_disagreement_mse"] = float(
                (first_prediction - second_prediction).square().mean()
            )
            relift_metrics["validation_relift_code_pair_rms"] = float(
                (diagnostic_relifted - second_relifted).square().mean().sqrt()
            )
            inversion_metrics.update(relift_metrics)
            diagnostic_action_residual = (
                first_prediction - reference_action_velocity
            )
            diagnostic_action_velocity_mse = (
                diagnostic_action_residual.square().mean()
            )
        else:
            diagnostic_action_residual = model.decoder_jvp(
                diagnostic_state,
                model.velocity(diagnostic_state, diagnostic_time),
            ) - (model.decoder(src) - tgt)
            diagnostic_action_velocity_mse = (
                diagnostic_action_residual.square().mean()
            )
        surface_kind = config.get("surface_kind", "torus")
        if surface_kind == "torus":
            surface_metrics = {
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
        elif surface_kind == "paraboloid":
            if "paraboloid_curvature" not in config:
                raise ValueError(
                    "paraboloid surface diagnostics require paraboloid_curvature"
                )
            surface_metrics = {
                "validation_paraboloid_surface_rmse": float(
                    SyntheticTrajectoryEval.paraboloid_surface_rmse(
                        generated, curvature=float(config["paraboloid_curvature"])
                    )
                )
            }
        elif surface_kind == "sphere":
            surface_metrics = {
                "validation_sphere_surface_rmse": float(
                    SyntheticTrajectoryEval.sphere_surface_rmse(
                        generated, radius=float(config["sphere_radius"])
                    )
                )
            }
        elif surface_kind == "cube":
            surface_metrics = {
                "validation_cube_surface_rmse": float(
                    SyntheticTrajectoryEval.cube_surface_rmse(
                        generated, half_extent=float(config["cube_half_extent"])
                    )
                )
            }
        else:
            raise ValueError(f"unknown surface_kind: {surface_kind}")
        summary.update(
            {
                "validation_path_consistency_mse": float(
                    model.path_consistency_loss(tgt, src, diagnostic_time)
                    if architecture == "action_adapter_flow"
                    else (
                        model.decoder_jvp(
                            diagnostic_state,
                            src - diagnostic_clean,
                        )
                        - (model.decoder(src) - tgt)
                    )
                    .square()
                    .mean()
                ),
                "validation_scale_loss": float(model.scale_loss(fixed_noise)),
                "validation_action_velocity_mse": float(
                    diagnostic_action_velocity_mse
                ),
                "validation_latent_code_rms": float(
                    diagnostic_clean.square().mean().sqrt()
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
                "validation_decoded_noise_mean_norm": float(
                    decoded_noise.mean(dim=0).norm()
                ),
                "validation_generated_endpoint_spread": float(
                    generated.var(dim=0, unbiased=True).mean()
                ),
                **inversion_metrics,
                **surface_metrics,
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
    if not all(np.isfinite(value) for value in summary.values() if np.isscalar(value)):
        raise FloatingPointError("non-finite terminal validation metric")
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
