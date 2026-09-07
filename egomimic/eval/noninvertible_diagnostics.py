"""Terminal checks for noninvertible torus trials and their direct-FM control.

Decoded training codes and Gaussian-start generation are different distributions.
The likelihood model instead reports its sampled reference boundary prediction,
which must pass through the shared denoiser before action decoding.
"""

from __future__ import annotations

import math

import torch
from torch.func import jacrev, vmap

from .synthetic_trajectory_eval import SyntheticTrajectoryEval


def _scale_health(decoded: torch.Tensor) -> torch.Tensor:
    mean = decoded.mean(0)
    centered = decoded - mean
    covariance = centered.T @ centered / (len(decoded) - 1)
    identity = torch.eye(decoded.shape[-1], device=decoded.device, dtype=decoded.dtype)
    return (mean.square().sum() + (covariance - identity).square().sum()) / len(mean)


def _identity(state):
    return state


def _singular_metrics(decoder, state, name):
    singular = torch.linalg.svdvals(vmap(jacrev(decoder))(state))
    return {
        f"validation_decoder_{name}_jacobian_singular_min": float(singular.min()),
        f"validation_decoder_{name}_jacobian_singular_median": float(singular.median()),
        f"validation_decoder_{name}_jacobian_singular_max": float(singular.max()),
    }


@torch.no_grad()
def noninvertible_diagnostics(model, source, target, trajectory, config):
    """Evaluate local losses without changing RNG state or the training objective.

The fixed-noise covariance value is health monitoring, including for likelihood
and direct FM; it is not silently added to either model's objective. Validation
losses retain their own path-noise scale value under ``validation_scale_loss``.
"""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("noninvertible diagnostics require no_grad, not inference_mode")
    architecture = config["architecture"]
    if architecture not in {
        "mmd_endpoint_flow", "graph_section_flow", "latent_bridge_likelihood",
        "direct_flow",
    }:
        raise ValueError(f"unsupported noninvertible diagnostic architecture: {architecture}")
    if len(target) < 2:
        raise ValueError("noninvertible diagnostics require at least two validation points")
    noise_count = int(config.get("diagnostic_noise_samples", 4096))
    if noise_count < 2:
        raise ValueError("diagnostic_noise_samples must be at least two")
    devices = [target.device.index] if target.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        seed = int(config["seed"]) + 30_000
        torch.manual_seed(seed)
        fixed_noise = torch.randn(
            noise_count, model.latent_dim, dtype=target.dtype,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to(target.device)
        if architecture == "direct_flow":
            losses = model.losses(source, target, flow_samples=1)
            decoder = _identity
            clean = target
            endpoint = target
            endpoint_prefix = "validation_decoded_training_code"
        else:
            arguments = {}
            if architecture != "latent_bridge_likelihood":
                arguments["lambda_scale"] = config.get("lambda_scale", 1.0)
            if architecture == "mmd_endpoint_flow":
                arguments["lambda_endpoint"] = config.get("lambda_endpoint", 10.0)
                if "flow_clean_gradient_mode" in config:
                    arguments["flow_clean_gradient_mode"] = config["flow_clean_gradient_mode"]
            losses = model.losses(
                target, flow_samples=1, noise=source, return_diagnostics=True,
                **arguments,
            )
            decoder = model.decoder
            if architecture == "latent_bridge_likelihood":
                clean = model.boundary_code(target, source)
                endpoint_prefix = "validation_reference_boundary_mean"
            else:
                clean = model.encoder(target)
                endpoint_prefix = "validation_decoded_training_code"
            endpoint = decoder(clean)

        decoded_noise = decoder(fixed_noise)
        radii = decoded_noise.norm(dim=-1)
        generated = trajectory[-1]
        summary = {f"validation_{key}": float(value) for key, value in losses.items()}
        summary.update({
            f"{endpoint_prefix}_symmetric_nn_mse": float(
                SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(endpoint, target)
            ),
            f"{endpoint_prefix}_paired_mse": float((endpoint - target).square().mean()),
            "validation_latent_code_rms": float(clean.square().mean().sqrt()),
            "validation_fixed_noise_scale_loss": float(_scale_health(decoded_noise)),
            "validation_decoded_noise_mean_norm": float(decoded_noise.mean(0).norm()),
            "validation_decoded_noise_radius_q50": float(torch.quantile(radii, 0.50)),
            "validation_decoded_noise_radius_q90": float(torch.quantile(radii, 0.90)),
            "validation_decoded_noise_radius_q99": float(torch.quantile(radii, 0.99)),
            "validation_decoded_noise_radius_max": float(radii.max()),
            "validation_generated_endpoint_spread": float(generated.var(dim=0).mean()),
            "validation_trajectory_max_displacement": float(
                (trajectory[-1] - trajectory[0]).norm(dim=-1).max()
            ),
            "validation_torus_surface_rmse": float(
                SyntheticTrajectoryEval.torus_surface_rmse(
                    generated, major_radius=float(config.get("torus_major_radius", 2.0)),
                    minor_radius=float(config.get("torus_minor_radius", 0.65)),
                )
            ),
        })
        summary.update({
            f"validation_{key}": float(value)
            for key, value in SyntheticTrajectoryEval.torus_angular_coverage(
                generated, target, bins=int(config.get("angular_bins", 16)),
                major_radius=float(config.get("torus_major_radius", 2.0)),
            ).items()
        })
        summary.update(_singular_metrics(decoder, fixed_noise[:128], "noise"))
        summary.update(_singular_metrics(decoder, clean[:128], "clean"))
        if not all(math.isfinite(value) for value in summary.values()):
            raise FloatingPointError("non-finite noninvertible terminal diagnostic")
        return summary
