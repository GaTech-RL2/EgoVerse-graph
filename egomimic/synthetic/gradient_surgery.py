"""Gradient routing utilities for synthetic Action Flow diagnostics."""

from __future__ import annotations

import torch

from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow

ENCODER_GRADIENT_SURGERIES = {"none", "protect_reconstruction_from_flow"}


def action_adapter_gradient_telemetry(
    losses: dict[str, torch.Tensor],
    model: SyntheticActionAdapterFlow,
) -> dict[str, torch.Tensor]:
    """Measure raw component-gradient cosines on shared parameters.

    FM/action velocity is compared on the field parameters reached by both
    losses. Action velocity/reconstruction is compared on their shared action
    adapter parameters. This diagnostic does not alter optimizer gradients.
    """
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise ValueError("gradient telemetry requires trainable parameters")

    components = {
        "fm": losses["flow_loss"],
        "action_velocity": losses["action_velocity_loss"],
        "reconstruction": losses["reconstruction_loss"],
    }
    gradients: dict[str, dict[int, torch.Tensor]] = {}
    metrics: dict[str, torch.Tensor] = {}
    parameters = tuple(parameter for _, parameter in named_parameters)
    for label, loss in components.items():
        raw = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        active = {
            index: gradient.detach()
            for index, gradient in enumerate(raw)
            if gradient is not None
        }
        if not active:
            raise RuntimeError(f"{label} reaches no trainable parameters")
        norm = sum(
            (gradient.float().square().sum() for gradient in active.values()),
            torch.zeros((), device=loss.device, dtype=torch.float32),
        ).sqrt()
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"non-finite {label} gradient norm")
        gradients[label] = active
        metrics[f"gradient_norm_{label}"] = norm

    for left, right in (
        ("fm", "action_velocity"),
        ("action_velocity", "reconstruction"),
    ):
        shared = tuple(index for index in gradients[left] if index in gradients[right])
        if not shared:
            raise RuntimeError(f"{left} and {right} share no trainable parameters")
        zero = torch.zeros((), device=components[left].device, dtype=torch.float32)
        dot = sum(
            (
                gradients[left][index].float()
                * gradients[right][index].float()
            ).sum()
            for index in shared
        )
        left_norm = sum(
            gradients[left][index].float().square().sum() for index in shared
        ).sqrt()
        right_norm = sum(
            gradients[right][index].float().square().sum() for index in shared
        ).sqrt()
        denominator = left_norm * right_norm
        epsilon = torch.finfo(denominator.dtype).eps
        defined = denominator > epsilon
        cosine = torch.where(
            defined,
            (dot / denominator.clamp_min(epsilon)).clamp(-1.0, 1.0),
            zero,
        )
        if not bool(torch.isfinite(cosine)):
            raise RuntimeError(f"non-finite {left}/{right} gradient cosine")
        pair = f"{left}_{right}"
        metrics[f"gradient_cosine_{pair}"] = cosine.detach()
        metrics[f"gradient_cosine_defined_{pair}"] = defined.to(torch.float32)
        metrics[f"gradient_intersection_parameter_count_{pair}"] = zero.new_tensor(
            float(sum(named_parameters[index][1].numel() for index in shared))
        )
    return metrics


def project_flow_gradient_against_reconstruction(
    flow_gradients: tuple[torch.Tensor, ...],
    reconstruction_gradients: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """Remove only the flow component that conflicts with reconstruction."""
    if len(flow_gradients) != len(reconstruction_gradients):
        raise ValueError("gradient tuples must have the same length")
    if not flow_gradients:
        raise ValueError("gradient tuples must not be empty")
    flow_norm_squared = sum(gradient.square().sum() for gradient in flow_gradients)
    reconstruction_norm_squared = sum(
        gradient.square().sum() for gradient in reconstruction_gradients
    )
    dot_product = sum(
        (flow * reconstruction).sum()
        for flow, reconstruction in zip(
            flow_gradients, reconstruction_gradients, strict=True
        )
    )
    epsilon = torch.finfo(dot_product.dtype).eps
    has_reference = reconstruction_norm_squared > epsilon
    conflicts = has_reference & (dot_product < 0)
    coefficient = torch.where(
        conflicts,
        dot_product / reconstruction_norm_squared.clamp_min(epsilon),
        torch.zeros_like(dot_product),
    )
    projected = tuple(
        flow - coefficient * reconstruction
        for flow, reconstruction in zip(
            flow_gradients, reconstruction_gradients, strict=True
        )
    )
    projected_norm_squared = sum(gradient.square().sum() for gradient in projected)
    cosine = dot_product / (
        flow_norm_squared.sqrt() * reconstruction_norm_squared.sqrt()
    ).clamp_min(epsilon)
    metrics = {
        "encoder_flow_reconstruction_gradient_cosine": cosine.detach(),
        "encoder_flow_projection_active": conflicts.to(dot_product.dtype).detach(),
        "encoder_flow_gradient_norm": flow_norm_squared.sqrt().detach(),
        "encoder_flow_projected_gradient_norm": projected_norm_squared.sqrt().detach(),
        "encoder_reconstruction_gradient_norm": reconstruction_norm_squared.sqrt().detach(),
    }
    return projected, metrics


def backward_with_encoder_gradient_surgery(
    losses: dict[str, torch.Tensor],
    model: SyntheticActionAdapterFlow,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Backpropagate, optionally replacing only the encoder's flow gradient."""
    if mode not in ENCODER_GRADIENT_SURGERIES:
        raise ValueError(f"unknown encoder gradient surgery: {mode}")
    if mode == "none":
        losses["loss"].backward()
        return {}
    parameters = tuple(
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("encoder gradient surgery requires a trainable encoder")
    flow_gradients_raw = torch.autograd.grad(
        losses["flow_loss"], parameters, retain_graph=True, allow_unused=True
    )
    reconstruction_gradients_raw = torch.autograd.grad(
        losses["reconstruction_loss"],
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    flow_gradients = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, flow_gradients_raw, strict=True)
    )
    reconstruction_gradients = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(
            parameters, reconstruction_gradients_raw, strict=True
        )
    )
    projected, metrics = project_flow_gradient_against_reconstruction(
        flow_gradients, reconstruction_gradients
    )
    losses["loss"].backward()
    with torch.no_grad():
        for parameter, flow, replacement in zip(
            parameters, flow_gradients, projected, strict=True
        ):
            if parameter.grad is None:
                parameter.grad = replacement.clone()
            else:
                parameter.grad.add_(replacement - flow)
    return metrics
