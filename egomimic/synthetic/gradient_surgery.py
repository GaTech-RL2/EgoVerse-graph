"""Gradient routing utilities for synthetic Action Flow diagnostics."""

from __future__ import annotations

import torch

from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow

ENCODER_GRADIENT_SURGERIES = {"none", "protect_reconstruction_from_flow"}


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
