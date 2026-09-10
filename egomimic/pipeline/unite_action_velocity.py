"""Decoder-aware action-velocity objective for UNITE latent flow."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def decoder_action_velocity_loss(
    decoder: nn.Module,
    latent: torch.Tensor,
    latent_velocity_residual: torch.Tensor,
) -> torch.Tensor:
    """Measure a latent velocity residual in the decoder-induced action metric.

    The returned scalar is ``mean(||J_decoder(latent) residual||^2)``.  A JVP is
    used instead of materializing the full decoder Jacobian.  ``create_graph``
    deliberately remains enabled during training so the loss differentiates
    through the decoder, the latent evaluation point, and the residual.
    """

    if not isinstance(decoder, nn.Module):
        raise TypeError("UNITE-AV decoder must be an nn.Module")
    if not torch.is_tensor(latent) or not torch.is_tensor(latent_velocity_residual):
        raise TypeError("UNITE-AV latent and velocity residual must be tensors")
    if latent.shape != latent_velocity_residual.shape:
        raise ValueError(
            "UNITE-AV latent/residual shape mismatch: "
            f"latent={tuple(latent.shape)} residual="
            f"{tuple(latent_velocity_residual.shape)}"
        )
    if latent.ndim != 3 or int(latent.shape[0]) <= 0:
        raise ValueError("UNITE-AV expects a non-empty (B, N, D) latent tensor")
    if not bool(torch.isfinite(latent).all()) or not bool(
        torch.isfinite(latent_velocity_residual).all()
    ):
        raise RuntimeError("UNITE-AV inputs must be finite")

    # Fused SDPA kernels and activation checkpointing do not provide the stable
    # double backward needed to train through this JVP. Restrict only this pass;
    # ordinary UNITE forward/inference keeps both optimizations.
    checkpointing = getattr(decoder, "gradient_checkpointing", None)
    if checkpointing is not None:
        decoder.gradient_checkpointing = False
    try:
        with sdpa_kernel(SDPBackend.MATH):
            _, action_velocity_residual = torch.autograd.functional.jvp(
                decoder,
                latent,
                latent_velocity_residual,
                create_graph=torch.is_grad_enabled(),
                strict=True,
            )
    finally:
        if checkpointing is not None:
            decoder.gradient_checkpointing = checkpointing
    if action_velocity_residual.ndim != 3 or int(
        action_velocity_residual.shape[0]
    ) != int(latent.shape[0]):
        raise RuntimeError(
            "UNITE-AV decoder JVP must return a batched action trajectory"
        )
    loss = action_velocity_residual.square().mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("UNITE-AV action-velocity loss must be a finite scalar")
    return loss
