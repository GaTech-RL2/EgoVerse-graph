"""Context-free sequence reference means for a discrete Gaussian latent model.

These are learned variational means, not ground-truth clean latents. The
terminal mean is exactly zero, so sigma_K=1 gives an action-independent prior.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from egomimic.models.action_flow_codec import ContextFreeSequenceEncoder


class TimeDependentSequenceMean(nn.Module):
    """mu(A,t)=(1-t) E_raw(A,t); no observation/context input exists."""

    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 8,
        horizon: int = 16,
        hidden_dim: int = 20,
        depth: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 80,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.network = ContextFreeSequenceEncoder(
            input_dim=self.input_dim + 1,
            latent_dim=self.latent_dim,
            horizon=self.horizon,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

    def forward(self, action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if action.ndim != 3 or tuple(action.shape[1:]) != (
            self.horizon,
            self.input_dim,
        ):
            raise ValueError("action must have shape (B, horizon, input_dim)")
        if time.shape != (len(action),) or time.device != action.device:
            raise ValueError("time must have shape (B,) on the action device")
        fraction = time.to(action).view(-1, 1, 1)
        raw = self.network(
            torch.cat((action, fraction.expand(-1, self.horizon, 1)), -1)
        )
        # Compute the gate in FP32: t=1 remains exactly zero under autocast.
        return (1.0 - time.float().view(-1, 1, 1)) * raw.float()


class GaussianBridgeSchedule(nn.Module):
    """Fixed variances of the learned reference and reverse transitions."""

    def __init__(self, num_levels=32, sigma_min=0.1, sigma_max=1.0, rho=0.95):
        super().__init__()
        self.num_levels = int(num_levels)
        self.sigma_min, self.sigma_max, self.rho = map(
            float, (sigma_min, sigma_max, rho)
        )
        if self.num_levels < 2:
            raise ValueError("num_levels must be at least two")
        if not 0 < self.sigma_min <= self.sigma_max or self.sigma_max != 1.0:
            raise ValueError(
                "require 0 < sigma_min <= sigma_max=1 for the standard Gaussian prior"
            )
        if not math.isfinite(self.rho) or not 0 <= self.rho < 1:
            raise ValueError("rho must lie in [0, 1)")
        self.register_buffer(
            "sigmas", torch.logspace(math.log10(self.sigma_min), 0, self.num_levels)
        )

    def previous_variance(self, levels: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.rho**2) * self.sigmas[levels - 2].float().square()
