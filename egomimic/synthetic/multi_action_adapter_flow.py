"""Multiple private action interfaces around one shared latent flow field."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .action_adapter_flow import ResidualActionAdapter, _fixed_lift
from .shared_latent_flow import _mlp


class SyntheticMultiActionAdapterFlow(nn.Module):
    """Action Flow with one private encoder/decoder pair per embodiment.

    Every embodiment uses the same latent Gaussian, time convention, and flow
    field. Only the nonlinear action interfaces are private.
    """

    def __init__(
        self,
        *,
        embodiments: Iterable[str],
        latent_dim: int = 8,
        residual_width: int = 32,
        residual_depth: int = 2,
        field_width: int = 128,
        field_depth: int = 4,
    ) -> None:
        super().__init__()
        names = list(embodiments)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("embodiments must contain at least two unique names")
        if any(not name or "." in name for name in names):
            raise ValueError("embodiment names must be non-empty and cannot contain dots")
        self.embodiments = tuple(names)
        self.latent_dim = int(latent_dim)
        lift = _fixed_lift(self.latent_dim)
        self.encoders = nn.ModuleDict(
            {
                name: ResidualActionAdapter(
                    3,
                    self.latent_dim,
                    lift,
                    residual_width=residual_width,
                    residual_depth=residual_depth,
                )
                for name in names
            }
        )
        self.decoders = nn.ModuleDict(
            {
                name: ResidualActionAdapter(
                    self.latent_dim,
                    3,
                    lift.T,
                    residual_width=residual_width,
                    residual_depth=residual_depth,
                )
                for name in names
            }
        )
        self.field = _mlp(
            self.latent_dim + 1, field_width, self.latent_dim, field_depth
        )

    def _check_embodiment(self, embodiment: str) -> None:
        if embodiment not in self.encoders:
            raise KeyError(f"unknown embodiment: {embodiment}")

    def encode(self, embodiment: str, action: torch.Tensor) -> torch.Tensor:
        self._check_embodiment(embodiment)
        return self.encoders[embodiment](action)

    def decode(self, embodiment: str, latent: torch.Tensor) -> torch.Tensor:
        self._check_embodiment(embodiment)
        return self.decoders[embodiment](latent)

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    def trajectory(
        self, noise: torch.Tensor, *, embodiment: str, steps: int = 32
    ) -> torch.Tensor:
        if noise.shape[-1] != self.latent_dim:
            raise ValueError("noise width does not match latent_dim")
        if steps <= 0:
            raise ValueError("steps must be positive")
        state = noise
        points = [self.decode(embodiment, state)]
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (len(state), 1), 1.0 - index * delta, device=state.device
            )
            state = state - delta * self.velocity(state, time)
            points.append(self.decode(embodiment, state))
        return torch.stack(points)

    def decoder_jvp(
        self, embodiment: str, state: torch.Tensor, tangent: torch.Tensor
    ) -> torch.Tensor:
        decoder = self.decoders[embodiment]
        return jvp(decoder, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(
        self, embodiment: str, state: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.svdvals(vmap(jacrev(self.decoders[embodiment]))(state))

    def reconstruction_loss(
        self, embodiment: str, action: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.decode(embodiment, self.encode(embodiment, action)) - action
        ).square().mean()

    def scale_loss(self, embodiment: str, noise: torch.Tensor) -> torch.Tensor:
        if len(noise) <= 1:
            raise ValueError("scale loss requires at least two samples")
        decoded = self.decode(embodiment, noise)
        mean = decoded.mean(dim=0)
        centered = decoded - mean
        covariance = centered.T @ centered / (len(decoded) - 1)
        identity = torch.eye(3, device=noise.device, dtype=noise.dtype)
        return mean.square().sum() / 3.0 + (covariance - identity).square().sum() / 3.0

    def losses_for_embodiment(
        self,
        embodiment: str,
        action: torch.Tensor,
        *,
        flow_samples: int = 14,
        lambda_reconstruction: float = 100.0,
        lambda_scale: float = 1.0,
        lambda_action_velocity: float = 1.0,
        clean_gradient_mode: str = "full",
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        if clean_gradient_mode not in {"full", "target_stopgrad", "all_stopgrad"}:
            raise ValueError(f"unknown clean gradient mode: {clean_gradient_mode}")
        clean = self.encode(embodiment, action)
        base_noise = torch.randn_like(clean) if noise is None else noise
        if base_noise.shape != clean.shape:
            raise ValueError("noise must match the unexpanded latent batch")
        clean_many = clean[:, None].expand(-1, flow_samples, -1).reshape(
            -1, self.latent_dim
        )
        noise_many = base_noise[:, None].expand(-1, flow_samples, -1).reshape(
            -1, self.latent_dim
        )
        if time is None:
            time = torch.rand(len(clean_many), 1, device=action.device)
        if time.shape != (len(clean_many), 1):
            raise ValueError("time does not match the expanded action batch")
        target_clean = clean_many if clean_gradient_mode == "full" else clean_many.detach()
        state_clean = clean_many.detach() if clean_gradient_mode == "all_stopgrad" else clean_many
        target_velocity = noise_many - target_clean
        state = (1.0 - time) * state_clean + time * noise_many
        residual = self.velocity(state, time) - target_velocity
        flow_loss = residual.square().mean()
        reconstruction_loss = self.reconstruction_loss(embodiment, action)
        scale_loss = self.scale_loss(embodiment, base_noise)
        action_velocity_loss = (
            self.decoder_jvp(embodiment, state, residual)
            .square()
            .sum(dim=-1)
            .mean()
            / 3.0
        )
        total = (
            flow_loss
            + float(lambda_reconstruction) * reconstruction_loss
            + float(lambda_scale) * scale_loss
            + float(lambda_action_velocity) * action_velocity_loss
        )
        return {
            "loss": total,
            "flow_loss": flow_loss,
            "reconstruction_loss": reconstruction_loss,
            "scale_loss": scale_loss,
            "action_velocity_loss": action_velocity_loss,
        }
