"""Projected latent flow with an analytically invertible action decoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .shared_latent_flow import _mlp


class _AffineCoupling(nn.Module):
    def __init__(
        self,
        dimension: int,
        mask: torch.Tensor,
        *,
        width: int,
        depth: int,
        max_log_scale: float,
    ) -> None:
        super().__init__()
        if mask.shape != (dimension,):
            raise ValueError("coupling mask has the wrong shape")
        self.register_buffer("mask", mask)
        self.max_log_scale = float(max_log_scale)
        self.net = _mlp(dimension, width, 2 * dimension, depth)
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("coupling network must end in a linear layer")
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()

    def parameters_from_masked(self, masked: torch.Tensor):
        log_scale, shift = self.net(masked).chunk(2, dim=-1)
        active = 1.0 - self.mask
        log_scale = active * self.max_log_scale * torch.tanh(log_scale)
        shift = active * shift
        return log_scale, shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        masked = self.mask * inputs
        log_scale, shift = self.parameters_from_masked(masked)
        return masked + (1.0 - self.mask) * (inputs * log_scale.exp() + shift)

    def inverse(self, outputs: torch.Tensor) -> torch.Tensor:
        masked = self.mask * outputs
        log_scale, shift = self.parameters_from_masked(masked)
        return masked + (1.0 - self.mask) * (outputs - shift) * (-log_scale).exp()


class InvertibleActionCoupling(nn.Module):
    def __init__(
        self,
        dimension: int,
        *,
        layers: int = 4,
        width: int = 32,
        depth: int = 2,
        max_log_scale: float = 1.5,
    ) -> None:
        super().__init__()
        if dimension < 2:
            raise ValueError("coupling transform requires dimension >= 2")
        if layers <= 0:
            raise ValueError("coupling layers must be positive")
        base = torch.tensor(
            [(index % 2) for index in range(dimension)], dtype=torch.float32
        )
        self.layers = nn.ModuleList(
            [
                _AffineCoupling(
                    dimension,
                    base if index % 2 == 0 else 1.0 - base,
                    width=width,
                    depth=depth,
                    max_log_scale=max_log_scale,
                )
                for index in range(layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def inverse(self, outputs: torch.Tensor) -> torch.Tensor:
        inputs = outputs
        for layer in reversed(self.layers):
            inputs = layer.inverse(inputs)
        return inputs


class SyntheticProjectedInvertibleFlow(nn.Module):
    """Exact randomized lift with a projection-then-diffeomorphism decoder."""

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        action_dim: int = 3,
        coupling_layers: int = 4,
        coupling_width: int = 32,
        coupling_depth: int = 2,
        max_log_scale: float = 1.5,
        field_width: int = 128,
        field_depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        if self.latent_dim < self.action_dim:
            raise ValueError("latent_dim must be at least action_dim")
        raw_projection = torch.zeros(self.latent_dim, self.action_dim)
        raw_projection[: self.action_dim] = torch.eye(self.action_dim)
        self.raw_projection = nn.Parameter(raw_projection)
        self.offset = nn.Parameter(torch.zeros(self.action_dim))
        self.transform = InvertibleActionCoupling(
            self.action_dim,
            layers=coupling_layers,
            width=coupling_width,
            depth=coupling_depth,
            max_log_scale=max_log_scale,
        )
        self.field = _mlp(
            self.latent_dim + 1,
            field_width,
            self.latent_dim,
            field_depth,
        )

    def projection(self) -> torch.Tensor:
        basis = torch.linalg.qr(self.raw_projection, mode="reduced").Q
        return basis.T

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        projected = latent @ self.projection().T + self.offset
        return self.transform(projected)

    def decoder(self, latent: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for shared synthetic trajectory diagnostics."""
        return self.decode(latent)

    def exact_lift(self, action: torch.Tensor, null_noise: torch.Tensor) -> torch.Tensor:
        if null_noise.shape != (len(action), self.latent_dim):
            raise ValueError("null_noise must have shape [batch, latent_dim]")
        projection = self.projection()
        projected_clean = self.transform.inverse(action) - self.offset
        null_component = null_noise - (null_noise @ projection.T) @ projection
        return projected_clean @ projection + null_component

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    def decoder_jvp(self, state: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return jvp(self.decode, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(self, state: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(vmap(jacrev(self.decode))(state))

    def action_velocity_loss(
        self, state: torch.Tensor, velocity_residual: torch.Tensor
    ) -> torch.Tensor:
        if state.shape != velocity_residual.shape:
            raise ValueError("state and velocity_residual must have matching shapes")
        return (
            self.decoder_jvp(state, velocity_residual)
            .square()
            .sum(dim=-1)
            .mean()
            / self.action_dim
        )

    def scale_loss(self, noise: torch.Tensor) -> torch.Tensor:
        decoded = self.decode(noise)
        mean = decoded.mean(dim=0)
        centered = decoded - mean
        covariance = centered.T @ centered / (len(decoded) - 1)
        identity = torch.eye(
            self.action_dim, device=noise.device, dtype=noise.dtype
        )
        return (
            mean.square().sum() / self.action_dim
            + (covariance - identity).square().sum() / self.action_dim
        )

    def trajectory(self, noise: torch.Tensor, steps: int = 32) -> torch.Tensor:
        if noise.shape[-1] != self.latent_dim:
            raise ValueError("noise width does not match latent_dim")
        if steps <= 0:
            raise ValueError("steps must be positive")
        state = noise
        points = [self.decode(state)]
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (len(state), 1), 1.0 - index * delta, device=state.device
            )
            state = state - delta * self.velocity(state, time)
            points.append(self.decode(state))
        return torch.stack(points)

    def losses(
        self,
        action: torch.Tensor,
        *,
        flow_samples: int = 1,
        lambda_scale: float = 1.0,
        lambda_latent_flow: float = 0.0,
        noise: torch.Tensor | None = None,
        null_noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        if noise is None:
            noise = torch.randn(len(action), self.latent_dim, device=action.device)
        if noise.shape != (len(action), self.latent_dim):
            raise ValueError("noise must have shape [batch, latent_dim]")
        if null_noise is None:
            null_noise = torch.randn_like(noise)
        clean = self.exact_lift(action, null_noise)
        clean_many = (
            clean[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        )
        noise_many = (
            noise[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        )
        if time is None:
            time = torch.rand(len(clean_many), 1, device=action.device)
        if time.shape != (len(clean_many), 1):
            raise ValueError("time does not match the expanded action batch")
        target_velocity = noise_many - clean_many
        state = (1.0 - time) * clean_many + time * noise_many
        residual = self.velocity(state, time) - target_velocity
        action_loss = (
            self.decoder_jvp(state, residual).square().sum(dim=-1).mean()
            / self.action_dim
        )
        latent_flow_loss = residual.square().mean()
        scale_loss = self.scale_loss(noise)
        zero = torch.zeros((), device=action.device, dtype=action.dtype)
        return {
            "loss": (
                action_loss
                + float(lambda_scale) * scale_loss
                + float(lambda_latent_flow) * latent_flow_loss
            ),
            "flow_loss": latent_flow_loss,
            "reconstruction_loss": zero,
            "scale_loss": scale_loss,
            "path_loss": zero,
            "action_velocity_loss": action_loss,
            "clean_roundtrip_mse": (self.decode(clean) - action).square().mean(),
            "clean_code_rms": clean.square().mean().sqrt(),
        }
