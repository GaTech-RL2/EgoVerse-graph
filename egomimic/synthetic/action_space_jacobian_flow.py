"""Action-space flow matching through a decoder Jacobian basis."""

from __future__ import annotations

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .action_adapter_flow import AffineActionAdapter, ResidualActionAdapter, _fixed_lift
from .shared_latent_flow import _mlp


class SyntheticActionSpaceJacobianFlow(nn.Module):
    """CFM on action bridges with no encoder or latent-space flow objective.

    The Gaussian seed remains fixed during action-space integration.  Its decoded
    value is the source endpoint and its decoder Jacobian maps the shared field's
    latent-width output into an action velocity.
    """

    _DECODER_FAMILIES = {"joint_affine", "nonlinear"}

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        action_dim: int = 3,
        decoder_family: str = "nonlinear",
        residual_width: int = 32,
        residual_depth: int = 2,
        field_width: int = 128,
        field_depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        if self.action_dim != 3:
            raise ValueError("the current synthetic datasets require action_dim=3")
        if decoder_family not in self._DECODER_FAMILIES:
            raise ValueError(f"unknown decoder_family: {decoder_family}")
        self.decoder_family = decoder_family
        projection = _fixed_lift(self.latent_dim).T
        if decoder_family == "nonlinear":
            self.decoder = ResidualActionAdapter(
                self.latent_dim,
                self.action_dim,
                projection,
                residual_width=residual_width,
                residual_depth=residual_depth,
            )
        else:
            self.decoder = AffineActionAdapter(
                self.latent_dim, self.action_dim, projection
            )
        self.field = _mlp(
            self.action_dim + 1,
            field_width,
            self.latent_dim,
            field_depth,
        )

    def latent_velocity(
        self, action_state: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((action_state, time.to(action_state)), dim=-1))

    def decoder_jvp(self, seed: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        return jvp(self.decoder, (seed,), (tangent,))[1]

    def action_velocity(
        self,
        action_state: torch.Tensor,
        seed: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder_jvp(seed, self.latent_velocity(action_state, time))

    def decoder_jacobian_singular_values(self, seed: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(vmap(jacrev(self.decoder))(seed))

    def scale_loss(self, seed: torch.Tensor) -> torch.Tensor:
        """Match decoded Gaussian-seed mean/covariance to a unit 3D Gaussian."""
        if seed.shape != (len(seed), self.latent_dim):
            raise ValueError("seed must have shape [batch, latent_dim]")
        if len(seed) <= 1:
            raise ValueError("nonlinear scale loss requires at least two samples")
        decoded = self.decoder(seed)
        mean = decoded.mean(dim=0)
        centered = decoded - mean
        covariance = centered.T @ centered / (len(decoded) - 1)
        identity = torch.eye(
            self.action_dim, device=seed.device, dtype=seed.dtype
        )
        return (
            mean.square().sum() / self.action_dim
            + (covariance - identity).square().sum() / self.action_dim
        )

    def trajectory(self, seed: torch.Tensor, steps: int = 32) -> torch.Tensor:
        if seed.shape[-1] != self.latent_dim:
            raise ValueError("seed width does not match latent_dim")
        if steps <= 0:
            raise ValueError("steps must be positive")
        state = self.decoder(seed)
        points = [state]
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (len(state), 1), 1.0 - index * delta, device=state.device
            )
            state = state - delta * self.action_velocity(state, seed, time)
            points.append(state)
        return torch.stack(points)

    def losses(
        self,
        action: torch.Tensor,
        *,
        flow_samples: int = 1,
        lambda_scale: float = 0.0,
        seed: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        if action.shape[-1] != self.action_dim:
            raise ValueError("action width does not match action_dim")
        if seed is None:
            seed = torch.randn(len(action), self.latent_dim, device=action.device)
        if seed.shape != (len(action), self.latent_dim):
            raise ValueError("seed must have shape [batch, latent_dim]")
        seed_many = (
            seed[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        )
        action_many = (
            action[:, None].expand(-1, flow_samples, -1).reshape(-1, self.action_dim)
        )
        source = self.decoder(seed_many)
        if time is None:
            time = torch.rand(len(seed_many), 1, device=action.device)
        if time.shape != (len(seed_many), 1):
            raise ValueError("time does not match the expanded action batch")
        state = (1.0 - time) * action_many + time * source
        target_velocity = source - action_many
        prediction = self.action_velocity(state, seed_many, time)
        flow = (prediction - target_velocity).square().mean()
        scale = self.scale_loss(seed)
        zero = torch.zeros((), device=action.device, dtype=action.dtype)
        return {
            "loss": flow + float(lambda_scale) * scale,
            "flow_loss": flow,
            "reconstruction_loss": zero,
            "scale_loss": scale,
            "path_loss": zero,
            "action_velocity_loss": flow,
        }
