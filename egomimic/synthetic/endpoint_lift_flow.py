"""Exact augmented endpoint lifts with latent or balanced flow regression.

The private full interface H is an R^D diffeomorphism; its first action_dim
coordinates form a many-to-one action decoder. Time 1 is Gaussian noise and
time 0 is the clean lift. All training paths, targets and JVPs remain attached.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .projected_invertible_flow import InvertibleActionCoupling
from .shared_latent_flow import _mlp


class SyntheticEndpointLiftFlow(nn.Module):
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
        if not 1 <= self.action_dim <= self.latent_dim:
            raise ValueError("require 1 <= action_dim <= latent_dim")
        self.transform = InvertibleActionCoupling(
            self.latent_dim,
            layers=coupling_layers,
            width=coupling_width,
            depth=coupling_depth,
            max_log_scale=max_log_scale,
        )
        self.field = _mlp(
            self.latent_dim + 1, field_width, self.latent_dim, field_depth
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.transform(latent)[..., : self.action_dim]

    def decoder(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)

    def exact_lift(
        self, action: torch.Tensor, aux_noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError("action must have shape [batch, action_dim]")
        shape = (len(action), self.latent_dim - self.action_dim)
        if aux_noise is None:
            aux_noise = action.new_empty(shape).normal_()
        if aux_noise.shape != shape:
            raise ValueError("aux_noise must have shape [batch, latent_dim-action_dim]")
        return self.transform.inverse(torch.cat((action, aux_noise), dim=-1))

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    @staticmethod
    def _check_forward_ad() -> None:
        # Inference mode silently zeroes torch.func.jvp tangents. Fail loudly;
        # torch.no_grad() is supported and preserves the needed forward AD.
        if torch.is_inference_mode_enabled():
            raise RuntimeError("JVP evaluation requires torch.no_grad(), not inference_mode()")

    def interface_jvp(self, state: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        self._check_forward_ad()
        return jvp(self.transform, (state,), (tangent,))[1]

    def decoder_jvp(self, state: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        self._check_forward_ad()
        return jvp(self.decode, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(self, state: torch.Tensor) -> torch.Tensor:
        self._check_forward_ad()
        return torch.linalg.svdvals(vmap(jacrev(self.decode))(state))

    def scale_loss(self, noise: torch.Tensor) -> torch.Tensor:
        """The existing decoded-noise mean/covariance regularizer, unchanged."""
        if len(noise) < 2:
            raise ValueError("scale_loss requires at least two noise samples")
        decoded = self.decode(noise)
        mean = decoded.mean(dim=0)
        centered = decoded - mean
        covariance = centered.T @ centered / (len(decoded) - 1)
        identity = torch.eye(self.action_dim, device=noise.device, dtype=noise.dtype)
        return (
            mean.square().sum() / self.action_dim
            + (covariance - identity).square().sum() / self.action_dim
        )

    def trajectory(self, noise: torch.Tensor, steps: int = 32) -> torch.Tensor:
        """Reverse Euler in latent space; return decoded states at all times."""
        if noise.ndim != 2 or noise.shape[-1] != self.latent_dim:
            raise ValueError("noise must have shape [batch, latent_dim]")
        if steps <= 0:
            raise ValueError("steps must be positive")
        state = noise
        points = [self.decode(state)]
        delta = 1.0 / steps
        for index in range(steps):
            time = state.new_full((len(state), 1), 1.0 - index * delta)
            state = state - delta * self.velocity(state, time)
            points.append(self.decode(state))
        return torch.stack(points)

    def losses(
        self,
        action: torch.Tensor,
        *,
        objective: str = "latent",
        flow_samples: int = 14,
        lambda_scale: float = 1.0,
        noise: torch.Tensor | None = None,
        aux_noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        if objective not in ("latent", "balanced"):
            raise ValueError("objective must be 'latent' or 'balanced'")
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        if noise is None:
            noise = action.new_empty((len(action), self.latent_dim)).normal_()
        if noise.shape != (len(action), self.latent_dim):
            raise ValueError("noise must have shape [batch, latent_dim]")
        clean = self.exact_lift(action, aux_noise)
        clean_many = clean[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        noise_many = noise[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        if time is None:
            time = action.new_empty((len(clean_many), 1)).uniform_()
        if time.shape != (len(clean_many), 1):
            raise ValueError("time must have shape [batch*flow_samples, 1]")
        target_velocity = noise_many - clean_many
        state = (1.0 - time) * clean_many + time * noise_many
        residual = self.velocity(state, time) - target_velocity
        latent_loss = residual.square().mean()
        scale_loss = self.scale_loss(noise)
        zero = action.new_zeros(())
        result = {
            "loss": latent_loss + float(lambda_scale) * scale_loss,
            "flow_loss": latent_loss,
            "latent_flow_loss": latent_loss,
            "scale_loss": scale_loss,
            "reconstruction_loss": zero,
            "path_loss": zero,
            "clean_code_rms": clean.square().mean().sqrt(),
        }
        if objective == "balanced" or return_diagnostics:
            full_residual = self.interface_jvp(state, residual)
            full_loss = full_residual.square().mean()
            action_component = full_residual[:, : self.action_dim].square()
            aux_component = full_residual[:, self.action_dim :].square()
            # Per-coordinate MSEs are interpretable across d and D. Contributions
            # are both divided by D and sum to the balanced objective's H term.
            result.update(
                full_velocity_loss=full_loss,
                action_velocity_loss=action_component.mean(),
                auxiliary_velocity_loss=aux_component.mean() if aux_component.numel() else zero,
                action_velocity_contribution=action_component.sum(dim=-1).mean() / self.latent_dim,
                auxiliary_velocity_contribution=aux_component.sum(dim=-1).mean() / self.latent_dim,
            )
            if objective == "balanced":
                result["loss"] = result["loss"] + full_loss
        if return_diagnostics:
            # Avoid another inverse/forward diagnostic on every optimizer step.
            result["clean_roundtrip_mse"] = (self.decode(clean) - action).square().mean()
        return result
