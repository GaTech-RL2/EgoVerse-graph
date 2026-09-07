"""Affine and nonlinear action adapters around a latent flow field."""

from __future__ import annotations

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .shared_latent_flow import _mlp


def _fixed_lift(latent_dim: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if latent_dim < 3:
        raise ValueError("action-adapter latent_dim must be at least 3")
    lift = torch.zeros(latent_dim, 3, dtype=dtype)
    lift[:3] = torch.eye(3, dtype=dtype)
    return lift


class AffineActionAdapter(nn.Module):
    """Affine map initialized to the canonical orthonormal lift or projection."""

    def __init__(self, input_dim: int, output_dim: int, weight: torch.Tensor) -> None:
        super().__init__()
        if tuple(weight.shape) != (output_dim, input_dim):
            raise ValueError("initial affine weight has the wrong shape")
        self.linear = nn.Linear(input_dim, output_dim)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.zero_()

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class ResidualActionAdapter(AffineActionAdapter):
    """Affine base plus a smooth residual MLP with zero initial output."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight: torch.Tensor,
        *,
        residual_width: int,
        residual_depth: int,
    ) -> None:
        super().__init__(input_dim, output_dim, weight)
        self.residual = _mlp(input_dim, residual_width, output_dim, residual_depth)
        final = self.residual[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("residual MLP must end in a linear layer")
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs) + self.residual(inputs)


class SyntheticActionAdapterFlow(nn.Module):
    """Latent flow with jointly learned action encoder and decoder adapters.

    Mathematical time follows the clean-to-noise bridge: clean action codes are
    at ``t=0`` and unit Gaussian noise is at ``t=1``. Generation integrates the
    learned field backward from 1 to 0 and decodes each latent state.
    """

    _ADAPTER_FAMILIES = {"fixed_affine", "joint_affine", "nonlinear"}

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        adapter_family: str = "fixed_affine",
        residual_width: int = 32,
        residual_depth: int = 2,
        field_width: int = 128,
        field_depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        if adapter_family not in self._ADAPTER_FAMILIES:
            raise ValueError(f"unknown adapter_family: {adapter_family}")
        self.adapter_family = adapter_family
        lift = _fixed_lift(self.latent_dim)
        if adapter_family == "nonlinear":
            self.encoder = ResidualActionAdapter(
                3,
                self.latent_dim,
                lift,
                residual_width=residual_width,
                residual_depth=residual_depth,
            )
            self.decoder = ResidualActionAdapter(
                self.latent_dim,
                3,
                lift.T,
                residual_width=residual_width,
                residual_depth=residual_depth,
            )
        else:
            self.encoder = AffineActionAdapter(3, self.latent_dim, lift)
            self.decoder = AffineActionAdapter(self.latent_dim, 3, lift.T)
        if adapter_family == "fixed_affine":
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
        self.field = _mlp(
            self.latent_dim + 1,
            field_width,
            self.latent_dim,
            field_depth,
        )

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    def integrate(self, noise: torch.Tensor, steps: int = 32) -> torch.Tensor:
        return self.trajectory(noise, steps=steps)[-1]

    def trajectory(self, noise: torch.Tensor, steps: int = 32) -> torch.Tensor:
        if noise.shape[-1] != self.latent_dim:
            raise ValueError(
                f"noise width {noise.shape[-1]} does not match latent_dim "
                f"{self.latent_dim}"
            )
        if steps <= 0:
            raise ValueError("steps must be positive")
        state = noise
        points = [self.decoder(state)]
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full((len(state), 1), 1.0 - index * delta, device=state.device)
            state = state - delta * self.velocity(state, time)
            points.append(self.decoder(state))
        return torch.stack(points)

    def decoder_jvp(self, state: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Return the differentiable per-example decoder Jacobian-vector product."""
        return jvp(self.decoder, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(self, state: torch.Tensor) -> torch.Tensor:
        """Return decoder Jacobian singular values for each sampled latent state."""
        jacobians = vmap(jacrev(self.decoder))(state)
        return torch.linalg.svdvals(jacobians)

    def reconstruction_loss(self, action: torch.Tensor) -> torch.Tensor:
        return (self.decoder(self.encoder(action)) - action).square().mean()

    def path_consistency_loss(
        self,
        action: torch.Tensor,
        noise: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        clean = self.encoder(action)
        velocity = noise - clean
        state = (1.0 - time) * clean + time * noise
        decoded_velocity = self.decoder_jvp(state, velocity)
        endpoint_velocity = self.decoder(noise) - action
        return (decoded_velocity - endpoint_velocity).square().mean()

    def action_velocity_loss(
        self,
        state: torch.Tensor,
        velocity_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Measure the latent FM residual in the decoder's action metric.

        For decoder ``g`` and latent residual ``R = v_theta(Z_t, t) - U``, this
        computes ``E[||J_g(Z_t) R||^2] / 3``.  The differentiable JVP keeps the
        full gradient path through the field, encoder-derived state and target,
        and decoder Jacobian.
        """
        if state.shape != velocity_residual.shape:
            raise ValueError("state and velocity_residual must have matching shapes")
        decoded_residual = self.decoder_jvp(state, velocity_residual)
        return decoded_residual.square().sum(dim=-1).mean() / 3.0

    def scale_loss(self, noise: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(3, device=noise.device, dtype=noise.dtype)
        if self.adapter_family != "nonlinear":
            mean = self.decoder.bias
            covariance = self.decoder.weight @ self.decoder.weight.T
        else:
            if len(noise) <= 1:
                raise ValueError("nonlinear scale loss requires at least two samples")
            decoded = self.decoder(noise)
            mean = decoded.mean(dim=0)
            centered = decoded - mean
            covariance = centered.T @ centered / (len(decoded) - 1)
        return mean.square().sum() / 3.0 + (covariance - identity).square().sum() / 3.0

    def noise_augmented_reconstruction_loss(
        self,
        action: torch.Tensor,
        *,
        t_min: float = 0.7,
        probability: float = 0.5,
    ) -> torch.Tensor:
        """Released UNITE's noise augmentation before decoding.

        For a random subset of the batch (``probability``), decode
        ``z = t z_0 + (1 - t) eps`` with ``t ~ U[t_min, 1]`` instead of the clean
        code, so the decoder learns to tolerate off-manifold latents. The
        encoder stays attached, exactly as in the clean reconstruction loss.
        """
        if not 0.0 <= t_min <= 1.0 or not 0.0 <= probability <= 1.0:
            raise ValueError("noise augmentation t_min and probability must be in [0, 1]")
        clean = self.encoder(action)
        t = torch.empty(len(clean), 1, device=clean.device, dtype=clean.dtype).uniform_(
            t_min, 1.0
        )
        apply = (torch.rand(len(clean), 1, device=clean.device) < probability).to(
            clean.dtype
        )
        t = apply * t + (1.0 - apply)
        noisy = t * clean + (1.0 - t) * torch.randn_like(clean)
        return (self.decoder(noisy) - action).square().mean()

    @staticmethod
    def _route_clean(clean_many: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (target_clean, state_clean) for a clean-gradient mode."""
        if mode not in {"full", "target_stopgrad", "all_stopgrad"}:
            raise ValueError(f"unknown clean gradient mode: {mode}")
        target_clean = clean_many if mode == "full" else clean_many.detach()
        state_clean = clean_many.detach() if mode == "all_stopgrad" else clean_many
        return target_clean, state_clean

    def losses(
        self,
        action: torch.Tensor,
        *,
        objective: str,
        flow_samples: int = 1,
        lambda_reconstruction: float = 1.0,
        lambda_scale: float = 1.0,
        lambda_path: float = 1.0,
        lambda_action_velocity: float = 1.0,
        clean_gradient_mode: str = "full",
        action_velocity_clean_gradient_mode: str | None = None,
        reconstruction_noise_aug: bool = False,
        noise_aug_t_min: float = 0.7,
        noise_aug_probability: float = 0.5,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if objective not in {"none", "reconstruction", "path", "action_velocity"}:
            raise ValueError(f"unknown adapter objective: {objective}")
        if objective == "path" and self.adapter_family != "nonlinear":
            raise ValueError(
                "affine path consistency duplicates reconstruction; use reconstruction"
            )
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        if clean_gradient_mode not in {"full", "target_stopgrad", "all_stopgrad"}:
            raise ValueError(f"unknown clean gradient mode: {clean_gradient_mode}")
        # The action-velocity term may attach to the encoder independently of
        # the flow term; by default it follows the flow term.
        if action_velocity_clean_gradient_mode is None:
            action_velocity_clean_gradient_mode = clean_gradient_mode
        clean = self.encoder(action)
        clean_many = (
            clean[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        )
        action_many = action[:, None].expand(-1, flow_samples, -1).reshape(-1, 3)
        if noise is None:
            base_noise = torch.randn_like(clean)
        elif noise.shape == clean.shape:
            base_noise = noise
        else:
            raise ValueError("noise must match the unexpanded action batch")
        noise_many = (
            base_noise[:, None]
            .expand(-1, flow_samples, -1)
            .reshape(-1, self.latent_dim)
        )
        if time is None:
            time = torch.rand(len(clean_many), 1, device=action.device)
        if time.shape != (len(clean_many), 1):
            raise ValueError("time does not match the expanded action batch")
        target_clean, state_clean = self._route_clean(clean_many, clean_gradient_mode)
        target_velocity = noise_many - target_clean
        state = (1.0 - time) * state_clean + time * noise_many
        velocity_residual = self.velocity(state, time) - target_velocity
        flow_loss = velocity_residual.square().mean()
        if reconstruction_noise_aug:
            reconstruction_loss = self.noise_augmented_reconstruction_loss(
                action, t_min=noise_aug_t_min, probability=noise_aug_probability
            )
        else:
            reconstruction_loss = self.reconstruction_loss(action)
        scale_loss = self.scale_loss(base_noise)
        if objective == "path":
            path_loss = self.path_consistency_loss(action_many, noise_many, time)
        else:
            path_loss = torch.zeros((), device=action.device, dtype=action.dtype)
        if objective == "action_velocity":
            if action_velocity_clean_gradient_mode == clean_gradient_mode:
                av_state, av_residual = state, velocity_residual
            else:
                av_target, av_state_clean = self._route_clean(
                    clean_many, action_velocity_clean_gradient_mode
                )
                av_state = (1.0 - time) * av_state_clean + time * noise_many
                av_residual = self.velocity(av_state, time) - (noise_many - av_target)
            action_velocity_loss = self.action_velocity_loss(av_state, av_residual)
        else:
            action_velocity_loss = torch.zeros(
                (), device=action.device, dtype=action.dtype
            )
        total = flow_loss + float(lambda_scale) * scale_loss
        if objective == "reconstruction":
            total = total + float(lambda_reconstruction) * reconstruction_loss
        elif objective == "path":
            total = total + float(lambda_path) * path_loss
        elif objective == "action_velocity":
            total = (
                total
                + float(lambda_reconstruction) * reconstruction_loss
                + float(lambda_action_velocity) * action_velocity_loss
            )
        return {
            "loss": total,
            "flow_loss": flow_loss,
            "reconstruction_loss": reconstruction_loss,
            "scale_loss": scale_loss,
            "path_loss": path_loss,
            "action_velocity_loss": action_velocity_loss,
        }
