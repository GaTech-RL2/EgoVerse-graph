"""Action Flow with differentiable decoder inversion and no encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .action_adapter_flow import (
    AffineActionAdapter,
    ResidualActionAdapter,
    _fixed_lift,
)
from .shared_latent_flow import _mlp


class SyntheticDecoderInversionFlow(nn.Module):
    """Shared latent field trained through an unrolled private-decoder search."""

    _DECODER_FAMILIES = {"joint_affine", "nonlinear"}
    _TRAINING_OBJECTIVES = {"endpoint_difference", "conditional_relifting"}

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        decoder_family: str = "nonlinear",
        residual_width: int = 32,
        residual_depth: int = 2,
        field_width: int = 128,
        field_depth: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        if decoder_family not in self._DECODER_FAMILIES:
            raise ValueError(f"unknown decoder_family: {decoder_family}")
        self.decoder_family = decoder_family
        projection = _fixed_lift(self.latent_dim).T
        if decoder_family == "nonlinear":
            self.decoder = ResidualActionAdapter(
                self.latent_dim,
                3,
                projection,
                residual_width=residual_width,
                residual_depth=residual_depth,
            )
        else:
            self.decoder = AffineActionAdapter(self.latent_dim, 3, projection)
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

    def trajectory(self, noise: torch.Tensor, steps: int = 32) -> torch.Tensor:
        if noise.shape[-1] != self.latent_dim:
            raise ValueError("noise width does not match latent_dim")
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
        return jvp(self.decoder, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(self, state: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(vmap(jacrev(self.decoder))(state))

    def scale_loss(self, noise: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(3, device=noise.device, dtype=noise.dtype)
        if self.decoder_family == "joint_affine":
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

    def infer_codes(
        self,
        action: torch.Tensor,
        initialization: torch.Tensor,
        *,
        steps: int,
        step_size: float,
        create_graph: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Unroll latent-only reconstruction descent through the current decoder."""
        if initialization.shape != (len(action), self.latent_dim):
            raise ValueError("initialization must have shape [batch, latent_dim]")
        if steps <= 0:
            raise ValueError("inversion steps must be positive")
        if step_size <= 0:
            raise ValueError("inversion step size must be positive")
        code = initialization.requires_grad_(True)
        before_per_example = (self.decoder(code) - action).square().mean(dim=-1)
        for _ in range(steps):
            half_squared_error = 0.5 * (self.decoder(code) - action).square().sum()
            (code_gradient,) = torch.autograd.grad(
                half_squared_error,
                code,
                create_graph=create_graph,
            )
            code = code - float(step_size) * code_gradient
        after_per_example = (self.decoder(code) - action).square().mean(dim=-1)
        return code, {
            "inversion_before_mse": before_per_example.mean(),
            "inversion_after_mse": after_per_example.mean(),
            "inversion_after_rmse_q50": after_per_example.sqrt().quantile(0.50),
            "inversion_after_rmse_q90": after_per_example.sqrt().quantile(0.90),
            "inversion_after_rmse_q99": after_per_example.sqrt().quantile(0.99),
            "inferred_code_rms": code.square().mean().sqrt(),
        }

    def losses(
        self,
        action: torch.Tensor,
        *,
        flow_samples: int,
        inversion_steps: int,
        inversion_step_size: float,
        lambda_scale: float,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        initialization: torch.Tensor | None = None,
        relift_initialization: torch.Tensor | None = None,
        training_objective: str = "endpoint_difference",
    ) -> dict[str, torch.Tensor]:
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        if training_objective not in self._TRAINING_OBJECTIVES:
            raise ValueError(f"unknown training objective: {training_objective}")
        flow_noise = torch.randn(len(action), self.latent_dim, device=action.device)
        if noise is not None:
            if noise.shape != flow_noise.shape:
                raise ValueError("noise must have shape [batch, latent_dim]")
            flow_noise = noise
        if initialization is None:
            initialization = torch.randn_like(flow_noise)
        clean, inversion = self.infer_codes(
            action,
            initialization,
            steps=inversion_steps,
            step_size=inversion_step_size,
            create_graph=True,
        )
        clean_many = clean[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        noise_many = flow_noise[:, None].expand(-1, flow_samples, -1).reshape(-1, self.latent_dim)
        action_many = action[:, None].expand(-1, flow_samples, -1).reshape(-1, 3)
        if time is None:
            time = torch.rand(len(clean_many), 1, device=action.device)
        if time.shape != (len(clean_many), 1):
            raise ValueError("time does not match the expanded action batch")
        state = (1.0 - time) * clean_many + time * noise_many
        if training_objective == "endpoint_difference":
            predicted_action_velocity = self.decoder_jvp(
                state, self.velocity(state, time)
            )
            target_action_velocity = self.decoder(noise_many) - action_many
            relift_metrics = {}
        else:
            reference_action = self.decoder(state)
            target_action_velocity = self.decoder_jvp(
                state, noise_many - clean_many
            )
            if relift_initialization is None:
                relift_initialization = torch.randn_like(state)
            relifted, relift_metrics = self.infer_codes(
                reference_action,
                relift_initialization,
                steps=inversion_steps,
                step_size=inversion_step_size,
                create_graph=True,
            )
            relift_metrics = {
                f"relift_{key}": value for key, value in relift_metrics.items()
            }
            predicted_action_velocity = self.decoder_jvp(
                relifted, self.velocity(relifted, time)
            )
        action_loss = (
            predicted_action_velocity - target_action_velocity
        ).square().mean()
        scale_loss = self.scale_loss(flow_noise)
        zero = torch.zeros((), device=action.device, dtype=action.dtype)
        return {
            "loss": action_loss + float(lambda_scale) * scale_loss,
            "flow_loss": zero,
            "reconstruction_loss": zero,
            "scale_loss": scale_loss,
            "path_loss": zero,
            "action_velocity_loss": action_loss,
            **inversion,
            **relift_metrics,
        }
