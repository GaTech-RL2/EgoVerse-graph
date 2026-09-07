"""Action-only flow regression with Gaussian-compatible independent relifting.

The decoder is ``g(z) = F(S(z)[:action_dim])``. Conditional pair rotations
make S exactly invertible and standard-Gaussian preserving; F is the existing
action-dimensional affine coupling. This is a restricted-interface diagnostic,
not a general many-to-one decoder architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .projected_invertible_flow import InvertibleActionCoupling
from .shared_latent_flow import _mlp


class _ConditionalPairRotation(nn.Module):
    """Keep half the coordinates fixed and rotate the other half in pairs.

Angles depend only on the unchanged half. The block Jacobian is triangular
with unit determinant, and every rotated pair preserves its radius. Together
these identities preserve the isotropic Gaussian density exactly, including
when angles are nonlinear functions of the conditioning coordinates.
    """

    def __init__(
        self,
        dimension: int,
        permutation: torch.Tensor,
        *,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        if dimension < 4 or dimension % 4:
            raise ValueError("rotation dimension must be a positive multiple of four")
        if permutation.shape != (dimension,) or not torch.equal(
            permutation.sort().values, torch.arange(dimension)
        ):
            raise ValueError("rotation permutation must contain each coordinate once")
        self.dimension = dimension
        self.half = dimension // 2
        self.register_buffer("permutation", permutation.clone())
        self.register_buffer("inverse_permutation", permutation.argsort())
        self.net = _mlp(self.half, width, self.half // 2, depth)
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("rotation angle network must end in a linear layer")
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()

    def _apply_rotation(self, inputs: torch.Tensor, direction: float) -> torch.Tensor:
        ordered = inputs.index_select(-1, self.permutation)
        fixed, active = ordered.split(self.half, dim=-1)
        angles = direction * self.net(fixed)
        pairs = active.reshape(*active.shape[:-1], self.half // 2, 2)
        first, second = pairs.unbind(-1)
        cosine, sine = angles.cos(), angles.sin()
        rotated = torch.stack(
            (cosine * first - sine * second, sine * first + cosine * second),
            dim=-1,
        ).flatten(-2)
        return torch.cat((fixed, rotated), dim=-1).index_select(
            -1, self.inverse_permutation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._apply_rotation(inputs, 1.0)

    def inverse(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._apply_rotation(outputs, -1.0)


class GaussianPreservingRotation(nn.Module):
    """A finite composition of nonlinear, radius-preserving rotation blocks."""

    def __init__(
        self,
        dimension: int = 8,
        *,
        layers: int = 4,
        width: int = 32,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if dimension < 4 or dimension % 4:
            raise ValueError("rotation dimension must be a positive multiple of four")
        if layers <= 0 or width <= 0 or depth <= 0:
            raise ValueError("rotation layers, width and depth must be positive")
        blocks = []
        for index in range(layers):
            # Each pair of blocks swaps the conditioning/rotated halves. A
            # deterministic cyclic shift then changes which coordinates pair.
            permutation = torch.arange(dimension).roll(-(index // 2))
            if index % 2:
                permutation = permutation.roll(dimension // 2)
            blocks.append(
                _ConditionalPairRotation(
                    dimension, permutation, width=width, depth=depth
                )
            )
        self.dimension = dimension
        self.layers = nn.ModuleList(blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = inputs
        for layer in self.layers:
            state = layer(state)
        return state

    def inverse(self, outputs: torch.Tensor) -> torch.Tensor:
        state = outputs
        for layer in reversed(self.layers):
            state = layer.inverse(state)
        return state


class SyntheticGaussianReliftFlow(nn.Module):
    """Candidate C: independent hidden-coordinate relifting and action loss.

At a fixed, realizable interface the projected regression optimum is a closed
action field. The inference sampler is the latent ODE, not a resampled latent
path. Its endpoint argument additionally assumes sufficient support, a
nonexplosive latent ODE, and action-field uniqueness. Joint conditioning and
unidentified null motion remain empirical questions.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        action_dim: int = 3,
        rotation_layers: int = 4,
        rotation_width: int = 32,
        rotation_depth: int = 2,
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
        if not 2 <= self.action_dim < self.latent_dim:
            raise ValueError("require 2 <= action_dim < latent_dim for relifting")
        if field_width <= 0 or field_depth <= 0:
            raise ValueError("field width and depth must be positive")
        self.aux_dim = self.latent_dim - self.action_dim
        self.gaussian_transform = GaussianPreservingRotation(
            self.latent_dim,
            layers=rotation_layers,
            width=rotation_width,
            depth=rotation_depth,
        )
        self.transform = InvertibleActionCoupling(
            self.action_dim,
            layers=coupling_layers,
            width=coupling_width,
            depth=coupling_depth,
            max_log_scale=max_log_scale,
        )
        self.field = _mlp(
            self.latent_dim + 1, field_width, self.latent_dim, field_depth
        )

    def _decode_with_projected(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.gaussian_transform(latent)[..., : self.action_dim]
        return self.transform(projected), projected

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decode_with_projected(latent)[0]

    def decoder(self, latent: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for exact-checkpoint trajectory diagnostics."""
        return self.decode(latent)

    def exact_lift(self, action: torch.Tensor, aux_noise: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError("action must have shape [batch, action_dim]")
        if aux_noise.shape != (len(action), self.aux_dim):
            raise ValueError("aux_noise must have shape [batch, latent_dim - action_dim]")
        projected = self.transform.inverse(action)
        return self.gaussian_transform.inverse(torch.cat((projected, aux_noise), -1))

    def relift(self, state: torch.Tensor, relift_noise: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.latent_dim:
            raise ValueError("state must have shape [batch, latent_dim]")
        if relift_noise.shape != (len(state), self.aux_dim):
            raise ValueError("relift_noise must have shape [batch, latent_dim - action_dim]")
        projected = self.gaussian_transform(state)[:, : self.action_dim]
        return self.gaussian_transform.inverse(torch.cat((projected, relift_noise), -1))

    def velocity(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    @staticmethod
    def _require_forward_ad() -> None:
        # inference_mode silently turns forward-mode JVPs into zero. The
        # earlier Jacobian-FM exporter hit exactly this failure mode.
        if torch.is_inference_mode_enabled():
            raise RuntimeError("decoder JVPs require torch.no_grad(), not inference_mode()")

    def decoder_jvp(self, state: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        self._require_forward_ad()
        return jvp(self.decode, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(self, state: torch.Tensor) -> torch.Tensor:
        self._require_forward_ad()
        return torch.linalg.svdvals(vmap(jacrev(self.decode))(state))

    def scale_loss(self, noise: torch.Tensor) -> torch.Tensor:
        if len(noise) < 2:
            raise ValueError("scale covariance requires at least two noise samples")
        decoded = self.decode(noise)
        mean = decoded.mean(dim=0)
        centered = decoded - mean
        covariance = centered.T @ centered / (len(decoded) - 1)
        identity = torch.eye(self.action_dim, device=noise.device, dtype=noise.dtype)
        return (
            mean.square().sum() + (covariance - identity).square().sum()
        ) / self.action_dim

    def trajectory(self, noise: torch.Tensor, steps: int = 32) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
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
        flow_samples: int = 1,
        lambda_scale: float = 1.0,
        noise: torch.Tensor | None = None,
        aux_noise: torch.Tensor | None = None,
        relift_noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._require_forward_ad()
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError("action must have shape [batch, action_dim]")
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        batch = len(action)
        tensor_options = {"device": action.device, "dtype": action.dtype}
        if noise is None:
            noise = torch.randn(batch, self.latent_dim, **tensor_options)
        if noise.shape != (batch, self.latent_dim):
            raise ValueError("noise must have shape [batch, latent_dim]")
        if aux_noise is None:
            aux_noise = torch.randn(batch, self.aux_dim, **tensor_options)
        clean = self.exact_lift(action, aux_noise)
        clean_many = clean.repeat_interleave(flow_samples, dim=0)
        noise_many = noise.repeat_interleave(flow_samples, dim=0)
        expanded_batch = batch * flow_samples
        if time is None:
            time = torch.rand(expanded_batch, 1, **tensor_options)
        if time.shape != (expanded_batch, 1):
            raise ValueError("time must have shape [batch * flow_samples, 1]")
        if relift_noise is None:
            # Sample after time expansion: every reference-time state receives
            # fresh independent nuisance coordinates, never a repeated eta.
            relift_noise = torch.randn(expanded_batch, self.aux_dim, **tensor_options)
        if relift_noise.shape != (expanded_batch, self.aux_dim):
            raise ValueError(
                "relift_noise must have shape [batch * flow_samples, latent_dim - action_dim]"
            )
        target_velocity = noise_many - clean_many
        state = (1.0 - time) * clean_many + time * noise_many
        # Reuse projected coordinates from the reference JVP. All primal and
        # tangent paths retain gradients, including the learned target.
        (reference_action, projected), (reference_velocity, _) = jvp(
            self._decode_with_projected, (state,), (target_velocity,)
        )
        relifted = self.gaussian_transform.inverse(
            torch.cat((projected, relift_noise), dim=-1)
        )
        prediction = self.decoder_jvp(relifted, self.velocity(relifted, time))
        action_loss = (prediction - reference_velocity).square().sum(-1).mean()
        action_loss = action_loss / self.action_dim
        scale = self.scale_loss(noise)
        zero = action.new_zeros(())
        losses = {
            "loss": action_loss + float(lambda_scale) * scale,
            "flow_loss": zero,
            "reconstruction_loss": zero,
            "scale_loss": scale,
            "path_loss": zero,
            "action_velocity_loss": action_loss,
        }
        if return_diagnostics:
            with torch.no_grad():
                losses.update(
                    clean_roundtrip_mse=(self.decode(clean) - action).square().mean(),
                    clean_code_rms=clean.square().mean().sqrt(),
                    relift_action_mse=(self.decode(relifted) - reference_action).square().mean(),
                    reference_action_velocity_rms=reference_velocity.square().mean().sqrt(),
                    predicted_action_velocity_rms=prediction.square().mean().sqrt(),
                    relift_displacement_rms=(relifted - state).square().mean().sqrt(),
                )
        return losses
