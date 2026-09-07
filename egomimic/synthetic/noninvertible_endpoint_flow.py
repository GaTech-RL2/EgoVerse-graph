"""Endpoint-distribution and exact-section controls with noninvertible decoders.

These unconditional torus models couple latent FM and action-Jacobian residuals.
Neither objective is a convergence guarantee for joint representation learning.
The graph section is a restricted diagnostic, not a general R6-compliant model.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .action_adapter_flow import ResidualActionAdapter, _fixed_lift
from .shared_latent_flow import _mlp


DEFAULT_MMD_BANDWIDTHS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _checked_bandwidths(bandwidths) -> tuple[float, ...]:
    values = tuple(float(value) for value in bandwidths)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("kernel bandwidths must be finite and positive")
    return values


def _multiscale_rbf(left, right, bandwidths):
    squared_distance = torch.cdist(left, right).square()
    return sum(
        torch.exp(-squared_distance / (2 * bandwidth * bandwidth))
        for bandwidth in bandwidths
    ) / len(bandwidths)


def paired_multiscale_mmd2(
    decoded: torch.Tensor,
    target: torch.Tensor,
    *,
    bandwidths=DEFAULT_MMD_BANDWIDTHS,
    context: torch.Tensor | None = None,
    context_bandwidths=(1.0,),
) -> torch.Tensor:
    """Paired unbiased U-statistic with a mean of fixed Gaussian RBF kernels.

Each row is an independent example; decoded[i] and target[i] may be dependent.
All three kernel sums omit i=j, including the cross sum. Negative finite-batch
values are valid and are never clamped. A provided fixed context uses the
product of action and context kernels to compare joint (context, action) laws.
Bandwidths are standard deviations in native input units, not variances.
"""
    if decoded.ndim != 2 or decoded.shape != target.shape or len(decoded) < 2:
        raise ValueError("paired MMD needs matching [batch, dimension] arrays, batch >= 2")
    bandwidths = _checked_bandwidths(bandwidths)
    decoded_kernel = _multiscale_rbf(decoded, decoded, bandwidths)
    target_kernel = _multiscale_rbf(target, target, bandwidths)
    cross_kernel = _multiscale_rbf(decoded, target, bandwidths)
    terms = decoded_kernel + target_kernel - 2.0 * cross_kernel
    if context is not None:
        if context.ndim != 2 or len(context) != len(decoded):
            raise ValueError("context must have shape [batch, context_dimension]")
        context_kernel = _multiscale_rbf(
            context, context, _checked_bandwidths(context_bandwidths)
        )
        terms = terms * context_kernel
    return (terms.sum() - terms.diagonal().sum()) / (len(decoded) * (len(decoded) - 1))


class _GeometricEndpointFlow(nn.Module):
    def __init__(self, *, latent_dim, action_dim, field_width, field_depth):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        if not 1 <= self.action_dim <= self.latent_dim:
            raise ValueError("require 1 <= action_dim <= latent_dim")
        self.field = _mlp(self.latent_dim + 1, field_width, self.latent_dim, field_depth)

    def velocity(self, state, time):
        if time.ndim == 1:
            time = time[:, None]
        return self.field(torch.cat((state, time.to(state)), dim=-1))

    @staticmethod
    def _require_forward_ad():
        if torch.is_inference_mode_enabled():
            raise RuntimeError("decoder JVPs require torch.no_grad(), not inference_mode()")

    def decoder_jvp(self, state, tangent):
        self._require_forward_ad()
        return jvp(self.decoder, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(self, state):
        self._require_forward_ad()
        return torch.linalg.svdvals(vmap(jacrev(self.decoder))(state))

    def scale_loss(self, noise):
        if len(noise) < 2:
            raise ValueError("scale covariance requires at least two samples")
        decoded = self.decoder(noise)
        mean = decoded.mean(0)
        centered = decoded - mean
        covariance = centered.T @ centered / (len(noise) - 1)
        identity = torch.eye(self.action_dim, device=noise.device, dtype=noise.dtype)
        return (mean.square().sum() + (covariance - identity).square().sum()) / self.action_dim

    def trajectory(self, noise, steps=32):
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError("noise must have shape [batch, latent_dim]")
        if steps <= 0:
            raise ValueError("steps must be positive")
        state = noise
        points = [self.decoder(state)]
        delta = 1.0 / steps
        for index in range(steps):
            time = state.new_full((len(state), 1), 1.0 - index * delta)
            state = state - delta * self.velocity(state, time)
            points.append(self.decoder(state))
        return torch.stack(points)

    def _geometric_losses(self, action, *, flow_samples, lambda_scale, noise, time):
        self._require_forward_ad()
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError("action must have shape [batch, action_dim]")
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        clean = self.encoder(action)
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("noise must have shape [batch, latent_dim]")
        clean_many = clean.repeat_interleave(flow_samples, dim=0)
        noise_many = noise.repeat_interleave(flow_samples, dim=0)
        if time is None:
            time = action.new_empty((len(clean_many), 1)).uniform_()
        if time.shape != (len(clean_many), 1):
            raise ValueError("time must have shape [batch*flow_samples, 1]")
        state = (1 - time) * clean_many + time * noise_many
        target = noise_many - clean_many
        residual = self.velocity(state, time) - target
        latent_loss = residual.square().mean()
        action_loss = self.decoder_jvp(state, residual).square().mean()
        scale = self.scale_loss(noise)
        zero = action.new_zeros(())
        return {
            "loss": latent_loss + action_loss + float(lambda_scale) * scale,
            "flow_loss": latent_loss,
            "latent_flow_loss": latent_loss,
            "action_velocity_loss": action_loss,
            "scale_loss": scale,
            "reconstruction_loss": zero,
            "path_loss": zero,
        }, clean


class SyntheticMMDEndpointFlow(_GeometricEndpointFlow):
    """Free nonlinear interfaces with a paired distributional endpoint penalty.

The torus trial is unconditional. The MMD helper supports a fixed context
product kernel; adding context only there does not make the field conditional.
Finite MMD weights are an optimization experiment, not an exact constraint.
"""

    def __init__(
        self, *, latent_dim=8, action_dim=3, residual_width=32, residual_depth=2,
        field_width=128, field_depth=4, mmd_bandwidths=DEFAULT_MMD_BANDWIDTHS,
    ):
        super().__init__(latent_dim=latent_dim, action_dim=action_dim,
                         field_width=field_width, field_depth=field_depth)
        if self.action_dim != 3:
            raise ValueError("the reused torus residual adapters require action_dim=3")
        lift = _fixed_lift(self.latent_dim)
        self.encoder = ResidualActionAdapter(
            self.action_dim, self.latent_dim, lift,
            residual_width=residual_width, residual_depth=residual_depth,
        )
        self.decoder = ResidualActionAdapter(
            self.latent_dim, self.action_dim, lift.T,
            residual_width=residual_width, residual_depth=residual_depth,
        )
        self.mmd_bandwidths = _checked_bandwidths(mmd_bandwidths)

    def losses(
        self, action, *, flow_samples=14, noise=None, time=None,
        lambda_endpoint=10.0, lambda_scale=1.0, return_diagnostics=False,
    ):
        losses, clean = self._geometric_losses(
            action, flow_samples=flow_samples, lambda_scale=lambda_scale,
            noise=noise, time=time,
        )
        endpoint = self.decoder(clean)
        endpoint_mmd2 = paired_multiscale_mmd2(
            endpoint, action, bandwidths=self.mmd_bandwidths
        )
        losses["loss"] = losses["loss"] + float(lambda_endpoint) * endpoint_mmd2
        losses["endpoint_mmd2"] = endpoint_mmd2
        if return_diagnostics:
            with torch.no_grad():
                losses["reconstruction_mse"] = (endpoint - action).square().mean()
                losses["clean_code_rms"] = clean.square().mean().sqrt()
        return losses


class SyntheticGraphSectionFlow(_GeometricEndpointFlow):
    """Restricted exact-section diagnostic: clean codes retain action coordinates.

E(a)=(a,f(a)); g(x,h)=x+R(x,h)-R(x,f(x)). This decoder can lose rank away
from E's graph and needs no inverse. Its mandatory raw-coordinate carrier
fails R6 for the unrestricted research design, despite exact clean decoding.
"""

    def __init__(
        self, *, latent_dim=8, action_dim=3, residual_width=32, residual_depth=2,
        field_width=128, field_depth=4,
    ):
        super().__init__(latent_dim=latent_dim, action_dim=action_dim,
                         field_width=field_width, field_depth=field_depth)
        if self.latent_dim <= self.action_dim:
            raise ValueError("graph sections require latent_dim > action_dim")
        self.graph = _mlp(self.action_dim, residual_width,
                          self.latent_dim - self.action_dim, residual_depth)
        self.residual = _mlp(self.latent_dim, residual_width,
                            self.action_dim, residual_depth)
        with torch.no_grad():
            for network in (self.graph, self.residual):
                network[-1].weight.zero_()
                network[-1].bias.zero_()

    def encoder(self, action):
        return torch.cat((action, self.graph(action)), dim=-1)

    def decoder(self, latent):
        action = latent[..., :self.action_dim]
        section = self.encoder(action)
        return action + self.residual(latent) - self.residual(section)

    def losses(
        self, action, *, flow_samples=14, noise=None, time=None,
        lambda_scale=1.0, return_diagnostics=False,
    ):
        losses, clean = self._geometric_losses(
            action, flow_samples=flow_samples, lambda_scale=lambda_scale,
            noise=noise, time=time,
        )
        if return_diagnostics:
            with torch.no_grad():
                mse = (self.decoder(clean) - action).square().mean()
                losses["reconstruction_mse"] = mse
                losses["clean_roundtrip_mse"] = mse
                losses["clean_code_rms"] = clean.square().mean().sqrt()
        return losses
