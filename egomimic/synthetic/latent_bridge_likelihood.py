"""Learned Gaussian latent bridge with the shared denoiser in the likelihood.

This is a stochastic discrete latent model, not a flow-matching ODE. The loss
is the complete fixed-variance negative ELBO up to parameter-independent
constants. A valid bound need not be tight: the learned Gaussian reference
family can leave a posterior gap, and positive action-output noise cannot
exactly match a noiseless torus surface. The pointwise likelihood bound still
applies there, but its population entropy/KL decomposition requires densities
and is not valid for the singular surface data law.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.func import jacrev, jvp, vmap

from .action_adapter_flow import ResidualActionAdapter
from .shared_latent_flow import _mlp


class SyntheticLatentBridgeLikelihood(nn.Module):
    """Non-invertible candidate B, specialized to the unconditioned toy task.

The private mean network supplies mu(A,k), with mu(A,K)=0 by construction.
The same shared reverse-mean network predicts every latent transition and the
latent inside the final action likelihood. Encoder, shared denoiser and
decoder all learn jointly from the boundary term; interior KL regressions
also update the learned reference means without detaching their targets.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        action_dim: int = 3,
        levels: int = 32,
        encoder_width: int = 32,
        encoder_depth: int = 2,
        decoder_width: int = 32,
        decoder_depth: int = 2,
        field_width: int = 128,
        field_depth: int = 4,
        sigma_min: float = 0.1,
        rho: float = 0.95,
        output_sigma: float = 0.02,
        sampler_seed: int = 424242,
    ) -> None:
        super().__init__()
        if latent_dim < action_dim or action_dim <= 0:
            raise ValueError("require latent_dim >= action_dim > 0")
        if levels < 2:
            raise ValueError("the bridge requires at least two levels")
        if not 0 < sigma_min <= 1:
            raise ValueError("sigma_min must lie in (0, 1]")
        if not 0 <= rho < 1:
            raise ValueError("rho must lie in [0, 1)")
        if not math.isfinite(output_sigma) or output_sigma <= 0:
            raise ValueError("output_sigma must be finite and strictly positive")
        if min(encoder_width, encoder_depth, decoder_width, decoder_depth,
               field_width, field_depth) <= 0:
            raise ValueError("network widths and depths must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.levels = int(levels)
        self.rho = float(rho)
        self.output_sigma = float(output_sigma)
        self.tau = self.output_sigma
        self.sampler_seed = int(sampler_seed)
        # Entry k-1 stores sigma_k. The terminal scale is exactly one, so the
        # reference terminal prior is exactly independent standard Gaussian.
        self.register_buffer("sigmas", torch.logspace(math.log10(sigma_min), 0, levels))
        self.mean_network = _mlp(
            self.action_dim + 1, encoder_width, self.latent_dim, encoder_depth
        )
        projection = torch.zeros(self.action_dim, self.latent_dim)
        projection[:, :self.action_dim] = torch.eye(self.action_dim)
        self.decoder = ResidualActionAdapter(
            self.latent_dim,
            self.action_dim,
            projection,
            residual_width=decoder_width,
            residual_depth=decoder_depth,
        )
        self.field = _mlp(
            self.latent_dim + 1, field_width, self.latent_dim, field_depth
        )

    def _level_vector(self, level: int | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if isinstance(level, int):
            if not 1 <= level <= self.levels:
                raise ValueError("level must lie in [1, levels]")
            return torch.full((len(reference),), level, dtype=torch.long, device=reference.device)
        if level.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("bridge levels must be integers, not continuous flow times")
        if level.shape == (len(reference), 1):
            level = level[:, 0]
        if level.shape != (len(reference),):
            raise ValueError("level must be an integer or a tensor of shape [batch]")
        return level.to(device=reference.device, dtype=torch.long)

    def bridge_mean(self, action: torch.Tensor, level: int | torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError("action must have shape [batch, action_dim]")
        level = self._level_vector(level, action)
        fraction = level[:, None].to(action) / self.levels
        raw = self.mean_network(torch.cat((action, fraction), dim=-1))
        return (1.0 - fraction) * raw

    def encoder(self, action: torch.Tensor) -> torch.Tensor:
        """Diagnostic code mean at level one; this is not an exact clean lift."""
        return self.bridge_mean(action, 1)

    def reverse_mean(self, state: torch.Tensor, level: int | torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.latent_dim:
            raise ValueError("state must have shape [batch, latent_dim]")
        level = self._level_vector(level, state)
        fraction = level[:, None].to(state) / self.levels
        return state + self.field(torch.cat((state, fraction), dim=-1))

    def reference_transition(
        self,
        action: torch.Tensor,
        level: int | torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Z_k, b_k and lambda_k^2 for k in 2,...,K.

Tensor levels are generated by the caller in the declared range. Scalar
levels are checked here; avoiding value extraction from generated GPU level
tensors keeps the training path free of device synchronization.
        """
        if isinstance(level, int) and level < 2:
            raise ValueError("reference transitions require level >= 2")
        level = self._level_vector(level, action)
        if noise.shape != (len(action), self.latent_dim):
            raise ValueError("noise must have shape [batch, latent_dim]")
        current_mean = self.bridge_mean(action, level)
        previous_mean = self.bridge_mean(action, level - 1)
        scales = self.sigmas.to(action)
        current_scale = scales[level - 1, None]
        previous_scale = scales[level - 2, None]
        state = current_mean + current_scale * noise
        # Algebraically identical to mu_prev + rho*sigma_prev/sigma_cur *
        # (state-mu_cur). Using epsilon directly avoids cancellation while
        # retaining the exact reparameterized gradient (mu_cur cancels).
        target = previous_mean + self.rho * previous_scale * noise
        variance = (1.0 - self.rho ** 2) * previous_scale.square()
        return state, target, variance

    def interior_kl_terms(
        self,
        action: torch.Tensor,
        level: int | torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Exact per-example equal-covariance Gaussian KL, summed over latent axes."""
        state, target, variance = self.reference_transition(action, level, noise)
        residual = self.reverse_mean(state, level) - target
        return residual.square().sum(-1) / (2.0 * variance[:, 0])

    def boundary_code(self, action: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Shared denoiser output used inside the final action likelihood."""
        if noise.shape != (len(action), self.latent_dim):
            raise ValueError("boundary noise must have shape [batch, latent_dim]")
        state = self.encoder(action) + self.sigmas[0].to(action) * noise
        return self.reverse_mean(state, 1)

    def boundary_prediction(self, action: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.boundary_code(action, noise))

    @staticmethod
    def _require_forward_ad() -> None:
        if torch.is_inference_mode_enabled():
            raise RuntimeError("decoder derivative diagnostics require torch.no_grad(), not inference_mode()")

    def decoder_jvp(self, state: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        self._require_forward_ad()
        return jvp(self.decoder, (state,), (tangent,))[1]

    def decoder_jacobian_singular_values(self, state: torch.Tensor) -> torch.Tensor:
        self._require_forward_ad()
        return torch.linalg.svdvals(vmap(jacrev(self.decoder))(state))

    def scale_loss(self, noise: torch.Tensor) -> torch.Tensor:
        """Decoded-noise health diagnostic, excluded from the likelihood objective."""
        if len(noise) < 2:
            raise ValueError("scale covariance requires at least two noise samples")
        decoded = self.decoder(noise)
        mean = decoded.mean(0)
        centered = decoded - mean
        covariance = centered.T @ centered / (len(decoded) - 1)
        identity = torch.eye(self.action_dim, device=noise.device, dtype=noise.dtype)
        return (mean.square().sum() + (covariance - identity).square().sum()) / self.action_dim

    def trajectory(
        self,
        noise: torch.Tensor,
        steps: int = 32,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the stated K-level chain, including final action-output noise.

A fresh local generator with the recorded sampler seed replays the same
independent innovations for evaluation and export of the same source cloud.
Passing a generator explicitly instead consumes that generator's state. No
global RNG state is changed by the default replay path.
        """
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError("noise must have shape [batch, latent_dim]")
        if steps != self.levels:
            raise ValueError("the stochastic likelihood sampler requires steps == levels")
        if generator is None:
            # CPU and CUDA generators use different streams. Draw the default
            # replay stream on CPU so a CPU strict reload reproduces the same
            # innovations as a GPU terminal metric/export.
            generator = torch.Generator(device="cpu").manual_seed(self.sampler_seed)

        def innovation_like(reference: torch.Tensor) -> torch.Tensor:
            return torch.randn(
                reference.shape,
                device=generator.device,
                dtype=reference.dtype,
                generator=generator,
            ).to(reference.device)

        state = noise
        points = [self.decoder(state)]
        for level in range(self.levels, 1, -1):
            innovation = innovation_like(state)
            scale = self.sigmas[level - 2].to(state) * math.sqrt(1.0 - self.rho ** 2)
            state = self.reverse_mean(state, level) + scale * innovation
            points.append(self.decoder(state))
        action_mean = self.decoder(self.reverse_mean(state, 1))
        output_noise = innovation_like(action_mean)
        points.append(action_mean + self.tau * output_noise)
        return torch.stack(points)

    def integrate(self, noise: torch.Tensor, steps: int = 32) -> torch.Tensor:
        return self.trajectory(noise, steps=steps)[-1]

    def losses(
        self,
        action: torch.Tensor,
        *,
        flow_samples: int = 14,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        if action.ndim != 2 or action.shape[1] != self.action_dim or len(action) == 0:
            raise ValueError("action must have nonempty shape [batch, action_dim]")
        if flow_samples <= 0:
            raise ValueError("flow_samples must be positive")
        if time is not None:
            raise ValueError("continuous time is not accepted by a discrete likelihood bridge")
        options = {"device": action.device, "dtype": action.dtype}
        if noise is None:
            noise = torch.randn(len(action), self.latent_dim, **options)
        if noise.shape != (len(action), self.latent_dim):
            raise ValueError("noise must match the unexpanded boundary action batch")
        expanded_action = action.repeat_interleave(flow_samples, 0)
        level = torch.randint(2, self.levels + 1, (len(expanded_action),), device=action.device)
        # Both levels and epsilons are independent for every expanded example.
        interior_noise = torch.randn(len(expanded_action), self.latent_dim, **options)
        interior_kl = (self.levels - 1) * self.interior_kl_terms(
            expanded_action, level, interior_noise
        ).mean()
        boundary_code = self.boundary_code(action, noise)
        endpoint_residual = self.decoder(boundary_code) - action
        endpoint_nll = endpoint_residual.square().sum(-1).mean() / (2.0 * self.tau ** 2)
        endpoint_mse = endpoint_residual.square().mean()
        result = {
            "loss": interior_kl + endpoint_nll,
            "interior_kl_loss": interior_kl,
            "endpoint_nll_loss": endpoint_nll,
            "endpoint_mse": endpoint_mse,
            "boundary_action_mse": endpoint_mse,
        }
        if return_diagnostics:
            with torch.no_grad():
                code = self.encoder(action)
                decoded_noise = self.decoder(noise)
                result.update(
                    code_rms=code.square().mean().sqrt(),
                    code_centered_rms=(code - code.mean(0)).square().mean().sqrt(),
                    boundary_code_rms=boundary_code.square().mean().sqrt(),
                    decoded_noise_rms=decoded_noise.square().mean().sqrt(),
                    sigma_first=self.sigmas[0].to(action),
                    sigma_terminal=self.sigmas[-1].to(action),
                    output_noise_std=action.new_tensor(self.tau),
                )
                if len(noise) > 1:
                    result["decoded_noise_scale_diagnostic"] = self.scale_loss(noise)
        return result
