"""Discrete learned Gaussian-bridge likelihood stages; not a CFM/Euler model.

The objective is the fixed-variance negative ELBO up to parameter-independent
constants: all reference-mean and target gradients remain attached. Coordinates
are summed across the complete chunk, then examples/sampled levels are averaged.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from egomimic.models.action_flow_likelihood import GaussianBridgeSchedule
from egomimic.pipeline.core import Stage


def _tensor(batch, key):
    value = batch[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{key} must be a tensor")
    return value


class LikelihoodReferenceStage(Stage):
    """Sample integer levels and evaluate attached private reference means."""

    train_only = True

    def __init__(
        self,
        mean_encoder: nn.Module,
        num_levels=32,
        interior_samples_per_content=14,
        input_key="target",
        prefix="likelihood/",
    ):
        super().__init__()
        if not isinstance(mean_encoder, nn.Module):
            raise TypeError("mean_encoder must be a module")
        self.mean_encoder = mean_encoder
        self.num_levels = int(num_levels)
        self.interior_samples_per_content = int(interior_samples_per_content)
        if self.num_levels < 2 or self.interior_samples_per_content < 1:
            raise ValueError("need >=2 levels and >=1 sampled interior per example")
        self.input_key, self.prefix = input_key, prefix
        self.reads = (input_key,)
        self.writes = tuple(
            prefix + key
            for key in (
                "levels",
                "base_index",
                "current_mean",
                "previous_mean",
                "boundary_mean",
            )
        )

    def forward(self, batch):
        action = _tensor(batch, self.input_key)
        if action.ndim != 3 or len(action) == 0:
            raise ValueError("target must be a nonempty (B,H,A) sequence")
        index = torch.arange(len(action), device=action.device).repeat_interleave(
            self.interior_samples_per_content
        )
        levels = torch.randint(
            2, self.num_levels + 1, (len(index),), device=action.device
        )
        repeated = action.index_select(0, index)
        current = self.mean_encoder(repeated, levels.float() / self.num_levels)
        previous = self.mean_encoder(repeated, (levels - 1).float() / self.num_levels)
        boundary = self.mean_encoder(
            action,
            torch.full((len(action),), 1.0 / self.num_levels, device=action.device),
        )
        if (
            current.shape != previous.shape
            or current.ndim != 3
            or boundary.shape[1:] != current.shape[1:]
        ):
            raise ValueError("reference means must be aligned (B,H,D) sequences")
        for key, value in zip(
            self.writes, (levels, index, current, previous, boundary), strict=True
        ):
            batch[key] = value
        return batch


class GaussianBridgeNoisingStage(Stage):
    """Independent Gaussian reparameterizations with exact reverse targets."""

    train_only = True

    def __init__(
        self,
        num_levels=32,
        sigma_min=0.1,
        sigma_max=1.0,
        rho=0.95,
        condition_dropout_probability=0.3,
        condition_key="condition",
        prefix="likelihood/",
    ):
        super().__init__()
        self.schedule = GaussianBridgeSchedule(num_levels, sigma_min, sigma_max, rho)
        self.num_levels = self.schedule.num_levels
        self.sigma_min, self.sigma_max, self.rho = (
            self.schedule.sigma_min,
            self.schedule.sigma_max,
            self.schedule.rho,
        )
        self.condition_dropout_probability = float(condition_dropout_probability)
        if not 0 <= self.condition_dropout_probability <= 1:
            raise ValueError("condition dropout must lie in [0,1]")
        self.condition_key, self.prefix = condition_key, prefix
        self.reads = tuple(
            prefix + key
            for key in (
                "levels",
                "base_index",
                "current_mean",
                "previous_mean",
                "boundary_mean",
            )
        ) + (condition_key,)
        self.writes = tuple(
            prefix + key
            for key in (
                "state",
                "time",
                "condition",
                "condition_drop_mask",
                "posterior_target",
                "posterior_variance",
                "interior_noise",
                "boundary_noise",
            )
        )

    def forward(self, batch):
        p = self.prefix
        levels, index, current, previous, boundary = (
            _tensor(batch, p + key)
            for key in (
                "levels",
                "base_index",
                "current_mean",
                "previous_mean",
                "boundary_mean",
            )
        )
        condition = _tensor(batch, self.condition_key)
        if len(condition) != len(boundary) or condition.device != current.device:
            raise ValueError("condition and reference means must align")
        if levels.dtype != torch.long or tuple(levels.shape) != (len(current),):
            raise ValueError("sampled levels must be an aligned int64 vector")
        current_scale = self.schedule.sigmas[levels - 1].float().view(-1, 1, 1)
        previous_scale = self.schedule.sigmas[levels - 2].float().view(-1, 1, 1)
        noise = torch.randn_like(current, dtype=torch.float32)
        boundary_noise = torch.randn_like(boundary, dtype=torch.float32)
        current_state = current.float() + current_scale * noise
        boundary_state = (
            boundary.float() + self.schedule.sigmas[0].float() * boundary_noise
        )
        # This equals mu_prev + rho*sigma_prev/sigma_k*(z_k-mu_k),
        # without cancellation. The target's learned mean is NOT detached.
        target = previous.float() + self.rho * previous_scale * noise
        probability = self.condition_dropout_probability
        mask = torch.rand(len(boundary), device=boundary.device) < probability
        values = (
            torch.cat((current_state, boundary_state)),
            torch.cat(
                (
                    levels.float() / self.num_levels,
                    torch.full(
                        (len(boundary),), 1.0 / self.num_levels, device=boundary.device
                    ),
                )
            ),
            torch.cat((condition.index_select(0, index), condition)),
            torch.cat((mask.index_select(0, index), mask)),
            target,
            self.schedule.previous_variance(levels),
            noise,
            boundary_noise,
        )
        for key, value in zip(self.writes, values, strict=True):
            batch[key] = value
        return batch


class ConditionalReverseMeanStage(Stage):
    """M(z,k,O)=z+field(z,k/K,O), with the actual stochastic K-level sampler."""

    def __init__(
        self,
        field: nn.Module,
        num_levels=32,
        sigma_min=0.1,
        sigma_max=1.0,
        rho=0.95,
        inference_noise_key="sampler/noise",
        inference_condition_key="condition",
        prefix="likelihood/",
    ):
        super().__init__()
        if not isinstance(field, nn.Module):
            raise TypeError("field must be a module")
        self.field = field
        self.schedule = GaussianBridgeSchedule(num_levels, sigma_min, sigma_max, rho)
        self.num_levels = self.schedule.num_levels
        self.sigma_min, self.sigma_max, self.rho = (
            self.schedule.sigma_min,
            self.schedule.sigma_max,
            self.schedule.rho,
        )
        self.prefix = prefix
        self.inference_noise_key, self.inference_condition_key = (
            inference_noise_key,
            inference_condition_key,
        )
        self.reads = tuple(
            prefix + key
            for key in (
                "state",
                "time",
                "condition",
                "condition_drop_mask",
                "posterior_target",
            )
        )
        self.writes = (prefix + "interior_prediction", prefix + "boundary_latent")
        self.reads_by_mode = {
            "inference": (inference_noise_key, inference_condition_key)
        }
        self.writes_by_mode = {
            "inference": (
                prefix + "generated_latent",
                prefix + "trajectory",
                "log/ActionFlow/SamplerCalls",
                "log/ActionFlow/LatentInnovations",
            )
        }

    def _mean(self, state, time, condition, mask):
        if state.ndim != 3 or len(state) == 0 or time.shape != (len(state),):
            raise ValueError(
                "reverse mean requires nonempty sequences and aligned times"
            )
        if (
            len(condition) != len(state)
            or mask.shape != (len(state),)
            or mask.dtype != torch.bool
        ):
            raise ValueError("reverse mean conditioning/mask must align")
        delta = self.field(state, time, condition, condition_drop_mask=mask)
        if not torch.is_tensor(delta) or delta.shape != state.shape:
            raise ValueError("reverse field output must match latent sequence shape")
        return state.float() + delta.float()

    def forward(self, batch):
        p = self.prefix
        state, time, condition, mask = (
            _tensor(batch, p + key)
            for key in (
                "state",
                "time",
                "condition",
                "condition_drop_mask",
            )
        )
        prediction = self._mean(state, time, condition, mask)
        n_interior = len(_tensor(batch, p + "posterior_target"))
        batch[p + "interior_prediction"] = prediction[:n_interior]
        batch[p + "boundary_latent"] = prediction[n_interior:]
        return batch

    @torch.no_grad()
    def _inference(self, batch):
        state = _tensor(batch, self.inference_noise_key).float()
        condition = _tensor(batch, self.inference_condition_key)
        mask = torch.zeros(len(state), dtype=torch.bool, device=state.device)
        trajectory = [state]
        for level in range(self.num_levels, 1, -1):
            time = torch.full(
                (len(state),), level / self.num_levels, device=state.device
            )
            mean = self._mean(state, time, condition, mask)
            scale = self.schedule.sigmas[level - 2].float() * math.sqrt(
                1.0 - self.rho**2
            )
            state = mean + scale * torch.randn_like(mean)
            trajectory.append(state)
        time = torch.full((len(state),), 1.0 / self.num_levels, device=state.device)
        state = self._mean(state, time, condition, mask)
        trajectory.append(state)
        batch[self.prefix + "generated_latent"] = state
        batch[self.prefix + "trajectory"] = torch.stack(trajectory)
        batch["log/ActionFlow/SamplerCalls"] = float(self.num_levels)
        batch["log/ActionFlow/LatentInnovations"] = float(self.num_levels - 1)
        return batch

    def execute(self, batch, *, mode):
        if mode == "inference":
            return self._inference(batch)
        return self(batch)


class LikelihoodDecoderStage(Stage):
    """Private boundary likelihood mean; inference samples its Gaussian output."""

    def __init__(
        self,
        decoder: nn.Module,
        tau=0.02,
        prefix="likelihood/",
        prediction_key="pred_action",
    ):
        super().__init__()
        if not isinstance(decoder, nn.Module):
            raise TypeError("decoder must be a module")
        self.decoder, self.tau, self.prefix, self.prediction_key = (
            decoder,
            float(tau),
            prefix,
            prediction_key,
        )
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be finite and positive")
        self.reads = (prefix + "boundary_latent",)
        self.writes = (prefix + "boundary_prediction",)
        self.reads_by_mode = {"inference": (prefix + "generated_latent",)}
        self.writes_by_mode = {"inference": (prediction_key, prefix + "action_mean")}

    def forward(self, batch):
        batch[self.prefix + "boundary_prediction"] = self.decoder(
            _tensor(batch, self.prefix + "boundary_latent")
        ).float()
        return batch

    def execute(self, batch, *, mode):
        if mode != "inference":
            return self(batch)
        mean = self.decoder(_tensor(batch, self.prefix + "generated_latent")).float()
        batch[self.prefix + "action_mean"] = mean
        batch[self.prediction_key] = mean + self.tau * torch.randn_like(mean)
        return batch


class GaussianBridgeObjectiveStage(Stage):
    """Gaussian negative-ELBO terms, excluding fixed additive constants.

    InteriorBridgeNLL is the equal-covariance KL sum (not ordinary latent FM).
    BoundaryNLL is ||A-action_mean||^2/(2*tau^2), not clean reconstruction.
    Both sum ALL sequence coordinates. No extra reconstruction/scale/JVP loss.
    """

    train_only = True
    reduction = "sum_chunk_coordinates_mean_examples_constant_free_gaussian_bound"

    def __init__(
        self, num_levels=32, tau=0.02, target_key="target", prefix="likelihood/"
    ):
        super().__init__()
        self.num_levels, self.tau = int(num_levels), float(tau)
        if self.num_levels < 2 or not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("require num_levels>=2 and finite positive tau")
        self.target_key, self.prefix = target_key, prefix
        self.reads = tuple(
            prefix + key
            for key in (
                "interior_prediction",
                "posterior_target",
                "posterior_variance",
                "boundary_prediction",
            )
        ) + (target_key,)
        self.writes = (
            "loss/likelihood",
            "log/ActionFlow/InteriorBridgeNLL",
            "log/ActionFlow/BoundaryNLL",
            "log/ActionFlow/TotalLoss",
            "log/MSE",
        )

    def forward(self, batch):
        p = self.prefix
        prediction, target, variance, boundary = (
            _tensor(batch, p + key)
            for key in (
                "interior_prediction",
                "posterior_target",
                "posterior_variance",
                "boundary_prediction",
            )
        )
        action = _tensor(batch, self.target_key)
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError(
                "interior predictions/targets must align as complete sequences"
            )
        if variance.shape != (len(prediction),) or boundary.shape != action.shape:
            raise ValueError("likelihood variance or action-boundary shape mismatch")
        interior = (self.num_levels - 1) * (
            (prediction.float() - target.float()).square().sum(dim=(-2, -1))
            / (2 * variance.float())
        ).mean()
        squared = (boundary.float() - action.float()).square()
        boundary_nll = squared.sum(dim=(-2, -1)).mean() / (2 * self.tau**2)
        total = interior + boundary_nll
        values = (total, interior, boundary_nll, total, squared.mean())
        for key, value in zip(self.writes, values, strict=True):
            batch[key] = value
        return batch
