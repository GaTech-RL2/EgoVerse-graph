"""Factorized Pipeline stages for epsilon-prediction Diffusion Policy."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Stage

_SCHEDULER_FIELDS = (
    "num_train_timesteps",
    "beta_start",
    "beta_end",
    "beta_schedule",
    "variance_type",
    "clip_sample",
    "set_alpha_to_one",
    "steps_offset",
    "prediction_type",
)


def _scheduler_value(scheduler, name: str):
    """Read a setting from either a repository-local or diffusers scheduler."""
    value = getattr(scheduler, name, None)
    if value is not None:
        return value
    config = getattr(scheduler, "config", None)
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _scheduler_signature(scheduler) -> tuple:
    """Return the forward/reverse-process identity shared across DP nodes."""
    return (
        f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        tuple(
            (name, repr(_scheduler_value(scheduler, name)))
            for name in _SCHEDULER_FIELDS
        ),
    )


def _validate_epsilon_scheduler(scheduler, domain: str) -> None:
    prediction_type = _scheduler_value(scheduler, "prediction_type")
    if prediction_type != "epsilon":
        raise ValueError(
            f"Scheduler for {domain!r} must use the epsilon noise objective; "
            f"prediction_type={prediction_type!r}"
        )
    train_steps = _scheduler_value(scheduler, "num_train_timesteps")
    if train_steps is None or int(train_steps) <= 0:
        raise ValueError(
            f"Scheduler for {domain!r} has no positive train timestep count"
        )


def _validate_action_shape(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    label: str,
) -> None:
    expected = (batch_size, action_horizon, action_dim)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{label} must be {expected}, got {tuple(tensor.shape)}")


class MultiDomainDiffusionNoisingStage(Stage):
    """Sample timesteps and construct noisy training actions explicitly."""

    train_only = True
    reads = ("target", "embodiment")
    writes = (
        "diffusion/noisy_action",
        "diffusion/noise_target",
        "diffusion/timestep",
        "diffusion/scheduler_signature",
    )
    objective = "epsilon"

    def __init__(
        self,
        noise_schedulers: Mapping[str, object],
        action_dims: Mapping[str, int],
        action_horizon: int,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dims = {str(key): int(value) for key, value in action_dims.items()}
        self.noise_schedulers = {
            str(key): value for key, value in dict(noise_schedulers).items()
        }
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not self.noise_schedulers or set(self.noise_schedulers) != set(
            self.action_dims
        ):
            raise ValueError("noise_schedulers and action_dims must share domain keys")
        for domain, scheduler in self.noise_schedulers.items():
            if self.action_dims[domain] <= 0:
                raise ValueError(f"Action width for {domain!r} must be positive")
            _validate_epsilon_scheduler(scheduler, domain)

    def forward(self, batch: dict) -> dict:
        domain = str(batch["embodiment"])
        if domain not in self.noise_schedulers:
            raise KeyError(
                f"Unknown diffusion domain {domain!r}; "
                f"configured={sorted(self.noise_schedulers)}"
            )
        target = batch["target"]
        _validate_action_shape(
            target,
            batch_size=int(target.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dims[domain],
            label=f"Diffusion target for {domain!r}",
        )
        scheduler = self.noise_schedulers[domain]
        noise = torch.randn(target.shape, device=target.device)
        timesteps = torch.randint(
            0,
            int(_scheduler_value(scheduler, "num_train_timesteps")),
            (int(target.shape[0]),),
            device=target.device,
        ).long()
        batch["diffusion/noisy_action"] = scheduler.add_noise(target, noise, timesteps)
        batch["diffusion/noise_target"] = noise
        batch["diffusion/timestep"] = timesteps
        batch["diffusion/scheduler_signature"] = _scheduler_signature(scheduler)
        return batch


class SingleDomainDiffusionNoisingStage(MultiDomainDiffusionNoisingStage):
    """Single-domain constructor for interpolated Hydra model configs."""

    def __init__(
        self,
        domain: str,
        noise_scheduler,
        action_dim: int,
        action_horizon: int,
    ):
        super().__init__(
            noise_schedulers={str(domain): noise_scheduler},
            action_dims={str(domain): int(action_dim)},
            action_horizon=action_horizon,
        )


class MultiDomainDiffusionDenoiserStage(Stage):
    """Predict epsilon in training and run the reverse process in rollout."""

    reads = (
        "condition",
        "diffusion/noisy_action",
        "diffusion/timestep",
        "diffusion/scheduler_signature",
        "embodiment",
    )
    writes = ("diffusion/predicted_noise",)
    reads_by_mode = {"rollout": ("condition", "embodiment")}
    writes_by_mode = {"rollout": ("pred_action", "log/*")}
    objective = "epsilon"

    def __init__(
        self,
        policies: Mapping[str, DiffusionPolicy],
        action_horizon: int,
        condition_input_dim: int,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.condition_input_dim = int(condition_input_dim)
        if self.action_horizon <= 0 or self.condition_input_dim <= 0:
            raise ValueError("action_horizon and condition_input_dim must be positive")
        if not policies:
            raise ValueError("policies must contain at least one domain")

        normalized = {}
        action_dims = {}
        for raw_domain, policy in dict(policies).items():
            domain = str(raw_domain)
            if not isinstance(policy, DiffusionPolicy):
                raise TypeError(
                    f"Policy for {domain!r} must be DiffusionPolicy, got "
                    f"{type(policy).__name__}"
                )
            if int(policy.action_horizon) != self.action_horizon:
                raise ValueError(
                    f"Policy {domain!r} horizon={policy.action_horizon}, expected "
                    f"{self.action_horizon}"
                )
            if set(policy.infer_ac_dims) != {domain}:
                raise ValueError(
                    f"Policy {domain!r} infer_ac_dims must contain only that "
                    f"domain, got {sorted(policy.infer_ac_dims)}"
                )
            action_dim = int(policy.infer_ac_dims[domain])
            if action_dim <= 0:
                raise ValueError(
                    f"Policy {domain!r} has invalid action width {action_dim}"
                )
            _validate_epsilon_scheduler(policy.noise_scheduler, domain)
            if (
                policy.num_inference_steps is None
                or int(policy.num_inference_steps) <= 0
            ):
                raise ValueError(
                    f"Policy {domain!r} must declare positive num_inference_steps"
                )
            proj_cond = getattr(policy.model, "proj_cond", None)
            if (
                proj_cond is not None
                and int(proj_cond.in_features) != self.condition_input_dim
            ):
                raise ValueError(
                    f"Policy {domain!r} expects {proj_cond.in_features} condition "
                    f"features, configured Pipeline condition has "
                    f"{self.condition_input_dim}"
                )
            global_cond_dim = getattr(policy.model, "global_cond_dim", None)
            if (
                global_cond_dim is not None
                and int(global_cond_dim) != self.condition_input_dim
            ):
                raise ValueError(
                    f"Policy {domain!r} expects {global_cond_dim} condition "
                    f"features, configured Pipeline condition has "
                    f"{self.condition_input_dim}"
                )
            normalized[domain] = policy
            action_dims[domain] = action_dim

        self.policies = nn.ModuleDict(normalized)
        self.action_dims = action_dims

    def policy_for(self, embodiment: str) -> DiffusionPolicy:
        domain = str(embodiment)
        if domain not in self.policies:
            raise KeyError(
                f"Unknown diffusion domain {domain!r}; "
                f"configured={sorted(self.policies)}"
            )
        return self.policies[domain]

    def _validate_condition(self, condition: torch.Tensor) -> None:
        if condition.ndim != 2 or int(condition.shape[-1]) != self.condition_input_dim:
            raise ValueError(
                "Diffusion condition must have shape "
                f"(B, {self.condition_input_dim}), got {tuple(condition.shape)}"
            )

    def forward(self, batch: dict) -> dict:
        domain = str(batch["embodiment"])
        policy = self.policy_for(domain)
        condition = batch["condition"]
        self._validate_condition(condition)

        if self.training:
            expected_signature = _scheduler_signature(policy.noise_scheduler)
            if batch["diffusion/scheduler_signature"] != expected_signature:
                raise ValueError(
                    f"Forward and reverse diffusion schedulers differ for {domain!r}"
                )
            noisy_action = batch["diffusion/noisy_action"]
            _validate_action_shape(
                noisy_action,
                batch_size=int(condition.shape[0]),
                action_horizon=self.action_horizon,
                action_dim=self.action_dims[domain],
                label=f"Noisy diffusion action for {domain!r}",
            )
            prediction = policy.model(
                noisy_action, batch["diffusion/timestep"], condition
            )
            if prediction.shape != noisy_action.shape:
                raise ValueError(
                    "Diffusion epsilon prediction shape mismatch: "
                    f"prediction={tuple(prediction.shape)} "
                    f"noisy_action={tuple(noisy_action.shape)}"
                )
            batch["diffusion/predicted_noise"] = prediction
            return batch

        prediction = policy.sample_action(condition, domain)
        _validate_action_shape(
            prediction,
            batch_size=int(condition.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dims[domain],
            label=f"Diffusion sample for {domain!r}",
        )
        batch["pred_action"] = prediction
        batch["log/diffusion_inference_steps"] = float(policy.num_inference_steps)
        batch["log/diffusion_prediction_rms"] = (
            prediction.detach().square().mean().sqrt()
        )
        return batch


class SingleDomainDiffusionDenoiserStage(MultiDomainDiffusionDenoiserStage):
    """Single-domain constructor preserving normal policy state-dict keys."""

    def __init__(
        self,
        domain: str,
        policy: DiffusionPolicy,
        action_horizon: int,
        condition_input_dim: int,
    ):
        super().__init__(
            policies={str(domain): policy},
            action_horizon=action_horizon,
            condition_input_dim=condition_input_dim,
        )


class DiffusionEpsilonLossStage(Stage):
    """Apply epsilon-prediction MSE as an explicit leaf node."""

    train_only = True
    reads = ("diffusion/predicted_noise", "diffusion/noise_target", "target")
    writes = ("loss/diffusion_noise", "log/*")
    objective = "epsilon"

    def forward(self, batch: dict) -> dict:
        prediction = batch["diffusion/predicted_noise"]
        noise_target = batch["diffusion/noise_target"]
        if prediction.shape != noise_target.shape:
            raise ValueError(
                "Diffusion epsilon prediction shape mismatch: "
                f"prediction={tuple(prediction.shape)} "
                f"target={tuple(noise_target.shape)}"
            )
        loss = F.mse_loss(prediction, noise_target)
        batch["loss/diffusion_noise"] = loss
        batch["log/diffusion_noise"] = loss.detach()
        batch["log/diffusion_target_rms"] = (
            batch["target"].detach().square().mean().sqrt()
        )
        return batch


class MultiDomainDiffusionPolicyStage(MultiDomainDiffusionDenoiserStage):
    """Compatibility adapter for historical two-node Pipeline checkpoints.

    New configurations should use the explicit noising, denoiser, and loss
    stages above. This class retains the prior state-dict and config surface.
    """

    reads = ("condition", "target", "embodiment")
    writes = ("loss/diffusion_noise", "log/*")

    def forward(self, batch: dict) -> dict:
        if not self.training:
            return super().forward(batch)
        domain = str(batch["embodiment"])
        policy = self.policy_for(domain)
        condition = batch["condition"]
        self._validate_condition(condition)
        target = batch["target"]
        _validate_action_shape(
            target,
            batch_size=int(condition.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dims[domain],
            label=f"Diffusion target for {domain!r}",
        )
        prediction, noise_target = policy.predict(target, condition)
        loss = policy.loss_fn(prediction, noise_target)
        batch["loss/diffusion_noise"] = loss
        batch["log/diffusion_noise"] = loss.detach()
        batch["log/diffusion_target_rms"] = target.detach().square().mean().sqrt()
        return batch
