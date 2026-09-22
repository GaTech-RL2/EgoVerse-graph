"""Dual flow-matching stages for separated shape and clock predictions.

The bimanual ARC tokenizer lays out ``per_waypoint`` and ``duration`` targets
as ``[M shape rows, M clock rows]``.  These stages keep those streams separate:
the shape head receives the first stream and the smaller clock head receives
the second.  At inference the two generated streams are concatenated back to
the tokenizer's canonical layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egomimic.pipeline.core import Stage


def _validate_shape(
    value: torch.Tensor,
    *,
    batch_size: int,
    horizon: int,
    action_dim: int,
    label: str,
) -> None:
    expected = (batch_size, horizon, action_dim)
    if tuple(value.shape) != expected:
        raise ValueError(f"{label} must be {expected}, got {tuple(value.shape)}")


class DualFlowNoisingStage(Stage):
    """Create independent flow paths for shape and clock targets."""

    train_only = True
    reads = ("target",)
    writes = (
        "dual_flow/noisy_shape",
        "dual_flow/noisy_clock",
        "dual_flow/shape_velocity_target",
        "dual_flow/clock_velocity_target",
        "dual_flow/time",
    )
    objective = "dual_flow"

    def __init__(
        self,
        waypoint_horizon: int,
        action_dim: int,
        time_dist: str = "beta",
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        dtype: str = "float32",
    ):
        super().__init__()
        self.waypoint_horizon = int(waypoint_horizon)
        self.action_dim = int(action_dim)
        if self.waypoint_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("waypoint_horizon and action_dim must be positive")
        if time_dist not in {"beta", "uniform"}:
            raise ValueError("time_dist must be 'beta' or 'uniform'")
        self.time_dist = time_dist
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        if self.beta_alpha <= 0 or self.beta_beta <= 0:
            raise ValueError("beta parameters must be positive")
        self.dtype = getattr(torch, str(dtype))
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise ValueError(f"dtype must name a floating torch dtype, got {dtype!r}")

    def _sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self.time_dist == "beta":
            time = torch.distributions.Beta(
                torch.tensor(self.beta_alpha, device=device, dtype=dtype),
                torch.tensor(self.beta_beta, device=device, dtype=dtype),
            ).sample((batch_size,))
        else:
            time = torch.rand(batch_size, device=device, dtype=dtype)
        return time * 0.999 + 0.001

    def forward(self, batch: dict) -> dict:
        target = batch["target"]
        if target.is_floating_point() and target.dtype != self.dtype:
            target = target.to(self.dtype)
            batch["target"] = target
        batch_size = int(target.shape[0])
        full_horizon = 2 * self.waypoint_horizon
        _validate_shape(
            target,
            batch_size=batch_size,
            horizon=full_horizon,
            action_dim=self.action_dim,
            label="Dual-flow target",
        )
        shape_target = target[:, : self.waypoint_horizon]
        clock_target = target[:, self.waypoint_horizon :]
        shape_noise = torch.randn_like(shape_target)
        clock_noise = torch.randn_like(clock_target)
        time = self._sample_time(batch_size, target.device, target.dtype)
        expanded = time[:, None, None]
        batch["dual_flow/noisy_shape"] = (
            expanded * shape_noise + (1.0 - expanded) * shape_target
        )
        batch["dual_flow/noisy_clock"] = (
            expanded * clock_noise + (1.0 - expanded) * clock_target
        )
        batch["dual_flow/shape_velocity_target"] = shape_noise - shape_target
        batch["dual_flow/clock_velocity_target"] = clock_noise - clock_target
        batch["dual_flow/time"] = time
        return batch


class DualFlowDenoiserStage(Stage):
    """Train/integrate separate shape and clock flow heads."""

    reads = (
        "condition",
        "dual_flow/noisy_shape",
        "dual_flow/noisy_clock",
        "dual_flow/time",
    )
    writes = (
        "dual_flow/predicted_shape_velocity",
        "dual_flow/predicted_clock_velocity",
    )
    reads_by_mode = {"inference": ("condition",)}
    writes_by_mode = {"inference": ("pred_action", "log/*")}
    objective = "dual_flow"

    def __init__(
        self,
        shape_model: torch.nn.Module,
        clock_model: torch.nn.Module,
        waypoint_horizon: int,
        action_dim: int,
        condition_input_dim: int,
        num_inference_steps: int = 50,
        condition_as_tokens: bool = True,
        shape_condition_key: str = "condition",
        clock_condition_key: str = "condition",
    ):
        super().__init__()
        self.shape_model = shape_model
        self.clock_model = clock_model
        self.waypoint_horizon = int(waypoint_horizon)
        self.action_dim = int(action_dim)
        self.condition_input_dim = int(condition_input_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.condition_as_tokens = bool(condition_as_tokens)
        self.shape_condition_key = str(shape_condition_key)
        self.clock_condition_key = str(clock_condition_key)
        self.reads = tuple(
            dict.fromkeys(
                (
                    self.shape_condition_key,
                    self.clock_condition_key,
                    "dual_flow/noisy_shape",
                    "dual_flow/noisy_clock",
                    "dual_flow/time",
                )
            )
        )
        self.reads_by_mode = {
            "inference": (self.shape_condition_key, self.clock_condition_key)
        }
        if self.waypoint_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("waypoint_horizon and action_dim must be positive")
        if self.condition_input_dim <= 0 or self.num_inference_steps <= 0:
            raise ValueError(
                "condition_input_dim and num_inference_steps must be positive"
            )

    def _condition(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2 or int(condition.shape[-1]) != self.condition_input_dim:
            raise ValueError(
                "Dual-flow condition must have shape "
                f"(B, {self.condition_input_dim}), got {tuple(condition.shape)}"
            )
        return condition.unsqueeze(1) if self.condition_as_tokens else condition

    def execute(self, batch: dict, *, mode: str) -> dict:
        if mode == "inference":
            return self._forward_inference(batch)
        shape_condition = self._condition(batch[self.shape_condition_key])
        clock_condition = self._condition(batch[self.clock_condition_key])
        shape = batch["dual_flow/noisy_shape"]
        clock = batch["dual_flow/noisy_clock"]
        batch_size = int(shape_condition.shape[0])
        if int(clock_condition.shape[0]) != batch_size:
            raise ValueError("Shape and clock condition batches must have equal size")
        _validate_shape(
            shape,
            batch_size=batch_size,
            horizon=self.waypoint_horizon,
            action_dim=self.action_dim,
            label="Dual-flow noisy shape",
        )
        _validate_shape(
            clock,
            batch_size=batch_size,
            horizon=self.waypoint_horizon,
            action_dim=self.action_dim,
            label="Dual-flow noisy clock",
        )
        time = batch["dual_flow/time"]
        batch["dual_flow/predicted_shape_velocity"] = self.shape_model(
            shape, time, shape_condition
        )
        batch["dual_flow/predicted_clock_velocity"] = self.clock_model(
            clock, time, clock_condition
        )
        return batch

    @torch.no_grad()
    def _forward_inference(self, batch: dict) -> dict:
        shape_condition = self._condition(batch[self.shape_condition_key])
        clock_condition = self._condition(batch[self.clock_condition_key])
        batch_size = int(shape_condition.shape[0])
        if int(clock_condition.shape[0]) != batch_size:
            raise ValueError("Shape and clock condition batches must have equal size")
        dtype = shape_condition.dtype
        device = shape_condition.device
        shape = torch.randn(
            batch_size,
            self.waypoint_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        clock = torch.randn_like(shape)
        step = -1.0 / self.num_inference_steps
        time = torch.ones(batch_size, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            shape = shape + step * self.shape_model(shape, time, shape_condition)
            clock = clock + step * self.clock_model(clock, time, clock_condition)
            time = time + step
        batch["pred_action"] = torch.cat([shape, clock], dim=1)
        batch["log/DualFlowInferenceSteps"] = torch.tensor(
            float(self.num_inference_steps), device=device
        )
        return batch


class DualFlowVelocityLossStage(Stage):
    """Separate shape/clock losses, with an optional clock weighting."""

    train_only = True
    reads = (
        "dual_flow/predicted_shape_velocity",
        "dual_flow/predicted_clock_velocity",
        "dual_flow/shape_velocity_target",
        "dual_flow/clock_velocity_target",
    )
    writes = ("loss/flow_shape", "loss/flow_clock", "log/*")

    def __init__(self, clock_loss_weight: float = 1.0):
        super().__init__()
        self.clock_loss_weight = float(clock_loss_weight)
        if self.clock_loss_weight < 0:
            raise ValueError("clock_loss_weight must be non-negative")

    def forward(self, batch: dict) -> dict:
        shape_prediction = batch["dual_flow/predicted_shape_velocity"]
        clock_prediction = batch["dual_flow/predicted_clock_velocity"]
        shape_target = batch["dual_flow/shape_velocity_target"]
        clock_target = batch["dual_flow/clock_velocity_target"]
        shape_loss = F.mse_loss(shape_prediction, shape_target)
        clock_loss = F.mse_loss(clock_prediction, clock_target)
        batch["loss/flow_shape"] = shape_loss
        batch["loss/flow_clock"] = self.clock_loss_weight * clock_loss
        batch["log/flow_shape"] = shape_loss.detach()
        batch["log/flow_clock"] = clock_loss.detach()
        batch["log/flow_clock_weighted"] = batch["loss/flow_clock"].detach()
        return batch
