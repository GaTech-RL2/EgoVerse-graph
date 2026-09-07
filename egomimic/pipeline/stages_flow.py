"""Flow-matching stages, the head HPT is trained with upstream.

Mirrors the diffusion trio in ``stages_diffusion.py`` so the two objectives are
interchangeable behind the same graph seam: both read ``condition`` and
``target``, both write ``loss/*`` in training and ``pred_action`` at inference.
Swapping one for the other is a config edit, not a code change.

Flow matching differs from epsilon-prediction diffusion in what the network is
asked to produce. Diffusion adds noise at a sampled timestep and predicts that
noise. Flow matching interpolates linearly between the action and noise,

    x_t = t * noise + (1 - t) * action

and predicts the constant velocity of that path, ``u_t = noise - action``.
Sampling then integrates the learned field backwards from ``t = 1`` to
``t = 0`` with plain Euler steps, so it needs no noise schedule at all -- which
is why these stages carry no scheduler and no scheduler-signature handshake.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egomimic.pipeline.core import Stage

_TIME_DISTRIBUTIONS = ("beta", "uniform")
# Keeps the sampled time off both endpoints: at exactly 0 the path carries no
# noise and at exactly 1 it carries no action, so neither end teaches the field
# anything about the interior.
_TIME_EPSILON = 0.001


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


class FlowNoisingStage(Stage):
    """Interpolate between the action and noise, and record the target field.

    ``time_dist`` picks where along the path to sample. HPT uses ``beta(1.5, 1)``
    upstream, which leans toward ``t = 1`` (the noisy end) and so spends more
    capacity where sampling starts and errors compound.
    """

    train_only = True
    reads = ("target",)
    writes = (
        "flow/noisy_action",
        "flow/velocity_target",
        "flow/time",
    )
    objective = "flow"

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        time_dist: str = "beta",
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        dtype: str = "float32",
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if time_dist not in _TIME_DISTRIBUTIONS:
            raise ValueError(
                f"time_dist must be one of {_TIME_DISTRIBUTIONS}, got {time_dist!r}"
            )
        self.time_dist = time_dist
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        if self.beta_alpha <= 0 or self.beta_beta <= 0:
            raise ValueError("beta_alpha and beta_beta must be positive")
        # The flow namespace's working precision. Zarr stamps actions as
        # float64 and torch.from_numpy preserves it, but autocast only handles
        # float32/float16/bfloat16 -- never Double -- so a float64 target would
        # reach the denoiser's Linear unchanged and raise. Normalising here, at
        # the single point where `target` enters the namespace, keeps
        # noisy_action, velocity_target and time in one dtype so the loss
        # compares like with like.
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
        return time * (1.0 - _TIME_EPSILON) + _TIME_EPSILON

    def forward(self, batch: dict) -> dict:
        target = batch["target"]
        if target.is_floating_point() and target.dtype != self.dtype:
            target = target.to(self.dtype)
            batch["target"] = target
        _validate_action_shape(
            target,
            batch_size=int(target.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            label="Flow target",
        )
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype)
        time = self._sample_time(int(target.shape[0]), target.device, target.dtype)
        expanded = time[:, None, None]
        batch["flow/noisy_action"] = expanded * noise + (1.0 - expanded) * target
        # The path is a straight line, so its velocity is constant in t.
        batch["flow/velocity_target"] = noise - target
        batch["flow/time"] = time
        return batch


class FlowDenoiserStage(Stage):
    """Predict the velocity field, or integrate it into an action.

    In training this reads the interpolated point and returns the field at it.
    At inference the noising stage is gone, so it starts from pure noise and
    walks ``num_inference_steps`` Euler steps from ``t = 1`` down to ``t = 0``.
    """

    reads = ("condition", "flow/noisy_action", "flow/time")
    writes = ("flow/predicted_velocity",)
    reads_by_mode = {"inference": ("condition",)}
    writes_by_mode = {"inference": ("pred_action", "log/*")}
    objective = "flow"

    def __init__(
        self,
        model: torch.nn.Module,
        action_horizon: int,
        action_dim: int,
        condition_input_dim: int,
        num_inference_steps: int = 50,
        condition_as_tokens: bool = False,
    ):
        super().__init__()
        self.model = model
        # Some denoisers cross-attend over the condition and so want a token
        # axis, (B, 1, C), rather than the pooled (B, C) vector HPT's trunk
        # emits. This only reshapes; the condition itself is unchanged.
        self.condition_as_tokens = bool(condition_as_tokens)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.condition_input_dim = int(condition_input_dim)
        self.num_inference_steps = int(num_inference_steps)
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if self.condition_input_dim <= 0:
            raise ValueError("condition_input_dim must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

    def _model_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return condition.unsqueeze(1) if self.condition_as_tokens else condition

    def _validate_condition(self, condition: torch.Tensor) -> None:
        if condition.ndim != 2 or int(condition.shape[-1]) != self.condition_input_dim:
            raise ValueError(
                "Flow condition must have shape "
                f"(B, {self.condition_input_dim}), got {tuple(condition.shape)}"
            )

    def execute(self, batch: dict, *, mode: str) -> dict:
        if mode == "inference":
            return self._forward_inference(batch)
        return self._forward_train(batch)

    def _forward_train(self, batch: dict) -> dict:
        condition = batch["condition"]
        self._validate_condition(condition)
        noisy = batch["flow/noisy_action"]
        _validate_action_shape(
            noisy,
            batch_size=int(condition.shape[0]),
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            label="Flow noisy action",
        )
        prediction = self.model(
            noisy, batch["flow/time"], self._model_condition(condition)
        )
        if prediction.shape != noisy.shape:
            raise ValueError(
                "Flow velocity prediction shape mismatch: "
                f"prediction={tuple(prediction.shape)} noisy={tuple(noisy.shape)}"
            )
        batch["flow/predicted_velocity"] = prediction
        return batch

    @torch.no_grad()
    def _forward_inference(self, batch: dict) -> dict:
        condition = batch["condition"]
        self._validate_condition(condition)
        batch_size = int(condition.shape[0])
        x_t = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=condition.device,
            dtype=condition.dtype,
        )
        # Integrate backwards: t runs 1 -> 0, so the step is negative.
        step = -1.0 / self.num_inference_steps
        time = torch.ones(batch_size, device=condition.device, dtype=condition.dtype)
        for _ in range(self.num_inference_steps):
            velocity = self.model(x_t, time, self._model_condition(condition))
            x_t = x_t + step * velocity
            time = time + step
        batch["pred_action"] = x_t
        batch["log/FlowInferenceSteps"] = torch.tensor(
            float(self.num_inference_steps), device=condition.device
        )
        return batch


class FlowVelocityLossStage(Stage):
    """MSE between the predicted and true velocity field."""

    train_only = True
    reads = ("flow/predicted_velocity", "flow/velocity_target")
    writes = ("loss/flow_velocity", "log/*")

    def forward(self, batch: dict) -> dict:
        prediction = batch["flow/predicted_velocity"]
        target = batch["flow/velocity_target"]
        if prediction.shape != target.shape:
            raise ValueError(
                "Flow loss shape mismatch: "
                f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
            )
        loss = F.mse_loss(prediction, target)
        batch["loss/flow_velocity"] = loss
        batch["log/flow_velocity"] = loss.detach()
        batch["log/flow_target_rms"] = target.detach().pow(2).mean().sqrt()
        return batch
