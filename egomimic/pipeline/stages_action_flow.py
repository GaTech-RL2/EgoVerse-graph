"""Generic Pipeline stages for latent conditional flow with a learned codec."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.func import jvp

from egomimic.pipeline.core import Stage


def _key(value: str, *, label: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _tensor(batch: dict, key: str) -> torch.Tensor:
    value = batch[key]
    if not torch.is_tensor(value):
        raise TypeError(f"{key} must be a tensor, got {type(value).__name__}")
    return value


def _module(value: nn.Module, *, label: str) -> nn.Module:
    if not isinstance(value, nn.Module):
        raise TypeError(f"{label} must be an nn.Module, got {type(value).__name__}")
    return value


class _SplitFieldPrediction(torch.autograd.Function):
    """Share one field value while isolating the FM gradient from its state.

    The action branch and FM branch see identical predictions. During backward,
    both cotangents reach the field parameters and conditioning path, while a
    direct state-gradient correction removes only the FM contribution from the
    bridge state. This is equivalent to evaluating the field a second time on
    ``state.detach()``, but avoids that additional forward evaluation.
    """

    @staticmethod
    def forward(
        ctx,
        prediction: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.save_for_backward(prediction, state)
        return prediction, prediction

    @staticmethod
    def backward(
        ctx,
        action_gradient: torch.Tensor | None,
        flow_gradient: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        prediction, state = ctx.saved_tensors
        if action_gradient is None and flow_gradient is None:
            return None, None
        if flow_gradient is None:
            return action_gradient, None

        combined_gradient = flow_gradient
        if action_gradient is not None:
            combined_gradient = combined_gradient + action_gradient
        if not state.requires_grad:
            return combined_gradient, None

        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            flow_state_gradient = torch.autograd.grad(
                prediction,
                state,
                flow_gradient,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=False,
            )[0]
        return combined_gradient, -flow_state_gradient


class ContentEncoderStage(Stage):
    """Encode paired target content into a clean latent endpoint."""

    train_only = True

    def __init__(
        self,
        encoder: nn.Module,
        input_key: str = "target",
        output_key: str = "action_flow/clean_latent",
    ):
        super().__init__()
        self.encoder = _module(encoder, label="encoder")
        self.input_key = _key(input_key, label="input_key")
        self.output_key = _key(output_key, label="output_key")
        self.reads = (self.input_key,)
        self.writes = (self.output_key,)

    def forward(self, batch: dict) -> dict:
        content = _tensor(batch, self.input_key)
        if content.ndim < 2 or int(content.shape[0]) <= 0:
            raise ValueError(
                f"{self.input_key} must have shape (B, ...), got {tuple(content.shape)}"
            )
        clean = self.encoder(content)
        if not torch.is_tensor(clean) or clean.ndim < 2:
            shape = tuple(clean.shape) if torch.is_tensor(clean) else None
            raise ValueError(f"encoder output must have shape (B, ...), got {shape}")
        if int(clean.shape[0]) != int(content.shape[0]):
            raise ValueError(
                "encoder output batch does not match target content: "
                f"{clean.shape[0]} != {content.shape[0]}"
            )
        batch[self.output_key] = clean
        return batch


class LatentBridgeStage(Stage):
    """Construct clean-to-Gaussian bridges with configurable sample coupling."""

    train_only = True

    def __init__(
        self,
        samples_per_content: int = 14,
        condition_dropout_probability: float = 0.3,
        time_sampling: str = "uniform",
        lognorm_mu: float = 0.0,
        lognorm_sigma: float = 1.0,
        timestep_shift_alpha: float = 0.5,
        independent_noise_per_sample: bool = False,
        independent_condition_dropout_per_sample: bool = False,
        clean_key: str = "action_flow/clean_latent",
        noise_key: str = "sampler/noise",
        condition_key: str = "condition",
        state_key: str = "action_flow/state",
        time_key: str = "action_flow/time",
        expanded_noise_key: str = "action_flow/noise",
        target_velocity_key: str = "action_flow/target_velocity",
        repeated_condition_key: str = "action_flow/condition",
        condition_drop_mask_key: str = "action_flow/condition_drop_mask",
        base_condition_drop_mask_key: str = ("action_flow/base_condition_drop_mask"),
        base_index_key: str = "action_flow/base_index",
    ):
        super().__init__()
        self.samples_per_content = int(samples_per_content)
        self.condition_dropout_probability = float(condition_dropout_probability)
        if self.samples_per_content <= 0:
            raise ValueError("samples_per_content must be positive")
        if not 0.0 <= self.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        if time_sampling not in {"uniform", "lognormal_shifted"}:
            raise ValueError("time_sampling must be uniform|lognormal_shifted")
        self.time_sampling = str(time_sampling)
        self.lognorm_mu = float(lognorm_mu)
        self.lognorm_sigma = float(lognorm_sigma)
        self.timestep_shift_alpha = float(timestep_shift_alpha)
        self.independent_noise_per_sample = bool(independent_noise_per_sample)
        self.independent_condition_dropout_per_sample = bool(
            independent_condition_dropout_per_sample
        )
        if self.lognorm_sigma <= 0.0 or self.timestep_shift_alpha <= 0.0:
            raise ValueError("log-normal sigma and timestep shift must be positive")

        self.clean_key = _key(clean_key, label="clean_key")
        self.noise_key = _key(noise_key, label="noise_key")
        self.condition_key = _key(condition_key, label="condition_key")
        self.state_key = _key(state_key, label="state_key")
        self.time_key = _key(time_key, label="time_key")
        self.expanded_noise_key = _key(expanded_noise_key, label="expanded_noise_key")
        self.target_velocity_key = _key(
            target_velocity_key, label="target_velocity_key"
        )
        self.repeated_condition_key = _key(
            repeated_condition_key, label="repeated_condition_key"
        )
        self.condition_drop_mask_key = _key(
            condition_drop_mask_key, label="condition_drop_mask_key"
        )
        self.base_condition_drop_mask_key = _key(
            base_condition_drop_mask_key,
            label="base_condition_drop_mask_key",
        )
        self.base_index_key = _key(base_index_key, label="base_index_key")
        self.reads = (self.clean_key, self.noise_key, self.condition_key)
        self.writes = (
            self.state_key,
            self.time_key,
            self.expanded_noise_key,
            self.target_velocity_key,
            self.repeated_condition_key,
            self.condition_drop_mask_key,
            self.base_condition_drop_mask_key,
            self.base_index_key,
        )

    def _base_drop_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        probability = self.condition_dropout_probability
        if probability == 0.0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if probability == 1.0:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return torch.rand(batch_size, device=device) < probability

    def _sample_time(self, count: int, device: torch.device) -> torch.Tensor:
        if self.time_sampling == "uniform":
            return torch.rand(count, dtype=torch.float32, device=device)
        normal = torch.randn(count, dtype=torch.float32, device=device)
        clean_fraction = torch.sigmoid(
            normal * self.lognorm_sigma + self.lognorm_mu
        )
        alpha = clean_fraction.new_tensor(self.timestep_shift_alpha)
        clean_fraction = alpha * clean_fraction / (
            1.0 + (alpha - 1.0) * clean_fraction
        )
        # This bridge uses t=0 clean and t=1 Gaussian.
        return 1.0 - clean_fraction

    def forward(self, batch: dict) -> dict:
        clean = _tensor(batch, self.clean_key)
        base_noise = _tensor(batch, self.noise_key)
        condition = _tensor(batch, self.condition_key)
        if clean.ndim < 2 or int(clean.shape[0]) <= 0:
            raise ValueError(
                f"{self.clean_key} must have shape (B, ...), got {tuple(clean.shape)}"
            )
        if tuple(base_noise.shape) != tuple(clean.shape):
            raise ValueError(
                "base Gaussian noise must match the clean latent shape: "
                f"{tuple(base_noise.shape)} != {tuple(clean.shape)}"
            )
        batch_size = int(clean.shape[0])
        if condition.ndim < 2 or int(condition.shape[0]) != batch_size:
            raise ValueError(
                f"{self.condition_key} must have first dimension {batch_size}, "
                f"got {tuple(condition.shape)}"
            )
        if clean.device != base_noise.device or clean.device != condition.device:
            raise ValueError("clean latent, noise, and condition must share a device")

        count = self.samples_per_content
        base_index = torch.arange(batch_size, device=clean.device).repeat_interleave(
            count
        )
        clean_many = clean.index_select(0, base_index)
        if self.independent_noise_per_sample:
            noise_many = torch.randn_like(clean_many)
        else:
            noise_many = base_noise.to(dtype=clean.dtype).index_select(0, base_index)
        condition_many = condition.index_select(0, base_index)

        time = self._sample_time(batch_size * count, clean.device)
        time_view = time.to(dtype=clean.dtype).reshape(-1, *([1] * (clean.ndim - 1)))
        state = (1.0 - time_view) * clean_many + time_view * noise_many
        target_velocity = noise_many - clean_many

        base_mask = self._base_drop_mask(batch_size, clean.device)
        repeated_mask = (
            self._base_drop_mask(batch_size * count, clean.device)
            if self.independent_condition_dropout_per_sample
            else base_mask.index_select(0, base_index)
        )
        batch[self.state_key] = state
        batch[self.time_key] = time
        batch[self.expanded_noise_key] = noise_many
        batch[self.target_velocity_key] = target_velocity
        batch[self.repeated_condition_key] = condition_many
        batch[self.condition_drop_mask_key] = repeated_mask
        batch[self.base_condition_drop_mask_key] = base_mask
        batch[self.base_index_key] = base_index
        return batch


class ConditionalVelocityStage(Stage):
    """Predict bridge velocity in training and integrate it during inference.

    ``all_stopgrad`` isolates only the latent-FM clean-state/target routes.
    The original state, prediction, and residual remain fully attached for
    the decoder JVP. Both loss branches share one field evaluation over the
    same vectorized bridge batch.
    """

    def __init__(
        self,
        field: nn.Module,
        num_inference_steps: int = 16,
        inference_method: str = "euler",
        timestep_shift_alpha: float = 0.5,
        dopri5_atol: float = 1.0e-6,
        dopri5_rtol: float = 1.0e-3,
        cfg_scale: float = 1.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        state_key: str = "action_flow/state",
        time_key: str = "action_flow/time",
        condition_key: str = "action_flow/condition",
        condition_drop_mask_key: str = "action_flow/condition_drop_mask",
        target_velocity_key: str = "action_flow/target_velocity",
        predicted_velocity_key: str = "action_flow/predicted_velocity",
        residual_key: str = "action_flow/velocity_residual",
        inference_noise_key: str = "sampler/noise",
        inference_condition_key: str = "condition",
        generated_latent_key: str = "action_flow/generated_latent",
        trajectory_key: str = "action_flow/trajectory",
        inference_steps_log_key: str = "log/action_flow_inference_steps",
        flow_clean_gradient_mode: str = "full",
        flow_residual_key: str = "action_flow/fm_velocity_residual",
    ):
        super().__init__()
        self.field = _module(field, label="field")
        self.num_inference_steps = int(num_inference_steps)
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if inference_method not in {"euler", "dopri5"}:
            raise ValueError("inference_method must be euler|dopri5")
        self.inference_method = str(inference_method)
        self.timestep_shift_alpha = float(timestep_shift_alpha)
        self.dopri5_atol = float(dopri5_atol)
        self.dopri5_rtol = float(dopri5_rtol)
        self.cfg_scale = float(cfg_scale)
        self.cfg_interval = tuple(float(value) for value in cfg_interval)
        if self.timestep_shift_alpha <= 0.0:
            raise ValueError("timestep_shift_alpha must be positive")
        if self.dopri5_atol <= 0.0 or self.dopri5_rtol <= 0.0:
            raise ValueError("Dopri5 tolerances must be positive")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0.0:
            raise ValueError("cfg_scale must be finite and non-negative")
        if (
            len(self.cfg_interval) != 2
            or not 0.0 <= self.cfg_interval[0] <= self.cfg_interval[1] <= 1.0
        ):
            raise ValueError("cfg_interval must be an ordered pair in [0, 1]")
        if flow_clean_gradient_mode not in {"full", "all_stopgrad"}:
            raise ValueError("flow_clean_gradient_mode must be full|all_stopgrad")
        self.flow_clean_gradient_mode = flow_clean_gradient_mode

        self.state_key = _key(state_key, label="state_key")
        self.time_key = _key(time_key, label="time_key")
        self.condition_key = _key(condition_key, label="condition_key")
        self.condition_drop_mask_key = _key(
            condition_drop_mask_key, label="condition_drop_mask_key"
        )
        self.target_velocity_key = _key(
            target_velocity_key, label="target_velocity_key"
        )
        self.predicted_velocity_key = _key(
            predicted_velocity_key, label="predicted_velocity_key"
        )
        self.residual_key = _key(residual_key, label="residual_key")
        self.flow_residual_key = _key(flow_residual_key, label="flow_residual_key")
        if len({self.predicted_velocity_key, self.residual_key, self.flow_residual_key}) != 3:
            raise ValueError("prediction, residual, and FM residual keys must be distinct")
        self.inference_noise_key = _key(
            inference_noise_key, label="inference_noise_key"
        )
        self.inference_condition_key = _key(
            inference_condition_key, label="inference_condition_key"
        )
        self.generated_latent_key = _key(
            generated_latent_key, label="generated_latent_key"
        )
        self.trajectory_key = _key(trajectory_key, label="trajectory_key")
        self.inference_steps_log_key = _key(
            inference_steps_log_key, label="inference_steps_log_key"
        )

        self.reads = (
            self.state_key,
            self.time_key,
            self.condition_key,
            self.condition_drop_mask_key,
            self.target_velocity_key,
        )
        self.writes = (
            self.predicted_velocity_key,
            self.residual_key,
            self.flow_residual_key,
        )
        self.reads_by_mode = {
            "inference": (self.inference_noise_key, self.inference_condition_key)
        }
        self.writes_by_mode = {
            "inference": (
                self.generated_latent_key,
                self.trajectory_key,
                self.inference_steps_log_key,
            )
        }

    @staticmethod
    def _validate_call_inputs(
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> None:
        batch_size = int(state.shape[0])
        if state.ndim < 2 or batch_size <= 0:
            raise ValueError(f"flow state must have shape (B, ...), got {state.shape}")
        if time.ndim != 1 or int(time.shape[0]) != batch_size:
            raise ValueError(f"flow time must have shape ({batch_size},)")
        if condition.ndim < 2 or int(condition.shape[0]) != batch_size:
            raise ValueError(f"flow condition must have first dimension {batch_size}")
        if drop_mask.dtype != torch.bool or tuple(drop_mask.shape) != (batch_size,):
            raise ValueError(f"condition drop mask must have shape ({batch_size},)")
        if not (state.device == time.device == condition.device == drop_mask.device):
            raise ValueError("flow inputs must share a device")

    def _predict(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_call_inputs(state, time, condition, drop_mask)
        prediction = self.field(
            state,
            time,
            condition,
            condition_drop_mask=drop_mask,
        )
        if not torch.is_tensor(prediction) or prediction.shape != state.shape:
            shape = tuple(prediction.shape) if torch.is_tensor(prediction) else None
            raise ValueError(
                "velocity field output must match the latent state shape: "
                f"{shape} != {tuple(state.shape)}"
            )
        return prediction

    def _forward_train(self, batch: dict) -> dict:
        state = _tensor(batch, self.state_key)
        time = _tensor(batch, self.time_key)
        condition = _tensor(batch, self.condition_key)
        drop_mask = _tensor(batch, self.condition_drop_mask_key)
        target_velocity = _tensor(batch, self.target_velocity_key)
        if target_velocity.shape != state.shape:
            raise ValueError("target velocity must match the latent state shape")
        prediction = self._predict(state, time, condition, drop_mask)
        residual = prediction - target_velocity
        if self.flow_clean_gradient_mode == "all_stopgrad":
            # The bridge consists only of the learned clean endpoint and
            # action-independent Gaussian noise. Share one field prediction,
            # but cancel only the FM state gradient and detach its target so
            # neither clean route reaches the encoder.
            prediction, flow_prediction = _SplitFieldPrediction.apply(
                prediction,
                state,
            )
            flow_residual = flow_prediction - target_velocity.detach()
        else:
            flow_residual = residual
        batch[self.predicted_velocity_key] = prediction
        batch[self.residual_key] = residual
        batch[self.flow_residual_key] = flow_residual
        return batch

    def _forward_inference(self, batch: dict) -> dict:
        state = _tensor(batch, self.inference_noise_key)
        condition = _tensor(batch, self.inference_condition_key)
        if state.ndim < 2 or int(state.shape[0]) <= 0:
            raise ValueError(f"{self.inference_noise_key} must have shape (B, ...)")
        batch_size = int(state.shape[0])
        if condition.ndim < 2 or int(condition.shape[0]) != batch_size:
            raise ValueError(
                f"{self.inference_condition_key} must have first dimension {batch_size}"
            )
        if state.device != condition.device:
            raise ValueError("inference noise and condition must share a device")

        conditioned_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=state.device
        )

        def guided_velocity(latent: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            conditioned = self._predict(
                latent, time, condition, conditioned_mask
            )
            if self.cfg_scale <= 1.0:
                return conditioned
            unconditioned = self._predict(
                latent, time, condition, ~conditioned_mask
            )
            guided = unconditioned + self.cfg_scale * (
                conditioned - unconditioned
            )
            start, end = self.cfg_interval
            active = ((time < end) & ((start == 0.0) | (time > start))).reshape(
                int(time.shape[0]), *([1] * (latent.ndim - 1))
            )
            return torch.where(active, guided, conditioned)
        if self.inference_method == "dopri5":
            try:
                from torchdiffeq import odeint
            except ImportError as exc:
                raise RuntimeError(
                    "Dopri5 Action Flow sampling requires torchdiffeq"
                ) from exc
            raw_grid = torch.linspace(
                0.0,
                1.0,
                self.num_inference_steps,
                device=state.device,
                dtype=torch.float32,
            )
            alpha = raw_grid.new_tensor(self.timestep_shift_alpha)
            clean_progress = alpha * raw_grid / (
                1.0 + (alpha - 1.0) * raw_grid
            )
            grid = 1.0 - clean_progress

            def velocity(
                time_scalar: torch.Tensor, latent: torch.Tensor
            ) -> torch.Tensor:
                time = time_scalar.expand(batch_size)
                return guided_velocity(latent, time).float()

            trajectory_tensor = odeint(
                velocity,
                state.float(),
                grid,
                method="dopri5",
                atol=self.dopri5_atol,
                rtol=self.dopri5_rtol,
            )
            if not bool(torch.isfinite(trajectory_tensor).all()):
                raise RuntimeError("Dopri5 Action Flow trajectory is non-finite")
            state = trajectory_tensor[-1].to(dtype=state.dtype)
            trajectory = trajectory_tensor.to(dtype=state.dtype)
        else:
            trajectory_values = [state]
            step_size = 1.0 / self.num_inference_steps
            for index in range(self.num_inference_steps):
                time = torch.full(
                    (batch_size,),
                    1.0 - index * step_size,
                    dtype=torch.float32,
                    device=state.device,
                )
                state = state - step_size * guided_velocity(state, time)
                trajectory_values.append(state)
            trajectory = torch.stack(trajectory_values)

        batch[self.generated_latent_key] = state
        batch[self.trajectory_key] = trajectory
        batch[self.inference_steps_log_key] = state.new_tensor(
            float(self.num_inference_steps)
        )
        return batch

    def execute(self, batch: dict, *, mode: str) -> dict:
        if mode == "train":
            return self._forward_train(batch)
        if mode == "inference":
            return self._forward_inference(batch)
        raise ValueError(f"unsupported flow execution mode {mode!r}")

    def forward(self, batch: dict) -> dict:
        return self._forward_train(batch)


class ContentDecoderStage(Stage):
    """Decode clean content and map latent residuals through the decoder JVP."""

    def __init__(
        self,
        decoder: nn.Module,
        reconstruction_noising_start: float = 1.0,
        reconstruction_noising_probability: float = 0.0,
        clean_key: str = "action_flow/clean_latent",
        state_key: str = "action_flow/state",
        residual_key: str = "action_flow/velocity_residual",
        reconstruction_key: str = "action_flow/reconstruction",
        decoded_residual_key: str = "action_flow/decoded_velocity_residual",
        decode_noise: bool = False,
        noise_key: str = "sampler/noise",
        decoded_noise_key: str = "action_flow/decoded_noise",
        inference_latent_key: str = "action_flow/generated_latent",
        prediction_key: str = "pred_action",
    ):
        super().__init__()
        self.decoder = _module(decoder, label="decoder")
        self.reconstruction_noising_start = float(reconstruction_noising_start)
        self.reconstruction_noising_probability = float(
            reconstruction_noising_probability
        )
        if not 0.0 <= self.reconstruction_noising_start <= 1.0:
            raise ValueError("reconstruction_noising_start must be in [0, 1]")
        if not 0.0 <= self.reconstruction_noising_probability <= 1.0:
            raise ValueError("reconstruction_noising_probability must be in [0, 1]")
        self.clean_key = _key(clean_key, label="clean_key")
        self.state_key = _key(state_key, label="state_key")
        self.residual_key = _key(residual_key, label="residual_key")
        self.reconstruction_key = _key(reconstruction_key, label="reconstruction_key")
        self.decoded_residual_key = _key(
            decoded_residual_key, label="decoded_residual_key"
        )
        self.decode_noise = bool(decode_noise)
        self.noise_key = _key(noise_key, label="noise_key")
        self.decoded_noise_key = _key(decoded_noise_key, label="decoded_noise_key")
        self.inference_latent_key = _key(
            inference_latent_key, label="inference_latent_key"
        )
        self.prediction_key = _key(prediction_key, label="prediction_key")
        self.reads = (
            self.clean_key,
            self.state_key,
            self.residual_key,
        ) + ((self.noise_key,) if self.decode_noise else ())
        self.writes = (
            self.reconstruction_key,
            self.decoded_residual_key,
        ) + ((self.decoded_noise_key,) if self.decode_noise else ())
        self.reads_by_mode = {"inference": (self.inference_latent_key,)}
        self.writes_by_mode = {"inference": (self.prediction_key,)}

    def _decode(self, value: torch.Tensor, *, label: str) -> torch.Tensor:
        decoded = self.decoder(value)
        if not torch.is_tensor(decoded) or decoded.ndim < 2:
            shape = tuple(decoded.shape) if torch.is_tensor(decoded) else None
            raise ValueError(f"decoder {label} must have shape (B, ...), got {shape}")
        if int(decoded.shape[0]) != int(value.shape[0]):
            raise ValueError(f"decoder {label} batch does not match its input")
        return decoded

    def _forward_train(self, batch: dict) -> dict:
        clean = _tensor(batch, self.clean_key)
        state = _tensor(batch, self.state_key)
        residual = _tensor(batch, self.residual_key)
        noise = _tensor(batch, self.noise_key) if self.decode_noise else None
        if state.shape != residual.shape:
            raise ValueError(
                "latent state and velocity residual must have matching shapes"
            )
        reconstruction_input = clean
        if self.reconstruction_noising_probability > 0.0:
            batch_size = int(clean.shape[0])
            clean_fraction = self.reconstruction_noising_start + (
                1.0 - self.reconstruction_noising_start
            ) * torch.rand(batch_size, device=clean.device, dtype=torch.float32)
            view = clean_fraction.to(clean).reshape(
                batch_size, *([1] * (clean.ndim - 1))
            )
            noised = view * clean + (1.0 - view) * torch.randn_like(clean)
            mask = (
                torch.rand(batch_size, device=clean.device)
                < self.reconstruction_noising_probability
            ).reshape(batch_size, *([1] * (clean.ndim - 1)))
            reconstruction_input = torch.where(mask, noised, clean)
        reconstruction = self._decode(reconstruction_input, label="reconstruction")
        decoded_noise = (
            self._decode(noise, label="noise") if noise is not None else None
        )
        # PyTorch's non-reentrant activation checkpointing installs saved-tensor
        # hooks that are incompatible with ``torch.func`` transforms. Preserve
        # checkpointing for the reconstruction pass, but disable it only while
        # computing this required forward-mode JVP.
        checkpointing = getattr(self.decoder, "gradient_checkpointing", None)
        if isinstance(checkpointing, bool):
            self.decoder.gradient_checkpointing = False
        # CUDA FlashAttention does not implement forward-mode AD. Restrict the
        # decoder JVP to the mathematically equivalent SDPA math kernel; normal
        # reconstruction, training, and inference forwards keep their default
        # optimized attention selection.
        attention_context = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
            if state.is_cuda
            else nullcontext()
        )
        # Higher-order backward through the math kernel also requires matching
        # primal/tangent dtypes, so keep this isolated derivative in FP32 when
        # the surrounding trainer uses CUDA mixed precision.
        precision_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if state.is_cuda
            else nullcontext()
        )
        jvp_state = state.float() if state.is_cuda else state
        jvp_residual = residual.float() if residual.is_cuda else residual
        try:
            with precision_context, attention_context:
                decoded_residual = jvp(
                    self.decoder,
                    (jvp_state,),
                    (jvp_residual,),
                )[1]
        finally:
            if isinstance(checkpointing, bool):
                self.decoder.gradient_checkpointing = checkpointing
        if not torch.is_tensor(decoded_residual) or decoded_residual.ndim < 2:
            shape = (
                tuple(decoded_residual.shape)
                if torch.is_tensor(decoded_residual)
                else None
            )
            raise ValueError(f"decoder JVP must have shape (B, ...), got {shape}")
        if int(decoded_residual.shape[0]) != int(state.shape[0]):
            raise ValueError("decoder JVP batch does not match the bridge state")
        batch[self.reconstruction_key] = reconstruction
        batch[self.decoded_residual_key] = decoded_residual
        if decoded_noise is not None:
            batch[self.decoded_noise_key] = decoded_noise
        return batch

    def _forward_inference(self, batch: dict) -> dict:
        latent = _tensor(batch, self.inference_latent_key)
        batch[self.prediction_key] = self._decode(latent, label="prediction")
        return batch

    def execute(self, batch: dict, *, mode: str) -> dict:
        if mode == "train":
            return self._forward_train(batch)
        if mode == "inference":
            return self._forward_inference(batch)
        raise ValueError(f"unsupported decoder execution mode {mode!r}")

    def forward(self, batch: dict) -> dict:
        return self._forward_train(batch)


class ActionFlowObjectiveStage(Stage):
    """Compute component means and one explicitly weighted optimizer loss."""

    train_only = True

    def __init__(
        self,
        flow_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        action_velocity_weight: float = 1.0,
        moment_weight: float = 0.0,
        flow_aggregation: str = "mean",
        flow_samples_per_content: int = 1,
        target_key: str = "target",
        residual_key: str = "action_flow/velocity_residual",
        reconstruction_key: str = "action_flow/reconstruction",
        decoded_residual_key: str = "action_flow/decoded_velocity_residual",
        decoded_noise_key: str = "action_flow/decoded_noise",
        loss_key: str = "loss/action_flow",
        log_prefix: str = "log/action_flow",
    ):
        super().__init__()
        self.flow_weight = float(flow_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.action_velocity_weight = float(action_velocity_weight)
        self.moment_weight = float(moment_weight)
        if flow_aggregation not in {"mean", "sum_samples"}:
            raise ValueError("flow_aggregation must be mean|sum_samples")
        self.flow_aggregation = str(flow_aggregation)
        self.flow_samples_per_content = int(flow_samples_per_content)
        if self.flow_samples_per_content <= 0:
            raise ValueError("flow_samples_per_content must be positive")
        weights = (
            self.flow_weight,
            self.reconstruction_weight,
            self.action_velocity_weight,
            self.moment_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("objective weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one objective weight must be positive")

        self.target_key = _key(target_key, label="target_key")
        self.residual_key = _key(residual_key, label="residual_key")
        self.reconstruction_key = _key(reconstruction_key, label="reconstruction_key")
        self.decoded_residual_key = _key(
            decoded_residual_key, label="decoded_residual_key"
        )
        self.decoded_noise_key = _key(decoded_noise_key, label="decoded_noise_key")
        self.loss_key = _key(loss_key, label="loss_key")
        self.log_prefix = _key(log_prefix, label="log_prefix").rstrip("/")
        self.total_log_key = f"{self.log_prefix}_total"
        self.flow_log_key = f"{self.log_prefix}_fm"
        self.reconstruction_log_key = f"{self.log_prefix}_reconstruction"
        self.reconstruction_l1_log_key = f"{self.log_prefix}_reconstruction_l1"
        self.action_velocity_log_key = f"{self.log_prefix}_action_velocity"
        self.moment_log_key = f"{self.log_prefix}_decoded_noise_moments"
        self.moment_mean_log_key = f"{self.log_prefix}_decoded_noise_mean_penalty"
        self.moment_covariance_log_key = (
            f"{self.log_prefix}_decoded_noise_covariance_penalty"
        )
        self.reads = (
            self.target_key,
            self.residual_key,
            self.reconstruction_key,
            self.decoded_residual_key,
        ) + ((self.decoded_noise_key,) if self.moment_weight > 0.0 else ())
        self.writes = (
            self.loss_key,
            self.total_log_key,
            self.flow_log_key,
            self.reconstruction_log_key,
            self.reconstruction_l1_log_key,
            self.action_velocity_log_key,
            self.moment_log_key,
            self.moment_mean_log_key,
            self.moment_covariance_log_key,
        )

    def forward(self, batch: dict) -> dict:
        target = _tensor(batch, self.target_key)
        residual = _tensor(batch, self.residual_key)
        reconstruction = _tensor(batch, self.reconstruction_key)
        decoded_residual = _tensor(batch, self.decoded_residual_key)
        moment_mean = residual.new_zeros(())
        moment_covariance = residual.new_zeros(())
        if self.moment_weight > 0.0:
            decoded_noise = _tensor(batch, self.decoded_noise_key)
            if decoded_noise.ndim < 2 or int(decoded_noise.shape[-1]) <= 0:
                raise ValueError(
                    f"{self.decoded_noise_key} must have shape (..., F)"
                )
            samples = decoded_noise.float().reshape(
                -1, int(decoded_noise.shape[-1])
            )
            if int(samples.shape[0]) <= 1:
                raise ValueError("moment matching requires at least two samples")
            feature_dim = int(samples.shape[-1])
            mean = samples.mean(dim=0)
            centered = samples - mean
            covariance = centered.T @ centered / (int(samples.shape[0]) - 1)
            identity = torch.eye(
                feature_dim,
                device=samples.device,
                dtype=samples.dtype,
            )
            moment_mean = mean.square().sum() / feature_dim
            moment_covariance = (
                (covariance - identity).square().sum() / feature_dim
            )
        moment_penalty = moment_mean + moment_covariance
        if reconstruction.shape != target.shape:
            raise ValueError(
                "decoded clean content must match the target shape: "
                f"{tuple(reconstruction.shape)} != {tuple(target.shape)}"
            )
        if residual.ndim < 2 or int(residual.shape[0]) <= 0:
            raise ValueError("velocity residual must have shape (B, ...)")
        if decoded_residual.ndim < 2 or int(decoded_residual.shape[0]) <= 0:
            raise ValueError("decoded velocity residual must have shape (B, ...)")

        flow = residual.square().mean()
        if self.flow_aggregation == "sum_samples":
            # Each repeated sample already has a mean over latent coordinates.
            # Multiplying the all-sample mean by K is exactly the sum of the K
            # per-sample means used by the reference multi-sample objective.
            flow = flow * self.flow_samples_per_content
        reconstruction_error = reconstruction - target
        reconstruction_loss = reconstruction_error.square().mean()
        reconstruction_l1 = reconstruction_error.abs().mean()
        action_velocity = decoded_residual.square().mean()
        total = (
            self.flow_weight * flow
            + self.reconstruction_weight * reconstruction_loss
            + self.action_velocity_weight * action_velocity
            + self.moment_weight * moment_penalty
        )
        batch[self.loss_key] = total
        batch[self.total_log_key] = total
        batch[self.flow_log_key] = flow
        batch[self.reconstruction_log_key] = reconstruction_loss
        batch[self.reconstruction_l1_log_key] = reconstruction_l1
        batch[self.action_velocity_log_key] = action_velocity
        batch[self.moment_log_key] = moment_penalty
        batch[self.moment_mean_log_key] = moment_mean
        batch[self.moment_covariance_log_key] = moment_covariance
        return batch
