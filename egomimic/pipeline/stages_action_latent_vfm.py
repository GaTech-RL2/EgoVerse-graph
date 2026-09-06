"""General action-latent velocity-flow stages.

The pipeline owns no embodiment semantics. Application adapters prepare the
``target`` and ``condition`` tensors before these stages run.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from egomimic.pipeline.core import Stage


def _positions(length: int, width: int) -> torch.Tensor:
    if length <= 0 or width <= 0 or width % 2:
        raise ValueError("position dimensions must be positive and width even")
    position = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    frequency = 1.0 / (
        10_000.0 ** (torch.arange(width // 2, dtype=torch.float64) / float(width // 2))
    )
    angle = position * frequency.unsqueeze(0)
    return torch.cat((angle.sin(), angle.cos()), dim=1).float().unsqueeze(0)


class TinyActionLatentEncoder(nn.Module):
    """Encode H action tokens into N compact latent registers."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        num_latent_tokens: int,
        latent_dim: int,
        hidden_dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.num_latent_tokens = int(num_latent_tokens)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        if (
            min(
                self.action_dim,
                self.action_horizon,
                self.num_latent_tokens,
                self.latent_dim,
                self.hidden_dim,
                int(depth),
                int(num_heads),
            )
            <= 0
        ):
            raise ValueError("encoder dimensions must be positive")
        if self.hidden_dim % int(num_heads):
            raise ValueError("encoder hidden_dim must be divisible by num_heads")
        self.action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.register_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.register_queries = nn.Parameter(
            torch.empty(1, self.num_latent_tokens, self.latent_dim).normal_(std=0.02)
        )
        self.register_buffer(
            "action_positions", _positions(self.action_horizon, self.hidden_dim)
        )
        self.register_buffer(
            "latent_positions", _positions(self.num_latent_tokens, self.hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.hidden_dim * float(mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=int(depth), norm=nn.LayerNorm(self.hidden_dim)
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        expected = (self.action_horizon, self.action_dim)
        if actions.ndim != 3 or tuple(actions.shape[1:]) != expected:
            raise ValueError(
                f"action encoder expected (B, {expected[0]}, {expected[1]}), "
                f"got {tuple(actions.shape)}"
            )
        content = self.action_projection(actions) + self.action_positions.to(actions)
        register_queries = self.register_queries.expand(int(actions.shape[0]), -1, -1)
        queries = self.register_projection(register_queries.to(actions))
        queries = queries + self.latent_positions.to(queries)
        hidden = self.blocks(torch.cat((queries, content), dim=1))
        return self.output_projection(hidden[:, : self.num_latent_tokens])

    def forward_with_activations(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        """Run the real encoder path and retain its latent-query activations."""

        activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        handles = []
        for index, block in enumerate(self.blocks.layers):
            name = f"block_{index:02d}"

            def capture(_module, _inputs, output, *, key=name):
                activations[key] = output[:, : self.num_latent_tokens]

            handles.append(block.register_forward_hook(capture))
        try:
            output = self(actions)
        finally:
            for handle in handles:
                handle.remove()
        if len(activations) != len(self.blocks.layers):
            raise RuntimeError("action encoder diagnostic capture is incomplete")
        return output, activations


class ActionLatentEncoderStage(Stage):
    train_only = True
    reads = ("target",)
    writes = ("latent/clean",)

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch: dict) -> dict:
        batch["latent/clean"] = self.encoder(batch["target"])
        return batch


class FlowBridgeNoisingStage(Stage):
    """Build target-coupled linear FM bridges with detached clean endpoints."""

    train_only = True
    reads = ("latent/clean",)
    writes = ("fm/noisy_latent", "fm/time", "fm/target_velocity", "fm/condition_repeat")

    def __init__(self, samples_per_reconstruction: int = 14):
        super().__init__()
        self.samples_per_reconstruction = int(samples_per_reconstruction)
        if self.samples_per_reconstruction <= 0:
            raise ValueError("samples_per_reconstruction must be positive")

    def forward(self, batch: dict) -> dict:
        clean = batch["latent/clean"].detach()
        repeats = self.samples_per_reconstruction
        clean = clean.repeat_interleave(repeats, dim=0)
        noise = torch.randn_like(clean)
        time = torch.rand(int(clean.shape[0]), device=clean.device, dtype=torch.float32)
        time_view = time.reshape(-1, 1, 1).to(clean)
        batch["fm/noisy_latent"] = time_view * clean + (1.0 - time_view) * noise
        batch["fm/time"] = time
        batch["fm/target_velocity"] = clean - noise
        batch["fm/condition_repeat"] = repeats
        return batch


class ReconstructionBridgeNoisingStage(Stage):
    """Always corrupt clean latents at t sampled uniformly from [0.5, 1)."""

    train_only = True
    reads = ("latent/clean",)
    writes = ("reconstruction/noisy_latent", "reconstruction/time")

    def __init__(self, time_min: float = 0.5, time_max: float = 1.0):
        super().__init__()
        self.time_min = float(time_min)
        self.time_max = float(time_max)
        if not 0.0 <= self.time_min < self.time_max <= 1.0:
            raise ValueError(
                "reconstruction time range must satisfy 0 <= min < max <= 1"
            )

    def forward(self, batch: dict) -> dict:
        clean = batch["latent/clean"]
        time = self.time_min + (self.time_max - self.time_min) * torch.rand(
            int(clean.shape[0]), device=clean.device, dtype=torch.float32
        )
        time_view = time.reshape(-1, 1, 1).to(clean)
        batch["reconstruction/noisy_latent"] = time_view * clean + (
            1.0 - time_view
        ) * torch.randn_like(clean)
        batch["reconstruction/time"] = time
        return batch


class ExpectedMonotonicNoisingStage(Stage):
    """Create paired bridge states ordered by noise severity.

    The repository's bridge time is a clean fraction, so noise severity is
    ``s = 1 - t``. Both states use the same Gaussian draw (common random
    numbers), which preserves their marginals while reducing ranking variance.
    """

    train_only = True
    reads = ("latent/clean",)
    writes = (
        "monotonic/noisy_less",
        "monotonic/time_less",
        "monotonic/noisy_more",
        "monotonic/time_more",
    )

    def __init__(self, severity_min: float = 0.0, severity_max: float = 0.5):
        super().__init__()
        self.severity_min = float(severity_min)
        self.severity_max = float(severity_max)
        if not 0.0 <= self.severity_min < self.severity_max <= 1.0:
            raise ValueError(
                "monotonic severity range must satisfy 0 <= min < max <= 1"
            )

    def forward(self, batch: dict) -> dict:
        # This auxiliary objective is a vector-field constraint. Keep its
        # gradient out of the action encoder; the reconstruction objective owns
        # encoder training.
        clean = batch["latent/clean"].detach()
        severities = self.severity_min + (
            self.severity_max - self.severity_min
        ) * torch.rand(2, device=clean.device, dtype=torch.float32)
        severity_less, severity_more = severities.sort().values.unbind()
        # One severity pair is shared across the minibatch so its two empirical
        # means estimate R(s_less) and R(s_more), rather than mixing many R(s).
        time_less = (1.0 - severity_less).expand(int(clean.shape[0]))
        time_more = (1.0 - severity_more).expand(int(clean.shape[0]))
        noise = torch.randn_like(clean)

        def bridge(time: torch.Tensor) -> torch.Tensor:
            time_view = time.reshape(-1, 1, 1).to(clean)
            return time_view * clean + (1.0 - time_view) * noise

        batch["monotonic/noisy_less"] = bridge(time_less)
        batch["monotonic/time_less"] = time_less
        batch["monotonic/noisy_more"] = bridge(time_more)
        batch["monotonic/time_more"] = time_more
        return batch


class LatentVelocityFieldStage(Stage):
    """Apply one shared AdaLN velocity field in training and integrate at inference."""

    reads = (
        "condition",
        "fm/noisy_latent",
        "fm/time",
        "fm/condition_repeat",
        "reconstruction/noisy_latent",
        "reconstruction/time",
        "monotonic/noisy_less",
        "monotonic/time_less",
        "monotonic/noisy_more",
        "monotonic/time_more",
    )
    writes = (
        "fm/pred_velocity",
        "reconstruction/pred_velocity",
        "log/action_latent_fm_condition_drop_fraction",
        "log/action_latent_reconstruction_condition_drop_fraction",
        "monotonic/pred_velocity_less",
        "monotonic/pred_velocity_more",
        "log/action_latent_monotonic_condition_drop_fraction",
    )
    reads_by_mode = {"inference": ("condition", "sampler/noise")}
    writes_by_mode = {"inference": ("sampler/endpoint",)}

    def __init__(
        self,
        denoising_module: nn.Module,
        condition_dim: int,
        num_inference_steps: int = 16,
        condition_dropout_probability: float = 0.1,
        cfg_scale: float = 4.0,
    ):
        super().__init__()
        self.denoising_module = denoising_module
        self.condition_dim = int(condition_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.condition_dropout_probability = float(condition_dropout_probability)
        self.cfg_scale = float(cfg_scale)
        if self.condition_dim <= 0 or self.num_inference_steps <= 0:
            raise ValueError("condition_dim and num_inference_steps must be positive")
        if not 0.0 <= self.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0.0:
            raise ValueError("cfg_scale must be finite and non-negative")
        self.null_condition = nn.Parameter(
            torch.empty(self.condition_dim).normal_(std=0.02)
        )

    def _null_condition_like(self, condition: torch.Tensor) -> torch.Tensor:
        if (
            condition.ndim not in {2, 3}
            or int(condition.shape[-1]) != self.condition_dim
        ):
            raise ValueError(
                "condition must have shape (B, D) or (B, C, D) with "
                f"D={self.condition_dim}, got {tuple(condition.shape)}"
            )
        shape = (1,) * (condition.ndim - 1) + (self.condition_dim,)
        return self.null_condition.to(condition).reshape(shape).expand_as(condition)

    def _drop_condition(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.condition_dropout_probability == 0.0:
            return condition, torch.zeros(
                int(condition.shape[0]), device=condition.device, dtype=torch.bool
            )
        mask = (
            torch.rand(int(condition.shape[0]), device=condition.device)
            < self.condition_dropout_probability
        )
        broadcast = mask.reshape(int(condition.shape[0]), *([1] * (condition.ndim - 1)))
        return (
            torch.where(broadcast, self._null_condition_like(condition), condition),
            mask,
        )

    def _guided_velocity(
        self, latent: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if self.cfg_scale == 1.0:
            return self.denoising_module(latent, time, condition)
        doubled_latent = torch.cat((latent, latent), dim=0)
        doubled_time = torch.cat((time, time), dim=0)
        doubled_condition = torch.cat(
            (condition, self._null_condition_like(condition)), dim=0
        )
        conditioned, unconditional = self.denoising_module(
            doubled_latent, doubled_time, doubled_condition
        ).chunk(2, dim=0)
        return unconditional + self.cfg_scale * (conditioned - unconditional)

    def execute(self, batch: dict, *, mode: str) -> dict:
        condition = batch["condition"]
        if mode == "train":
            repeats = int(batch["fm/condition_repeat"])
            fm_condition = condition.repeat_interleave(repeats, dim=0)
            fm_condition, fm_dropped = self._drop_condition(fm_condition)
            batch["fm/pred_velocity"] = self.denoising_module(
                batch["fm/noisy_latent"],
                batch["fm/time"],
                fm_condition,
            )
            reconstruction_condition, reconstruction_dropped = self._drop_condition(
                condition
            )
            batch["reconstruction/pred_velocity"] = self.denoising_module(
                batch["reconstruction/noisy_latent"],
                batch["reconstruction/time"],
                reconstruction_condition,
            )
            monotonic_condition, monotonic_dropped = self._drop_condition(condition)
            paired_velocity = self.denoising_module(
                torch.cat(
                    (
                        batch["monotonic/noisy_less"],
                        batch["monotonic/noisy_more"],
                    ),
                    dim=0,
                ),
                torch.cat(
                    (batch["monotonic/time_less"], batch["monotonic/time_more"]),
                    dim=0,
                ),
                torch.cat((monotonic_condition, monotonic_condition), dim=0),
            )
            (
                batch["monotonic/pred_velocity_less"],
                batch["monotonic/pred_velocity_more"],
            ) = paired_velocity.chunk(2, dim=0)
            batch["log/action_latent_fm_condition_drop_fraction"] = (
                fm_dropped.float().mean()
            )
            batch["log/action_latent_reconstruction_condition_drop_fraction"] = (
                reconstruction_dropped.float().mean()
            )
            batch["log/action_latent_monotonic_condition_drop_fraction"] = (
                monotonic_dropped.float().mean()
            )
            return batch
        latent = batch["sampler/noise"]
        step = 1.0 / self.num_inference_steps
        for index in range(self.num_inference_steps):
            time = torch.full(
                (int(latent.shape[0]),), index * step, device=latent.device
            )
            latent = latent + step * self._guided_velocity(latent, time, condition)
        batch["sampler/endpoint"] = latent
        return batch

    def diagnostic_rollout(
        self,
        *,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        condition: torch.Tensor,
        raw_noise_levels: tuple[float, ...] | list[float],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        OrderedDict[str, torch.Tensor],
    ]:
        """Return sampler states and fixed-level denoising diagnostics."""

        if tuple(noise.shape) != tuple(clean_latent.shape):
            raise ValueError("diagnostic clean latent and noise shapes must match")
        levels = tuple(float(value) for value in raw_noise_levels)
        if not levels or any(not 0.0 <= value <= 1.0 for value in levels):
            raise ValueError("diagnostic raw noise levels must be in [0, 1]")

        step = 1.0 / self.num_inference_steps
        latent = noise
        sampler_states = [latent]
        for index in range(self.num_inference_steps):
            time = torch.full(
                (int(latent.shape[0]),), index * step, device=latent.device
            )
            latent = latent + step * self._guided_velocity(latent, time, condition)
            sampler_states.append(latent)

        final_predictions = []
        captured: dict[str, list[torch.Tensor]] = {}
        for raw_level in levels:
            start_time = 1.0 - raw_level
            state = start_time * clean_latent + raw_level * noise
            remaining_step = raw_level / self.num_inference_steps
            first_activations = None
            for index in range(self.num_inference_steps):
                time = torch.full(
                    (int(state.shape[0]),),
                    start_time + index * remaining_step,
                    device=state.device,
                )
                if index == 0:
                    velocity, first_activations = (
                        self.denoising_module.forward_with_activations(
                            state, time, condition
                        )
                    )
                else:
                    velocity = self.denoising_module(state, time, condition)
                state = state + remaining_step * velocity
            if first_activations is None:
                raise RuntimeError("diagnostic denoising activation capture failed")
            final_predictions.append(state)
            for name, value in first_activations.items():
                captured.setdefault(name, []).append(value)
        return (
            torch.stack(sampler_states, dim=0),
            torch.stack(final_predictions, dim=0),
            OrderedDict(
                (name, torch.stack(values, dim=0)) for name, values in captured.items()
            ),
        )


class VelocityFlowObjectiveStage(Stage):
    train_only = True
    reads = ("fm/pred_velocity", "fm/target_velocity")
    writes = ("loss/action_latent_fm", "log/action_latent_fm")

    def __init__(self, samples_per_reconstruction: int = 14):
        super().__init__()
        self.samples_per_reconstruction = int(samples_per_reconstruction)
        if self.samples_per_reconstruction <= 0:
            raise ValueError("samples_per_reconstruction must be positive")

    def forward(self, batch: dict) -> dict:
        # Match the released UNITE 14:1 objective convention: each independently
        # sampled FM target contributes one full mean loss, so the terms are
        # summed rather than averaged away.
        loss = (
            batch["fm/pred_velocity"] - batch["fm/target_velocity"]
        ).square().mean() * self.samples_per_reconstruction
        batch["loss/action_latent_fm"] = loss
        batch["log/action_latent_fm"] = loss.detach()
        return batch


class VelocityEndpointStage(Stage):
    train_only = True
    reads = (
        "reconstruction/noisy_latent",
        "reconstruction/time",
        "reconstruction/pred_velocity",
        "monotonic/noisy_less",
        "monotonic/time_less",
        "monotonic/pred_velocity_less",
        "monotonic/noisy_more",
        "monotonic/time_more",
        "monotonic/pred_velocity_more",
    )
    writes = (
        "reconstruction/pred_clean_latent",
        "monotonic/pred_clean_less",
        "monotonic/pred_clean_more",
    )

    def forward(self, batch: dict) -> dict:
        def endpoint(prefix: str, suffix: str = "") -> torch.Tensor:
            key_suffix = f"_{suffix}" if suffix else ""
            noisy = batch[
                f"{prefix}/noisy{key_suffix}" if suffix else f"{prefix}/noisy_latent"
            ]
            time = batch[f"{prefix}/time{key_suffix}" if suffix else f"{prefix}/time"]
            velocity = batch[
                f"{prefix}/pred_velocity{key_suffix}"
                if suffix
                else f"{prefix}/pred_velocity"
            ]
            return noisy + (1.0 - time.reshape(-1, 1, 1).to(noisy)) * velocity

        batch["reconstruction/pred_clean_latent"] = endpoint("reconstruction")
        batch["monotonic/pred_clean_less"] = endpoint("monotonic", "less")
        batch["monotonic/pred_clean_more"] = endpoint("monotonic", "more")
        return batch


class ActionLatentDecoderStage(Stage):
    reads = (
        "reconstruction/pred_clean_latent",
        "monotonic/pred_clean_less",
        "monotonic/pred_clean_more",
    )
    writes = (
        "reconstruction/pred_action",
        "monotonic/pred_action_less",
        "monotonic/pred_action_more",
    )
    reads_by_mode = {"inference": ("sampler/endpoint",)}
    writes_by_mode = {"inference": ("pred_action",)}

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def _decode_with_frozen_parameters(self, latent: torch.Tensor) -> torch.Tensor:
        """Backpropagate to latent while excluding decoder parameter gradients."""

        frozen_state = {
            name: value.detach() for name, value in self.decoder.named_parameters()
        }
        frozen_state.update(dict(self.decoder.named_buffers()))
        return torch.func.functional_call(self.decoder, frozen_state, (latent,))

    def execute(self, batch: dict, *, mode: str) -> dict:
        if mode == "train":
            batch["reconstruction/pred_action"] = self.decoder(
                batch["reconstruction/pred_clean_latent"]
            )
            paired_action = self._decode_with_frozen_parameters(
                torch.cat(
                    (
                        batch["monotonic/pred_clean_less"],
                        batch["monotonic/pred_clean_more"],
                    ),
                    dim=0,
                )
            )
            (
                batch["monotonic/pred_action_less"],
                batch["monotonic/pred_action_more"],
            ) = paired_action.chunk(2, dim=0)
        else:
            batch["pred_action"] = self.decoder(batch["sampler/endpoint"])
        return batch


class NoisyReconstructionObjectiveStage(Stage):
    train_only = True
    reads = ("reconstruction/pred_action", "target", "latent/clean")
    writes = (
        "loss/action_latent_reconstruction",
        "log/action_latent_reconstruction",
        "log/action_latent_reconstruction_l1",
        "log/action_latent_clean_rms",
        "log/action_latent_clean_std",
    )

    def forward(self, batch: dict) -> dict:
        error = batch["reconstruction/pred_action"] - batch["target"]
        loss = error.square().mean()
        batch["loss/action_latent_reconstruction"] = loss
        batch["log/action_latent_reconstruction"] = loss.detach()
        batch["log/action_latent_reconstruction_l1"] = error.abs().mean().detach()
        clean = batch["latent/clean"].detach().float()
        batch["log/action_latent_clean_rms"] = clean.square().mean().sqrt()
        batch["log/action_latent_clean_std"] = clean.std(unbiased=False)
        return batch


class ExpectedMonotonicRankingObjectiveStage(Stage):
    """Penalize violations between batch-mean risks at ordered severities."""

    train_only = True
    reads = (
        "monotonic/pred_action_less",
        "monotonic/pred_action_more",
        "target",
    )
    writes = (
        "loss/action_latent_monotonic",
        "log/action_latent_monotonic",
        "log/action_latent_monotonic_risk_less",
        "log/action_latent_monotonic_risk_more",
        "log/action_latent_monotonic_order_gap",
    )

    def __init__(self, margin: float = 0.0, weight: float = 1.0):
        super().__init__()
        self.margin = float(margin)
        self.weight = float(weight)
        if not math.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError("monotonic margin must be finite and non-negative")
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("monotonic weight must be finite and non-negative")

    def forward(self, batch: dict) -> dict:
        target = batch["target"]
        risk_less = (batch["monotonic/pred_action_less"] - target).square().mean()
        risk_more = (batch["monotonic/pred_action_more"] - target).square().mean()
        loss = self.weight * torch.relu(risk_less - risk_more + self.margin)
        batch["loss/action_latent_monotonic"] = loss
        batch["log/action_latent_monotonic"] = loss.detach()
        batch["log/action_latent_monotonic_risk_less"] = risk_less.detach()
        batch["log/action_latent_monotonic_risk_more"] = risk_more.detach()
        batch["log/action_latent_monotonic_order_gap"] = (
            risk_more - risk_less
        ).detach()
        return batch
