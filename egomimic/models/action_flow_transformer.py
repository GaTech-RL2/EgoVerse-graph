"""Generic observation-conditioned AdaLN sequence vector field."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate(
    value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _sinusoidal_time_embedding(
    time: torch.Tensor, width: int, *, time_scale: float
) -> torch.Tensor:
    """Embed normalized continuous time with diffusion-style frequency scale.

    The bridge supplies ``time`` in ``[0, 1]``.  Scaling it to the conventional
    diffusion timestep range before applying log-spaced frequencies prevents
    almost all channels from becoming effectively constant on that interval.
    """

    if width <= 0 or width % 2:
        raise ValueError("time_embedding_dim must be a positive even integer")
    half = width // 2
    exponent = (
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / float(max(half - 1, 1))
    )
    angle = time.float().unsqueeze(-1) * float(time_scale) * exponent.exp().unsqueeze(0)
    return torch.cat((angle.cos(), angle.sin()), dim=-1)


class _FieldSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        query, key, content = (
            self.qkv(value)
            .reshape(
                batch_size,
                sequence_length,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            content,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_dim
        )
        return self.output(attended)


class _AdaLNFieldBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1.0e-6
        )
        self.attention = _FieldSelfAttention(hidden_dim, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False, eps=1.0e-6
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            feedforward_shift,
            feedforward_scale,
            feedforward_gate,
        ) = modulation.unbind(dim=1)
        attended = self.attention(
            _modulate(self.attention_norm(value), attention_shift, attention_scale)
        )
        value = value + attention_gate.unsqueeze(1) * self.attention_dropout(attended)
        transformed = self.feedforward(
            _modulate(
                self.feedforward_norm(value),
                feedforward_shift,
                feedforward_scale,
            )
        )
        return value + feedforward_gate.unsqueeze(1) * transformed


class AdaLNSequenceField(nn.Module):
    """Map a noised sequence, scalar time, and condition to a vector field.

    A single learned null condition supports classifier-free condition dropout.
    Callers that expand each base example across multiple bridge times should
    sample one base-example mask and pass its expanded form through
    ``condition_drop_mask``.  Omitting the mask during training samples one mask
    per input row using ``condition_dropout_probability``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        condition_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        feedforward_dim: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
        condition_dropout_probability: float = 0.0,
        time_scale: float = 1_000.0,
    ) -> None:
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "horizon": horizon,
            "condition_dim": condition_dim,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "num_heads": num_heads,
            "feedforward_dim": feedforward_dim,
            "time_embedding_dim": time_embedding_dim,
        }
        invalid = [name for name, value in dimensions.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"field dimensions must be positive: {invalid}")
        if int(hidden_dim) % int(num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads")
        if int(time_embedding_dim) % 2:
            raise ValueError("time_embedding_dim must be even")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= float(condition_dropout_probability) <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        if not math.isfinite(float(time_scale)) or float(time_scale) <= 0.0:
            raise ValueError("time_scale must be finite and positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.dropout = float(dropout)
        self.condition_dropout_probability = float(condition_dropout_probability)
        self.time_scale = float(time_scale)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.horizon, self.hidden_dim)
        )
        self.null_condition = nn.Parameter(torch.zeros(self.condition_dim))
        self.condition_norm = nn.LayerNorm(self.condition_dim)
        self.condition_projection = nn.Linear(self.condition_dim, self.hidden_dim)
        self.time_projection = (
            nn.Identity()
            if self.time_embedding_dim == self.hidden_dim
            else nn.Linear(self.time_embedding_dim, self.hidden_dim)
        )
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, 6 * self.hidden_dim)
        )
        self.block_modulation_scale = nn.Parameter(
            torch.ones(self.depth, 6, self.hidden_dim)
        )
        self.block_modulation_bias = nn.Parameter(
            torch.zeros(self.depth, 6, self.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            _AdaLNFieldBlock(
                self.hidden_dim,
                self.num_heads,
                self.feedforward_dim,
                self.dropout,
            )
            for _ in range(self.depth)
        )
        self.output_norm = nn.LayerNorm(
            self.hidden_dim, elementwise_affine=False, eps=1.0e-6
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def apply_condition_dropout(
        self,
        condition: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace selected rows by the learned null condition.

        ``drop_mask=True`` means that the corresponding condition is dropped.
        An explicit mask is honored in both training and evaluation mode.
        """

        expected = (self.condition_dim,)
        if condition.ndim != 2 or tuple(condition.shape[1:]) != expected:
            raise ValueError(
                f"expected condition shape (B, {self.condition_dim}), got "
                f"{tuple(condition.shape)}"
            )
        batch_size = int(condition.shape[0])
        if drop_mask is None:
            probability = self.condition_dropout_probability if self.training else 0.0
            drop_mask = torch.rand(batch_size, device=condition.device) < probability
        else:
            if drop_mask.ndim != 1 or tuple(drop_mask.shape) != (batch_size,):
                raise ValueError(
                    f"condition_drop_mask must have shape ({batch_size},), got "
                    f"{tuple(drop_mask.shape)}"
                )
            if drop_mask.dtype is not torch.bool:
                raise TypeError("condition_drop_mask must have dtype torch.bool")
            drop_mask = drop_mask.to(device=condition.device)
        null = self.null_condition.to(condition).unsqueeze(0).expand_as(condition)
        effective = torch.where(drop_mask.unsqueeze(-1), null, condition)
        return effective, drop_mask

    @staticmethod
    def _normalize_time(time: torch.Tensor, batch_size: int) -> torch.Tensor:
        if time.ndim == 2 and tuple(time.shape) == (batch_size, 1):
            time = time[:, 0]
        if time.ndim != 1 or tuple(time.shape) != (batch_size,):
            raise ValueError(
                f"time must have shape ({batch_size},) or ({batch_size}, 1), "
                f"got {tuple(time.shape)}"
            )
        return time

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        *,
        condition_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.horizon, self.input_dim)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"expected value shape (B, {self.horizon}, {self.input_dim}), "
                f"got {tuple(value.shape)}"
            )
        batch_size = int(value.shape[0])
        time = self._normalize_time(time, batch_size)
        effective_condition, _ = self.apply_condition_dropout(
            condition, drop_mask=condition_drop_mask
        )
        time_features = _sinusoidal_time_embedding(
            time,
            self.time_embedding_dim,
            time_scale=self.time_scale,
        )
        context = self.condition_projection(
            self.condition_norm(effective_condition)
        ) + self.time_projection(time_features).to(value.dtype)
        context = F.silu(self.context_norm(context))
        base_modulation = self.adaln_modulation(context).reshape(
            batch_size, 6, self.hidden_dim
        )

        hidden = self.input_projection(value) + self.position_embedding.to(value)
        for index, block in enumerate(self.blocks):
            modulation = base_modulation * self.block_modulation_scale[index].to(
                base_modulation
            ) + self.block_modulation_bias[index].to(base_modulation)
            hidden = block(hidden, modulation)
        final_shift, final_scale = base_modulation[:, :2].unbind(dim=1)
        hidden = _modulate(self.output_norm(hidden), final_shift, final_scale)
        return self.output_projection(hidden)
