"""Small context-free sequence codecs for latent generative models.

The modules in this file deliberately accept only a sequence tensor.  In
particular, they have no observation, task, or routing input, so conditional
generation cannot bypass the latent sequence through either codec.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SequenceSelfAttention(nn.Module):
    """Ordinary multi-head self-attention with forward-AD-safe operations."""

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
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, content)
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_dim
        )
        return self.output(attended)


class _PreNormSequenceBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = _SequenceSelfAttention(hidden_dim, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attention_dropout(
            self.attention(self.attention_norm(value))
        )
        return value + self.feedforward(self.feedforward_norm(value))


class _ContextFreeSequenceCodec(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        horizon: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "horizon": horizon,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "num_heads": num_heads,
            "feedforward_dim": feedforward_dim,
        }
        invalid = [name for name, value in dimensions.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"codec dimensions must be positive: {invalid}")
        if int(hidden_dim) % int(num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.dropout = float(dropout)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.horizon, self.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            _PreNormSequenceBlock(
                self.hidden_dim,
                self.num_heads,
                self.feedforward_dim,
                self.dropout,
            )
            for _ in range(self.depth)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        expected = (self.horizon, self.input_dim)
        if content.ndim != 3 or tuple(content.shape[1:]) != expected:
            raise ValueError(
                f"expected sequence shape (B, {self.horizon}, {self.input_dim}), "
                f"got {tuple(content.shape)}"
            )
        hidden = self.input_projection(content) + self.position_embedding.to(content)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_norm(hidden))


class ContextFreeSequenceEncoder(_ContextFreeSequenceCodec):
    """Encode a fixed-length content sequence without observation context."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        horizon: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        self.latent_dim = int(latent_dim)
        super().__init__(
            input_dim=input_dim,
            output_dim=latent_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )


class ContextFreeSequenceDecoder(_ContextFreeSequenceCodec):
    """Decode a fixed-length latent sequence without observation context."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        horizon: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        self.latent_dim = int(latent_dim)
        super().__init__(
            input_dim=latent_dim,
            output_dim=output_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
