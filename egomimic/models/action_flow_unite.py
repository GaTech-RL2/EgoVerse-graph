"""UNITE backbone adapters for the Action Flow objective.

These modules keep the Action Flow stage interfaces while using the released
UNITE tokenizer/denoiser machinery.  They deliberately remain separate module
instances so reconstruction and latent flow do not share Transformer weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UniteActionFlowContentEncoder(nn.Module):
    """Tokenize a clean action chunk into compact Gaussian-query registers."""

    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        action_horizon: int,
        latent_dim: int,
        num_latent_tokens: int,
        condition_dim: int,
        tokenization_time_max: float = 0.01,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.latent_dim = int(latent_dim)
        self.num_latent_tokens = int(num_latent_tokens)
        self.condition_dim = int(condition_dim)
        self.tokenization_time_max = float(tokenization_time_max)
        if min(
            self.input_dim,
            self.action_horizon,
            self.latent_dim,
            self.num_latent_tokens,
            self.condition_dim,
        ) <= 0:
            raise ValueError("UNITE Action Flow encoder dimensions must be positive")
        if not 0.0 <= self.tokenization_time_max <= 1.0:
            raise ValueError("tokenization_time_max must be in [0, 1]")
        self.content_projection = nn.Linear(self.input_dim, self.condition_dim)
        self.null_condition_input = nn.Parameter(
            torch.empty(self.condition_dim).normal_(std=0.02)
        )
        self.condition_projection = nn.Linear(self.condition_dim, self.condition_dim)
        self.domain_embedding = nn.Parameter(
            torch.empty(self.condition_dim).normal_(std=0.02)
        )
        self.output_norm = nn.LayerNorm(self.latent_dim)

    @property
    def blocks(self) -> nn.ModuleList:
        """Expose released backbone blocks to maintained diagnostics."""

        return self.backbone.blocks

    def canonicalize_diagnostic_block_output(
        self, output: torch.Tensor, block_index: int
    ) -> torch.Tensor:
        """Return aligned register activations across in-context insertion."""

        start = self.backbone.in_context_start
        if start is not None and block_index >= start:
            output = output[:, self.backbone.in_context_len :]
        return output[:, : self.num_latent_tokens]

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        expected = (self.action_horizon, self.input_dim)
        if content.ndim != 3 or tuple(content.shape[1:]) != expected:
            raise ValueError(
                "UNITE Action Flow encoder expected content shape "
                f"(B, {expected[0]}, {expected[1]}), got {tuple(content.shape)}"
            )
        batch_size = int(content.shape[0])
        registers = torch.randn(
            batch_size,
            self.num_latent_tokens,
            self.latent_dim,
            device=content.device,
            dtype=content.dtype,
        )
        action_tokens = self.content_projection(content)
        condition = self.condition_projection(
            self.null_condition_input.to(content)
        ) + self.domain_embedding.to(content)
        condition = condition.reshape(1, 1, -1).expand(batch_size, -1, -1)
        time = (
            torch.rand(batch_size, device=content.device, dtype=torch.float32)
            * self.tokenization_time_max
        )
        encoded = self.backbone(
            registers,
            time,
            condition=condition,
            content_tokens=action_tokens,
        )
        return self.output_norm(encoded)


class UniteActionFlowVelocityField(nn.Module):
    """Predict Action Flow velocity with the released UNITE DiT backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        output_dim: int,
        horizon: int,
        condition_dim: int,
        condition_dropout_probability: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.condition_dim = int(condition_dim)
        self.condition_dropout_probability = float(condition_dropout_probability)
        if self.input_dim != self.output_dim:
            raise ValueError("Action Flow velocity input/output dimensions must match")
        if not 0.0 <= self.condition_dropout_probability <= 1.0:
            raise ValueError("condition_dropout_probability must be in [0, 1]")
        self.condition_projection = nn.Linear(self.condition_dim, self.condition_dim)
        self.null_condition = nn.Parameter(
            torch.empty(self.condition_dim).normal_(std=0.02)
        )
        self.domain_embedding = nn.Parameter(
            torch.empty(self.condition_dim).normal_(std=0.02)
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    @property
    def blocks(self) -> nn.ModuleList:
        """Expose released backbone blocks to maintained diagnostics."""

        return self.backbone.blocks

    def canonicalize_diagnostic_block_output(
        self, output: torch.Tensor, block_index: int
    ) -> torch.Tensor:
        """Return aligned register activations across in-context insertion."""

        start = self.backbone.in_context_start
        if start is not None and block_index >= start:
            output = output[:, self.backbone.in_context_len :]
        return output[:, : self.horizon]

    def apply_condition_dropout(
        self,
        condition: torch.Tensor,
        *,
        drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(condition.shape[0])
        if drop_mask is None:
            if self.training and self.condition_dropout_probability > 0.0:
                drop_mask = (
                    torch.rand(batch_size, device=condition.device)
                    < self.condition_dropout_probability
                )
            else:
                drop_mask = torch.zeros(
                    batch_size, dtype=torch.bool, device=condition.device
                )
        if drop_mask.dtype != torch.bool or tuple(drop_mask.shape) != (batch_size,):
            raise ValueError("condition drop mask has the wrong shape or dtype")
        shape = (batch_size,) + (1,) * (condition.ndim - 2) + (self.condition_dim,)
        null = self.null_condition.to(condition).reshape(
            *((1,) * (condition.ndim - 1)), self.condition_dim
        ).expand(shape)
        mask = drop_mask.reshape(batch_size, *((1,) * (condition.ndim - 1)))
        return torch.where(mask, null, condition), drop_mask

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        condition_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.horizon, self.input_dim)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"UNITE velocity field expected (B, {expected[0]}, {expected[1]})"
            )
        if (
            condition.ndim not in {2, 3}
            or int(condition.shape[-1]) != self.condition_dim
        ):
            raise ValueError("UNITE velocity condition has the wrong shape")
        effective, _ = self.apply_condition_dropout(
            condition, drop_mask=condition_drop_mask
        )
        projected = self.condition_projection(effective)
        projected = projected + self.domain_embedding.to(projected).reshape(
            *((1,) * (projected.ndim - 1)), -1
        )
        return self.output_norm(self.backbone(value, time, condition=projected))
