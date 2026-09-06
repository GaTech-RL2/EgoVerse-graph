"""General AdaLN Transformer over a fixed sequence of state tokens."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from egomimic.models.unite_dit import (
    _AdaLNBlock,
    _AdaLNFinalLayer,
    _GaussianFourierEmbedding,
    _sincos_positions,
)


class AdaLNBackbone(nn.Module):
    """Map state tokens, continuous time, and one condition vector to tokens.

    State tokens enter through the token projection and retain their ordered
    geometry through positional embeddings and RoPE. Time and the external
    condition are embedded separately, then combined only as the global AdaLN
    control signal. The module has no flow, velocity, sampling, or application
    semantics.
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
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        block_norm: bool = True,
        time_fourier_dim: int = 256,
        trainable_position_embeddings: bool = True,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if (
            min(
                self.input_dim,
                self.output_dim,
                self.horizon,
                self.condition_dim,
                self.hidden_dim,
                self.depth,
                self.num_heads,
            )
            <= 0
        ):
            raise ValueError("AdaLN backbone dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.condition_projection = nn.Linear(self.condition_dim, self.hidden_dim)
        self.time_embedder = _GaussianFourierEmbedding(
            self.hidden_dim, int(time_fourier_dim)
        )
        positions = _sincos_positions(self.horizon, self.hidden_dim)
        if trainable_position_embeddings:
            self.position_embedding = nn.Parameter(positions)
        else:
            self.register_buffer("position_embedding", positions, persistent=True)
        self.blocks = nn.ModuleList(
            _AdaLNBlock(
                self.hidden_dim,
                self.num_heads,
                float(mlp_ratio),
                qk_norm=bool(qk_norm),
                use_rope=bool(use_rope),
                use_rmsnorm=bool(use_rmsnorm),
                use_swiglu=bool(use_swiglu),
                block_norm=bool(block_norm),
                dropout=float(dropout),
            )
            for _ in range(self.depth)
        )
        self.final_layer = _AdaLNFinalLayer(self.hidden_dim, bool(use_rmsnorm))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.horizon, self.input_dim)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"state tokens must have shape (B, {self.horizon}, "
                f"{self.input_dim}), got {tuple(value.shape)}"
            )
        batch_size = int(value.shape[0])
        if time.ndim == 0:
            time = time.expand(batch_size)
        if time.ndim != 1 or int(time.shape[0]) != batch_size:
            raise ValueError(
                f"time must have shape ({batch_size},), got {tuple(time.shape)}"
            )
        if condition.ndim != 2 or tuple(condition.shape) != (
            batch_size,
            self.condition_dim,
        ):
            raise ValueError(
                f"condition must have shape (B, {self.condition_dim}), got "
                f"{tuple(condition.shape)}"
            )

        hidden = self.input_projection(value) + self.position_embedding.to(value)
        conditioning = self.time_embedder(time) + self.condition_projection(condition)
        conditioning = conditioning.to(hidden.dtype)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(
                    block,
                    hidden,
                    conditioning,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden = block(hidden, conditioning, None)
        return self.output_projection(self.final_layer(hidden, conditioning))

    def forward_with_activations(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        """Run the ordinary forward path and retain real block outputs."""

        activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        handles = []
        for index, block in enumerate(self.blocks):
            name = f"block_{index:02d}"

            def capture(_module, _inputs, output, *, key=name):
                activations[key] = output

            handles.append(block.register_forward_hook(capture))
        try:
            output = self(value, time, condition)
        finally:
            for handle in handles:
                handle.remove()
        if len(activations) != len(self.blocks):
            raise RuntimeError("AdaLN diagnostic activation capture is incomplete")
        return output, activations
