"""Flow-matching denoisers for ARC shape and per-waypoint velocity streams.

The public action contract remains the wide ``(B, M, 28)`` ARC token used by
the bimanual tokenizer. Internally, these models view its two 14-D halves as
two length-M token streams. Keeping the split inside the denoiser lets the
existing ARC data transform, flow stages, evaluator, and detokenizer stay
unchanged.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from egomimic.models.denoising_nets import CrossBlock, CrossTransformer, posemb_sincos

logger = logging.getLogger(__name__)


def _stream_pair(value: torch.Tensor, *, stream_dim: int, horizon: int) -> torch.Tensor:
    if value.ndim != 3 or tuple(value.shape[1:]) != (horizon, 2 * stream_dim):
        raise ValueError(
            "ARC flow input must have shape "
            f"(B, {horizon}, {2 * stream_dim}), got {tuple(value.shape)}"
        )
    shape = value[..., :stream_dim]
    velocity = value[..., stream_dim:]
    return torch.cat((shape, velocity), dim=1)


def _pack_stream_pair(
    value: torch.Tensor, *, stream_dim: int, horizon: int
) -> torch.Tensor:
    expected = (horizon * 2, stream_dim)
    if value.ndim != 3 or tuple(value.shape[1:]) != expected:
        raise ValueError(
            f"ARC stream output must have shape (B, {expected[0]}, {expected[1]}), "
            f"got {tuple(value.shape)}"
        )
    shape = value[:, :horizon]
    velocity = value[:, horizon:]
    return torch.cat((shape, velocity), dim=-1)


def _directional_mask(horizon: int, *, device: torch.device) -> torch.Tensor:
    """Boolean MHA mask: shape queries cannot read velocity keys."""
    length = 2 * horizon
    mask = torch.zeros((length, length), dtype=torch.bool, device=device)
    mask[:horizon, horizon:] = True
    return mask


def _time_features(timesteps: torch.Tensor, hidden_dim: int, tokens: torch.Tensor):
    if timesteps.ndim == 0:
        timesteps = timesteps.expand(tokens.shape[0])
    if timesteps.ndim != 1 or int(timesteps.shape[0]) != int(tokens.shape[0]):
        raise ValueError(
            f"time must be shape ({tokens.shape[0]},), got {tuple(timesteps.shape)}"
        )
    return (
        posemb_sincos(timesteps, hidden_dim, min_period=4e-3, max_period=4.0)
        .unsqueeze(1)
        .to(device=tokens.device, dtype=tokens.dtype)
    )


class _ConditionedBlocks(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        cond_dim: int,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        mlp_layers: int,
        mlp_ratio: float,
    ):
        super().__init__()
        if int(depth) <= 0:
            raise ValueError("transformer depth must be positive")
        self.layers = nn.ModuleList(
            [
                CrossBlock(
                    cond_dim=cond_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    mlp_layers=mlp_layers,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, tokens, condition, attention_mask=None):
        for layer in self.layers:
            tokens = layer(tokens, condition, self_attn_mask=attention_mask)
        return tokens


class _StreamEncoder(nn.Module):
    """One parameter-separated stream encoder with HPT cross-attention."""

    def __init__(
        self,
        *,
        stream_dim: int,
        horizon: int,
        cond_dim: int,
        hidden_dim: int,
        depth: int,
        n_heads: int,
        dropout: float,
        mlp_layers: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(stream_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.stream_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.stream_embedding, std=0.02)
        self.blocks = _ConditionedBlocks(
            depth=depth,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            mlp_layers=mlp_layers,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, sample, timesteps, condition):
        tokens = self.input_proj(sample)
        tokens = tokens + self.position_embedding + self.stream_embedding
        tokens = tokens + _time_features(timesteps, self.hidden_dim, tokens)
        return self.blocks(tokens, condition)


class ArcStreamFlowDenoiser(nn.Module):
    """One of four HPT flow heads that factor ARC shape and velocity tokens.

    Modes:
      ``velocity_decoder``: independent shape and velocity flow heads; a small
      decoder maps HPT's shared pooled trunk condition for the velocity head.
      ``shared_directional``: one shared transformer stack over both streams,
      with velocity allowed to attend to shape and shape blocked from velocity.
      ``mot_parallel``: independent shape and velocity transformer branches,
      followed by a shared bidirectional flow-matching head.
      ``mot_directional``: the same MoT branches, with one-way shape-to-velocity
      attention in the shared flow-matching head.

    ``act_dim`` is the external wide-token width (28); ``stream_dim`` is the
    width of each stream (14). All modes return the external wide-token layout.
    """

    MODES = (
        "velocity_decoder",
        "shared_directional",
        "mot_parallel",
        "mot_directional",
    )

    def __init__(
        self,
        nblocks: int,
        cond_dim: int,
        hidden_dim: int,
        act_dim: int,
        act_seq: int,
        n_heads: int,
        dropout: float,
        mlp_layers: int,
        mlp_ratio: float,
        architecture_mode: str,
        stream_dim: int,
        shape_blocks: int = 6,
        velocity_blocks: int = 4,
        fusion_blocks: int = 4,
        decoder_hidden_dim: int | None = None,
        time_conditioning: str = "additive",
    ):
        super().__init__()
        if architecture_mode not in self.MODES:
            raise ValueError(
                f"architecture_mode must be one of {self.MODES}, "
                f"got {architecture_mode!r}"
            )
        self.architecture_mode = architecture_mode
        self.stream_dim = int(stream_dim)
        self.horizon = int(act_seq)
        self.hidden_dim = int(hidden_dim)
        if int(act_dim) != 2 * self.stream_dim:
            raise ValueError(
                f"act_dim must be twice stream_dim ({2 * self.stream_dim}), "
                f"got {act_dim}"
            )
        if self.horizon <= 0 or self.stream_dim <= 0:
            raise ValueError("act_seq and stream_dim must be positive")
        if self.hidden_dim % int(n_heads):
            raise ValueError("hidden_dim must be divisible by n_heads")
        if time_conditioning != "additive":
            raise ValueError(
                "ARC stream flow denoisers require additive time conditioning"
            )

        shared_args = dict(
            cond_dim=int(cond_dim),
            hidden_dim=self.hidden_dim,
            n_heads=int(n_heads),
            dropout=float(dropout),
            mlp_layers=int(mlp_layers),
            mlp_ratio=float(mlp_ratio),
        )

        if architecture_mode == "velocity_decoder":
            self.shape_model = CrossTransformer(
                nblocks=int(shape_blocks),
                cond_dim=int(cond_dim),
                hidden_dim=self.hidden_dim,
                act_dim=self.stream_dim,
                act_seq=self.horizon,
                n_heads=int(n_heads),
                dropout=float(dropout),
                mlp_layers=int(mlp_layers),
                mlp_ratio=float(mlp_ratio),
                time_conditioning="additive",
            )
            self.velocity_condition_decoder = nn.Sequential(
                nn.LayerNorm(int(cond_dim)),
                nn.Linear(int(cond_dim), int(decoder_hidden_dim or cond_dim)),
                nn.GELU(),
                nn.Linear(int(decoder_hidden_dim or cond_dim), int(cond_dim)),
            )
            self.velocity_model = CrossTransformer(
                nblocks=int(velocity_blocks),
                cond_dim=int(cond_dim),
                hidden_dim=self.hidden_dim,
                act_dim=self.stream_dim,
                act_seq=self.horizon,
                n_heads=int(n_heads),
                dropout=float(dropout),
                mlp_layers=int(mlp_layers),
                mlp_ratio=float(mlp_ratio),
                time_conditioning="additive",
            )
        elif architecture_mode == "shared_directional":
            self.shared_input_proj = nn.Linear(self.stream_dim, self.hidden_dim)
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.horizon, self.hidden_dim)
            )
            self.stream_embedding = nn.Parameter(torch.zeros(1, 2, self.hidden_dim))
            nn.init.normal_(self.position_embedding, std=0.02)
            nn.init.normal_(self.stream_embedding, std=0.02)
            self.shared_blocks = _ConditionedBlocks(depth=int(nblocks), **shared_args)
            self.shared_output_proj = nn.Linear(self.hidden_dim, self.stream_dim)
        else:
            self.shape_encoder = _StreamEncoder(
                stream_dim=self.stream_dim,
                horizon=self.horizon,
                depth=int(shape_blocks),
                **shared_args,
            )
            self.velocity_encoder = _StreamEncoder(
                stream_dim=self.stream_dim,
                horizon=self.horizon,
                depth=int(velocity_blocks),
                **shared_args,
            )
            self.fusion_stream_embedding = nn.Parameter(
                torch.zeros(1, 2, self.hidden_dim)
            )
            nn.init.normal_(self.fusion_stream_embedding, std=0.02)
            self.fusion_blocks = _ConditionedBlocks(
                depth=int(fusion_blocks), **shared_args
            )
            self.fusion_output_proj = nn.Linear(self.hidden_dim, self.stream_dim)

        logger.info(
            "ARC stream flow denoiser mode=%s parameters=%d",
            self.architecture_mode,
            sum(parameter.numel() for parameter in self.parameters()),
        )

    def _condition(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if condition.ndim != 3:
            raise ValueError(
                "ARC stream condition must be (B, C) or (B, N, C), "
                f"got {tuple(condition.shape)}"
            )
        return condition

    def forward(self, sample, timesteps, condition, *args, **kwargs):
        streams = _stream_pair(sample, stream_dim=self.stream_dim, horizon=self.horizon)
        shape_sample = streams[:, : self.horizon]
        velocity_sample = streams[:, self.horizon :]
        condition = self._condition(condition)

        if self.architecture_mode == "velocity_decoder":
            shape_velocity = self.shape_model(shape_sample, timesteps, condition)
            velocity_condition = condition + self.velocity_condition_decoder(condition)
            waypoint_velocity = self.velocity_model(
                velocity_sample, timesteps, velocity_condition
            )
            return torch.cat((shape_velocity, waypoint_velocity), dim=-1)

        if self.architecture_mode == "shared_directional":
            tokens = self.shared_input_proj(streams)
            position = torch.cat(
                (self.position_embedding, self.position_embedding), dim=1
            )
            stream_identity = torch.cat(
                (
                    self.stream_embedding[:, 0:1].expand(-1, self.horizon, -1),
                    self.stream_embedding[:, 1:2].expand(-1, self.horizon, -1),
                ),
                dim=1,
            )
            tokens = tokens + position + stream_identity
            tokens = tokens + _time_features(timesteps, self.hidden_dim, tokens)
            tokens = self.shared_blocks(
                tokens,
                condition,
                attention_mask=_directional_mask(self.horizon, device=tokens.device),
            )
            return _pack_stream_pair(
                self.shared_output_proj(tokens),
                stream_dim=self.stream_dim,
                horizon=self.horizon,
            )

        shape_features = self.shape_encoder(shape_sample, timesteps, condition)
        velocity_features = self.velocity_encoder(velocity_sample, timesteps, condition)
        tokens = torch.cat((shape_features, velocity_features), dim=1)
        stream_identity = torch.cat(
            (
                self.fusion_stream_embedding[:, 0:1].expand(-1, self.horizon, -1),
                self.fusion_stream_embedding[:, 1:2].expand(-1, self.horizon, -1),
            ),
            dim=1,
        )
        tokens = tokens + stream_identity
        attention_mask = None
        if self.architecture_mode == "mot_directional":
            attention_mask = _directional_mask(self.horizon, device=tokens.device)
        tokens = self.fusion_blocks(tokens, condition, attention_mask=attention_mask)
        return _pack_stream_pair(
            self.fusion_output_proj(tokens),
            stream_dim=self.stream_dim,
            horizon=self.horizon,
        )
