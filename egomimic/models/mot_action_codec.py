"""MoT action codecs: shared latent-token weights and private action-token weights.

Attention spans latent and action tokens within each example. Embodiments never
share an attention batch; they share actual Parameter objects on the latent stream.
Encoder and decoder banks are independent. This is an architecture candidate,
not a checkpoint-compatible replacement for released UNITE codecs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as decoder_checkpoint

from egomimic.models.unite_action_decoder import _sincos_positions


class _TokenParameters(nn.Module):
    def __init__(self, width, mlp_ratio):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(width * mlp_ratio), width),
        )


class _ContextParameters(nn.Module):
    """Terminal tokens supply keys/values without an unused output update."""

    def __init__(self, width):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.kv = nn.Linear(width, 2 * width)


class MoTActionBlock(nn.Module):
    """Two parameter types with one global, noncausal attention operation."""

    def __init__(self, latent_parameters, *, width, heads, mlp_ratio, readout="both"):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("MoT width must be positive and divisible by heads")
        self.latent = latent_parameters
        if readout not in {"both", "latent", "action"}:
            raise ValueError("Unknown MoT block readout")
        if isinstance(latent_parameters, _ContextParameters) != (readout == "action"):
            raise ValueError("MoT latent parameter bank differs from block readout")
        self.readout = readout
        self.action = (
            _ContextParameters(width)
            if readout == "latent"
            else _TokenParameters(width, mlp_ratio)
        )
        self.heads = heads

    def forward(self, latent, action):
        if latent.shape[0] != action.shape[0] or latent.shape[-1] != action.shape[-1]:
            raise ValueError("MoT token streams need matching batch and width")
        streams = (latent, action)
        parameters = (self.latent, self.action)
        batch, _, width = latent.shape
        queries, keys, values, lengths = [], [], [], []
        for x, p in zip(streams, parameters):
            normalized = p.attention_norm(x)
            if isinstance(p, _ContextParameters):
                key, value = p.kv(normalized).chunk(2, dim=-1)
                lengths.append(0)
            else:
                query, key, value = p.qkv(normalized).chunk(3, dim=-1)
                queries.append(query)
                lengths.append(x.shape[1])
            keys.append(key)
            values.append(value)

        def heads(value):
            return value.reshape(batch, -1, self.heads, width // self.heads).transpose(
                1, 2
            )

        query, key, value = map(
            heads, (torch.cat(queries, 1), torch.cat(keys, 1), torch.cat(values, 1))
        )
        # Math attention supports the forward-over-reverse derivatives needed
        # by action-velocity JVP losses; no fused-attention assumption here.
        weights = (
            (query @ key.transpose(-2, -1)) / (width // self.heads) ** 0.5
        ).softmax(-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, sum(lengths), width)
        outputs = []
        for x, update, p in zip(
            streams,
            attended.split(lengths, dim=1),
            parameters,
        ):
            if isinstance(p, _ContextParameters):
                outputs.append(x)
                continue
            x = x + p.output(update)
            outputs.append(x + p.ffn(p.ffn_norm(x)))
        return tuple(outputs)


class _LatentParameters(nn.Module):
    def __init__(self, latent_dim, tokens, width, depth, mlp_ratio, role):
        super().__init__()
        self.proj_u = nn.Linear(latent_dim, width)
        self.proj_d = nn.Linear(width, latent_dim) if role == "encoder" else None
        self.norm = nn.LayerNorm(width) if role == "encoder" else None
        self.register_buffer("positions", _sincos_positions(tokens, width))
        self.layers = nn.ModuleList(
            _ContextParameters(width)
            if role == "decoder" and index == depth - 1
            else _TokenParameters(width, mlp_ratio)
            for index in range(depth)
        )


class MoTActionCodec(nn.Module):
    """One embodiment view; its latent parameters are shared by identity."""

    def __init__(
        self,
        shared,
        *,
        role,
        action_dim,
        horizon,
        tokens,
        latent_dim,
        width,
        heads,
        mlp_ratio,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.shared = shared
        self.role = role
        self.action_dim = action_dim
        self.horizon = horizon
        self.tokens = tokens
        self.latent_dim = latent_dim
        self.content_projection = (
            nn.Linear(action_dim, width) if role == "encoder" else None
        )
        self.action_output = nn.Linear(width, action_dim) if role == "decoder" else None
        self.query = (
            nn.Parameter(torch.randn(1, 1, width) * 0.02) if role == "decoder" else None
        )
        self.register_buffer("action_positions", _sincos_positions(horizon, width))
        self.blocks = nn.ModuleList(
            MoTActionBlock(
                layer,
                width=width,
                heads=heads,
                mlp_ratio=mlp_ratio,
                readout=("latent" if role == "encoder" else "action")
                if index == len(shared.layers) - 1
                else "both",
            )
            for index, layer in enumerate(shared.layers)
        )
        self.action_norm = nn.LayerNorm(width) if role == "decoder" else None

    def canonicalize_diagnostic_block_output(self, output, block_index):
        del block_index
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("MoT diagnostic block must return two token streams")
        return output[0]

    def forward(self, value):
        expected = (
            (self.horizon, self.action_dim)
            if self.role == "encoder"
            else (self.tokens, self.latent_dim)
        )
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"MoT {self.role} expected (B,{expected[0]},{expected[1]})"
            )
        if self.role == "encoder":
            registers = torch.randn(
                value.shape[0],
                self.tokens,
                self.latent_dim,
                device=value.device,
                dtype=value.dtype,
            )
            latent = self.shared.proj_u(registers)
            action = self.content_projection(value)
        else:
            latent = self.shared.proj_u(value)
            action = self.query.expand(value.shape[0], self.horizon, -1)
        latent = latent + self.shared.positions.to(latent)
        action = action + self.action_positions.to(action)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                if self.role == "encoder":
                    # Use the explicitly audited shared checkpoint dispatcher.
                    from egomimic.models import unite_dit

                    latent, action = unite_dit.checkpoint(
                        block, latent, action, use_reentrant=False
                    )
                else:
                    latent, action = decoder_checkpoint(
                        block, latent, action, use_reentrant=False
                    )
            else:
                latent, action = block(latent, action)
        if self.role == "encoder":
            return self.shared.proj_d(self.shared.norm(latent))
        return self.action_output(self.action_norm(action))


def make_mot_codecs(
    *,
    action_dims,
    role,
    horizon,
    tokens,
    latent_dim,
    width,
    depth,
    heads,
    mlp_ratio=4.0,
    gradient_checkpointing=False,
):
    """Build one canonical shared bank and the routed embodiment module views.

    Invoke separately for encoders and decoders. PyTorch named_parameters
    deduplicates aliases; checkpoint/EMA consumers must preserve these aliases.
    """
    if role not in {"encoder", "decoder"}:
        raise ValueError("MoT role must be encoder or decoder")
    if len(action_dims) < 2 or any(
        not isinstance(k, str) or not k or "." in k for k in action_dims
    ):
        raise ValueError("MoT requires at least two valid embodiment keys")
    if any(
        type(v) is not int or v <= 0
        for v in [
            *action_dims.values(),
            horizon,
            tokens,
            latent_dim,
            width,
            depth,
            heads,
        ]
    ):
        raise ValueError("MoT dimensions must be positive integers")
    if width % heads or width % 2 or mlp_ratio <= 0:
        raise ValueError(
            "MoT requires even width divisible by heads and positive mlp ratio"
        )
    shared = _LatentParameters(latent_dim, tokens, width, depth, mlp_ratio, role)
    return nn.ModuleDict(
        {
            name: MoTActionCodec(
                shared,
                role=role,
                action_dim=dim,
                horizon=horizon,
                tokens=tokens,
                latent_dim=latent_dim,
                width=width,
                heads=heads,
                mlp_ratio=mlp_ratio,
                gradient_checkpointing=gradient_checkpointing,
            )
            for name, dim in action_dims.items()
        }
    )
