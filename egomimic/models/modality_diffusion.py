"""Modality-specific diffusion Transformers with joint action attention.

Applies the Mixture-of-Transformers design (arXiv:2411.04996) to the released
OAT diffusion decoder: each modality owns its Q/K/V/output projections, FFN,
and norms. Projected action keys/values are concatenated before attention.
Observation/time conditioning stays in the common, action-free memory.
"""

import torch
from torch import nn
from torch.nn import functional as F

from egomimic.models.oat.diffusion import GraphDiffusionTransformer


class ModalityDecoderLayer(nn.Module):
    """Pre-norm decoder experts sharing one attention operation across modalities.

    Each expert also owns its observation cross-attention projections. The
    underlying TransformerDecoderLayer supplies the same norms, GELU, dropout,
    and bias conventions as the OAT decoder. No attention parameters are tied.
    """

    def __init__(self, modalities, *, width, heads, feedforward, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.experts = nn.ModuleDict(
            {
                name: nn.TransformerDecoderLayer(
                    width,
                    heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for name in modalities
            }
        )

    def forward(self, streams, memory, *, mask, memory_mask):
        # Project each modality with its own expert before mixing keys/values.
        projected = [
            F.linear(
                expert.norm1(stream),
                expert.self_attn.in_proj_weight,
                expert.self_attn.in_proj_bias,
            )
            for stream, expert in zip(streams, self.experts.values())
        ]
        q, k, v = torch.cat(projected, dim=1).chunk(3, dim=-1)
        batch, tokens, width = q.shape

        def heads(value):
            return value.reshape(
                batch, tokens, self.heads, width // self.heads
            ).transpose(1, 2)

        attended = (
            F.scaled_dot_product_attention(
                heads(q),
                heads(k),
                heads(v),
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            .transpose(1, 2)
            .reshape(batch, tokens, width)
        )
        attended = attended.split([stream.shape[1] for stream in streams], dim=1)
        outputs = []
        for stream, attention, expert in zip(streams, attended, self.experts.values()):
            stream = stream + expert.dropout1(expert.self_attn.out_proj(attention))
            cross = expert.multihead_attn(
                expert.norm2(stream),
                memory,
                memory,
                attn_mask=memory_mask,
                need_weights=False,
            )[0]
            stream = stream + expert.dropout2(cross)
            feedforward = expert.linear2(
                expert.dropout(expert.activation(expert.linear1(expert.norm3(stream))))
            )
            outputs.append(stream + expert.dropout3(feedforward))
        return outputs


class ModalityDiffusionTransformer(GraphDiffusionTransformer):
    """One token per modality per waypoint, returning the original action layout.

    ``modality_columns`` must partition every input/output channel exactly once.
    ``blocked_attention`` contains [query_modality, key_modality] pairs. All
    modalities otherwise attend across streams to the same or earlier waypoint;
    the mask is applied in every layer and every diffusion denoising step.
    """

    def __init__(
        self, *, modality_columns, blocked_attention=(), dim_feedforward=None, **kwargs
    ):
        columns = {
            str(name): tuple(indices) for name, indices in modality_columns.items()
        }
        flat = tuple(index for indices in columns.values() for index in indices)
        channels = kwargs.get("input_dim")
        if (
            not columns
            or any(
                not name or "." in name or not indices
                for name, indices in columns.items()
            )
            or any(type(index) is not int for index in flat)
            or sorted(flat) != list(range(channels or 0))
            or channels != kwargs.get("output_dim")
        ):
            raise ValueError(
                "Modalities must partition all input/output channels exactly once"
            )
        edges = tuple(tuple(edge) for edge in blocked_attention)
        if any(
            len(edge) != 2
            or any(name not in columns for name in edge)
            or edge[0] == edge[1]
            for edge in edges
        ):
            raise ValueError(
                "Blocked attention must name distinct query/key modalities"
            )
        layers = int(kwargs.setdefault("n_layer", 4))
        if layers < 1 or not (
            kwargs.get("causal_attn", True)
            and kwargs.get("time_as_cond", True)
            and kwargs.get("obs_as_cond", True)
            and kwargs.get("cond_dim", 0) > 0
        ):
            raise ValueError(
                "MoT requires positive depth and causal observation/time conditioning"
            )
        super().__init__(**kwargs)
        width = self.input_emb.out_features
        heads = self.decoder.layers[0].self_attn.num_heads
        dropout = self.decoder.layers[0].dropout.p
        feedforward = int(dim_feedforward if dim_feedforward is not None else 4 * width)
        if feedforward < 1:
            raise ValueError("MoT feedforward dimension must be positive")
        self.modality_columns = columns
        self.blocked_attention = edges
        self.modalities = tuple(columns)
        self.input_emb = nn.ModuleDict(
            {name: nn.Linear(len(cols), width) for name, cols in columns.items()}
        )
        self.head = nn.ModuleDict(
            {name: nn.Linear(width, len(cols)) for name, cols in columns.items()}
        )
        self.ln_f = nn.ModuleDict({name: nn.LayerNorm(width) for name in columns})
        self.stream_emb = nn.Embedding(len(columns), width)
        self.decoder = nn.ModuleList(
            ModalityDecoderLayer(
                self.modalities,
                width=width,
                heads=heads,
                feedforward=feedforward,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        for module in (
            self.input_emb,
            self.head,
            self.ln_f,
            self.stream_emb,
            self.decoder,
        ):
            module.apply(self._init_weights)
        for name, cols in columns.items():
            self.register_buffer(
                f"{name}_columns", torch.tensor(cols), persistent=False
            )
        self.register_buffer(
            "output_order", torch.argsort(torch.tensor(flat)), persistent=False
        )
        self.channels = channels
        waypoint = torch.arange(self.horizon).repeat(len(columns))
        blocked = waypoint[None, :] > waypoint[:, None]
        for query, key in edges:
            q, k = self.modalities.index(query), self.modalities.index(key)
            blocked[
                q * self.horizon : (q + 1) * self.horizon,
                k * self.horizon : (k + 1) * self.horizon,
            ] = True
        self.mask = torch.zeros_like(blocked, dtype=torch.float32).masked_fill(
            blocked, float("-inf")
        )
        # Every stream uses the original waypoint-aligned observation mask.
        # The memory is never updated from action tokens, so a forbidden path
        # cannot leak through a later layer's observation cross-attention.

    def _init_weights(self, module):
        if isinstance(module, (ModalityDecoderLayer, nn.ModuleDict)):
            return
        super()._init_weights(module)

    def forward(self, sample, timestep, global_cond):
        if sample.ndim != 3 or tuple(sample.shape[1:]) != (self.horizon, self.channels):
            raise ValueError(
                f"Expected action input (B, {self.horizon}, {self.channels})"
            )
        if tuple(global_cond.shape) != (sample.shape[0], self.global_cond_dim):
            raise ValueError("MoT observation condition has the wrong shape")
        times = torch.as_tensor(timestep, device=sample.device)
        if times.ndim == 0:
            times = times[None]
        times = times.expand(sample.shape[0])
        observations = global_cond.reshape(
            sample.shape[0], self.observation_steps, self.observation_dim
        )
        condition = torch.cat(
            [self.time_emb(times).unsqueeze(1), self.cond_obs_emb(observations)], dim=1
        )
        memory = self.encoder(self.drop(condition + self.cond_pos_emb))
        streams = [
            self.drop(
                self.input_emb[name](
                    sample.index_select(-1, getattr(self, f"{name}_columns"))
                )
                + self.pos_emb
                + self.stream_emb.weight[index]
            )
            for index, name in enumerate(self.modalities)
        ]
        for layer in self.decoder:
            streams = layer(
                streams, memory, mask=self.mask, memory_mask=self.memory_mask
            )
        packed = torch.cat(
            [
                self.head[name](self.ln_f[name](stream))
                for name, stream in zip(self.modalities, streams)
            ],
            dim=-1,
        )
        return packed.index_select(-1, self.output_order)
