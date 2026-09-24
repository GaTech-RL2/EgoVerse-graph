"""Deterministic HPT decoder retained from EgoVerse ec5c903c.

The head owns only neural modules; the graph stage owns targets and losses.
"""

import torch
from torch import nn

from egomimic.models.cores.hpt_transformer import Attention, CrossAttention
from egomimic.models.cores.hpt_utils import get_sinusoid_encoding_table

INIT_CONST = 0.02


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention = Attention(
            dim=input_dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.cross_attention = CrossAttention(
            input_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.SiLU(), nn.Linear(input_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, tokens, context):
        query = self.self_attention(self.norm1(tokens))
        query = tokens + query

        out = self.cross_attention(self.norm2(query), context)
        out = query + out

        mlp_out = self.mlp(self.norm3(out))
        tokens = mlp_out + out
        return tokens


class MultiBlockTransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 10,
        action_horizon: int = 16,
        latent_token_len: int = 8,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        num_layers: int = 4,
        final_norm: bool = False,
    ):
        super().__init__()
        self.tokens = nn.Parameter(
            torch.randn(1, action_horizon, input_dim) * INIT_CONST
        )
        self.pos_token = nn.Parameter(
            get_sinusoid_encoding_table(0, action_horizon, input_dim)
        )
        self.pos_context = nn.Parameter(
            get_sinusoid_encoding_table(0, latent_token_len, input_dim)
        )

        self.context_norm = nn.LayerNorm(input_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, output_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = final_norm
        if self.final_norm:
            self.last_layer_norm = nn.LayerNorm(input_dim)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[MultiBlockTransformerDecoder] Total trainable parameters: {total_params / 1e6:.2f}M"
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = self.tokens.expand(B, -1, -1) + self.pos_token.expand(B, -1, -1)
        context = self.context_norm(x + self.pos_context.expand(B, -1, -1))

        for block in self.blocks:
            tokens = block(tokens, context)

        if self.final_norm:
            tokens = self.last_layer_norm(tokens)

        return self.out_proj(tokens)
