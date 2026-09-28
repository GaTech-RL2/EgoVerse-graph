"""Random byte-language features with explicit self/cross-attention padding masks."""

from collections.abc import Mapping

import torch
from torch import nn

from egomimic.models.stems.hpt_stems import PolicyStem

PAD, BOS, EOS = 256, 257, 258


def encode_bytes(instructions, *, max_tokens=256, device=None):
    """UTF-8 with boundaries. Refuse truncation of object/target instructions."""
    if not instructions or isinstance(instructions, str):
        raise ValueError("instructions must be a nonempty sequence of strings")
    rows = []
    for instruction in instructions:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Instructions must be nonempty strings")
        row = [BOS, *instruction.encode("utf-8"), EOS]
        if len(row) > max_tokens:
            raise ValueError(f"Instruction exceeds {max_tokens} byte tokens")
        rows.append(row)
    ids = torch.full(
        (len(rows), max(map(len, rows))), PAD, dtype=torch.long, device=device
    )
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, device=device)
    return {"input_ids": ids, "attention_mask": ids != PAD}


class ByteLanguageStem(PolicyStem):
    """HPT-compatible language stem; no tokenizer files, corpus, or weight loads."""

    def __init__(
        self,
        width=256,
        layers=2,
        heads=8,
        ffn_dim=1024,
        dropout=0.1,
        max_tokens=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_tokens = int(max_tokens)
        self.embedding = nn.Embedding(259, width, padding_idx=PAD)
        self.positions = nn.Parameter(torch.randn(1, self.max_tokens, width) * 0.02)
        # Construct independently: TransformerEncoder's copied layers otherwise
        # start with identical values, despite having distinct Parameter objects.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width, heads, ffn_dim, dropout, activation="gelu", batch_first=True
                )
                for _ in range(layers)
            ]
        )

    def _inputs(self, value):
        if not isinstance(value, Mapping):
            value = encode_bytes(value, max_tokens=self.max_tokens, device=self.device)
        if set(value) != {"input_ids", "attention_mask"}:
            raise ValueError(
                "Byte language inputs must contain IDs and attention_mask only"
            )
        ids, mask = value["input_ids"], value["attention_mask"]
        if (
            ids.ndim != 2
            or ids.dtype != torch.long
            or mask.dtype != torch.bool
            or mask.shape != ids.shape
        ):
            raise ValueError(
                "Byte IDs must be [B,L] int64 with a same-shaped boolean mask"
            )
        if not 2 <= ids.shape[1] <= self.max_tokens or ids.shape[0] == 0:
            raise ValueError("Invalid byte sequence length")
        if bool(((ids < 0) | (ids > EOS)).any()) or not torch.equal(mask, ids != PAD):
            raise ValueError("Invalid byte vocabulary or padding mask")
        lengths = mask.sum(1)
        expected = (
            torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        )
        if bool((lengths < 3).any()) or not torch.equal(mask, expected):
            raise ValueError(
                "Byte tokens require nonempty content and contiguous right padding"
            )
        if not bool((ids[:, 0] == BOS).all()) or not bool(
            (ids.gather(1, (lengths - 1)[:, None]) == EOS).all()
        ):
            raise ValueError("Byte tokens require BOS/EOS boundaries")
        interior = (
            mask
            & (torch.arange(ids.shape[1], device=ids.device)[None] > 0)
            & (
                torch.arange(ids.shape[1], device=ids.device)[None]
                < lengths[:, None] - 1
            )
        )
        if bool((ids[interior] > 255).any()):
            raise ValueError("Only UTF-8 byte IDs are allowed inside boundaries")
        return ids, mask

    def compute_latent(self, value):
        ids, mask = self._inputs(value)
        hidden = self.embedding(ids) + self.positions[:, : ids.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=~mask)
        tokens = self.tokens.expand(ids.shape[0], -1, -1)
        return self.cross_attention(tokens, hidden, mask=mask)
