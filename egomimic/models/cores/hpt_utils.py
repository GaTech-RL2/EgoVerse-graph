"""Small tensor helpers the HPT trunk and stems need.

This fork dropped ``egomimic/utils/tensor_utils.py`` during the graph
consolidation. Only these two helpers are still required, so they live here --
scoped to the HPT cores -- rather than resurrecting a module the fork
deliberately removed.
"""

from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn


def get_sinusoid_encoding_table(
    position_start: int, position_end: int, d_hid: int
) -> torch.Tensor:
    """Sinusoid position encoding table for ``[position_start, position_end)``.

    Returns ``(1, position_end - position_start, d_hid)`` so it broadcasts
    directly onto a ``(B, T, d_hid)`` token sequence.
    """
    if position_end <= position_start:
        raise ValueError("position_end must be greater than position_start")
    if d_hid <= 0:
        raise ValueError("d_hid must be positive")

    positions = torch.arange(position_start, position_end, dtype=torch.float32)
    div_term = torch.exp(
        torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid)
    )
    table = torch.zeros((position_end - position_start, d_hid))
    table[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
    # An odd d_hid leaves the cosine half one column short, hence the slice.
    table[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term[: d_hid // 2])
    return table.unsqueeze(0)


class EinOpsRearrange(nn.Module):
    """``einops.rearrange`` as a module, so it can sit inside a ``Sequential``."""

    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = str(rearrange_expr)
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"EinOpsRearrange expects a tensor, got {type(x).__name__}")
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)
