"""Opt-in transformer primitives shared by native image and vector experts."""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Normalize the final axis in FP32, retaining the activation dtype.

    Q/K use a learned scale of head_dim, shared across heads within one private
    expert. AdaLN residual and final norms use no affine parameters: their
    scale and shift still come from the existing conditioning projections.
    """

    def __init__(self, width, eps=1e-6, elementwise_affine=True):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("RMSNorm width must be a positive integer")
        if type(elementwise_affine) is not bool:
            raise ValueError("RMSNorm elementwise_affine must be boolean")
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(torch.ones(width)) if elementwise_affine else None

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized.to(value.dtype)
        return normalized if self.weight is None else normalized * self.weight.to(value.dtype)


def swiglu_hidden_dim(width, mlp_ratio, multiple_of=64):
    """Match two-matrix GELU capacity with three SwiGLU matrices, then align.

    The target is (2/3) * width * mlp_ratio. Nearest-multiple rounding uses an
    integer half-up rule; the aligned width is at least one positive multiple.
    Biases make the resulting parameter budget approximate, not identical.
    """
    for name, value in (("width", width), ("mlp_ratio", mlp_ratio), ("multiple_of", multiple_of)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    expanded = width * mlp_ratio
    return max(1, (4 * expanded + 3 * multiple_of) // (6 * multiple_of)) * multiple_of


class SwiGLU(nn.Module):
    def __init__(self, width, mlp_ratio=4, multiple_of=64):
        super().__init__()
        self.hidden_dim = swiglu_hidden_dim(width, mlp_ratio, multiple_of)
        self.gate_value = nn.Linear(width, 2 * self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, width)

    def forward(self, value):
        gate, content = self.gate_value(value).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)
