"""Repeated-index gather with grouped gradient reduction.

Forward is the original index_select. Backward sums each contiguous group's
contributions without repeated-index scatter accumulation. This changes numeric
addition order, not the mathematical derivative, and is not a global CUDA
determinism guarantee.
"""
import torch


class _GroupedIndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, index, count):
        count = int(count)
        if count <= 0 or value.ndim < 1 or value.shape[0] <= 0:
            raise ValueError("grouped gather requires nonempty input and positive count")
        expected = torch.arange(value.shape[0], device=value.device).repeat_interleave(count)
        if index.dtype != torch.long or not torch.equal(index, expected):
            raise ValueError("grouped gather requires contiguous repeat_interleave indices")
        ctx.shape = tuple(value.shape)
        ctx.count = count
        return value.index_select(0, index)

    @staticmethod
    def backward(ctx, gradient):
        grouped = gradient.reshape(ctx.shape[0], ctx.count, *ctx.shape[1:])
        return grouped.sum(dim=1), None, None


def grouped_index_select(value, index, count):
    return _GroupedIndexSelect.apply(value, index, count)
