"""Shared, model-agnostic mechanics for training metrics and telemetry."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


def finite_scalar(value: Any, label: str) -> torch.Tensor:
    """Return a finite scalar tensor or fail closed with its source label."""

    if not torch.is_tensor(value) or value.ndim != 0:
        raise TypeError(f"{label} must be a scalar tensor")
    if not bool(torch.isfinite(value.detach())):
        raise RuntimeError(f"Non-finite metric {label}")
    return value


def distributed_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Average an autograd value without mutating the graph or optimizer state."""

    value = gradient.detach().float().contiguous().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        value.div_(torch.distributed.get_world_size())
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("Non-finite distributed component gradient")
    return value


def gradient_norm(
    gradients: Sequence[torch.Tensor], *, label: str
) -> torch.Tensor:
    try:
        values = [distributed_gradient(gradient) for gradient in gradients]
    except RuntimeError as error:
        raise RuntimeError(
            f"{label} gradient norm is zero or non-finite"
        ) from error
    if not values:
        raise RuntimeError(f"{label} received no gradients")
    norm = sum(
        (value.square().sum() for value in values), values[0].new_zeros(())
    ).sqrt()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise RuntimeError(f"{label} gradient norm is zero or non-finite")
    return norm


def component_gradients(
    loss: torch.Tensor,
    named: Sequence[tuple[str, nn.Parameter]],
    *,
    allow_unused: bool,
    label: str,
) -> tuple[tuple[nn.Parameter, ...], tuple[torch.Tensor | None, ...]]:
    if not loss.requires_grad:
        raise RuntimeError(f"{label} must retain its autograd graph for telemetry")
    parameters = tuple(parameter for _, parameter in named)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=allow_unused,
    )
    return parameters, gradients


def reduce_weighted_sums(
    sums: Mapping[str, torch.Tensor],
    count: int,
    *,
    label: str,
) -> tuple[OrderedDict[str, torch.Tensor], int]:
    """All-reduce weighted metric sums and their exact sample count."""

    if count <= 0 or not sums:
        raise RuntimeError(f"{label} metric reduction has no samples")
    names = tuple(sums)
    first = sums[names[0]]
    payload = torch.stack(
        (
            *(sums[name].detach().double() for name in names),
            torch.tensor(float(count), device=first.device, dtype=torch.float64),
        )
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
    if not bool(torch.isfinite(payload).all()) or float(payload[-1]) <= 0.0:
        raise RuntimeError(f"non-finite {label} metric reduction")
    return (
        OrderedDict(
            (name, (payload[index] / payload[-1]).to(first.dtype))
            for index, name in enumerate(names)
        ),
        int(payload[-1].item()),
    )


def reduce_component_means(
    components: Mapping[str, torch.Tensor],
    count: int,
    *,
    label: str,
) -> tuple[OrderedDict[str, torch.Tensor], int]:
    return reduce_weighted_sums(
        OrderedDict(
            (name, value.detach().double() * count)
            for name, value in components.items()
        ),
        count,
        label=label,
    )


class MetricAccumulator:
    """Accumulate exact sample-weighted validation values between hooks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sums: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.count = 0

    def add(
        self,
        metrics: Mapping[str, torch.Tensor],
        count: int,
        *,
        update_count: bool = True,
    ) -> None:
        if count <= 0:
            raise RuntimeError("metric accumulator count must be positive")
        for name, value in metrics.items():
            weighted = finite_scalar(value, name).detach().double() * count
            self.sums[name] = self.sums.get(name, 0.0) + weighted
        if update_count:
            self.count += count

    def reduce(self, *, label: str) -> tuple[OrderedDict[str, torch.Tensor], int]:
        return reduce_weighted_sums(self.sums, self.count, label=label)
