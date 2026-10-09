"""Hydra-friendly learning-rate scheduler factories."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    max_steps: int,
    warmup_steps: int = 500,
    warmup_start_factor: float = 0.01,
    eta_min: float = 2e-5,
) -> LRScheduler:
    """Linearly warm up, then cosine-anneal for the remaining optimizer steps."""
    warmup_steps = max(1, int(warmup_steps))
    max_steps = max(warmup_steps + 1, int(max_steps))
    warmup = LinearLR(
        optimizer,
        start_factor=float(warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=float(eta_min),
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def warmup_then_cosine(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 0.0,
    warmup_start_factor: float = 1.0e-3,
) -> LRScheduler:
    """Retain the source epoch schedule, including its validation and defaults.

    Configure ``scheduler_interval: epoch`` when selecting this factory.
    """
    if warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be > 0")
    if total_epochs <= warmup_epochs:
        raise ValueError("total_epochs must be > warmup_epochs")
    return warmup_cosine_scheduler(
        optimizer,
        max_steps=total_epochs,
        warmup_steps=warmup_epochs,
        eta_min=eta_min,
        warmup_start_factor=warmup_start_factor,
    )
