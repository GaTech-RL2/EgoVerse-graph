"""Gradient-conflict diagnostics that do not mutate ``parameter.grad``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch


def gradient_cosine_similarity(
    first_loss: torch.Tensor,
    second_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return the cosine between two losses' gradients on shared parameters.

    ``torch.autograd.grad`` leaves optimizer gradients untouched. The graph is
    retained so the normal training backward can still run afterward.
    Parameters unused by either branch contribute a zero vector.
    """

    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not params:
        raise ValueError("gradient cosine requires at least one trainable parameter")
    if first_loss.ndim != 0 or second_loss.ndim != 0:
        raise ValueError("gradient cosine losses must be scalar tensors")

    first = torch.autograd.grad(
        first_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    second = torch.autograd.grad(
        second_loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    reference = first_loss.detach().float()
    dot = torch.zeros((), device=reference.device, dtype=torch.float32)
    first_sq = torch.zeros_like(dot)
    second_sq = torch.zeros_like(dot)
    for first_grad, second_grad, parameter in zip(first, second, params):
        first_value = (
            (
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                if first_grad is None
                else first_grad
            )
            .detach()
            .float()
        )
        second_value = (
            (
                torch.zeros_like(parameter, memory_format=torch.preserve_format)
                if second_grad is None
                else second_grad
            )
            .detach()
            .float()
        )
        dot = dot + torch.sum(first_value * second_value)
        first_sq = first_sq + torch.sum(first_value.square())
        second_sq = second_sq + torch.sum(second_value.square())

    denominator = torch.sqrt(first_sq * second_sq)
    cosine = dot / denominator.clamp_min(float(epsilon))
    return torch.where(denominator > float(epsilon), cosine, torch.zeros_like(cosine))


def render_gradient_conflict(
    steps: Sequence[int], cosines: Sequence[float], output_path: str | Path
) -> Path:
    """Render an offline positive/neutral/negative loss-conflict chart."""

    steps_array = np.asarray(steps, dtype=np.int64)
    cosine_array = np.asarray(cosines, dtype=np.float64)
    if steps_array.ndim != 1 or cosine_array.ndim != 1:
        raise ValueError("steps and cosines must be one-dimensional")
    if steps_array.size != cosine_array.size or steps_array.size == 0:
        raise ValueError("steps and cosines must have the same non-zero length")
    if not np.all(np.isfinite(cosine_array)):
        raise ValueError("cosines contain non-finite values")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.axhspan(0.0, 1.0, color="#2ca02c", alpha=0.08, label="agree")
    axis.axhspan(-1.0, 0.0, color="#d62728", alpha=0.08, label="conflict")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.plot(steps_array, cosine_array, color="#1f77b4", linewidth=1.3)
    axis.set(xlabel="optimizer step", ylabel="gradient cosine", ylim=(-1.02, 1.02))
    axis.set_title("UNITE reconstruction vs denoising gradient alignment")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
