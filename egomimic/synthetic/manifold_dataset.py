"""Deterministic 2D-Gaussian to 3D-manifold benchmark data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GaussianTorusBatch:
    source_2d: torch.Tensor
    source_3d: torch.Tensor
    target_3d: torch.Tensor
    angles: torch.Tensor


def generate_gaussian_torus(
    count: int,
    *,
    seed: int = 42,
    major_radius: float = 2.0,
    minor_radius: float = 0.65,
    dtype: torch.dtype = torch.float32,
) -> GaussianTorusBatch:
    """Map a planar standard Gaussian deterministically onto a 3D torus.

    Gaussian coordinates pass through the standard-normal CDF, producing two
    uniform angular coordinates. The source is also returned embedded in the
    z=0 plane so a same-dimensional vector field can transport it in 3D.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if major_radius <= 0 or minor_radius <= 0 or minor_radius >= major_radius:
        raise ValueError("require 0 < minor_radius < major_radius")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    source_2d = torch.randn((count, 2), generator=generator, dtype=dtype)
    uniform = 0.5 * (1.0 + torch.erf(source_2d / math.sqrt(2.0)))
    angles = uniform * (2.0 * math.pi)
    theta, phi = angles.unbind(dim=-1)
    tube = major_radius + minor_radius * phi.cos()
    target_3d = torch.stack(
        (tube * theta.cos(), tube * theta.sin(), minor_radius * phi.sin()), dim=-1
    )
    source_3d = torch.nn.functional.pad(source_2d, (0, 1))
    return GaussianTorusBatch(source_2d, source_3d, target_3d, angles)


class GaussianTorusDataset(Dataset):
    """Indexable paired source/target dataset with an exact CFM bridge."""

    def __init__(self, count: int, **kwargs):
        self.data = generate_gaussian_torus(count, **kwargs)

    def __len__(self) -> int:
        return int(self.data.source_2d.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "source_2d": self.data.source_2d[index],
            "source_3d": self.data.source_3d[index],
            "target_3d": self.data.target_3d[index],
            "angles": self.data.angles[index],
        }

    def cfm_state_velocity(
        self, index: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return x_t and the exact constant velocity for linear CFM paths."""
        source = self.data.source_3d[index]
        target = self.data.target_3d[index]
        time = time.to(source).reshape(-1, 1)
        velocity = target - source
        return source + time * velocity, velocity
