"""Deterministic 2D-Gaussian to 3D-manifold benchmark data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class GaussianTorusBatch:
    source_latent: torch.Tensor
    source_gaussian_latent: torch.Tensor
    source_2d: torch.Tensor
    source_3d: torch.Tensor
    source_gaussian_3d: torch.Tensor
    target_3d: torch.Tensor
    angles: torch.Tensor


@dataclass(frozen=True)
class GaussianParaboloidBatch:
    source_latent: torch.Tensor
    source_gaussian_latent: torch.Tensor
    source_2d: torch.Tensor
    source_3d: torch.Tensor
    source_gaussian_3d: torch.Tensor
    target_3d: torch.Tensor


def _independent_gaussian(
    count: int, dimension: int, seed: int, dtype: torch.dtype
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 10_000)
    return torch.randn((count, dimension), generator=generator, dtype=dtype)


def _independent_gaussian_pair(
    count: int, latent_dim: int, seed: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matching 3D and latent noise with identical first coordinates."""
    noise_3d = _independent_gaussian(count, 3, seed, dtype)
    if latent_dim <= 3:
        return noise_3d, noise_3d[:, :latent_dim]
    extra = _independent_gaussian(count, latent_dim - 3, seed + 10_000, dtype)
    return noise_3d, torch.cat((noise_3d, extra), dim=-1)


def generate_gaussian_torus(
    count: int,
    *,
    seed: int = 42,
    major_radius: float = 2.0,
    minor_radius: float = 0.65,
    source_dim: int = 2,
    dtype: torch.dtype = torch.float32,
) -> GaussianTorusBatch:
    """Map a planar standard Gaussian deterministically onto a 3D torus.

    Gaussian coordinates pass through the standard-normal CDF, producing two
    uniform angular coordinates. The source is also returned embedded in the
    z=0 plane so a same-dimensional vector field can transport it in 3D.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if source_dim < 2:
        raise ValueError("source_dim must be at least 2")
    if major_radius <= 0 or minor_radius <= 0 or minor_radius >= major_radius:
        raise ValueError("require 0 < minor_radius < major_radius")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    source_2d = torch.randn((count, 2), generator=generator, dtype=dtype)
    if source_dim == 2:
        source_latent = source_2d
    else:
        nuisance = torch.randn(
            (count, source_dim - 2), generator=generator, dtype=dtype
        )
        source_latent = torch.cat((source_2d, nuisance), dim=-1)
    uniform = 0.5 * (1.0 + torch.erf(source_2d / math.sqrt(2.0)))
    angles = uniform * (2.0 * math.pi)
    theta, phi = angles.unbind(dim=-1)
    tube = major_radius + minor_radius * phi.cos()
    target_3d = torch.stack(
        (tube * theta.cos(), tube * theta.sin(), minor_radius * phi.sin()), dim=-1
    )
    source_3d = torch.nn.functional.pad(source_2d, (0, 1))
    source_gaussian_3d, source_gaussian_latent = _independent_gaussian_pair(
        count, source_dim, seed, dtype
    )
    return GaussianTorusBatch(
        source_latent,
        source_gaussian_latent,
        source_2d,
        source_3d,
        source_gaussian_3d,
        target_3d,
        angles,
    )


def generate_gaussian_paraboloid(
    count: int,
    *,
    seed: int = 42,
    curvature: float = 0.25,
    source_dim: int = 2,
    dtype: torch.dtype = torch.float32,
) -> GaussianParaboloidBatch:
    """Lift a planar Gaussian onto a smooth, injective 3D paraboloid.

    The first two coordinates are preserved exactly and only the height changes:
    ``(u, v, 0) -> (u, v, curvature * (u**2 + v**2))``. This avoids the
    periodic seams and topology change of the torus benchmark, making it a
    deliberately easy control for the shared-latent training objectives.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if source_dim < 2:
        raise ValueError("source_dim must be at least 2")
    if curvature <= 0:
        raise ValueError("curvature must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    source_2d = torch.randn((count, 2), generator=generator, dtype=dtype)
    if source_dim == 2:
        source_latent = source_2d
    else:
        nuisance = torch.randn(
            (count, source_dim - 2), generator=generator, dtype=dtype
        )
        source_latent = torch.cat((source_2d, nuisance), dim=-1)
    source_3d = torch.nn.functional.pad(source_2d, (0, 1))
    height = curvature * source_2d.square().sum(dim=-1, keepdim=True)
    target_3d = torch.cat((source_2d, height), dim=-1)
    source_gaussian_3d, source_gaussian_latent = _independent_gaussian_pair(
        count, source_dim, seed, dtype
    )
    return GaussianParaboloidBatch(
        source_latent,
        source_gaussian_latent,
        source_2d,
        source_3d,
        source_gaussian_3d,
        target_3d,
    )


class GaussianTorusDataset(Dataset):
    """Indexable paired source/target dataset with an exact CFM bridge."""

    def __init__(self, count: int, **kwargs):
        self.data = generate_gaussian_torus(count, **kwargs)

    def __len__(self) -> int:
        return int(self.data.source_latent.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "source_latent": self.data.source_latent[index],
            "source_gaussian_latent": self.data.source_gaussian_latent[index],
            "source_2d": self.data.source_2d[index],
            "source_3d": self.data.source_3d[index],
            "source_gaussian_3d": self.data.source_gaussian_3d[index],
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


class GaussianParaboloidDataset(Dataset):
    """Indexable paired Gaussian-to-paraboloid control dataset."""

    def __init__(self, count: int, **kwargs):
        self.data = generate_gaussian_paraboloid(count, **kwargs)

    def __len__(self) -> int:
        return int(self.data.source_latent.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "source_latent": self.data.source_latent[index],
            "source_gaussian_latent": self.data.source_gaussian_latent[index],
            "source_2d": self.data.source_2d[index],
            "source_3d": self.data.source_3d[index],
            "source_gaussian_3d": self.data.source_gaussian_3d[index],
            "target_3d": self.data.target_3d[index],
        }

    def cfm_state_velocity(
        self, index: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.data.source_3d[index]
        target = self.data.target_3d[index]
        time = time.to(source).reshape(-1, 1)
        velocity = target - source
        return source + time * velocity, velocity
