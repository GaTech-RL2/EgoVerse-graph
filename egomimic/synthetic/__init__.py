"""Synthetic distribution benchmarks for latent generative models."""

from .manifold_dataset import (
    GaussianParaboloidDataset,
    GaussianTorusDataset,
    generate_gaussian_paraboloid,
    generate_gaussian_torus,
)

__all__ = [
    "GaussianParaboloidDataset",
    "GaussianTorusDataset",
    "generate_gaussian_paraboloid",
    "generate_gaussian_torus",
]
