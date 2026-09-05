"""Synthetic distribution benchmarks for latent generative models."""

from .action_adapter_flow import SyntheticActionAdapterFlow
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
    "SyntheticActionAdapterFlow",
]
