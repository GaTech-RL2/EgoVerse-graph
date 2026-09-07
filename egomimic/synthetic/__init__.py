"""Synthetic distribution benchmarks for latent generative models."""

from .action_adapter_flow import SyntheticActionAdapterFlow
from .decoder_inversion_flow import SyntheticDecoderInversionFlow
from .manifold_dataset import (
    GaussianParaboloidDataset,
    GaussianTorusDataset,
    generate_gaussian_paraboloid,
    generate_gaussian_sphere_cube,
    generate_gaussian_torus,
)
from .multi_action_adapter_flow import SyntheticMultiActionAdapterFlow
from .projected_invertible_flow import SyntheticProjectedInvertibleFlow

__all__ = [
    "GaussianParaboloidDataset",
    "GaussianTorusDataset",
    "generate_gaussian_paraboloid",
    "generate_gaussian_sphere_cube",
    "generate_gaussian_torus",
    "SyntheticActionAdapterFlow",
    "SyntheticDecoderInversionFlow",
    "SyntheticMultiActionAdapterFlow",
    "SyntheticProjectedInvertibleFlow",
]
