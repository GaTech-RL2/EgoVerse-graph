"""Synthetic distribution benchmarks for latent generative models."""

from .manifold_dataset import GaussianTorusDataset, generate_gaussian_torus

__all__ = ["GaussianTorusDataset", "generate_gaussian_torus"]
