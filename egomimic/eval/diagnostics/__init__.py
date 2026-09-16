"""Reusable diagnostics for generative action policies.

Capture stays separate from rendering so a stochastic model is evaluated once
and plots can be regenerated without another inference pass.
"""

from egomimic.eval.diagnostics.gradients import gradient_cosine_similarity
from egomimic.eval.diagnostics.latent_projection import (
    project_pca,
    project_umap,
    write_latent_projection,
)
from egomimic.eval.diagnostics.trajectory import (
    chunk_seam_metrics,
    render_chunk_seam_report,
)

__all__ = [
    "chunk_seam_metrics",
    "gradient_cosine_similarity",
    "project_pca",
    "project_umap",
    "render_chunk_seam_report",
    "write_latent_projection",
]
