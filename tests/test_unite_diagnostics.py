import json

import numpy as np
import pytest
import torch

from egomimic.eval.diagnostics.gradients import gradient_cosine_similarity
from egomimic.eval.diagnostics.latent_projection import project_pca
from egomimic.eval.diagnostics.trajectory import (
    chunk_seam_metrics,
    validate_chunk_seam_artifact,
    write_chunk_seam_artifact,
)
from egomimic.eval.diagnostics.unite_artifact import (
    UniteDiagnosticArtifactWriter,
    validate_unite_artifact,
)


def test_gradient_cosine_detects_exact_conflict_without_populating_grad():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    first = parameter.sum()
    second = -parameter.sum()

    cosine = gradient_cosine_similarity(first, second, [parameter])

    assert cosine.item() == pytest.approx(-1.0)
    assert parameter.grad is None


def test_chunk_seam_metrics_use_circular_rotation_and_derivatives():
    old = np.asarray(
        [
            [0.0, 0.0, 3.10, 0.0],
            [1.0, 0.0, 3.11, 0.5],
            [2.0, 0.0, 3.12, 1.0],
            [3.0, 0.0, 3.13, 1.0],
        ],
        dtype=np.float32,
    )
    new = old.copy()
    new[:, 2] = -3.10
    metrics = chunk_seam_metrics(old, new)

    assert metrics["aligned_steps"] == 4
    assert metrics["position_mean_l2"] == pytest.approx(0.0)
    assert metrics["rotation_mean_abs_radians"] < 0.1
    assert metrics["velocity_rms_difference"] == pytest.approx(0.0)
    assert metrics["jerk_rms_difference"] == pytest.approx(0.0)


def test_chunk_seam_artifact_is_immutable_and_hash_validated(tmp_path):
    event = {
        "t": 8,
        "embodiment_id": 19,
        "episode_index": 0,
        "previous_tail": np.zeros((8, 4), dtype=np.float32),
        "new_head": np.ones((8, 4), dtype=np.float32),
        "executed_head": np.ones((8, 4), dtype=np.float32),
    }
    artifact = write_chunk_seam_artifact([event], tmp_path / "seams")

    manifest = validate_chunk_seam_artifact(artifact)

    assert manifest["event_count"] == 1
    with pytest.raises(FileExistsError):
        write_chunk_seam_artifact([event], artifact)


def test_shared_latent_pca_has_stable_shape():
    features = np.arange(48, dtype=np.float32).reshape(8, 6)

    projected, ratio = project_pca(features, 3)

    assert projected.shape == (8, 3)
    assert ratio.shape == (3,)
    assert np.all(np.isfinite(projected))


def test_unite_artifact_round_trip(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"schema_version": 1, "checkpoint_sha256": "abc"}))
    writer = UniteDiagnosticArtifactWriter(
        str(tmp_path / "unite"), request_path=str(request)
    )
    batch, horizon, action_dim, tokens, latent_dim = 2, 4, 3, 4, 8
    steps, samples = 3, 5
    writer.append(
        embodiment_id=17,
        embodiment_name="pushshapes_sim_u_socket",
        action_key="actions",
        target_action=np.zeros((batch, horizon, action_dim), dtype=np.float32),
        reconstructed_action=np.zeros((batch, horizon, action_dim), dtype=np.float32),
        generated_action=np.zeros((batch, horizon, action_dim), dtype=np.float32),
        clean_latent=np.zeros((batch, tokens, latent_dim), dtype=np.float32),
        generated_latent=np.zeros((batch, tokens, latent_dim), dtype=np.float32),
        denoising_latents=np.zeros(
            (batch, steps, tokens, latent_dim), dtype=np.float32
        ),
        denoising_actions=np.zeros(
            (batch, steps, horizon, action_dim), dtype=np.float32
        ),
        diversity_latents=np.zeros(
            (batch, samples, tokens, latent_dim), dtype=np.float32
        ),
        diversity_actions=np.zeros(
            (batch, samples, horizon, action_dim), dtype=np.float32
        ),
        episode_hashes=["episode-a", "episode-b"],
    )
    artifact = writer.finalize()

    manifest = validate_unite_artifact(artifact, request)

    assert manifest["sample_count"] == 2
    assert manifest["parts"][0]["denoising_steps"] == steps
    assert manifest["parts"][0]["diversity_samples"] == samples
