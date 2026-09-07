"""Two-step trainer/reload checks for the endpoint-first torus candidates."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.endpoint_lift_flow import SyntheticEndpointLiftFlow
from egomimic.synthetic.gaussian_relift_flow import SyntheticGaussianReliftFlow


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [(SyntheticEndpointLiftFlow, 59_336), (SyntheticGaussianReliftFlow, 62_504)],
)
def test_default_endpoint_parameter_counts(model_type, expected):
    model = model_type()
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert sum(parameter.numel() for parameter in model.field.parameters()) == 51_848


def _dataset(path, seed):
    rng = np.random.default_rng(seed)
    theta, phi = rng.uniform(0, 2 * np.pi, size=(2, 40))
    radius = 2.0 + 0.65 * np.cos(phi)
    target = np.stack(
        (radius * np.cos(theta), radius * np.sin(theta), 0.65 * np.sin(phi)), axis=-1
    ).astype(np.float32)
    np.savez_compressed(
        path,
        source_gaussian_latent=rng.normal(size=(40, 8)).astype(np.float32),
        target_3d=target,
        split=np.array([0] * 30 + [1] * 5 + [2] * 5, dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("architecture", "objective"),
    [
        ("endpoint_lift_flow", "latent"),
        ("endpoint_lift_flow", "balanced"),
        ("gaussian_relift_flow", None),
    ],
    ids=["A-latent", "B-balanced", "C-relift"],
)
def test_endpoint_training_validates_and_strict_reload_reproduces_trajectory(
    tmp_path, architecture, objective
):
    source = Path(__file__).resolve().parents[1]
    dataset = tmp_path / "train.npz"
    evaluation_dataset = tmp_path / "evaluation.npz"
    _dataset(dataset, 31)
    _dataset(evaluation_dataset, 32)
    output = tmp_path / "run"
    model_config = {
        "latent_dim": 8,
        "action_dim": 3,
        "coupling_layers": 4,
        "coupling_width": 8,
        "coupling_depth": 1,
        "field_width": 8,
        "field_depth": 1,
    }
    if architecture == "gaussian_relift_flow":
        model_config.update(rotation_layers=4, rotation_width=8, rotation_depth=1)
    config = {
        "architecture": architecture,
        "seed": 42,
        "dataset": str(dataset),
        "evaluation_dataset": str(evaluation_dataset),
        "evaluation_particles": 5,
        "source_key": "source_gaussian_latent",
        "training_noise": "fresh_gaussian",
        "output_dir": str(output),
        "model": model_config,
        "flow_samples": 2,
        "lambda_scale": 1.0,
        "learning_rate": 0.0003,
        "batch_size": 4,
        "max_steps": 2,
        "inference_steps": 2,
        "diagnostic_noise_samples": 16,
        "angular_bins": 4,
        "log_every": 1,
        "checkpoint_every": 2,
    }
    if objective is not None:
        config["endpoint_objective"] = objective
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    subprocess.run(
        [sys.executable, str(source / "scripts/train/train_synthetic_manifold.py"),
         "--config", str(config_path)],
        cwd=source, check=True, timeout=120,
    )
    checkpoint_paths = list((output / "checkpoints").glob("*.pt"))
    assert len(checkpoint_paths) == 1
    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["config"] == config
    assert checkpoint["optimizer"]["state"]
    model_type = (
        SyntheticEndpointLiftFlow if architecture == "endpoint_lift_flow"
        else SyntheticGaussianReliftFlow
    )
    model = model_type(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    for row in rows:
        assert all(np.isfinite(value) for value in row.values())
        assert row["reconstruction_loss"] == 0
        assert row["clean_roundtrip_mse"] < 1e-10
        assert row["training_elapsed_seconds"] > 0
        assert row["training_steps_per_second"] > 0
    summary = json.loads((output / "summary.json").read_text())
    for key in (
        "validation_loss", "validation_generation_symmetric_nn_mse",
        "validation_action_velocity_loss", "validation_scale_loss",
        "validation_torus_surface_rmse", "validation_angular_histogram_l1",
        "validation_decoder_jacobian_singular_min",
        "validation_decoder_jacobian_singular_max",
        "validation_decoded_noise_radius_q50", "validation_generated_endpoint_spread",
        "validation_exact_lift_mse", "validation_trajectory_max_displacement",
    ):
        assert np.isfinite(summary[key]), key
    assert summary["parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert summary["validation_exact_lift_mse"] < 1e-10
    assert summary["validation_decoder_jacobian_singular_min"] > 0
    assert summary["validation_trajectory_max_displacement"] > 0
    if architecture == "endpoint_lift_flow":
        np.testing.assert_allclose(
            summary["validation_full_velocity_loss"],
            summary["validation_action_velocity_contribution"]
            + summary["validation_auxiliary_velocity_contribution"], rtol=1e-6,
        )
    else:
        assert summary["validation_relift_action_mse"] < 1e-10
        assert summary["validation_flow_loss"] == 0
        assert summary["validation_reference_action_velocity_rms"] > 0

    val_source, val_target = SyntheticTrajectoryEval.load_validation_data(
        evaluation_dataset, config["source_key"], 5
    )
    with torch.no_grad():
        expected = model.trajectory(val_source, steps=2)
    with np.load(output / "validation_trajectory.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"times", "points", "target"}
        np.testing.assert_array_equal(saved["target"], val_target.numpy())
        np.testing.assert_allclose(saved["points"], expected.numpy(), rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(saved["times"], np.linspace(0, 1, 3, dtype=np.float32))
    expected_nn = SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(expected[-1], val_target)
    np.testing.assert_allclose(summary["validation_generation_symmetric_nn_mse"], float(expected_nn), rtol=2e-5, atol=2e-6)

    # Also exercise the maintained exporter: exact evaluation data are explicit,
    # so its checkpoint reload cannot silently substitute the training cloud.
    reexport = tmp_path / "reexport.npz"
    subprocess.run(
        [sys.executable, str(source / "scripts/eval/export_synthetic_trajectory_npz.py"),
         "--checkpoint", str(checkpoint_paths[0]), "--dataset", str(evaluation_dataset),
         "--particles", "5", "--steps", "2", "--device", "cpu", "--output", str(reexport)],
        cwd=source, check=True, timeout=120,
    )
    with np.load(reexport, allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["target"], val_target.numpy())
        np.testing.assert_allclose(exported["points"], expected.numpy(), rtol=1e-6, atol=1e-7)
