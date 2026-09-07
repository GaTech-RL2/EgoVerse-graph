"""Short optimizer/validation/reload checks for noninvertible torus trials."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.eval.noninvertible_diagnostics import noninvertible_diagnostics
from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.latent_bridge_likelihood import SyntheticLatentBridgeLikelihood
from egomimic.synthetic.noninvertible_endpoint_flow import (
    SyntheticGraphSectionFlow,
    SyntheticMMDEndpointFlow,
)
from egomimic.synthetic.shared_latent_flow import SyntheticDirectFlow


MODELS = {
    "mmd_endpoint_flow": SyntheticMMDEndpointFlow,
    "graph_section_flow": SyntheticGraphSectionFlow,
    "latent_bridge_likelihood": SyntheticLatentBridgeLikelihood,
    "direct_flow": SyntheticDirectFlow,
}


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [("mmd_endpoint_flow", 54_798), ("graph_section_flow", 54_640),
     ("latent_bridge_likelihood", 54_798), ("direct_flow", 50_563)],
)
def test_default_noninvertible_comparison_parameter_counts(architecture, expected):
    model = MODELS[architecture]()
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    field_count = sum(parameter.numel() for parameter in model.field.parameters())
    assert field_count == (50_563 if architecture == "direct_flow" else 51_848)


def _dataset(path, seed):
    rng = np.random.default_rng(seed)
    theta, phi = rng.uniform(0, 2 * np.pi, size=(2, 40))
    radius = 2.0 + 0.65 * np.cos(phi)
    target = np.stack(
        (radius * np.cos(theta), radius * np.sin(theta), 0.65 * np.sin(phi)), axis=-1
    ).astype(np.float32)
    source = rng.normal(size=(40, 8)).astype(np.float32)
    np.savez_compressed(
        path, source_gaussian_latent=source, source_gaussian_3d=source[:, :3],
        target_3d=target,
        split=np.array([0] * 30 + [1] * 5 + [2] * 5, dtype=np.uint8),
    )


def _small_config(architecture):
    if architecture == "direct_flow":
        return {"data_dim": 3, "field_width": 8, "field_depth": 1}
    model = {"latent_dim": 8, "action_dim": 3, "field_width": 8, "field_depth": 1}
    if architecture == "latent_bridge_likelihood":
        model.update(levels=2, encoder_width=8, encoder_depth=1,
                     decoder_width=8, decoder_depth=1)
    else:
        model.update(residual_width=8, residual_depth=1)
    return model


@pytest.mark.parametrize(
    ("architecture", "endpoint_weight"),
    [("mmd_endpoint_flow", 10.0), ("mmd_endpoint_flow", 100.0),
     ("graph_section_flow", None), ("latent_bridge_likelihood", None),
     ("direct_flow", None)],
)
def test_noninvertible_training_validates_and_exact_reload_reexports(
    tmp_path, architecture, endpoint_weight
):
    repository = Path(__file__).resolve().parents[1]
    dataset = tmp_path / "train.npz"
    evaluation_dataset = tmp_path / "evaluation.npz"
    _dataset(dataset, 71)
    _dataset(evaluation_dataset, 72)
    output = tmp_path / "run"
    config = {
        "architecture": architecture, "seed": 42,
        "dataset": str(dataset), "evaluation_dataset": str(evaluation_dataset),
        "evaluation_particles": 5,
        "source_key": "source_gaussian_3d" if architecture == "direct_flow"
        else "source_gaussian_latent",
        "training_noise": "fresh_gaussian", "output_dir": str(output),
        "model": _small_config(architecture), "flow_samples": 2,
        "learning_rate": 0.0003, "batch_size": 4, "max_steps": 2,
        "inference_steps": 2, "diagnostic_noise_samples": 16,
        "angular_bins": 4, "log_every": 1, "checkpoint_every": 2,
    }
    if architecture in {"mmd_endpoint_flow", "graph_section_flow"}:
        config["lambda_scale"] = 1.0
    if architecture == "mmd_endpoint_flow":
        config["lambda_endpoint"] = endpoint_weight
    if architecture == "direct_flow":
        config["noninvertible_diagnostics"] = True
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    subprocess.run(
        [sys.executable, str(repository / "scripts/train/train_synthetic_manifold.py"),
         "--config", str(config_path)], cwd=repository, check=True, timeout=120,
    )
    checkpoint_paths = list((output / "checkpoints").glob("*.pt"))
    assert len(checkpoint_paths) == 1
    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["config"] == config
    assert checkpoint["optimizer"]["state"]
    model = MODELS[architecture](**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    for row in rows:
        assert all(np.isfinite(value) for value in row.values())
        assert row["training_elapsed_seconds"] > 0
        assert row["training_steps_per_second"] > 0
        if architecture == "mmd_endpoint_flow":
            np.testing.assert_allclose(
                row["loss"], row["flow_loss"] + row["action_velocity_loss"]
                + row["scale_loss"] + endpoint_weight * row["endpoint_mmd2"],
                rtol=2e-6, atol=1e-7,
            )
    summary = json.loads((output / "summary.json").read_text())
    for key in (
        "validation_loss", "validation_generation_symmetric_nn_mse",
        "validation_generation_energy_distance", "validation_torus_surface_rmse",
        "validation_angular_histogram_l1", "validation_angular_support_recall",
        "validation_fixed_noise_scale_loss", "validation_latent_code_rms",
        "validation_decoder_noise_jacobian_singular_min",
        "validation_decoder_clean_jacobian_singular_max",
        "validation_decoded_noise_radius_q50", "validation_generated_endpoint_spread",
        "validation_trajectory_max_displacement",
    ):
        assert np.isfinite(summary[key]), key
    assert summary["parameters"] == sum(parameter.numel() for parameter in model.parameters())
    assert summary["validation_trajectory_max_displacement"] > 0
    prefix = ("validation_reference_boundary_mean" if architecture == "latent_bridge_likelihood"
              else "validation_decoded_training_code")
    for suffix in ("symmetric_nn_mse", "paired_mse"):
        assert np.isfinite(summary[f"{prefix}_{suffix}"])
    if architecture == "graph_section_flow":
        assert summary[f"{prefix}_paired_mse"] < 1e-10
        assert summary["validation_clean_roundtrip_mse"] < 1e-10
    elif architecture == "mmd_endpoint_flow":
        assert np.isfinite(summary["validation_endpoint_mmd2"])
    elif architecture == "latent_bridge_likelihood":
        assert "validation_action_velocity_loss" not in summary
        assert "validation_scale_loss" not in summary
        assert "validation_decoded_training_code_paired_mse" not in summary
        np.testing.assert_allclose(
            summary["validation_loss"],
            summary["validation_interior_kl_loss"] + summary["validation_endpoint_nll_loss"],
            rtol=1e-6,
        )
    else:
        assert "validation_scale_loss" not in summary
        assert summary[f"{prefix}_paired_mse"] == 0

    val_source, val_target = SyntheticTrajectoryEval.load_validation_data(
        evaluation_dataset, config["source_key"], 5
    )
    with torch.no_grad():
        expected = model.trajectory(val_source, steps=2)
        if architecture == "latent_bridge_likelihood":
            endpoint = model.boundary_prediction(val_target, val_source)
            np.testing.assert_allclose(
                summary[f"{prefix}_paired_mse"],
                float((endpoint - val_target).square().mean()), rtol=2e-5, atol=2e-6,
            )
        elif architecture != "direct_flow":
            np.testing.assert_allclose(summary["validation_scale_loss"],
                                       float(model.scale_loss(val_source)), rtol=2e-5, atol=2e-6)
    with np.load(output / "validation_trajectory.npz", allow_pickle=False) as saved:
        assert set(saved.files) == {"times", "points", "target"}
        np.testing.assert_array_equal(saved["target"], val_target.numpy())
        np.testing.assert_allclose(saved["points"], expected.numpy(), rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(saved["times"], np.linspace(0, 1, 3, dtype=np.float32))
    with np.load(output / "validation_particles.npz", allow_pickle=False) as particles:
        np.testing.assert_array_equal(particles["target"], val_target.numpy())
        np.testing.assert_allclose(particles["generation"], expected[-1].numpy(), rtol=2e-5, atol=2e-6)
    expected_nn = SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(expected[-1], val_target)
    np.testing.assert_allclose(summary["validation_generation_symmetric_nn_mse"],
                               float(expected_nn), rtol=2e-5, atol=2e-6)

    # The maintained exporter must replay the actual stochastic likelihood,
    # including output noise, on the same explicitly selected held-out cloud.
    reexport = tmp_path / "reexport.npz"
    subprocess.run(
        [sys.executable, str(repository / "scripts/eval/export_synthetic_trajectory_npz.py"),
         "--checkpoint", str(checkpoint_paths[0]), "--dataset", str(evaluation_dataset),
         "--particles", "5", "--steps", "2", "--device", "cpu", "--output", str(reexport)],
        cwd=repository, check=True, timeout=120,
    )
    with np.load(reexport, allow_pickle=False) as exported:
        np.testing.assert_array_equal(exported["target"], val_target.numpy())
        np.testing.assert_allclose(exported["points"], expected.numpy(), rtol=1e-6, atol=1e-7)


def test_terminal_diagnostics_preserve_rng_and_keep_fixed_noise_health_separate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyntheticMMDEndpointFlow(**_small_config("mmd_endpoint_flow")).to(device).eval()
    source = torch.randn(5, 8, device=device)
    target = torch.randn(5, 3, device=device)
    config = {"architecture": "mmd_endpoint_flow", "seed": 42,
              "diagnostic_noise_samples": 16, "angular_bins": 4}
    with torch.no_grad():
        trajectory = model.trajectory(source, steps=2)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    summary = noninvertible_diagnostics(model, source, target, trajectory, config)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        fixed = torch.randn(16, 8, generator=torch.Generator().manual_seed(30_042)).to(device)
        assert summary["validation_scale_loss"] == float(model.scale_loss(source))
        assert summary["validation_fixed_noise_scale_loss"] == float(model.scale_loss(fixed))
    assert summary == noninvertible_diagnostics(model, source, target, trajectory, config)
