import json

import numpy as np
import pytest
import torch

from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow
from scripts.experiments.aggregate_distribution_benchmark import aggregate
from scripts.experiments.prepare_distribution_benchmark import (
    DISTRIBUTIONS,
    _high_features,
    _native,
)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_distribution_generators_are_deterministic_finite_and_nonconstant(distribution):
    first = _native(distribution, 64, 123)
    second = _native(distribution, 64, 123)
    np.testing.assert_array_equal(first, second)
    assert first.ndim == 2
    assert len(first) == 64
    assert np.isfinite(first).all()
    assert np.all(first.std(axis=0) > 0)


def test_high_dimensional_embedding_is_deterministic_and_injective():
    native = _native("henon", 64, 123)
    high = _high_features(native, seed=456)
    assert high.shape == (64, 32)
    np.testing.assert_array_equal(high[:, : native.shape[1]], native)
    np.testing.assert_array_equal(high, _high_features(native, seed=456))


def test_action_adapter_supports_high_dimensional_actions():
    model = SyntheticActionAdapterFlow(
        action_dim=32,
        latent_dim=64,
        adapter_family="nonlinear",
        residual_width=8,
        residual_depth=1,
        field_width=8,
        field_depth=1,
    )
    action = torch.randn(6, 32)
    noise = torch.randn(6, 64)
    losses = model.losses(
        action,
        objective="action_velocity",
        flow_samples=2,
        lambda_scale=0.0,
        clean_gradient_mode="all_stopgrad",
        noise=noise,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.decoder(model.encoder(action)).shape == action.shape


def test_all_stopgrad_reserves_encoder_updates_for_reconstruction():
    model = SyntheticActionAdapterFlow(
        action_dim=2,
        latent_dim=8,
        adapter_family="nonlinear",
        residual_width=8,
        residual_depth=1,
        field_width=8,
        field_depth=1,
    )
    with torch.no_grad():
        model.decoder.residual[-1].weight.normal_(std=0.02)
        model.decoder.residual[-1].bias.normal_(std=0.02)
    losses = model.losses(
        torch.randn(6, 2),
        objective="action_velocity",
        flow_samples=2,
        lambda_scale=0.0,
        clean_gradient_mode="all_stopgrad",
    )
    losses["flow_loss"].backward(retain_graph=True)
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    model.zero_grad(set_to_none=True)
    losses["action_velocity_loss"].backward(retain_graph=True)
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    model.zero_grad(set_to_none=True)
    losses["reconstruction_loss"].backward()
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum())
        for parameter in model.encoder.parameters()
    )


def test_aggregate_requires_and_pairs_all_72_finite_runs(tmp_path):
    runs = []
    for distribution in DISTRIBUTIONS:
        for dimension in ("low", "high32"):
            for seed in (42, 43, 44):
                for method, energy in (("direct", 2.0), ("action_flow", 1.0)):
                    run_id = f"{distribution}-{dimension}-{method}-s{seed}"
                    output = tmp_path / "runs" / run_id
                    output.mkdir(parents=True)
                    (output / "summary.json").write_text(
                        json.dumps(
                            {
                                "trainable_parameters": 100,
                                "validation_generation_energy_distance": energy,
                                "validation_generation_symmetric_nn_mse": energy / 2,
                            }
                        )
                    )
                    config = tmp_path / "configs" / f"{run_id}.json"
                    config.parent.mkdir(exist_ok=True)
                    config.write_text(
                        json.dumps(
                            {"source_commit": "a" * 40, "output_dir": str(output)}
                        )
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "config": str(config),
                            "distribution": distribution,
                            "dimension_regime": dimension,
                            "seed": seed,
                            "method": method,
                        }
                    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_commit": "a" * 40, "runs": runs}))
    rows, paired = aggregate(manifest)
    assert len(rows) == 72
    assert len(paired) == 36
    assert all(
        row["action_minus_direct_validation_generation_energy_distance"] == -1.0
        for row in paired
    )
