"""Guards for Candidate C's exact maps, targets, and sampler contract."""

import pytest
import torch
from torch.func import jacrev, jvp, vmap

from egomimic.synthetic.gaussian_relift_flow import (
    GaussianPreservingRotation,
    SyntheticGaussianReliftFlow,
)


def _small_model():
    torch.manual_seed(123)
    model = SyntheticGaussianReliftFlow(
        latent_dim=8,
        action_dim=3,
        rotation_layers=4,
        rotation_width=8,
        rotation_depth=2,
        coupling_layers=4,
        coupling_width=8,
        coupling_depth=2,
        field_width=12,
        field_depth=2,
    ).double()
    with torch.no_grad():
        for layer in model.gaussian_transform.layers:
            layer.net[-1].weight.normal_(std=0.3)
            layer.net[-1].bias.normal_(std=0.3)
        for layer in model.transform.layers:
            layer.net[-1].weight.normal_(std=0.04)
            layer.net[-1].bias.normal_(std=0.04)
    return model


def _draw_inputs(batch=7, samples=2):
    return {
        "action": torch.randn(batch, 3, dtype=torch.float64),
        "noise": torch.randn(batch, 8, dtype=torch.float64),
        "aux_noise": torch.randn(batch, 5, dtype=torch.float64),
        "relift_noise": torch.randn(batch * samples, 5, dtype=torch.float64),
        "time": torch.rand(batch * samples, 1, dtype=torch.float64),
    }


def test_rotation_starts_at_identity_and_has_exact_nonlinear_inverse():
    identity = GaussianPreservingRotation().double()
    points = torch.randn(17, 8, dtype=torch.float64)
    torch.testing.assert_close(identity(points), points, atol=0, rtol=0)
    rotation = _small_model().gaussian_transform
    moved = rotation(points)
    assert (moved - points).abs().max() > 0.1
    torch.testing.assert_close(rotation.inverse(moved), points, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(rotation(rotation.inverse(points)), points, atol=2e-12, rtol=2e-12)


def test_rotation_preserves_radius_and_volume_hence_gaussian_density():
    rotation = _small_model().gaussian_transform
    points = torch.randn(9, 8, dtype=torch.float64)
    torch.testing.assert_close(
        rotation(points).square().sum(-1), points.square().sum(-1), atol=2e-12, rtol=2e-12
    )
    jacobian = vmap(jacrev(rotation))(points)
    torch.testing.assert_close(
        torch.linalg.det(jacobian), torch.ones(9, dtype=torch.float64), atol=2e-12, rtol=2e-12
    )
    # Radius alone would not prove Gaussian preservation: this independent
    # volume assertion catches transformations that distort angular density.


def test_exact_clean_lift_and_relift_keep_decoded_action():
    model = _small_model()
    inputs = _draw_inputs()
    clean = model.exact_lift(inputs["action"], inputs["aux_noise"])
    torch.testing.assert_close(model.decoder(clean), inputs["action"], atol=3e-12, rtol=3e-12)
    state = torch.randn(14, 8, dtype=torch.float64)
    relifted = model.relift(state, inputs["relift_noise"])
    torch.testing.assert_close(model.decoder(relifted), model.decoder(state), atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(
        model.gaussian_transform(relifted)[:, 3:], inputs["relift_noise"], atol=3e-12, rtol=3e-12
    )
    assert (relifted - state).abs().max() > 0.5


def test_gaussian_relifted_noise_boundary_has_standard_gaussian_moments():
    model = _small_model()
    torch.manual_seed(904)
    with torch.no_grad():
        source = torch.randn(8192, 8, dtype=torch.float64)
        relifted = model.relift(source, torch.randn(8192, 5, dtype=torch.float64))
        assert relifted.mean(0).abs().max() < 0.055
        centered = relifted - relifted.mean(0)
        covariance = centered.T @ centered / (len(relifted) - 1)
        assert (covariance - torch.eye(8)).abs().max() < 0.065


def test_decoder_jvp_matches_centered_finite_difference_and_works_without_grad():
    model = _small_model()
    state = torch.randn(7, 8, dtype=torch.float64)
    tangent = torch.randn_like(state)
    epsilon = 1e-5
    expected = (model.decoder(state + epsilon * tangent) - model.decoder(state - epsilon * tangent)) / (2 * epsilon)
    with torch.no_grad():
        actual = model.decoder_jvp(state, tangent)
    assert actual.abs().max() > 0.1
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="torch.no_grad"):
        model.decoder_jvp(state, tangent)


def test_loss_matches_two_distinct_jvps_and_standard_scale_regularizer():
    model = _small_model()
    inputs = _draw_inputs()
    losses = model.losses(**inputs, flow_samples=2, lambda_scale=0.4, return_diagnostics=True)
    clean = model.exact_lift(inputs["action"], inputs["aux_noise"]).repeat_interleave(2, 0)
    noise = inputs["noise"].repeat_interleave(2, 0)
    time = inputs["time"]
    state = (1 - time) * clean + time * noise
    relifted = model.relift(state, inputs["relift_noise"])
    target = jvp(model.decoder, (state,), (noise - clean,))[1]
    prediction = jvp(model.decoder, (relifted,), (model.velocity(relifted, time),))[1]
    action_loss = (prediction - target).square().mean()
    torch.testing.assert_close(losses["action_velocity_loss"], action_loss)
    torch.testing.assert_close(losses["loss"], action_loss + 0.4 * model.scale_loss(inputs["noise"]))
    assert losses["flow_loss"] == 0
    assert losses["reconstruction_loss"] == 0
    assert losses["clean_roundtrip_mse"] < 1e-23
    assert losses["relift_action_mse"] < 1e-23


def test_action_loss_gradients_reach_maps_field_and_all_sample_paths():
    model = _small_model()
    inputs = {name: value.requires_grad_() for name, value in _draw_inputs().items()}
    model.losses(**inputs, flow_samples=2, lambda_scale=0)["loss"].backward()
    for module in (model.gaussian_transform, model.transform, model.field):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 1e-8
    for value in inputs.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 1e-8


def test_learned_reference_target_retains_gradients_into_both_maps():
    model = _small_model()
    inputs = _draw_inputs(samples=1)
    clean = model.exact_lift(inputs["action"], inputs["aux_noise"])
    time = inputs["time"]
    state = (1 - time) * clean + time * inputs["noise"]
    reference = model.decoder_jvp(state, inputs["noise"] - clean)
    reference.square().mean().backward()
    for module in (model.gaussian_transform, model.transform):
        assert sum(parameter.grad.abs().sum() for parameter in module.parameters() if parameter.grad is not None) > 1e-8


def test_fresh_hidden_noise_for_every_expanded_time():
    model = _small_model()
    inputs = _draw_inputs(batch=8, samples=4)
    inputs.pop("relift_noise")
    inputs["time"] = torch.full((32, 1), 0.4, dtype=torch.float64)
    seen = []
    handle = model.field.register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach()))
    try:
        model.losses(**inputs, flow_samples=4)
    finally:
        handle.remove()
    assert len(seen) == 1
    relift_coordinates = model.gaussian_transform(seen[0][:, :8]).reshape(8, 4, 8)
    # Fixed epsilon, eta and time within a group give identical decoded
    # coordinates, but independent xi must differ across all four repeats.
    torch.testing.assert_close(relift_coordinates[:, 1:, :3], relift_coordinates[:, :1, :3].expand(-1, 3, -1), atol=2e-12, rtol=2e-12)
    assert (relift_coordinates[:, 1:, 3:] - relift_coordinates[:, :1, 3:]).abs().sum(-1).min() > 0.1
    with pytest.raises(ValueError, match="relift_noise must have shape"):
        model.losses(**inputs, flow_samples=4, relift_noise=torch.randn(8, 5, dtype=torch.float64))


def test_no_grad_sampler_integrates_latent_ode_with_backward_time():
    model = _small_model()
    noise = torch.randn(6, 8, dtype=torch.float64)
    with torch.no_grad():
        trajectory = model.trajectory(noise, steps=4)
        state = noise
        expected = [model.decoder(state)]
        for step in range(4):
            state = state - 0.25 * model.velocity(state, state.new_full((6, 1), 1 - 0.25 * step))
            expected.append(model.decoder(state))
    assert trajectory.shape == (5, 6, 3)
    assert trajectory.dtype == torch.float64
    assert torch.isfinite(trajectory).all()
    assert (trajectory[-1] - trajectory[0]).abs().max() > 1e-3
    torch.testing.assert_close(trajectory, torch.stack(expected), atol=0, rtol=0)


def test_default_draws_follow_model_dtype_and_diagnostics_are_optional():
    model = _small_model()
    loss = model.losses(torch.randn(4, 3, dtype=torch.float64), flow_samples=2)
    assert all(value.dtype == torch.float64 and torch.isfinite(value) for value in loss.values())
    assert "clean_roundtrip_mse" not in loss
    with pytest.raises(ValueError, match="at least two noise"):
        model.scale_loss(torch.randn(1, 8, dtype=torch.float64))
