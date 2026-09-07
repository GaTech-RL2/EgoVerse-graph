import pytest
import torch

from egomimic.synthetic.endpoint_lift_flow import SyntheticEndpointLiftFlow


def _model():
    torch.manual_seed(42)
    model = SyntheticEndpointLiftFlow(
        coupling_width=8, coupling_depth=1, field_width=8, field_depth=1
    ).double()
    with torch.no_grad():
        for layer in model.transform.layers:
            layer.net[-1].weight.normal_(std=0.03)
            layer.net[-1].bias.normal_(std=0.03)
    return model


def _batch():
    return dict(
        action=torch.randn(5, 3, dtype=torch.float64),
        noise=torch.randn(5, 8, dtype=torch.float64),
        aux_noise=torch.randn(5, 5, dtype=torch.float64),
        time=torch.rand(15, 1, dtype=torch.float64),
        flow_samples=3,
    )


def _reference(model, batch):
    clean = model.exact_lift(batch["action"], batch["aux_noise"])
    clean_many = clean.repeat_interleave(batch["flow_samples"], dim=0)
    noise_many = batch["noise"].repeat_interleave(batch["flow_samples"], dim=0)
    time = batch["time"]
    target = noise_many - clean_many
    state = (1 - time) * clean_many + time * noise_many
    residual = model.velocity(state, time) - target
    return clean, state, target, residual


def test_exact_lift_recovers_both_action_and_auxiliary_coordinates():
    model = _model()
    batch = _batch()
    clean = model.exact_lift(batch["action"], batch["aux_noise"])
    augmented = torch.cat((batch["action"], batch["aux_noise"]), dim=-1)
    torch.testing.assert_close(model.transform(clean), augmented, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(model.decoder(clean), batch["action"], atol=1e-11, rtol=1e-11)
    latent = torch.randn(5, 8, dtype=torch.float64)
    torch.testing.assert_close(model.transform.inverse(model.transform(latent)), latent)


def test_complete_and_action_jvps_match_finite_differences_under_no_grad():
    model = _model()
    state = torch.randn(5, 8, dtype=torch.float64)
    tangent = torch.randn_like(state)
    epsilon = 1e-5
    with torch.no_grad():
        finite = (model.transform(state + epsilon * tangent) - model.transform(state - epsilon * tangent)) / (2 * epsilon)
        complete = model.interface_jvp(state, tangent)
        torch.testing.assert_close(complete, finite, atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(model.decoder_jvp(state, tangent), finite[:, :3], atol=1e-9, rtol=1e-8)
        assert complete.abs().sum() > 0
        assert (model.decoder_jacobian_singular_values(state) > 0).all()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        model.decoder_jvp(state, tangent)


def test_balanced_minus_latent_is_complete_jvp_loss_with_identical_randomness():
    model = _model()
    batch = _batch()
    latent = model.losses(**batch, objective="latent", lambda_scale=0.4)
    balanced = model.losses(**batch, objective="balanced", lambda_scale=0.4)
    _, state, _, residual = _reference(model, batch)
    complete_loss = model.interface_jvp(state, residual).square().mean()
    torch.testing.assert_close(balanced["loss"] - latent["loss"], complete_loss)
    torch.testing.assert_close(balanced["full_velocity_loss"], complete_loss)
    torch.testing.assert_close(
        balanced["action_velocity_contribution"] + balanced["auxiliary_velocity_contribution"],
        complete_loss,
    )
    torch.testing.assert_close(latent["loss"], residual.square().mean() + 0.4 * model.scale_loss(batch["noise"]))
    assert "full_velocity_loss" not in latent
    assert "clean_roundtrip_mse" not in balanced
    diagnostics = model.losses(**batch, objective="latent", return_diagnostics=True)
    assert diagnostics["clean_roundtrip_mse"] < 1e-20
    assert diagnostics["reconstruction_loss"] == 0


@pytest.mark.parametrize("objective", ["latent", "balanced"])
def test_loss_has_full_gradients_through_lift_state_target_and_interface(objective):
    model = _model()
    batch = _batch()
    actual = model.losses(**batch, objective=objective, lambda_scale=0.0)["loss"]
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))
    clean, state, target, residual = _reference(model, batch)
    reference = residual.square().mean()
    if objective == "balanced":
        reference = reference + model.interface_jvp(state, residual).square().mean()
    expected_gradients = torch.autograd.grad(reference, tuple(model.parameters()), retain_graph=True)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.isfinite(actual_gradient).all()
    for tensor in (clean, state, target):
        gradient = torch.autograd.grad(reference, tensor, retain_graph=True)[0]
        assert gradient.abs().sum() > 0
    # This also rules out the common false positive where regularization alone
    # reaches the interface: lambda_scale above is exactly zero.
    interface_count = len(tuple(model.transform.parameters()))
    assert sum(g.abs().sum() for g in actual_gradients[:interface_count]) > 0
    assert sum(g.abs().sum() for g in actual_gradients[interface_count:]) > 0


def test_scale_regularizer_matches_existing_mean_and_covariance_convention():
    model = _model()
    noise = _batch()["noise"]
    decoded = model.decoder(noise)
    expected = decoded.mean(0).square().sum() / 3
    expected += (torch.cov(decoded.T) - torch.eye(3, dtype=noise.dtype)).square().sum() / 3
    torch.testing.assert_close(model.scale_loss(noise), expected)


def test_sampler_is_reverse_latent_euler_and_preserves_float64():
    model = _model()
    noise = _batch()["noise"]
    with torch.no_grad():
        actual = model.trajectory(noise, steps=4)
        state = noise.clone()
        expected = [model.decoder(state)]
        for index in range(4):
            time = noise.new_full((len(noise), 1), 1 - index / 4)
            state = state - model.velocity(state, time) / 4
            expected.append(model.decoder(state))
    torch.testing.assert_close(actual, torch.stack(expected))
    assert actual.shape == (5, 5, 3)
    assert actual.dtype == torch.float64
    assert torch.isfinite(actual).all()
    assert (actual[-1] - actual[0]).abs().sum() > 0


def test_default_sampled_inputs_preserve_model_dtype():
    model = _model()
    losses = model.losses(torch.randn(5, 3, dtype=torch.float64), objective="balanced")
    assert all(value.dtype == torch.float64 for value in losses.values())
    assert all(torch.isfinite(value) for value in losses.values())
