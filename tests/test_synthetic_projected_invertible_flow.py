import torch

from egomimic.synthetic.projected_invertible_flow import (
    SyntheticProjectedInvertibleFlow,
)


def _small_model():
    return SyntheticProjectedInvertibleFlow(
        latent_dim=8,
        action_dim=3,
        coupling_layers=4,
        coupling_width=8,
        coupling_depth=1,
        field_width=8,
        field_depth=1,
    )


def test_projection_is_orthonormal_and_lift_roundtrips_exactly():
    model = _small_model()
    with torch.no_grad():
        model.raw_projection.add_(0.1 * torch.randn_like(model.raw_projection))
        for layer in model.transform.layers:
            layer.net[-1].weight.normal_(std=0.02)
            layer.net[-1].bias.normal_(std=0.02)
    projection = model.projection()
    torch.testing.assert_close(
        projection @ projection.T,
        torch.eye(3),
        atol=2e-6,
        rtol=2e-6,
    )
    action = torch.randn(11, 3)
    clean = model.exact_lift(action, torch.randn(11, 8))
    torch.testing.assert_close(model.decode(clean), action, atol=2e-6, rtol=2e-6)


def test_decoder_jvp_matches_centered_finite_difference():
    model = _small_model()
    with torch.no_grad():
        for layer in model.transform.layers:
            layer.net[-1].weight.normal_(std=0.02)
    state = torch.randn(7, 8)
    tangent = torch.randn(7, 8)
    epsilon = 1e-3
    finite_difference = (
        model.decode(state + epsilon * tangent)
        - model.decode(state - epsilon * tangent)
    ) / (2 * epsilon)
    torch.testing.assert_close(
        model.decoder_jvp(state, tangent),
        finite_difference,
        atol=5e-4,
        rtol=5e-4,
    )


def test_only_action_and_scale_losses_are_active_for_proposed_diagnostic():
    model = _small_model()
    action = torch.randn(5, 3)
    noise = torch.randn(5, 8)
    null_noise = torch.randn(5, 8)
    time = torch.rand(15, 1)
    losses = model.losses(
        action,
        flow_samples=3,
        lambda_scale=0.4,
        lambda_latent_flow=0.0,
        noise=noise,
        null_noise=null_noise,
        time=time,
    )
    expected = losses["action_velocity_loss"] + 0.4 * losses["scale_loss"]
    torch.testing.assert_close(losses["loss"], expected)
    torch.testing.assert_close(losses["reconstruction_loss"], torch.tensor(0.0))
    torch.testing.assert_close(losses["clean_roundtrip_mse"], torch.tensor(0.0))


def test_action_loss_gradients_reach_projection_transform_and_field():
    model = _small_model()
    with torch.no_grad():
        for layer in model.transform.layers:
            layer.net[-1].weight.normal_(std=0.02)
    losses = model.losses(
        torch.randn(7, 3),
        flow_samples=2,
        lambda_scale=0.0,
        lambda_latent_flow=0.0,
    )
    losses["loss"].backward()
    assert model.raw_projection.grad is not None
    assert bool(model.raw_projection.grad.abs().sum())
    for module in (model.transform, model.field):
        assert any(
            parameter.grad is not None and bool(parameter.grad.abs().sum())
            for parameter in module.parameters()
        )


def test_trajectory_integrates_shared_latent_and_decodes_each_state():
    model = _small_model()
    trajectory = model.trajectory(torch.randn(6, 8), steps=4)
    assert trajectory.shape == (5, 6, 3)
    assert torch.isfinite(trajectory).all()
