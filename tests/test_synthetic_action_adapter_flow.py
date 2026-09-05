import pytest
import torch

from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow


def _perturb_residual_outputs(model: SyntheticActionAdapterFlow) -> None:
    with torch.no_grad():
        for adapter in (model.encoder, model.decoder):
            adapter.residual[-1].weight.normal_(std=0.02)
            adapter.residual[-1].bias.normal_(std=0.02)


def test_fixed_lift_reconstructs_and_projects_unit_noise_exactly():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="fixed_affine")
    action = torch.randn(11, 3)
    torch.testing.assert_close(model.decoder(model.encoder(action)), action)
    torch.testing.assert_close(
        model.decoder.weight @ model.decoder.weight.T, torch.eye(3)
    )
    torch.testing.assert_close(model.scale_loss(torch.randn(16, 8)), torch.tensor(0.0))
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.decoder.parameters())


def test_affine_path_identity_equals_reconstruction():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="joint_affine")
    with torch.no_grad():
        model.encoder.weight.add_(0.05 * torch.randn_like(model.encoder.weight))
        model.decoder.weight.add_(0.05 * torch.randn_like(model.decoder.weight))
        model.encoder.bias.add_(0.02 * torch.randn_like(model.encoder.bias))
        model.decoder.bias.add_(0.02 * torch.randn_like(model.decoder.bias))
    action = torch.randn(13, 3)
    noise = torch.randn(13, 8)
    time = torch.rand(13, 1)
    torch.testing.assert_close(
        model.path_consistency_loss(action, noise, time),
        model.reconstruction_loss(action),
        rtol=2e-5,
        atol=2e-6,
    )


def test_nonlinear_adapters_start_at_the_same_fixed_lift():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="nonlinear")
    action = torch.randn(9, 3)
    torch.testing.assert_close(model.decoder(model.encoder(action)), action)


def test_nonlinear_decoder_jvp_matches_centered_finite_difference():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="nonlinear")
    _perturb_residual_outputs(model)
    state = torch.randn(7, 8)
    tangent = torch.randn(7, 8)
    epsilon = 1e-3
    finite_difference = (
        model.decoder(state + epsilon * tangent)
        - model.decoder(state - epsilon * tangent)
    ) / (2 * epsilon)
    torch.testing.assert_close(
        model.decoder_jvp(state, tangent),
        finite_difference,
        rtol=3e-3,
        atol=3e-4,
    )


def test_flow_gradient_reaches_trainable_encoder_through_state_and_target():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="joint_affine")
    losses = model.losses(
        torch.randn(12, 3), objective="reconstruction", flow_samples=3
    )
    losses["flow_loss"].backward()
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())
    assert (
        sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
        )
        > 0
    )


def test_loss_uses_the_supplied_fixed_noise_cloud_for_all_flow_samples():
    model = SyntheticActionAdapterFlow(
        latent_dim=8, adapter_family="fixed_affine", field_width=16, field_depth=2
    )
    action = torch.randn(5, 3)
    noise = torch.randn(5, 8)
    time = torch.rand(15, 1)
    losses = model.losses(
        action,
        objective="none",
        flow_samples=3,
        noise=noise,
        time=time,
    )
    clean = model.encoder(action)
    clean_many = clean[:, None].expand(-1, 3, -1).reshape(-1, 8)
    noise_many = noise[:, None].expand(-1, 3, -1).reshape(-1, 8)
    state = (1.0 - time) * clean_many + time * noise_many
    expected = (model.velocity(state, time) - (noise_many - clean_many)).square().mean()
    torch.testing.assert_close(losses["flow_loss"], expected)


def test_path_loss_gradients_reach_both_nonlinear_adapters():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="nonlinear")
    _perturb_residual_outputs(model)
    losses = model.losses(torch.randn(10, 3), objective="path", flow_samples=2)
    losses["path_loss"].backward()
    for adapter in (model.encoder, model.decoder):
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ]
        assert any(
            gradient is not None and bool(gradient.abs().sum())
            for gradient in gradients
        )


def test_affine_path_objective_is_rejected_as_duplicate():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="joint_affine")
    with pytest.raises(ValueError, match="duplicates reconstruction"):
        model.losses(torch.randn(8, 3), objective="path")


def test_reverse_time_trajectory_decodes_every_state():
    model = SyntheticActionAdapterFlow(
        latent_dim=8, adapter_family="fixed_affine", field_width=16, field_depth=2
    )
    trajectory = model.trajectory(torch.randn(6, 8), steps=4)
    assert trajectory.shape == (5, 6, 3)
    assert torch.isfinite(trajectory).all()
