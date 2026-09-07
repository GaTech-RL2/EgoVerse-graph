import pytest
import torch

from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow
from egomimic.synthetic.gradient_surgery import (
    backward_with_encoder_gradient_surgery,
    project_flow_gradient_against_reconstruction,
)


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


def test_clean_gradient_modes_keep_forward_loss_identical_and_change_encoder_route():
    model = SyntheticActionAdapterFlow(
        latent_dim=8, adapter_family="nonlinear", field_width=16, field_depth=2
    )
    action, noise, time = torch.randn(12, 3), torch.randn(12, 8), torch.rand(36, 1)
    losses = {}
    gradients = {}
    for mode in ("full", "target_stopgrad", "all_stopgrad"):
        model.zero_grad(set_to_none=True)
        losses[mode] = model.losses(
            action,
            objective="action_velocity",
            flow_samples=3,
            noise=noise,
            time=time,
            clean_gradient_mode=mode,
        )["flow_loss"]
        losses[mode].backward()
        gradients[mode] = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
    torch.testing.assert_close(losses["full"], losses["target_stopgrad"])
    torch.testing.assert_close(losses["full"], losses["all_stopgrad"])
    assert gradients["full"] > 0
    assert gradients["target_stopgrad"] > 0
    assert gradients["all_stopgrad"] == 0


def test_all_stopgrad_also_removes_action_velocity_encoder_gradient():
    model = SyntheticActionAdapterFlow(
        latent_dim=8, adapter_family="nonlinear", field_width=16, field_depth=2
    )
    _perturb_residual_outputs(model)
    action, noise, time = torch.randn(12, 3), torch.randn(12, 8), torch.rand(36, 1)
    gradients = {}
    for mode in ("target_stopgrad", "all_stopgrad"):
        model.zero_grad(set_to_none=True)
        action_velocity_loss = model.losses(
            action,
            objective="action_velocity",
            flow_samples=3,
            noise=noise,
            time=time,
            clean_gradient_mode=mode,
        )["action_velocity_loss"]
        action_velocity_loss.backward()
        gradients[mode] = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
    assert gradients["target_stopgrad"] > 0
    assert gradients["all_stopgrad"] == 0


def test_unknown_clean_gradient_mode_is_rejected():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="nonlinear")
    with pytest.raises(ValueError, match="unknown clean gradient mode"):
        model.losses(
            torch.randn(8, 3), objective="action_velocity", clean_gradient_mode="bad"
        )


def test_asymmetric_gradient_projection_removes_only_conflicting_component():
    flow = (torch.tensor([-2.0, 3.0]), torch.tensor([1.0]))
    reconstruction = (torch.tensor([1.0, 0.0]), torch.tensor([0.0]))
    projected, metrics = project_flow_gradient_against_reconstruction(
        flow, reconstruction
    )

    torch.testing.assert_close(projected[0], torch.tensor([0.0, 3.0]))
    torch.testing.assert_close(projected[1], flow[1])
    assert metrics["encoder_flow_projection_active"] == 1
    assert metrics["encoder_flow_reconstruction_gradient_cosine"] < 0

    aligned, aligned_metrics = project_flow_gradient_against_reconstruction(
        (torch.tensor([2.0, 3.0]),), (torch.tensor([1.0, 0.0]),)
    )
    torch.testing.assert_close(aligned[0], torch.tensor([2.0, 3.0]))
    assert aligned_metrics["encoder_flow_projection_active"] == 0


def test_gradient_surgery_replaces_only_encoder_flow_gradient():
    model = SyntheticActionAdapterFlow(
        latent_dim=8,
        adapter_family="nonlinear",
        residual_width=8,
        residual_depth=1,
        field_width=8,
        field_depth=1,
    )
    _perturb_residual_outputs(model)
    losses = model.losses(
        torch.randn(7, 3),
        objective="action_velocity",
        flow_samples=2,
        lambda_reconstruction=1.0,
    )
    parameters = tuple(model.encoder.parameters())
    flow = torch.autograd.grad(losses["flow_loss"], parameters, retain_graph=True)
    reconstruction = torch.autograd.grad(
        losses["reconstruction_loss"], parameters, retain_graph=True
    )
    total = torch.autograd.grad(losses["loss"], parameters, retain_graph=True)
    projected, _ = project_flow_gradient_against_reconstruction(
        flow, reconstruction
    )

    model.zero_grad(set_to_none=True)
    metrics = backward_with_encoder_gradient_surgery(
        losses, model, "protect_reconstruction_from_flow"
    )

    for parameter, total_gradient, flow_gradient, projected_gradient in zip(
        parameters, total, flow, projected, strict=True
    ):
        torch.testing.assert_close(
            parameter.grad,
            total_gradient - flow_gradient + projected_gradient,
        )
    projected_dot = sum(
        (flow_gradient * reconstruction_gradient).sum()
        for flow_gradient, reconstruction_gradient in zip(
            projected, reconstruction, strict=True
        )
    )
    assert projected_dot >= -1e-6
    assert set(metrics) == {
        "encoder_flow_reconstruction_gradient_cosine",
        "encoder_flow_projection_active",
        "encoder_flow_gradient_norm",
        "encoder_flow_projected_gradient_norm",
        "encoder_reconstruction_gradient_norm",
    }


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


def test_action_velocity_loss_matches_explicit_decoder_jacobian_product():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="nonlinear")
    _perturb_residual_outputs(model)
    state = torch.randn(7, 8)
    residual = torch.randn(7, 8)

    jacobians = torch.func.vmap(torch.func.jacrev(model.decoder))(state)
    explicit_action_residual = torch.einsum("bai,bi->ba", jacobians, residual)
    expected = explicit_action_residual.square().sum(dim=-1).mean() / 3.0

    torch.testing.assert_close(model.action_velocity_loss(state, residual), expected)


def test_action_velocity_objective_reaches_encoder_decoder_and_field():
    model = SyntheticActionAdapterFlow(
        latent_dim=8,
        adapter_family="nonlinear",
        field_width=16,
        field_depth=2,
    )
    _perturb_residual_outputs(model)
    losses = model.losses(
        torch.randn(10, 3),
        objective="action_velocity",
        flow_samples=2,
        lambda_reconstruction=0.7,
        lambda_scale=0.2,
        lambda_action_velocity=0.4,
    )
    losses["action_velocity_loss"].backward()

    for module in (model.encoder, model.decoder, model.field):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert any(
            gradient is not None and bool(gradient.abs().sum())
            for gradient in gradients
        )


def test_action_velocity_objective_combines_reconstruction_scale_and_metric_loss():
    model = SyntheticActionAdapterFlow(latent_dim=8, adapter_family="joint_affine")
    losses = model.losses(
        torch.randn(9, 3),
        objective="action_velocity",
        lambda_reconstruction=0.7,
        lambda_scale=0.2,
        lambda_action_velocity=0.4,
    )
    expected = (
        losses["flow_loss"]
        + 0.7 * losses["reconstruction_loss"]
        + 0.2 * losses["scale_loss"]
        + 0.4 * losses["action_velocity_loss"]
    )
    torch.testing.assert_close(losses["loss"], expected)


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
