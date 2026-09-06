import pytest
import torch

from egomimic.synthetic.decoder_inversion_flow import SyntheticDecoderInversionFlow


def _small_model(decoder_family: str) -> SyntheticDecoderInversionFlow:
    return SyntheticDecoderInversionFlow(
        latent_dim=8,
        decoder_family=decoder_family,
        residual_width=8,
        residual_depth=1,
        field_width=8,
        field_depth=1,
    )


def test_model_has_no_encoder_and_affine_inversion_is_exact_in_one_step():
    model = _small_model("joint_affine")
    assert not hasattr(model, "encoder")
    assert all("encoder" not in name for name, _ in model.named_parameters())

    action = torch.randn(11, 3)
    initialization = torch.randn(11, 8)
    code, metrics = model.infer_codes(
        action,
        initialization,
        steps=1,
        step_size=1.0,
        create_graph=True,
    )
    torch.testing.assert_close(model.decoder(code), action, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        metrics["inversion_after_mse"], torch.tensor(0.0), atol=1e-12, rtol=0
    )


def test_loss_matches_action_endpoint_objective_and_has_no_outer_reconstruction():
    model = _small_model("joint_affine")
    action = torch.randn(5, 3)
    noise = torch.randn(5, 8)
    initialization = torch.randn(5, 8)
    time = torch.rand(15, 1)
    losses = model.losses(
        action,
        flow_samples=3,
        inversion_steps=1,
        inversion_step_size=1.0,
        lambda_scale=0.4,
        noise=noise,
        time=time,
        initialization=initialization,
    )
    clean, _ = model.infer_codes(
        action,
        initialization.detach().clone(),
        steps=1,
        step_size=1.0,
        create_graph=True,
    )
    clean_many = clean[:, None].expand(-1, 3, -1).reshape(-1, 8)
    noise_many = noise[:, None].expand(-1, 3, -1).reshape(-1, 8)
    action_many = action[:, None].expand(-1, 3, -1).reshape(-1, 3)
    state = (1.0 - time) * clean_many + time * noise_many
    expected_action_loss = (
        model.decoder_jvp(state, model.velocity(state, time))
        - (model.decoder(noise_many) - action_many)
    ).square().mean()
    torch.testing.assert_close(losses["action_velocity_loss"], expected_action_loss)
    torch.testing.assert_close(losses["reconstruction_loss"], torch.tensor(0.0))
    torch.testing.assert_close(losses["flow_loss"], torch.tensor(0.0))
    torch.testing.assert_close(
        losses["loss"], expected_action_loss + 0.4 * losses["scale_loss"]
    )


def test_nonlinear_action_loss_backpropagates_through_inversion_to_decoder_and_field():
    model = _small_model("nonlinear")
    with torch.no_grad():
        model.decoder.residual[-1].weight.normal_(std=0.02)
        model.decoder.residual[-1].bias.normal_(std=0.02)
    losses = model.losses(
        torch.randn(7, 3),
        flow_samples=2,
        inversion_steps=2,
        inversion_step_size=0.5,
        lambda_scale=0.0,
    )
    losses["loss"].backward()
    for module in (model.decoder, model.field):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        assert any(
            gradient is not None and bool(gradient.abs().sum())
            for gradient in gradients
        )


@pytest.mark.parametrize(("steps", "step_size"), ((0, 0.5), (1, 0.0), (1, -0.1)))
def test_inversion_rejects_invalid_hyperparameters(steps, step_size):
    model = _small_model("joint_affine")
    with pytest.raises(ValueError):
        model.infer_codes(
            torch.randn(3, 3),
            torch.randn(3, 8),
            steps=steps,
            step_size=step_size,
            create_graph=False,
        )


def test_reverse_trajectory_decodes_every_latent_state():
    model = _small_model("nonlinear")
    trajectory = model.trajectory(torch.randn(6, 8), steps=4)
    assert trajectory.shape == (5, 6, 3)
    assert torch.isfinite(trajectory).all()
