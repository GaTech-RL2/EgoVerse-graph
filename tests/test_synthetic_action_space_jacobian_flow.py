import torch
from torch.func import jacrev, vmap

from egomimic.synthetic.action_space_jacobian_flow import (
    SyntheticActionSpaceJacobianFlow,
)


def _small_model(decoder_family: str = "nonlinear"):
    return SyntheticActionSpaceJacobianFlow(
        latent_dim=8,
        action_dim=3,
        decoder_family=decoder_family,
        residual_width=8,
        residual_depth=1,
        field_width=8,
        field_depth=1,
    )


def test_model_has_no_encoder_or_latent_state_bridge():
    model = _small_model()
    assert not hasattr(model, "encoder")
    assert all("encoder" not in name for name, _ in model.named_parameters())
    assert model.field[0].in_features == 4
    assert model.field[-1].out_features == 8


def test_loss_is_action_space_cfm_through_seed_decoder_jacobian():
    model = _small_model()
    with torch.no_grad():
        model.decoder.residual[-1].weight.normal_(std=0.02)
        model.decoder.residual[-1].bias.normal_(std=0.02)
    action = torch.randn(5, 3)
    seed = torch.randn(5, 8)
    time = torch.rand(15, 1)
    losses = model.losses(action, flow_samples=3, seed=seed, time=time)

    seed_many = seed[:, None].expand(-1, 3, -1).reshape(-1, 8)
    action_many = action[:, None].expand(-1, 3, -1).reshape(-1, 3)
    source = model.decoder(seed_many)
    state = (1.0 - time) * action_many + time * source
    jacobian = vmap(jacrev(model.decoder))(seed_many)
    latent_velocity = model.latent_velocity(state, time)
    prediction = torch.einsum("bad,bd->ba", jacobian, latent_velocity)
    expected = (prediction - (source - action_many)).square().mean()

    torch.testing.assert_close(losses["loss"], expected)
    torch.testing.assert_close(losses["flow_loss"], expected)
    torch.testing.assert_close(losses["reconstruction_loss"], torch.tensor(0.0))


def test_only_action_loss_updates_decoder_and_shared_field():
    model = _small_model()
    with torch.no_grad():
        model.decoder.residual[-1].weight.normal_(std=0.02)
    losses = model.losses(torch.randn(7, 3), flow_samples=2)
    losses["loss"].backward()
    for module in (model.decoder, model.field):
        assert any(
            parameter.grad is not None and bool(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.requires_grad
        )


def test_sampling_integrates_actions_while_seed_stays_fixed():
    model = _small_model()
    seed = torch.randn(6, 8)
    trajectory = model.trajectory(seed, steps=4)
    assert trajectory.shape == (5, 6, 3)
    torch.testing.assert_close(trajectory[0], model.decoder(seed))
    assert torch.isfinite(trajectory).all()
