"""The complete corrected action velocity must never depend on source labels."""

import pytest
import torch
from torch.func import jacrev, vmap

from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.action_space_jacobian_flow import SyntheticActionSpaceJacobianFlow


def small_model(mode="action_state", family="nonlinear"):
    model = SyntheticActionSpaceJacobianFlow(
        latent_dim=8, action_dim=3, decoder_family=family,
        residual_width=8, residual_depth=2, field_width=8, field_depth=2,
        jacobian_basepoint=mode,
    )
    if family == "nonlinear":
        with torch.no_grad():
            model.decoder.residual[-1].weight.normal_(std=0.3)
    return model


def test_nonlinear_state_jvp_matches_explicit_jacobian():
    model = small_model().double()
    state, seed, time = torch.randn(7, 3, dtype=torch.float64), torch.randn(7, 8, dtype=torch.float64), torch.rand(7, 1, dtype=torch.float64)
    base = torch.nn.functional.pad(state, (0, 5))
    expected = torch.einsum(
        "bad,bd->ba", vmap(jacrev(model.decoder))(base), model.latent_velocity(state, time)
    )
    torch.testing.assert_close(model.action_velocity(state, seed, time), expected)
    assert model._action_lift.dtype == torch.float64


def test_fixed_state_velocity_is_invariant_to_original_seed():
    model = small_model()
    state, time = torch.randn(7, 3), torch.rand(7, 1)
    first_seed, second_seed = torch.randn(7, 8), torch.randn(7, 8) * 5
    expected = model.action_velocity(state, first_seed, time)
    torch.testing.assert_close(
        model.action_velocity(state, second_seed, time), expected, rtol=0, atol=0
    )
    # Make sure this nonlinearly perturbed fixture detects the historical bug.
    model.jacobian_basepoint = "seed"
    assert not torch.allclose(
        model.action_velocity(state, first_seed, time),
        model.action_velocity(state, second_seed, time),
    )


def test_same_decoded_start_produces_identical_full_paths():
    class SquaredDecoder(torch.nn.Module):
        def forward(self, z):
            return z[..., :3].square()
    model = small_model()
    model.decoder = SquaredDecoder()
    seed = torch.randn(7, 8)
    with torch.no_grad():
        torch.testing.assert_close(model.trajectory(seed, 5), model.trajectory(-seed, 5), rtol=0, atol=0)


def test_sampler_recomputes_basis_from_every_current_action(monkeypatch):
    model = small_model()
    observed = []
    original = model.decoder_jvp
    def record(base, tangent):
        observed.append(base.detach().clone())
        return original(base, tangent)
    monkeypatch.setattr(model, "decoder_jvp", record)
    with torch.no_grad():
        points = model.trajectory(torch.randn(7, 8), steps=4)
    assert len(observed) == 4
    for index, base in enumerate(observed):
        torch.testing.assert_close(base, torch.nn.functional.pad(points[index], (0, 5)))
    assert not torch.equal(observed[0], observed[-1])


def test_state_loss_and_all_attached_parameter_gradients_match_explicit_formula():
    model = small_model().double()
    action, seed, time = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 8, dtype=torch.float64), torch.rand(10, 1, dtype=torch.float64)
    actual = model.losses(action, seed=seed, time=time, flow_samples=2, lambda_scale=1)
    noises = seed[:, None].expand(-1, 2, -1).reshape(-1, 8)
    actions = action[:, None].expand(-1, 2, -1).reshape(-1, 3)
    decoded = model.decoder(noises)
    state = (1 - time) * actions + time * decoded
    jacobian = vmap(jacrev(model.decoder))(torch.nn.functional.pad(state, (0, 5)))
    predicted = torch.einsum("bad,bd->ba", jacobian, model.latent_velocity(state, time))
    expected_flow = (predicted - (decoded - actions)).square().mean()
    torch.testing.assert_close(actual["flow_loss"], expected_flow)
    params = tuple(model.parameters())
    actual_grads = torch.autograd.grad(actual["flow_loss"], params, retain_graph=True)
    expected_grads = torch.autograd.grad(expected_flow, params, retain_graph=True)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-8)
    for module in (model.decoder, model.field):
        grads = torch.autograd.grad(actual["flow_loss"], tuple(module.parameters()), retain_graph=True)
        assert sum(float(grad.abs().sum()) for grad in grads) > 0
    assert not hasattr(model, "encoder")
    assert float(actual["reconstruction_loss"]) == 0


def test_affine_case_is_same_constant_matrix_field_for_either_mode():
    model = small_model(family="joint_affine")
    state, seed, time = torch.randn(7, 3), torch.randn(7, 8), torch.rand(7, 1)
    expected = model.latent_velocity(state, time) @ model.decoder.weight.T
    torch.testing.assert_close(model.action_velocity(state, seed, time), expected)
    model.jacobian_basepoint = "seed"
    torch.testing.assert_close(model.action_velocity(state, seed, time), expected)


def test_legacy_checkpoint_keys_and_initial_parameters_unchanged():
    torch.manual_seed(42)
    legacy = SyntheticActionSpaceJacobianFlow()
    torch.manual_seed(42)
    current = SyntheticActionSpaceJacobianFlow(jacobian_basepoint="action_state")
    assert legacy.state_dict().keys() == current.state_dict().keys()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
    current.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(p.numel() for p in current.parameters()) == 52678
    assert sum(p.numel() for p in current.parameters() if p.requires_grad) == 52678


def test_state_basis_diagnostic_detects_rank_loss_hidden_by_noise_scale():
    class NullAtLift(torch.nn.Module):
        def forward(self, z):
            return (z[..., 3:6].square() - 1) / (2 ** 0.5)
    model = small_model()
    model.decoder = NullAtLift()
    action, seed = torch.randn(7, 3), torch.randn(7, 8)
    assert bool((model.decoder_jacobian_singular_values(seed) > 0).all())
    assert not bool(model.velocity_basis_singular_values(action, seed).any())


def test_shared_evaluator_keeps_corrected_jvp_active():
    model = small_model(family="joint_affine")
    with torch.no_grad():
        for p in model.field.parameters():
            p.zero_()
        model.field[-1].bias.fill_(1)
    points = SyntheticTrajectoryEval.evaluate(model, torch.randn(7, 8), torch.randn(7, 3), steps=4)
    assert float((points[-1] - points[0]).abs().max()) > 0


def test_invalid_basis_mode_is_rejected():
    with pytest.raises(ValueError, match="Jacobian basepoint"):
        small_model(mode="unresolved")
