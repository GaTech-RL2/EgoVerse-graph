"""Guards for the complete likelihood objective and its actual stochastic sampler."""

import copy

import pytest
import torch

from egomimic.synthetic.latent_bridge_likelihood import SyntheticLatentBridgeLikelihood


def _model(levels=4):
    torch.manual_seed(117)
    return SyntheticLatentBridgeLikelihood(
        levels=levels,
        encoder_width=8,
        encoder_depth=2,
        decoder_width=8,
        decoder_depth=2,
        field_width=12,
        field_depth=2,
        sampler_seed=818,
    ).double()


def test_terminal_mean_is_exactly_zero_independent_of_action_and_parameters():
    model = _model()
    action = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    terminal_mean = model.bridge_mean(action, model.levels)
    assert torch.equal(terminal_mean, torch.zeros_like(terminal_mean))
    assert model.sigmas[-1].item() == 1.0
    terminal_mean.sum().backward()
    assert torch.equal(action.grad, torch.zeros_like(action.grad))
    for parameter in model.mean_network.parameters():
        assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
    noise = torch.randn(7, 8, dtype=torch.float64)
    state, _, _ = model.reference_transition(action, model.levels, noise)
    torch.testing.assert_close(state, noise, atol=0, rtol=0)


def test_geometric_schedule_and_reference_marginal_identity():
    model = _model(levels=8)
    torch.testing.assert_close(model.sigmas[0], torch.tensor(0.1, dtype=torch.float64), atol=2e-9, rtol=0)
    ratios = model.sigmas[1:] / model.sigmas[:-1]
    torch.testing.assert_close(ratios, ratios.mean().expand_as(ratios), atol=2e-7, rtol=0)
    action = torch.randn(7, 3, dtype=torch.float64)
    noise = torch.randn(7, 8, dtype=torch.float64)
    levels = torch.arange(2, 9)
    state, target, variance = model.reference_transition(action, levels, noise)
    current_mean = model.bridge_mean(action, levels)
    previous_mean = model.bridge_mean(action, levels - 1)
    current_sigma = model.sigmas[levels - 1, None]
    previous_sigma = model.sigmas[levels - 2, None]
    torch.testing.assert_close(target, previous_mean + model.rho * previous_sigma / current_sigma * (state - current_mean))
    torch.testing.assert_close(model.rho ** 2 * previous_sigma.square() + variance, previous_sigma.square())


def test_interior_regression_equals_full_gaussian_kl_without_dimension_average():
    model = _model()
    action = torch.randn(6, 3, dtype=torch.float64)
    noise = torch.randn(6, 8, dtype=torch.float64)
    levels = torch.tensor([2, 3, 4, 2, 3, 4])
    state, target, variance = model.reference_transition(action, levels, noise)
    prediction = model.reverse_mean(state, levels)
    reference = torch.distributions.Independent(torch.distributions.Normal(target, variance.sqrt()), 1)
    generative = torch.distributions.Independent(torch.distributions.Normal(prediction, variance.sqrt()), 1)
    expected = torch.distributions.kl_divergence(reference, generative)
    actual = model.interior_kl_terms(action, levels, noise)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-12)
    assert actual.mean() > 1.0


def test_uniform_level_estimator_recovers_sum_and_boundary_likelihood(monkeypatch):
    model = _model()
    action = torch.randn(2, 3, dtype=torch.float64)
    boundary_noise = torch.randn(2, 8, dtype=torch.float64)
    levels = torch.tensor([2, 3, 4, 2, 3, 4])
    interior_noise = torch.randn(6, 8, dtype=torch.float64)
    original_randn = torch.randn
    seen_shapes = []

    def fixed_randint(low, high, size, *, device):
        assert (low, high, size) == (2, model.levels + 1, (6,))
        return levels.to(device)

    def fixed_randn(*shape, **kwargs):
        seen_shapes.append(shape)
        if shape == (6, 8):
            return interior_noise.to(**kwargs)
        return original_randn(*shape, **kwargs)

    monkeypatch.setattr(torch, "randint", fixed_randint)
    monkeypatch.setattr(torch, "randn", fixed_randn)
    losses = model.losses(action, flow_samples=3, noise=boundary_noise)
    expected_kl = model.interior_kl_terms(action.repeat_interleave(3, 0), levels, interior_noise).reshape(2, 3).sum(1).mean()
    prediction = model.boundary_prediction(action, boundary_noise)
    boundary_nll = -torch.distributions.Independent(torch.distributions.Normal(prediction, model.tau), 1).log_prob(action)
    normalization = 0.5 * model.action_dim * torch.log(torch.tensor(2 * torch.pi * model.tau ** 2, dtype=torch.float64))
    torch.testing.assert_close(losses["interior_kl_loss"], expected_kl)
    torch.testing.assert_close(losses["endpoint_nll_loss"], boundary_nll.mean() - normalization)
    torch.testing.assert_close(losses["loss"], expected_kl + losses["endpoint_nll_loss"])
    assert seen_shapes == [(6, 8)]
    assert "scale_loss" not in losses
    assert "flow_loss" not in losses
    assert "action_velocity_loss" not in losses
    assert "reconstruction_loss" not in losses


def test_boundary_likelihood_updates_private_encoder_shared_denoiser_and_decoder():
    model = _model()
    action = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    noise = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    losses = model.losses(action, flow_samples=2, noise=noise)
    losses["endpoint_nll_loss"].backward()
    for module in (model.mean_network, model.field, model.decoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 1e-6
    assert action.grad is not None and action.grad.abs().sum() > 1e-6
    assert noise.grad is not None and noise.grad.abs().sum() > 1e-6


def test_interior_target_and_state_both_retain_learned_mean_gradients():
    model = _model()
    action = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    noise = torch.randn(5, 8, dtype=torch.float64)
    state, target, _ = model.reference_transition(action, 3, noise)
    parameters = tuple(model.mean_network.parameters())
    state_grad = torch.autograd.grad(state.square().sum(), parameters, retain_graph=True)
    target_grad = torch.autograd.grad(target.square().sum(), parameters)
    assert all(torch.isfinite(gradient).all() for gradient in state_grad + target_grad)
    assert sum(gradient.abs().sum() for gradient in state_grad) > 1e-6
    assert sum(gradient.abs().sum() for gradient in target_grad) > 1e-6


def test_sampler_replays_all_independent_innovations_and_output_noise():
    model = _model()
    noise = torch.randn(6, 8, dtype=torch.float64)
    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        actual = model.trajectory(noise, steps=4)
        torch.testing.assert_close(model.trajectory(noise, steps=4), actual, atol=0, rtol=0)
        torch.testing.assert_close(model.integrate(noise, steps=4), actual[-1], atol=0, rtol=0)
        generator = torch.Generator().manual_seed(model.sampler_seed)
        state = noise
        expected = [model.decoder(state)]
        for level in range(4, 1, -1):
            innovation = torch.randn(state.shape, dtype=state.dtype, generator=generator)
            scale = model.sigmas[level - 2] * (1 - model.rho ** 2) ** 0.5
            state = model.reverse_mean(state, level) + scale * innovation
            expected.append(model.decoder(state))
        final_mean = model.decoder(model.reverse_mean(state, 1))
        output_noise = torch.randn(final_mean.shape, dtype=state.dtype, generator=generator)
        expected.append(final_mean + model.tau * output_noise)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert actual.shape == (5, 6, 3)
    torch.testing.assert_close(actual, torch.stack(expected), atol=0, rtol=0)
    torch.testing.assert_close(actual[-1] - final_mean, model.tau * output_noise, atol=2e-16, rtol=2e-13)
    assert (actual[-1] - final_mean).abs().sum() > 0.01
    assert (actual[1] - model.decoder(model.reverse_mean(noise, 4))).abs().sum() > 0.01


def test_explicit_sampler_generator_changes_draws_and_consumes_its_state():
    model = _model()
    noise = torch.randn(5, 8, dtype=torch.float64)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        first = model.trajectory(noise, steps=4, generator=generator)
        second = model.trajectory(noise, steps=4, generator=generator)
        replay = model.trajectory(noise, steps=4, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(first, replay, atol=0, rtol=0)
    assert (first[-1] - second[-1]).abs().max() > 0.02
    with pytest.raises(ValueError, match="steps == levels"):
        model.trajectory(noise, steps=3)
    assert not hasattr(model, "velocity")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device replay requires CUDA")
def test_cpu_and_cuda_default_samplers_share_innovation_stream():
    cpu_model = _model()
    cuda_model = copy.deepcopy(cpu_model).cuda()
    noise = torch.randn(11, 8, dtype=torch.float64)
    with torch.no_grad():
        cpu_points = cpu_model.trajectory(noise, steps=4)
        cuda_points = cuda_model.trajectory(noise.cuda(), steps=4).cpu()
    torch.testing.assert_close(cpu_points, cuda_points, atol=2e-11, rtol=2e-11)


def test_default_architecture_parameter_count_and_sampler_budget():
    model = SyntheticLatentBridgeLikelihood()
    assert sum(parameter.numel() for parameter in model.parameters()) == 54798
    assert model.levels == 32
    assert model.output_sigma == 0.02
    assert model.sigmas.shape == (32,)


def test_finite_dtype_losses_optimizer_and_fixed_schedule_parameters():
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    action = torch.randn(9, 3, dtype=torch.float64)
    losses = model.losses(action, flow_samples=3, return_diagnostics=True)
    assert all(value.dtype == torch.float64 and bool(torch.isfinite(value)) for value in losses.values())
    losses["loss"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    before = [parameter.detach().clone() for parameter in model.field.parameters()]
    optimizer.step()
    assert any(not torch.equal(previous, current) for previous, current in zip(before, model.field.parameters()))
    assert not model.sigmas.requires_grad
    assert "decoded_noise_scale_diagnostic" in losses
    assert losses["sigma_terminal"] == 1
    assert not any(name.startswith("sigmas") for name, _ in model.named_parameters())


def test_decoder_derivative_diagnostics_survive_no_grad_and_reject_inference_mode():
    model = _model()
    state = torch.randn(5, 8, dtype=torch.float64)
    tangent = torch.randn_like(state)
    with torch.no_grad():
        actual = model.decoder_jvp(state, tangent)
        singular_values = model.decoder_jacobian_singular_values(state)
    torch.testing.assert_close(actual, tangent[:, :3])
    torch.testing.assert_close(singular_values, torch.ones(5, 3, dtype=torch.float64))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="torch.no_grad"):
        model.decoder_jvp(state, tangent)


def test_discrete_and_shape_contracts_fail_explicitly():
    model = _model()
    action = torch.randn(4, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="continuous time"):
        model.losses(action, time=torch.rand(4, 1))
    with pytest.raises(ValueError, match="boundary action batch"):
        model.losses(action, noise=torch.randn(8, 8))
    with pytest.raises(ValueError, match="positive"):
        model.losses(action, flow_samples=0)
    with pytest.raises(ValueError, match="integers"):
        model.bridge_mean(action, torch.ones(4))
    with pytest.raises(ValueError, match="at least two"):
        SyntheticLatentBridgeLikelihood(levels=1)
    with pytest.raises(ValueError, match="output_sigma"):
        SyntheticLatentBridgeLikelihood(output_sigma=0)
