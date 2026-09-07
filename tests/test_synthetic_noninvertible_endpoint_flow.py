import pytest
import torch
from torch import nn
from torch.func import jacrev, jvp

from egomimic.synthetic.noninvertible_endpoint_flow import (
    DEFAULT_MMD_BANDWIDTHS,
    SyntheticGraphSectionFlow,
    SyntheticMMDEndpointFlow,
    paired_multiscale_mmd2,
)


def _small(model_type):
    torch.manual_seed(42)
    return model_type(residual_width=8, residual_depth=1,
                      field_width=8, field_depth=1).double()


def _inputs():
    return dict(action=torch.randn(7, 3, dtype=torch.float64),
                noise=torch.randn(7, 8, dtype=torch.float64),
                time=torch.rand(14, 1, dtype=torch.float64), flow_samples=2)


@pytest.mark.parametrize("conditional", [False, True])
def test_paired_mmd_matches_explicit_off_diagonal_estimator(conditional):
    torch.manual_seed(7)
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, 3, dtype=torch.float64)
    context = torch.randn(4, 2, dtype=torch.float64) if conditional else None

    def kernel(a, b):
        return sum(torch.exp(-(a - b).square().sum() / (2 * sigma**2))
                   for sigma in DEFAULT_MMD_BANDWIDTHS) / len(DEFAULT_MMD_BANDWIDTHS)

    expected = x.new_zeros(())
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            value = kernel(x[i], x[j]) + kernel(y[i], y[j]) - kernel(x[i], y[j]) - kernel(y[i], x[j])
            if context is not None:
                value *= torch.exp(-(context[i] - context[j]).square().sum() / 2)
            expected = expected + value / 12
    actual = paired_multiscale_mmd2(x, y, context=context)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, x)[0]
    expected_gradient = torch.autograd.grad(expected, x)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_same_action_distribution_can_have_large_reconstruction_error_and_negative_mmd():
    action = torch.tensor([[-3.], [-1.], [1.], [3.]], dtype=torch.float64)
    decoded = action.flip(0)
    assert (decoded - action).square().mean() == 20
    # Both empirical distributions are identical. The paired U-statistic has a
    # finite-batch diagonal correction and may be negative; do not clamp it.
    value = paired_multiscale_mmd2(decoded, action)
    assert value < 0
    assert paired_multiscale_mmd2(action, action) == 0
    assert paired_multiscale_mmd2(action + 30, action) > value


def test_joint_context_kernel_detects_context_swapping_hidden_by_marginal_mmd():
    context = torch.tensor([[-2.], [2.]], dtype=torch.float64).repeat(32, 1)
    action = context.clone()
    swapped = -action
    assert paired_multiscale_mmd2(swapped, action) < 0
    assert paired_multiscale_mmd2(swapped, action, context=context) > 0.4


def test_mmd_loss_is_geometric_plus_endpoint_and_standard_scale_only():
    model = _small(SyntheticMMDEndpointFlow)
    with torch.no_grad():
        model.decoder.linear.bias.add_(0.3)
    inputs = _inputs()
    low = model.losses(**inputs, lambda_endpoint=10, return_diagnostics=True)
    high = model.losses(**inputs, lambda_endpoint=100)
    torch.testing.assert_close(high["loss"] - low["loss"], 90 * low["endpoint_mmd2"])
    expected = low["flow_loss"] + low["action_velocity_loss"] + low["scale_loss"] + 10 * low["endpoint_mmd2"]
    torch.testing.assert_close(low["loss"], expected)
    assert low["reconstruction_mse"] > 0
    assert low["reconstruction_loss"] == 0
    direct = paired_multiscale_mmd2(model.decoder(model.encoder(inputs["action"])), inputs["action"])
    torch.testing.assert_close(low["endpoint_mmd2"], direct)


@pytest.mark.parametrize("model_type", [SyntheticMMDEndpointFlow, SyntheticGraphSectionFlow])
def test_geometric_loss_keeps_encoder_target_state_and_decoder_gradients(model_type):
    model = _small(model_type)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    inputs = _inputs()
    extras = {"lambda_endpoint": 0.0} if model_type is SyntheticMMDEndpointFlow else {}
    actual = model.losses(**inputs, lambda_scale=0.0, **extras)["loss"]
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))
    clean = model.encoder(inputs["action"])
    many = clean.repeat_interleave(2, 0)
    noise = inputs["noise"].repeat_interleave(2, 0)
    state = (1 - inputs["time"]) * many + inputs["time"] * noise
    target = noise - many
    residual = model.velocity(state, inputs["time"]) - target
    expected = residual.square().mean() + model.decoder_jvp(state, residual).square().mean()
    expected_gradients = torch.autograd.grad(
        expected, tuple(model.parameters()), retain_graph=True, allow_unused=True
    )
    # Pure geometric FM is invariant to an additive decoder translation. Its
    # JVP graph therefore omits these output biases, while 0*scale/endpoint in
    # the implementation materializes their mathematically zero gradients.
    # Do not excuse unused tensors elsewhere: they would hide a broken route.
    translation_biases = (
        (model.decoder.linear.bias, model.decoder.residual[-1].bias)
        if model_type is SyntheticMMDEndpointFlow else (model.residual[-1].bias,)
    )
    allowed_unused = {id(parameter) for parameter in translation_biases}
    for (name, parameter), observed, reference in zip(
        model.named_parameters(), actual_gradients, expected_gradients
    ):
        if reference is None:
            assert id(parameter) in allowed_unused, f"unexpected disconnected parameter: {name}"
            reference = torch.zeros_like(parameter)
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        torch.testing.assert_close(observed, reference)
    for tensor in (clean, state, target):
        assert torch.autograd.grad(expected, tensor, retain_graph=True)[0].abs().sum() > 0
    full_clean_gradient = torch.autograd.grad(expected, clean, retain_graph=True)[0]
    detached_residual = model.velocity(state, inputs["time"]) - target.detach()
    detached_loss = (
        detached_residual.square().mean()
        + model.decoder_jvp(state, detached_residual).square().mean()
    )
    detached_clean_gradient = torch.autograd.grad(detached_loss, clean)[0]
    assert not torch.allclose(full_clean_gradient, detached_clean_gradient, rtol=1e-8, atol=1e-10)
    named_gradients = dict(zip((name for name, _ in model.named_parameters()), actual_gradients))
    prefixes = ("encoder", "decoder", "field") if model_type is SyntheticMMDEndpointFlow else ("graph", "residual", "field")
    for prefix in prefixes:
        assert sum(gradient.abs().sum() for name, gradient in named_gradients.items() if name.startswith(prefix)) > 0


def test_mmd_endpoint_gradient_reaches_both_free_interfaces():
    model = _small(SyntheticMMDEndpointFlow)
    with torch.no_grad():
        model.decoder.linear.bias.add_(0.25)
    model.losses(**_inputs())["endpoint_mmd2"].backward()
    for module in (model.encoder, model.decoder):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in module.parameters())
    assert all(parameter.grad is None for parameter in model.field.parameters())


def test_graph_section_exact_identity_and_tangent_survive_optimizer_update():
    model = _small(SyntheticGraphSectionFlow)
    inputs = _inputs()
    torch.testing.assert_close(model.encoder(inputs["action"])[:, 3:], torch.zeros(7, 5, dtype=torch.float64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_graph = model.graph[-1].weight.detach().clone()
    initial_residual = model.residual[-1].weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        losses = model.losses(**inputs, return_diagnostics=True)
        assert losses["reconstruction_loss"] == 0
        losses["loss"].backward()
        optimizer.step()
    assert not torch.equal(model.graph[-1].weight, initial_graph)
    assert not torch.equal(model.residual[-1].weight, initial_residual)
    action = inputs["action"]
    direction = torch.randn_like(action)
    torch.testing.assert_close(model.decoder(model.encoder(action)), action, atol=1e-12, rtol=1e-12)
    _, derivative = jvp(lambda value: model.decoder(model.encoder(value)), (action,), (direction,))
    torch.testing.assert_close(derivative, direction, atol=1e-12, rtol=1e-12)


def test_graph_section_can_have_a_true_decoder_rank_defect():
    class ZeroGraph(nn.Module):
        def forward(self, action):
            return torch.zeros_like(action)

    class ProductResidual(nn.Module):
        def forward(self, latent):
            return latent[..., :1] * latent[..., 1:2]

    model = SyntheticGraphSectionFlow(latent_dim=2, action_dim=1,
                                     residual_width=4, residual_depth=1,
                                     field_width=4, field_depth=1).double()
    model.graph = ZeroGraph()
    model.residual = ProductResidual()
    defect = torch.tensor([0., -1.], dtype=torch.float64)
    torch.testing.assert_close(jacrev(model.decoder)(defect), torch.zeros(1, 2, dtype=torch.float64))
    action = torch.tensor([[2.], [-3.]], dtype=torch.float64)
    torch.testing.assert_close(model.decoder(model.encoder(action)), action)


@pytest.mark.parametrize("model_type", [SyntheticMMDEndpointFlow, SyntheticGraphSectionFlow])
def test_decoder_jvp_and_sampler_use_correct_no_grad_semantics(model_type):
    model = _small(model_type)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
        state = torch.randn(7, 8, dtype=torch.float64)
        direction = torch.randn_like(state)
        epsilon = 1e-5
        finite = (model.decoder(state + epsilon * direction) - model.decoder(state - epsilon * direction)) / (2 * epsilon)
        torch.testing.assert_close(model.decoder_jvp(state, direction), finite, atol=1e-9, rtol=1e-8)
        trajectory = model.trajectory(state, steps=2)
        first = state - 0.5 * model.velocity(state, state.new_ones((7, 1)))
        final = first - 0.5 * model.velocity(first, state.new_full((7, 1), 0.5))
        torch.testing.assert_close(trajectory[-1], model.decoder(final))
        assert trajectory.dtype == torch.float64
        assert (trajectory[-1] - trajectory[0]).abs().sum() > 0
    with torch.inference_mode(), pytest.raises(RuntimeError, match="no_grad"):
        model.decoder_jvp(state, direction)


@pytest.mark.parametrize(("model_type", "expected"),
                         [(SyntheticMMDEndpointFlow, 54_798), (SyntheticGraphSectionFlow, 54_640)])
def test_default_parameter_counts(model_type, expected):
    model = model_type()
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert sum(parameter.numel() for parameter in model.field.parameters()) == 51_848


def test_invalid_mmd_batch_and_bandwidth_fail_explicitly():
    with pytest.raises(ValueError, match="batch"):
        paired_multiscale_mmd2(torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="bandwidth"):
        paired_multiscale_mmd2(torch.zeros(2, 3), torch.zeros(2, 3), bandwidths=[0.0])
