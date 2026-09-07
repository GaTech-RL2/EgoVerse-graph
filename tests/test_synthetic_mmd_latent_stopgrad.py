"""Latent-only stop-gradient must retain Action Flow and MMD representation learning."""

import pytest
import torch

from egomimic.synthetic.noninvertible_endpoint_flow import (
    SyntheticMMDEndpointFlow,
    paired_multiscale_mmd2,
)


def _case():
    torch.manual_seed(61)
    model = SyntheticMMDEndpointFlow(
        residual_width=8, residual_depth=1, field_width=8, field_depth=1,
    ).double()
    # A moved nonlinear interface makes every intended route observable;
    # identity initialization would make the endpoint MMD gradient vanish.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.03 * torch.randn_like(parameter))
        model.decoder.linear.bias.add_(0.2)
    inputs = {
        "action": torch.randn(7, 3, dtype=torch.float64),
        "noise": torch.randn(7, 8, dtype=torch.float64),
        "time": torch.rand(14, 1, dtype=torch.float64),
        "flow_samples": 2, "return_diagnostics": True,
    }
    return model, inputs


@pytest.mark.parametrize("weight", [10.0, 100.0])
@pytest.mark.parametrize("mode", ["target_stopgrad", "all_stopgrad"])
def test_mmd_latent_override_preserves_values_and_field_gradients(weight, mode):
    model, inputs = _case()
    full = model.losses(**inputs, lambda_endpoint=weight)
    changed = model.losses(**inputs, lambda_endpoint=weight, flow_clean_gradient_mode=mode)
    assert full.keys() == changed.keys()
    for key in full:
        torch.testing.assert_close(full[key], changed[key], rtol=0, atol=0)
    field = tuple(model.field.parameters())
    for key in ("latent_flow_loss", "action_velocity_loss", "loss"):
        expected = torch.autograd.grad(full[key], field, retain_graph=True)
        observed = torch.autograd.grad(changed[key], field, retain_graph=True)
        for reference, gradient in zip(expected, observed):
            torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("weight", [10.0, 100.0])
def test_all_clean_stopgrad_removes_only_latent_fm_encoder_update(weight):
    model, inputs = _case()
    full = model.losses(**inputs, lambda_endpoint=weight)
    changed = model.losses(
        **inputs, lambda_endpoint=weight, flow_clean_gradient_mode="all_stopgrad"
    )
    encoder = tuple(model.encoder.parameters())
    original_fm = torch.autograd.grad(full["flow_loss"], encoder, retain_graph=True)
    assert sum(gradient.abs().sum() for gradient in original_fm) > 0
    stopped_fm = torch.autograd.grad(
        changed["flow_loss"], encoder, retain_graph=True, allow_unused=True,
    )
    assert all(gradient is None or not bool(gradient.abs().sum()) for gradient in stopped_fm)
    retained = []
    for key in ("action_velocity_loss", "endpoint_mmd2"):
        original = torch.autograd.grad(full[key], encoder, retain_graph=True)
        observed = torch.autograd.grad(changed[key], encoder, retain_graph=True)
        assert sum(gradient.abs().sum() for gradient in observed) > 0
        for before, after in zip(original, observed):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        retained.append(observed)
    total = torch.autograd.grad(changed["loss"], encoder, retain_graph=True)
    for actual, action_gradient, endpoint_gradient in zip(total, *retained):
        torch.testing.assert_close(
            actual, action_gradient + weight * endpoint_gradient, rtol=1e-11, atol=1e-12,
        )
    # Decoder supervision is also identical, including nonlinear JVP work.
    decoder = tuple(model.decoder.parameters())
    before = torch.autograd.grad(full["loss"], decoder, retain_graph=True)
    after = torch.autograd.grad(changed["loss"], decoder)
    for original, preserved in zip(before, after):
        torch.testing.assert_close(original, preserved, rtol=1e-12, atol=1e-12)


def test_target_only_stopgrad_keeps_the_latent_fm_state_route():
    model, inputs = _case()
    changed = model.losses(**inputs, flow_clean_gradient_mode="target_stopgrad")
    clean = model.encoder(inputs["action"]).repeat_interleave(2, 0)
    noise = inputs["noise"].repeat_interleave(2, 0)
    state = (1 - inputs["time"]) * clean + inputs["time"] * noise
    expected = (model.velocity(state, inputs["time"]) - (noise - clean.detach())).square().mean()
    encoder = tuple(model.encoder.parameters())
    reference = torch.autograd.grad(expected, encoder)
    observed = torch.autograd.grad(changed["flow_loss"], encoder)
    assert sum(gradient.abs().sum() for gradient in observed) > 0
    for actual, target in zip(observed, reference):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [None, "full"])
def test_legacy_mmd_uses_one_field_and_preserves_exact_attached_objective(mode):
    model, inputs = _case()
    calls = []
    handle = model.field.register_forward_hook(lambda *_: calls.append(True))
    try:
        original = model.losses(**inputs)
        assert len(calls) == 1
        explicit = model.losses(**inputs, flow_clean_gradient_mode=mode)
        assert len(calls) == 2
        model.losses(**inputs, flow_clean_gradient_mode="all_stopgrad")
        assert len(calls) == 4
    finally:
        handle.remove()
    for key in original:
        torch.testing.assert_close(original[key], explicit[key], rtol=0, atol=0)
    clean = model.encoder(inputs["action"])
    state = (1 - inputs["time"]) * clean.repeat_interleave(2, 0) + inputs["time"] * inputs["noise"].repeat_interleave(2, 0)
    residual = model.velocity(state, inputs["time"]) - (inputs["noise"] - clean).repeat_interleave(2, 0)
    legacy = (residual.square().mean() + model.decoder_jvp(state, residual).square().mean()
              + model.scale_loss(inputs["noise"])
              + 10 * paired_multiscale_mmd2(model.decoder(clean), inputs["action"]))
    torch.testing.assert_close(original["loss"], legacy, rtol=0, atol=0)
    expected = torch.autograd.grad(legacy, tuple(model.parameters()))
    observed = torch.autograd.grad(explicit["loss"], tuple(model.parameters()))
    for gradient, reference in zip(observed, expected):
        torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-12)


def test_mmd_rejects_unknown_latent_clean_mode():
    model, inputs = _case()
    with pytest.raises(ValueError, match="unknown flow clean gradient mode"):
        model.losses(**inputs, flow_clean_gradient_mode="detach_everything")
