"""Gradient routes for the separately approved latent-FM-only SG candidate."""

import copy

import pytest
import torch
import torch.nn as nn

from egomimic.pipeline.stages_action_flow import (
    ActionFlowObjectiveStage,
    ConditionalVelocityStage,
    ContentDecoderStage,
    ContentEncoderStage,
    LatentBridgeStage,
)


class _Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.condition = nn.Linear(3, 8).double()
        self.calls = []

    def forward(self, state, time, condition, *, condition_drop_mask):
        self.calls.append((state, time, condition, condition_drop_mask))
        masked = condition.masked_fill(condition_drop_mask[:, None], 0)
        return (
            self.weight * state.tanh()
            + self.condition(masked)[:, None]
            + time[:, None, None]
        )


def _forward(mode, encoder, decoder, field, target, noise, condition):
    stages = [
        ContentEncoderStage(encoder),
        LatentBridgeStage(samples_per_content=3, condition_dropout_probability=0.3),
        ConditionalVelocityStage(field, flow_clean_gradient_mode=mode),
        ContentDecoderStage(decoder),
        ActionFlowObjectiveStage(residual_key="action_flow/fm_velocity_residual"),
    ]
    batch = {"target": target, "sampler/noise": noise, "condition": condition}
    torch.manual_seed(72)
    for stage in stages:
        batch = stage(batch)
    return batch


def _models_and_inputs():
    torch.manual_seed(41)
    return (
        nn.Sequential(nn.Linear(4, 8), nn.Tanh()).double(),
        nn.Sequential(nn.Linear(8, 6), nn.SiLU(), nn.Linear(6, 4)).double(),
        _Field(),
        torch.randn(3, 4, 4, dtype=torch.float64),
        torch.randn(3, 4, 8, dtype=torch.float64),
        torch.randn(3, 3, dtype=torch.float64, requires_grad=True),
    )


def test_latent_only_stopgrad_blocks_both_clean_fm_routes_not_shared_condition():
    encoder, decoder, field, target, noise, condition = _models_and_inputs()
    batch = _forward("all_stopgrad", encoder, decoder, field, target, noise, condition)
    fm = batch["log/action_flow_fm"]
    gradients = torch.autograd.grad(
        fm, tuple(encoder.parameters()), allow_unused=True, retain_graph=True
    )
    assert all(gradient is None for gradient in gradients)
    field_gradients = torch.autograd.grad(
        fm, tuple(field.parameters()), retain_graph=True
    )
    assert all(torch.isfinite(gradient).all() for gradient in field_gradients)
    assert sum(gradient.abs().sum() for gradient in field_gradients) > 0
    condition_gradient = torch.autograd.grad(fm, condition)[0]
    assert torch.isfinite(condition_gradient).all()
    assert condition_gradient.abs().sum() > 0

    attached, detached = field.calls
    assert attached[0].requires_grad and not detached[0].requires_grad
    torch.testing.assert_close(attached[0], detached[0], rtol=0, atol=0)
    # No new bridge noise, time, observation encoding, or dropout mask draw.
    for index in (1, 2, 3):
        assert attached[index] is detached[index]
    torch.testing.assert_close(
        batch["action_flow/fm_velocity_residual"],
        batch["action_flow/velocity_residual"], rtol=0, atol=0,
    )


def test_latent_only_sg_preserves_all_action_flow_gradients_and_diagnostics():
    originals = _models_and_inputs()
    results = []
    for mode in ("full", "all_stopgrad"):
        encoder, decoder, field = [copy.deepcopy(module) for module in originals[:3]]
        target, noise = originals[3:5]
        condition = originals[5].detach().clone().requires_grad_()
        batch = _forward(mode, encoder, decoder, field, target, noise, condition)
        params = (*encoder.parameters(), *decoder.parameters(), *field.parameters(), condition)
        gradients = torch.autograd.grad(
            batch["log/action_flow_action_velocity"], params,
            allow_unused=True, retain_graph=True,
        )
        # The encoder must still receive generative supervision through the
        # bridge state, target tangent, and decoder Jacobian evaluation point.
        encoder_gradients = gradients[:len(tuple(encoder.parameters()))]
        assert all(gradient is not None for gradient in encoder_gradients)
        assert sum(gradient.abs().sum() for gradient in encoder_gradients) > 0
        if mode == "full":
            fm_gradients = torch.autograd.grad(
                batch["log/action_flow_fm"] , tuple(encoder.parameters())
            )
            assert sum(gradient.abs().sum() for gradient in fm_gradients) > 0
            assert len(field.calls) == 1
            assert batch["action_flow/fm_velocity_residual"] is batch["action_flow/velocity_residual"]
        results.append((batch, gradients))
    for key in ("action_flow/state", "action_flow/target_velocity",
                "action_flow/predicted_velocity", "action_flow/velocity_residual",
                "action_flow/decoded_velocity_residual"):
        torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
    for full, isolated in zip(results[0][1], results[1][1]):
        if full is None:
            assert isolated is None
        else:
            torch.testing.assert_close(full, isolated, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["target_stopgrad", "all", None])
def test_candidate_rejects_unsupported_gradient_modes(mode):
    with pytest.raises(ValueError, match="flow_clean_gradient_mode"):
        ConditionalVelocityStage(_Field(), flow_clean_gradient_mode=mode)


def test_fm_key_cannot_clobber_action_flow_residual():
    with pytest.raises(ValueError, match="must be distinct"):
        ConditionalVelocityStage(_Field(), flow_residual_key="action_flow/velocity_residual")
