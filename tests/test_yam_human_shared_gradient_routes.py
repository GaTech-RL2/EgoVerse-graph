"""Focused two-embodiment gradient contract for the provisional co-train graph."""

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


class _SharedField(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.condition = nn.Linear(3, 8).double()

    def forward(self, state, time, condition, *, condition_drop_mask):
        visible = condition.masked_fill(condition_drop_mask[:, None], 0)
        return (
            self.scale * state.tanh()
            + self.condition(visible)[:, None]
            + time[:, None, None]
        )


def _norm(gradients):
    return sum(
        float(gradient.abs().sum()) for gradient in gradients if gradient is not None
    )


@pytest.mark.parametrize("embodiment,action_dim", [(7, 14), (3, 138)])
def test_both_domains_supervise_one_shared_field_and_their_own_codec(
    embodiment, action_dim
):
    torch.manual_seed(19)
    encoders = {
        "yam_bimanual": nn.Sequential(nn.Linear(14, 8), nn.Tanh()).double(),
        "human_bimanual": nn.Sequential(nn.Linear(138, 8), nn.Tanh()).double(),
    }
    decoders = {
        "yam_bimanual": nn.Sequential(
            nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 14)
        ).double(),
        "human_bimanual": nn.Sequential(
            nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 138)
        ).double(),
    }
    aliases = {"7": "yam_bimanual", "3": "human_bimanual"}
    field = _SharedField()
    stages = (
        ContentEncoderStage(encoders=encoders, selector_aliases=aliases),
        LatentBridgeStage(samples_per_content=3, condition_dropout_probability=0.0),
        ConditionalVelocityStage(field, flow_clean_gradient_mode="all_stopgrad"),
        ContentDecoderStage(decoders=decoders, selector_aliases=aliases),
        ActionFlowObjectiveStage(residual_key="action_flow/fm_velocity_residual"),
    )
    batch = {
        "target": torch.randn(2, 4, action_dim, dtype=torch.float64),
        "sampler/noise": torch.randn(2, 4, 8, dtype=torch.float64),
        "condition": torch.randn(2, 3, dtype=torch.float64),
        "embodiment": torch.full((2,), embodiment),
    }
    for stage in stages:
        batch = stage(batch)
    selected = aliases[str(embodiment)]
    other = next(name for name in encoders if name != selected)
    selected_encoder = tuple(encoders[selected].parameters())
    selected_decoder = tuple(decoders[selected].parameters())
    shared = tuple(field.parameters())
    fm = batch["log/action_flow_fm"]
    velocity = batch["log/action_flow_action_velocity"]
    total = batch["loss/action_flow"]
    assert torch.isfinite(torch.stack((fm, velocity, total))).all()
    assert _norm(torch.autograd.grad(fm, shared, retain_graph=True)) > 0
    assert (
        _norm(
            torch.autograd.grad(
                fm, selected_encoder, retain_graph=True, allow_unused=True
            )
        )
        == 0
    )
    assert _norm(torch.autograd.grad(velocity, selected_encoder, retain_graph=True)) > 0
    assert _norm(torch.autograd.grad(total, selected_decoder, retain_graph=True)) > 0
    assert _norm(torch.autograd.grad(total, shared, retain_graph=True)) > 0
    assert (
        _norm(
            torch.autograd.grad(
                total,
                tuple(encoders[other].parameters()),
                retain_graph=True,
                allow_unused=True,
            )
        )
        == 0
    )
    assert (
        _norm(
            torch.autograd.grad(
                total, tuple(decoders[other].parameters()), allow_unused=True
            )
        )
        == 0
    )
