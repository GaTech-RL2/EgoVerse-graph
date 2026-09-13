"""Exact-section and JVP behavior of the restricted graph diagnostic."""

import pytest
import torch
import torch.nn as nn
from torch.func import jvp

from egomimic.models.action_flow_codec import GraphSectionSequenceCodec
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_action_flow import (
    ActionFlowObjectiveStage,
    ConditionalVelocityStage,
    ContentDecoderStage,
    ContentEncoderStage,
    LatentBridgeStage,
)


class _EncoderView(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.graph = codec.graph

    def forward(self, content):
        return torch.cat((content, self.graph(content)), dim=-1)


class _Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.condition = nn.Linear(3, 8)

    def forward(self, value, time, condition, *, condition_drop_mask):
        condition = condition.masked_fill(condition_drop_mask[:, None], 0)
        return self.linear(value).tanh() + self.condition(condition)[:, None]


def _codec():
    return GraphSectionSequenceCodec(action_dim=4, latent_dim=8, horizon=4)


def test_graph_section_identity_survives_joint_optimizer_updates_and_shared_ownership():
    torch.manual_seed(46)
    codec = _codec().double()
    encoder = ContentEncoderStage(_EncoderView(codec))
    decoder = ContentDecoderStage(codec)
    field = ConditionalVelocityStage(_Field().double())
    objective = ActionFlowObjectiveStage(
        reconstruction_weight=0.0, residual_key="action_flow/fm_velocity_residual"
    )
    pipeline = Pipeline([
        encoder, LatentBridgeStage(samples_per_content=2), field, decoder, objective
    ])
    parameters = list(pipeline.parameters())
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert encoder.encoder.graph is decoder.decoder.graph
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    assert len(optimizer.param_groups[0]["params"]) == len(parameters)
    snapshots = {
        name: [parameter.detach().clone() for parameter in module.parameters()]
        for name, module in (("graph", codec.graph), ("residual", codec.residual))
    }
    target = torch.randn(2, 4, 4, dtype=torch.float64)
    for _ in range(2):
        latent = codec.encode(target)
        torch.testing.assert_close(latent[..., :4], target, rtol=0, atol=0)
        torch.testing.assert_close(codec(latent), target, rtol=0, atol=1e-12)
        batch = pipeline({
            "target": target,
            "sampler/noise": torch.randn(2, 4, 8, dtype=torch.float64),
            "condition": torch.randn(2, 3, dtype=torch.float64),
        })
        torch.testing.assert_close(
            batch["loss/action_flow"],
            batch["log/action_flow_fm"] + batch["log/action_flow_action_velocity"],
            rtol=0, atol=0,
        )
        optimizer.zero_grad()
        batch["loss/action_flow"].backward()
        for module in (codec.graph, codec.residual, field.field):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert sum(g.abs().sum() for g in gradients) > 0
        optimizer.step()
    torch.testing.assert_close(codec(codec.encode(target)), target, rtol=0, atol=1e-12)
    for name, module in (("graph", codec.graph), ("residual", codec.residual)):
        assert any(not torch.equal(old, new) for old, new in zip(snapshots[name], module.parameters()))


def test_graph_section_jvp_matches_finite_difference_with_trainable_graph():
    torch.manual_seed(7)
    codec = _codec().double()
    latent = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    tangent = torch.randn_like(latent, requires_grad=True)
    _, predicted = jvp(codec, (latent,), (tangent,))
    delta = 1e-6
    expected = (codec(latent + delta*tangent) - codec(latent - delta*tangent))/(2*delta)
    torch.testing.assert_close(predicted, expected, rtol=2e-5, atol=2e-6)
    gradients = torch.autograd.grad(predicted.square().mean(), (latent, tangent, *codec.graph.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients[2:]) > 0


def test_graph_section_context_free_shape_and_small_private_networks():
    codec = GraphSectionSequenceCodec(action_dim=4, latent_dim=8, horizon=16)
    for module in (codec.graph, codec.residual):
        assert module.depth == 2
        assert sum(parameter.numel() for parameter in module.parameters()) < 12_000
    with pytest.raises(ValueError, match="expected sequence shape"):
        codec(torch.randn(2, 15, 8))
    with pytest.raises(ValueError, match="expected sequence shape"):
        codec.encode(torch.randn(2, 16, 3))


def test_graph_section_mixed_precision_identity_and_jvp_backward_are_finite():
    torch.manual_seed(19)
    codec = _codec()
    content = torch.randn(2, 4, 4)
    latent = torch.randn(2, 4, 8, requires_grad=True)
    tangent = torch.randn_like(latent, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reconstructed = codec(codec.encode(content))
        _, decoded_tangent = jvp(codec, (latent,), (tangent,))
        loss = decoded_tangent.square().mean()
    torch.testing.assert_close(reconstructed, content, rtol=0, atol=0)
    assert torch.isfinite(decoded_tangent).all()
    loss.backward()
    for module in (codec.graph, codec.residual):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


@pytest.mark.parametrize("kwargs", [{"action_dim":8}, {"latent_dim":3}, {"dropout":0.1}])
def test_graph_section_rejects_invalid_or_stochastic_section(kwargs):
    options = {"action_dim":4, "latent_dim":8, "horizon":4, **kwargs}
    with pytest.raises(ValueError):
        GraphSectionSequenceCodec(**options)
