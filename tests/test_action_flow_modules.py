import inspect

import pytest
import torch
from torch.func import jvp

from egomimic.models.action_flow_codec import (
    ContextFreeSequenceDecoder,
    ContextFreeSequenceEncoder,
)
from egomimic.models.action_flow_transformer import (
    AdaLNSequenceField,
    _sinusoidal_time_embedding,
)


def _encoder() -> ContextFreeSequenceEncoder:
    return ContextFreeSequenceEncoder(
        input_dim=4,
        latent_dim=8,
        horizon=16,
        hidden_dim=20,
        depth=2,
        num_heads=4,
        feedforward_dim=80,
        dropout=0.0,
    )


def _decoder() -> ContextFreeSequenceDecoder:
    return ContextFreeSequenceDecoder(
        latent_dim=8,
        output_dim=4,
        horizon=16,
        hidden_dim=20,
        depth=2,
        num_heads=4,
        feedforward_dim=80,
        dropout=0.0,
    )


def _small_field(**overrides) -> AdaLNSequenceField:
    arguments = {
        "input_dim": 8,
        "output_dim": 8,
        "horizon": 16,
        "condition_dim": 7,
        "hidden_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "feedforward_dim": 64,
        "time_embedding_dim": 32,
        "dropout": 0.0,
        "condition_dropout_probability": 0.3,
    }
    arguments.update(overrides)
    return AdaLNSequenceField(**arguments)


def test_context_free_codec_shapes_signatures_and_parameter_budgets():
    encoder = _encoder()
    decoder = _decoder()
    action = torch.randn(3, 16, 4)
    latent = encoder(action)
    reconstruction = decoder(latent)

    assert latent.shape == (3, 16, 8)
    assert reconstruction.shape == (3, 16, 4)
    assert tuple(inspect.signature(encoder.forward).parameters) == ("content",)
    assert tuple(inspect.signature(decoder.forward).parameters) == ("content",)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 10_748
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 10_744
    assert sum(parameter.numel() for parameter in encoder.parameters()) <= 12_000
    assert sum(parameter.numel() for parameter in decoder.parameters()) <= 12_000


def test_context_free_codec_rejects_wrong_sequence_contract():
    encoder = _encoder()
    decoder = _decoder()
    with pytest.raises(ValueError, match="expected sequence shape"):
        encoder(torch.randn(2, 15, 4))
    with pytest.raises(ValueError, match="expected sequence shape"):
        decoder(torch.randn(2, 16, 7))


def test_decoder_jvp_matches_finite_difference_and_has_higher_order_gradients():
    torch.manual_seed(11)
    decoder = _decoder().double()
    latent = torch.randn(2, 16, 8, dtype=torch.float64, requires_grad=True)
    tangent = torch.randn_like(latent, requires_grad=True)

    _, decoded_tangent = jvp(decoder, (latent,), (tangent,))
    step = 1.0e-6
    finite_difference = (
        decoder(latent + step * tangent) - decoder(latent - step * tangent)
    ) / (2.0 * step)
    torch.testing.assert_close(
        decoded_tangent, finite_difference, rtol=2.0e-5, atol=2.0e-6
    )

    loss = decoded_tangent.square().mean()
    named_parameters = tuple(decoder.named_parameters())
    parameter_gradients = torch.autograd.grad(
        loss,
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
        retain_graph=True,
    )
    missing = {
        name
        for (name, _), gradient in zip(named_parameters, parameter_gradients)
        if gradient is None
    }
    # An additive output bias cannot affect a Jacobian-vector product.  Every
    # other decoder parameter must remain reachable through the JVP objective.
    assert missing == {"output_projection.bias"}
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in parameter_gradients
        if gradient is not None
    )
    latent_gradient, tangent_gradient = torch.autograd.grad(loss, (latent, tangent))
    assert bool(torch.isfinite(latent_gradient).all())
    assert bool(torch.isfinite(tangent_gradient).all())


def test_decoder_bfloat16_jvp_tracks_float32_reference():
    torch.manual_seed(17)
    decoder = _decoder().eval()
    latent = torch.randn(2, 16, 8)
    tangent = torch.randn_like(latent)
    _, float32_jvp = jvp(decoder, (latent,), (tangent,))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, bfloat16_jvp = jvp(decoder, (latent,), (tangent,))

    difference_rms = (float32_jvp - bfloat16_jvp.float()).square().mean().sqrt()
    reference_rms = float32_jvp.square().mean().sqrt()
    assert bool(torch.isfinite(bfloat16_jvp).all())
    assert float(difference_rms / reference_rms) < 0.02


def test_adaln_field_shapes_condition_dropout_and_input_gradients():
    torch.manual_seed(13)
    field = _small_field().train()
    value = torch.randn(4, 16, 8, requires_grad=True)
    time = torch.rand(4, 1)
    condition = torch.randn(4, 7, requires_grad=True)
    drop_mask = torch.tensor([False, True, True, False])

    effective, returned_mask = field.apply_condition_dropout(
        condition, drop_mask=drop_mask
    )
    torch.testing.assert_close(effective[~drop_mask], condition[~drop_mask])
    torch.testing.assert_close(effective[drop_mask], field.null_condition.expand(2, -1))
    torch.testing.assert_close(returned_mask, drop_mask)

    output = field(
        value,
        time,
        condition,
        condition_drop_mask=drop_mask,
    )
    assert output.shape == value.shape
    output.square().mean().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())
    assert condition.grad is not None
    assert bool(torch.isfinite(condition.grad).all())
    assert field.null_condition.grad is not None
    assert bool(torch.isfinite(field.null_condition.grad).all())


def test_normalized_time_embedding_resolves_endpoints_and_euler_steps():
    times = torch.tensor([0.0, 1.0 / 16.0, 0.5, 9.0 / 16.0, 1.0])
    embedding = _sinusoidal_time_embedding(times, 512, time_scale=1_000.0)

    endpoint_rms = (embedding[0] - embedding[-1]).square().mean().sqrt()
    first_step_rms = (embedding[0] - embedding[1]).square().mean().sqrt()
    middle_step_rms = (embedding[2] - embedding[3]).square().mean().sqrt()
    assert float(endpoint_rms) > 0.75
    assert float(first_step_rms) > 0.6
    assert float(middle_step_rms) > 0.6

    field = _small_field().eval()
    value = torch.randn(1, 16, 8).expand(2, -1, -1)
    condition = torch.randn(1, 7).expand(2, -1)
    output = field(value, torch.tensor([0.0, 1.0]), condition)
    assert not torch.allclose(output[0], output[1])


def test_adaln_field_default_manifest_is_about_forty_million_parameters():
    field = AdaLNSequenceField(
        input_dim=8,
        output_dim=8,
        horizon=16,
        condition_dim=67,
        hidden_dim=512,
        depth=12,
        num_heads=8,
        feedforward_dim=2048,
        time_embedding_dim=512,
        dropout=0.0,
        condition_dropout_probability=0.3,
    )
    parameter_count = sum(parameter.numel() for parameter in field.parameters())
    assert 39_000_000 <= parameter_count <= 41_000_000


def test_adaln_field_eval_uses_condition_unless_mask_is_explicit():
    field = _small_field().eval()
    condition = torch.randn(2, 7)
    effective, mask = field.apply_condition_dropout(condition)
    torch.testing.assert_close(effective, condition)
    assert not bool(mask.any())

    explicit_mask = torch.ones(2, dtype=torch.bool)
    effective, mask = field.apply_condition_dropout(condition, drop_mask=explicit_mask)
    torch.testing.assert_close(effective, field.null_condition.expand_as(condition))
    assert bool(mask.all())
