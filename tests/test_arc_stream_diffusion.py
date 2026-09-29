"""Information-flow, parameter-budget, diffusion, and checkpoint contracts."""

import pytest
import torch

from egomimic.models.arc_diffusion import (
    DECODER_VARIANTS,
    SHAPE_COLUMNS,
    TIMING_COLUMNS,
    ArcStreamDiffusionTransformer,
)
from egomimic.models.diffusion_policy import DiffusionPolicy
from tests.test_oat_diffusion import dimensions, scheduler


def model(variant, *, full_size=False):
    kwargs = dimensions(horizon=32 if full_size else 6)
    kwargs.update(p_drop_emb=0, p_drop_attn=0)
    if not full_size:
        kwargs.update(n_emb=32)
    torch.manual_seed(72)
    return ArcStreamDiffusionTransformer(decoder_variant=variant, **kwargs).eval()


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
def test_timing_cannot_influence_shape_in_restricted_variants(variant):
    net = model(variant)
    sample = torch.randn(2, 6, 12, requires_grad=True)
    condition = torch.randn(2, 276)
    output = net(sample, torch.tensor([7, 53]), condition)
    derivative = torch.autograd.grad(
        output[..., list(SHAPE_COLUMNS)].square().sum(), sample
    )[0]
    timing_gradient = derivative[..., list(TIMING_COLUMNS)]
    if variant == "shared":
        assert timing_gradient.abs().max() > 1e-7
    else:
        assert torch.count_nonzero(timing_gradient) == 0
    changed = sample.detach().clone()
    changed[..., list(TIMING_COLUMNS)] += 10 * torch.randn(2, 6, 2)
    other = net(changed, torch.tensor([7, 53]), condition)
    if variant != "shared":
        torch.testing.assert_close(
            output[..., list(SHAPE_COLUMNS)],
            other[..., list(SHAPE_COLUMNS)],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
def test_shape_to_timing_direction_remains_open_only_when_requested(variant):
    net = model(variant)
    sample = torch.randn(2, 6, 12, requires_grad=True)
    output = net(sample, 10, torch.randn(2, 276))
    derivative = torch.autograd.grad(
        output[..., list(TIMING_COLUMNS)].square().sum(), sample
    )[0]
    shape_gradient = derivative[..., list(SHAPE_COLUMNS)]
    if variant == "separate":
        assert torch.count_nonzero(shape_gradient) == 0
    else:
        assert shape_gradient.abs().max() > 1e-7


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
def test_temporal_causality_applies_to_both_streams(variant):
    net = model(variant)
    sample, condition = torch.randn(2, 6, 12), torch.randn(2, 276)
    changed = sample.clone()
    changed[:, 3:] += 10 * torch.randn_like(changed[:, 3:])
    torch.testing.assert_close(
        net(sample, 5, condition)[:, :3],
        net(changed, 5, condition)[:, :3],
        rtol=0,
        atol=0,
    )


def test_equal_parameter_budget_and_identical_initialization():
    networks = {variant: model(variant, full_size=True) for variant in DECODER_VARIANTS}
    counts = {
        key: sum(p.numel() for p in net.parameters()) for key, net in networks.items()
    }
    assert set(counts.values()) == {4_791_564}
    shared = dict(networks["shared"].named_parameters())
    for variant, net in networks.items():
        for name, parameter in net.named_parameters():
            original_name = name
            if name.startswith("shape_decoder.layers."):
                original_name = name.replace("shape_decoder", "decoder", 1)
            elif name.startswith("timing_decoder.layers."):
                parts = name.split(".")
                parts[0], parts[2] = "decoder", str(int(parts[2]) + 2)
                original_name = ".".join(parts)
            torch.testing.assert_close(parameter, shared[original_name], rtol=0, atol=0)
        groups = net.get_optim_groups(lr=5e-5, weight_decay=0)
        ids = [id(p) for group in groups for p in group["params"]]
        assert len(ids) == len(set(ids)) == len(list(net.parameters()))
        assert set(ids) == {id(p) for p in net.parameters()}
    separate = networks["separate"]
    assert (
        len(separate.shape_decoder.layers) == len(separate.timing_decoder.layers) == 2
    )
    assert not {id(p) for p in separate.shape_decoder.parameters()} & {
        id(p) for p in separate.timing_decoder.parameters()
    }


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
def test_outputs_restore_original_codec_columns(variant):
    net = model(variant)
    with torch.no_grad():
        for stream, columns in (("shape", SHAPE_COLUMNS), ("timing", TIMING_COLUMNS)):
            net.head[stream].weight.zero_()
            net.head[stream].bias.copy_(torch.tensor(columns))
    output = net(torch.randn(2, 6, 12), 5, torch.randn(2, 276))
    torch.testing.assert_close(output, torch.arange(12).expand(2, 6, 12).float())


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
def test_real_diffusion_loss_trains_every_nonempty_parameter_and_reloads(variant):
    net = model(variant)
    condition = torch.randn(2, 276)
    noise, action = torch.randn(2, 6, 12), torch.randn(2, 6, 12)
    timestep = torch.tensor([4, 67])
    noisy = scheduler().add_noise(action, noise, timestep)
    prediction = net(noisy, timestep, condition)
    loss = (prediction - noise).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in net.parameters()
        if p.numel()
    )
    restored = model(variant)
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(
        restored(noisy, timestep, condition), prediction, rtol=0, atol=0
    )


@pytest.mark.parametrize("variant", ("shape_masked", "separate"))
def test_repeated_ddim_sampling_cannot_leak_timing_back_into_shape(variant):
    net = model(variant)
    policy = DiffusionPolicy(net, scheduler(), action_horizon=6, num_inference_steps=10)
    condition = torch.randn(2, 276)
    initial = torch.randn(2, 6, 12)
    changed = initial.clone()
    changed[..., list(TIMING_COLUMNS)] = torch.randn(2, 6, 2)
    with torch.no_grad():
        first = policy.inference(initial.clone(), condition)
        second = policy.inference(changed, condition)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(
        first[..., list(SHAPE_COLUMNS)],
        second[..., list(SHAPE_COLUMNS)],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "override", [dict(input_dim=11), dict(n_layer=3), dict(causal_attn=False)]
)
def test_rejects_incompatible_comparison_contract(override):
    kwargs = dimensions()
    kwargs.update(override)
    with pytest.raises(ValueError):
        ArcStreamDiffusionTransformer(decoder_variant="shared", **kwargs)
