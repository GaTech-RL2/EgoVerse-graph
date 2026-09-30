"""MoT routing, attention parity, forbidden paths, diffusion, and optimizer contracts."""

import copy

import pytest
import torch
from torch import nn

from egomimic.benchmarks.libero.arc_decoders import (
    MOT_VARIANTS,
    decoder_metadata,
    mot_architecture,
)
from egomimic.models.arc_diffusion import SHAPE_COLUMNS, TIMING_COLUMNS
from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.models.modality_diffusion import (
    ModalityDecoderLayer,
    ModalityDiffusionTransformer,
)
from tests.test_oat_diffusion import dimensions, scheduler


def model(variant="mot_shape_velocity", *, full_size=False):
    architecture = mot_architecture(variant)
    kwargs = dimensions(horizon=32 if full_size else 6)
    kwargs.update(
        n_emb=architecture["width"] if full_size else 32, p_drop_emb=0, p_drop_attn=0
    )
    torch.manual_seed(72)
    return ModalityDiffusionTransformer(
        modality_columns=architecture["modalities"],
        blocked_attention=architecture["blocked_attention"],
        dim_feedforward=architecture["feedforward"] if full_size else 80,
        **kwargs,
    ).eval()


def test_identically_weighted_experts_equal_joint_pytorch_decoder_outputs_and_gradients():
    """Check the new attention operation against a distinct trusted implementation."""
    torch.manual_seed(91)
    layer = ModalityDecoderLayer(
        ("a", "b", "c"), width=32, heads=4, feedforward=64, dropout=0
    ).eval()
    original = copy.deepcopy(layer.experts["a"])
    for expert in layer.experts.values():
        expert.load_state_dict(original.state_dict())
    sample = torch.randn(2, 12, 32, requires_grad=True)
    other = sample.detach().clone().requires_grad_()
    memory = torch.randn(2, 3, 32)
    waypoint = torch.arange(4).repeat(3)
    mask = torch.zeros(12, 12).masked_fill(
        waypoint[None, :] > waypoint[:, None], float("-inf")
    )
    memory_mask = torch.zeros(4, 3)
    memory_mask[0, 2] = float("-inf")
    actual = torch.cat(
        layer(sample.split(4, dim=1), memory, mask=mask, memory_mask=memory_mask), dim=1
    )
    expected = original(
        other, memory, tgt_mask=mask, memory_mask=memory_mask.repeat(3, 1)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(sample.grad, other.grad, atol=2e-6, rtol=1e-5)
    for name, parameter in original.named_parameters():
        expert_gradients = sum(
            dict(expert.named_parameters())[name].grad
            for expert in layer.experts.values()
        )
        torch.testing.assert_close(
            expert_gradients, parameter.grad, atol=2e-5, rtol=1e-4
        )


@pytest.mark.parametrize("variant", MOT_VARIANTS)
def test_modalities_own_every_projection_ffn_and_norm_and_restore_channels(variant):
    net = model(variant)
    for layer in net.decoder:
        seen = set()
        for expert in layer.experts.values():
            parameters = {id(p) for p in expert.parameters()}
            assert not parameters & seen
            seen |= parameters
            assert isinstance(expert.norm1, nn.LayerNorm)
            assert isinstance(expert.norm2, nn.LayerNorm)
            assert isinstance(expert.norm3, nn.LayerNorm)
    with torch.no_grad():
        for name, columns in net.modality_columns.items():
            net.head[name].weight.zero_()
            net.head[name].bias.copy_(torch.tensor(columns))
    torch.testing.assert_close(
        net(torch.randn(2, 6, 12), 5, torch.randn(2, 276)),
        torch.arange(12).expand(2, 6, 12).float(),
    )


@pytest.mark.parametrize("variant", MOT_VARIANTS)
def test_temporal_causality_and_gradients_for_cross_modality_information_flow(variant):
    net = model(variant)
    sample = torch.randn(2, 6, 12, requires_grad=True)
    condition = torch.randn(2, 276)
    prediction = net(sample, torch.tensor([5, 9]), condition)
    changed = sample.detach().clone()
    changed[:, 3:] += 10 * torch.randn_like(changed[:, 3:])
    torch.testing.assert_close(
        prediction[:, :3],
        net(changed, torch.tensor([5, 9]), condition)[:, :3],
        atol=0,
        rtol=0,
    )
    for query, query_columns in net.modality_columns.items():
        derivative = torch.autograd.grad(
            prediction[..., list(query_columns)].square().sum(),
            sample,
            retain_graph=True,
        )[0]
        for key, key_columns in net.modality_columns.items():
            gradient = derivative[..., list(key_columns)]
            if (query, key) in net.blocked_attention:
                assert torch.count_nonzero(gradient) == 0
            else:
                assert gradient.abs().max() > 1e-7, (query, key)


def test_masked_shape_is_invariant_to_velocity_after_all_ten_ddim_steps():
    net = model("mot_shape_velocity_masked")
    policy = DiffusionPolicy(net, scheduler(), action_horizon=6, num_inference_steps=10)
    condition, initial = torch.randn(2, 276), torch.randn(2, 6, 12)
    changed = initial.clone()
    changed[..., list(TIMING_COLUMNS)] += 10 * torch.randn(2, 6, 2)
    with torch.no_grad():
        first, second = (
            policy.inference(initial, condition),
            policy.inference(changed, condition),
        )
    assert torch.isfinite(first).all()
    torch.testing.assert_close(
        first[..., list(SHAPE_COLUMNS)],
        second[..., list(SHAPE_COLUMNS)],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        first[..., list(TIMING_COLUMNS)], second[..., list(TIMING_COLUMNS)]
    )


@pytest.mark.parametrize("variant", MOT_VARIANTS)
def test_epsilon_training_bfloat16_checkpoint_and_optimizer_include_all_experts(
    variant,
):
    net = model(variant)
    noise, action, condition = (
        torch.randn(2, 6, 12),
        torch.randn(2, 6, 12),
        torch.randn(2, 276),
    )
    timestep = torch.tensor([5, 19])
    noisy = scheduler().add_noise(action, noise, timestep)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = net(noisy, timestep, condition)
        loss = (prediction - noise).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
        for p in net.parameters()
        if p.numel()
    )
    groups = net.get_optim_groups(lr=5e-5, weight_decay=0)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(net.parameters()))
    restored = model(variant)
    restored.load_state_dict(net.state_dict(), strict=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(
            restored(noisy, timestep, condition), prediction, atol=0, rtol=0
        )


def test_full_parameter_budget_and_mask_only_pair_initialization():
    counts = {
        variant: decoder_metadata(variant, 32)["decoder_parameters"]
        for variant in MOT_VARIANTS
    }
    assert counts == {
        "mot_xyz_rot_gripper": 4_790_284,
        "mot_shape_velocity": 4_787_532,
        "mot_shape_velocity_masked": 4_787_532,
    }
    assert all(abs(count / 4_791_564 - 1) < 0.002 for count in counts.values())
    open_net, masked = (
        model("mot_shape_velocity", full_size=True),
        model("mot_shape_velocity_masked", full_size=True),
    )
    for (name, parameter), (other_name, other) in zip(
        open_net.named_parameters(), masked.named_parameters()
    ):
        assert name == other_name
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)


@pytest.mark.parametrize(
    "columns,edges",
    [
        ({"a": [0, 1], "b": [1, 2]}, []),
        ({"a": [0]}, []),
        ({"a": list(range(12))}, [["a", "unknown"]]),
        ({"a": list(range(12))}, [["a", "a"]]),
    ],
)
def test_invalid_partitions_and_masks_are_rejected(columns, edges):
    with pytest.raises(ValueError):
        ModalityDiffusionTransformer(
            modality_columns=columns, blocked_attention=edges, **dimensions()
        )
