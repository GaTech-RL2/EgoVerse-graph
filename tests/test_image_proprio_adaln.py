"""DiT-style image noise conditioning with per-frame shared modulation."""

import copy
from contextlib import nullcontext

import pytest
import torch

from egomimic.models.image_proprio import ImageProprioModel
from scripts.world_model.benchmark_pixel_execution import fixed_corruption
from tests.test_pixel_packed_execution import (
    assert_gradient_parity,
    batch,
    reference_kernels,  # noqa: F401 -- explicitly shared CPU kernel fixture
)
from tests.test_pixel_packed_execution import (
    model as packed_model,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(**kwargs):
    settings = dict(noise_conditioning="adaln_zero", use_phase_embedding=False)
    settings.update(kwargs)
    return packed_model(backend="sdpa", attention_dtype="native", **settings)


def activate_conditioning(network):
    with torch.no_grad():
        network.image_out.weight.normal_(std=0.08)
        network.image_out.bias.normal_(std=0.02)
        network.final_adaln[-1].weight.normal_(std=0.1)
        for block in network.blocks:
            block.adaln_image[-1].weight.normal_(std=0.1)
            block.adaln_image[-1].bias.normal_(std=0.05)


def test_additive_default_preserves_seeded_state_rng_and_outputs():
    torch.manual_seed(12)
    original = model(noise_conditioning="additive")
    rng = torch.get_rng_state().clone()
    torch.manual_seed(12)
    explicit = packed_model(backend="sdpa", attention_dtype="native", use_phase_embedding=False)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, explicit.state_dict()[name]) for name, value in original.state_dict().items())
    world = batch(nonprefix=True)
    noisy, levels = fixed_corruption(world, 8)
    assert torch.equal(original(world, "libero_gripper", noisy, levels)["image"],
                       explicit(world, "libero_gripper", noisy, levels)["image"])


def test_standard_zero_initialization_and_exact_parameter_count():
    with torch.device("meta"):
        full = ImageProprioModel(objective="visual_only", phase_a_condition_on_proprio=False,
                                 max_cameras=2, use_phase_embedding=False, noise_conditioning="adaln_zero")
    assert sum(parameter.numel() for parameter in full.parameters()) == 88_180_992
    network = model()
    assert not network.norm[0].elementwise_affine
    assert all(not block.norm1[0].elementwise_affine and not block.norm2[0].elementwise_affine for block in network.blocks)
    world = batch()
    noisy, levels = fixed_corruption(world, 7)
    prediction = network(world, "libero_gripper", noisy, levels)["image"]
    assert prediction.count_nonzero() == 0
    network.training_losses(world, "libero_gripper")["total"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in network.parameters())
    assert network.image_out.weight.grad.abs().sum() > 0
    assert network.blocks[0].qkv[0].weight.grad.count_nonzero() == 0


def test_frame_modulation_is_shared_but_examples_and_levels_remain_distinct():
    network = model()
    activate_conditioning(network)
    world = batch(nonprefix=True)
    noisy, levels = fixed_corruption(world, 11)
    frames = network._prepare_noise_frame_map(world["image"])
    assert frames.representatives.numel() == 5  #2 and3frames, each8camera/patchtokens.
    shapes = []
    handles = [block.adaln_image[-1].register_forward_pre_hook(lambda _, args: shapes.append(args[0].shape))
               for block in network.blocks]
    first = network(world, "libero_gripper", noisy, levels)["image"]
    for handle in handles:
        handle.remove()
    assert shapes == [torch.Size([5, 16])] * 2
    changed = levels.clone()
    changed[1] = changed[1] * 0.5
    second = network(world, "libero_gripper", noisy, changed)["image"]
    assert torch.equal(first[0], second[0])
    assert not torch.allclose(first[1], second[1], atol=1e-7, rtol=1e-7)
    # Input embeddings no longer receive additive noise vectors.
    image = world["image"]
    assert torch.equal(network._embed_image(image, noisy, levels, image["valid"], "A", False),
                       network._embed_image(image, noisy, changed, image["valid"], "A", False))
    query = image["valid"] & ~image["observed"]
    row, token = query.nonzero()[0].tolist()
    invalid = levels.clone()
    invalid[row, token] += 0.01
    with pytest.raises(ValueError, match="agree within each timestamp"):
        network(world, "libero_gripper", noisy, invalid)


def test_sampler_caches_only_frame_mapping_and_recomputes_noise_embedding(monkeypatch):
    network = model()
    activate_conditioning(network)
    world = batch()
    calls, levels_seen = [], []
    original = network._prepare_noise_frame_map

    def record_map(image):
        calls.append(True)
        return original(image)

    monkeypatch.setattr(network, "_prepare_noise_frame_map", record_map)
    original_embed = network._embed_noise_frames_execute

    def record_embed(levels):
        levels_seen.append(levels.clone())
        return original_embed(levels)

    object.__setattr__(network, "_embed_noise_frames_execute", record_embed)
    output = network.sample(world, "libero_gripper", steps=4, generator=torch.Generator().manual_seed(4))["image"]
    assert len(calls) == 1 and len(levels_seen) == 4
    assert [float(level.max()) for level in levels_seen] == [1.0, 0.75, 0.5, 0.25]
    assert all((level == 0).sum() == 2 for level in levels_seen)
    assert torch.isfinite(output).all()
    observed = world["image"]["observed"]
    assert torch.equal(output[observed], world["image"]["value"][observed])


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("time_mode", ["continuous", "discrete"])
@pytest.mark.usefixtures("reference_kernels")
def test_nonzero_packed_conditioning_matches_padded_outputs_and_gradients(mixed, time_mode):
    arguments = dict(noise_conditioning="adaln_zero", use_phase_embedding=False, time_mode=time_mode)
    padded = packed_model(**arguments)
    activate_conditioning(padded)
    packed = packed_model(token_execution="packed_valid", **arguments)
    packed.load_state_dict(padded.state_dict(), strict=True)
    world = batch(nonprefix=True)
    noisy, levels = fixed_corruption(world, 19)
    outputs = []
    for network in (padded, packed):
        with torch.autocast("cpu", dtype=torch.bfloat16) if mixed else nullcontext():
            prediction = network(world, "libero_gripper", noisy, levels)["image"]
            loss = prediction.float().square().sum()
        loss.backward()
        outputs.append(prediction.detach())
    assert outputs[0].abs().max() > 0.01
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0.03, atol=0.003)
    assert_gradient_parity(packed, padded)
    assert packed.noise_level[0].weight.grad.abs().sum() > 0
    changed = copy.deepcopy(world)
    changed["image"]["value"][1] *= -3
    altered = noisy.clone()
    altered[1] *= 2
    with torch.no_grad():
        expected = packed(world, "libero_gripper", noisy, levels)["image"][0]
        isolated = packed(changed, "libero_gripper", altered, levels)["image"][0]
    assert torch.equal(expected, isolated)


def test_fullgraph_tensor_regions_match_eager_with_nonzero_conditioning():
    eager = model()
    activate_conditioning(eager)
    compiled = model(compile_mode="default", compile_backend="aot_eager", compile_dynamic=True)
    compiled.load_state_dict(eager.state_dict(), strict=True)
    world = batch(nonprefix=True)
    noisy, levels = fixed_corruption(world, 23)
    outputs = []
    for network in (eager, compiled):
        prediction = network(world, "libero_gripper", noisy, levels)["image"]
        prediction.square().mean().backward()
        outputs.append(prediction.detach())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    for name, parameter in eager.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(compiled.named_parameters())[name].grad,
                                   rtol=2e-5, atol=2e-6, msg=name)
