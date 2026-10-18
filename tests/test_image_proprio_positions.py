"""Axial time/XY RoPE geometry, legacy identity and packed execution parity."""

from contextlib import nullcontext

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.models.attention_execution import (
    apply_temporal_rope,
    prepare_spatiotemporal_rope,
    resolve_spatiotemporal_rope_geometry,
)
from egomimic.models.image_proprio import ImageProprioModel
from scripts.world_model.benchmark_pixel_execution import fixed_corruption
from tests.test_image_proprio_adaln import activate_conditioning
from tests.test_pixel_packed_execution import (
    assert_gradient_parity,
    batch,
    model,
    reference_kernels,  # noqa: F401 -- shared explicit CPU attention fixture
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def spatial_model(**kwargs):
    settings = dict(
        noise_conditioning="adaln_zero",
        use_phase_embedding=False,
        position_encoding="spatiotemporal_rope",
        rope_axis_dims=[4, 2, 2],
        spatial_grid_size=[2, 2],
    )
    settings.update(kwargs)
    return model(**settings)


def test_axes_use_centered_patch_units_and_correct_half_split_pairs():
    times = torch.tensor([[0.0, 1.0]])
    xy = torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]])
    rope = prepare_spatiotemporal_rope(
        times, xy, 8, axis_dims=[4, 2, 2], spatial_grid_size=[2, 2]
    )
    expected_phase = torch.tensor([[[[0.0, 0.0, -0.5, -0.5], [1.0, 0.01, 0.5, 0.5]]]])
    torch.testing.assert_close(rope[0], expected_phase.cos())
    torch.testing.assert_close(rope[1], expected_phase.sin())
    basis = torch.zeros(1, 1, 2, 8)
    basis[..., 2] = 1  # x cosine coordinate pairs with index6, not adjacent3.
    rotated = apply_temporal_rope(basis, rope)
    torch.testing.assert_close(rotated[..., 2], expected_phase[..., 2].cos())
    torch.testing.assert_close(rotated[..., 6], expected_phase[..., 2].sin())
    assert rotated[..., [0, 1, 3, 4, 5, 7]].count_nonzero() == 0


def test_joint_coordinate_translation_preserves_attention_dot_products():
    torch.manual_seed(6)
    q, k = torch.randn(2, 3, 5, 16), torch.randn(2, 3, 5, 16)
    times, xy = torch.randn(2, 5), torch.randn(2, 5, 2)
    arguments = dict(head_dim=16, axis_dims=[8, 4, 4], spatial_grid_size=[10, 10])
    original = prepare_spatiotemporal_rope(times, xy, **arguments)
    shifted = prepare_spatiotemporal_rope(
        times + 13, xy + torch.tensor([0.2, -0.3]), **arguments
    )

    def logits(table):
        return apply_temporal_rope(q, table) @ apply_temporal_rope(k, table).transpose(
            -1, -2
        )

    torch.testing.assert_close(logits(original), logits(shifted), rtol=1e-5, atol=1e-5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = prepare_spatiotemporal_rope(times, xy, **arguments)
    assert all(value.dtype == torch.float32 for value in mixed)
    assert all(torch.equal(a, b) for a, b in zip(original, mixed))


def test_nonimage_tokens_share_temporal_frequencies_with_zero_spatial_rotation():
    times = torch.tensor([[0.2, 2.1]])
    kwargs = dict(head_dim=128, axis_dims=[64, 32, 32], spatial_grid_size=[10, 10])
    image = prepare_spatiotemporal_rope(times, torch.ones(1, 2, 2), **kwargs)
    pose = prepare_spatiotemporal_rope(times, None, **kwargs)
    assert all(torch.equal(a[..., :32], b[..., :32]) for a, b in zip(image, pose))
    assert torch.equal(pose[0][..., 32:], torch.ones_like(pose[0][..., 32:]))
    assert pose[1][..., 32:].count_nonzero() == 0


def test_hydra_geometry_and_discrete_integer_clock_match_float_tables():
    axes, grid = OmegaConf.create([4, 2, 2]), OmegaConf.create([2, 2])
    assert resolve_spatiotemporal_rope_geometry(8, axes, grid) == ((4, 2, 2), (2, 2))
    times = torch.tensor([[0, 1, 3]], dtype=torch.int64)
    xy = torch.zeros(1, 3, 2)
    expected = prepare_spatiotemporal_rope(
        times.float(), xy, 8, axis_dims=axes, spatial_grid_size=grid
    )
    actual = prepare_spatiotemporal_rope(
        times, xy, 8, axis_dims=axes, spatial_grid_size=grid
    )
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    network = spatial_model(
        backend="sdpa",
        attention_dtype="native",
        time_mode="discrete",
        rope_axis_dims=axes,
        spatial_grid_size=grid,
    )
    world = batch()
    world["image"]["index"] = world["image"]["index"].long()
    noisy, levels = fixed_corruption(world, 8)
    assert torch.isfinite(
        network(world, "libero_gripper", noisy, levels)["image"]
    ).all()


@pytest.mark.parametrize(
    "axes,grid",
    [([4, 3, 1], [2, 2]), ([4, 2, 4], [2, 2]), ([4, 2, 2], None), ([4, 2, 2], [0, 2])],
)
def test_explicit_spatial_geometry_is_validated(axes, grid):
    with pytest.raises(ValueError):
        resolve_spatiotemporal_rope_geometry(8, axes, grid)


def test_new_parameter_count_and_legacy_rng_state_compatibility():
    with torch.device("meta"):
        full = ImageProprioModel(
            objective="visual_only",
            phase_a_condition_on_proprio=False,
            noise_conditioning="adaln_zero",
            use_phase_embedding=False,
            max_cameras=2,
            position_encoding="spatiotemporal_rope",
            rope_axis_dims=[64, 32, 32],
            spatial_grid_size=[10, 10],
        )
    assert sum(parameter.numel() for parameter in full.parameters()) == 88_178_688
    assert not hasattr(full, "xy") and hasattr(full, "camera")
    assert not any(name.startswith("xy.") for name in full.state_dict())
    torch.manual_seed(29)
    default = model()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(29)
    explicit = model(
        position_encoding="linear_xy_temporal_rope",
        rope_axis_dims=None,
        spatial_grid_size=None,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(value, explicit.state_dict()[name])
        for name, value in default.state_dict().items()
    )
    assert resolve_spatiotemporal_rope_geometry(8, None, [2, 2])[0] == (4, 2, 2)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("time_mode", ["continuous", "discrete"])
@pytest.mark.usefixtures("reference_kernels")
def test_nonzero_spatial_packed_outputs_and_all_gradients_match_padded(
    mixed, time_mode
):
    padded = spatial_model(time_mode=time_mode)
    activate_conditioning(padded)
    packed = spatial_model(time_mode=time_mode, token_execution="packed_valid")
    packed.load_state_dict(padded.state_dict(), strict=True)
    world = batch(nonprefix=True)
    noisy, levels = fixed_corruption(world, 20)
    outputs = []
    for network in (padded, packed):
        with torch.autocast("cpu", dtype=torch.bfloat16) if mixed else nullcontext():
            output = network(world, "libero_gripper", noisy, levels)["image"]
            loss = output.float().square().sum()
        loss.backward()
        outputs.append(output.detach())
    assert outputs[0].abs().max() > 0.01
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.03, atol=0.003)
    assert_gradient_parity(packed, padded)


def test_sampler_reuses_tables_and_phase_b_nonimage_axes_are_aligned(monkeypatch):
    network = spatial_model(backend="sdpa", attention_dtype="native")
    activate_conditioning(network)
    world = batch()
    calls = []
    original = network._prepare_rope

    def prepare(times, xy=None):
        result = original(times, xy)
        calls.append(result)
        return result

    monkeypatch.setattr(network, "_prepare_rope", prepare)
    output = network.sample(world, "libero_gripper", steps=4)["image"]
    assert len(calls) == 1 and torch.isfinite(output).all()
    times = torch.tensor([[1.0, 2.0]])
    _, rope = network._prepare_execution(
        [times, times], [torch.ones(1, 2, dtype=torch.bool)] * 2, torch.ones(1, 2, 2)
    )
    assert all(torch.equal(a[..., :2], b[..., :2]) for a, b in zip(*rope))
    assert rope[1][1][..., 2:].count_nonzero() == 0


def test_spatial_fullgraph_tensor_regions_match_eager():
    eager = spatial_model(backend="sdpa", attention_dtype="native")
    activate_conditioning(eager)
    compiled = spatial_model(
        backend="sdpa",
        attention_dtype="native",
        compile_mode="default",
        compile_backend="aot_eager",
        compile_dynamic=True,
    )
    compiled.load_state_dict(eager.state_dict(), strict=True)
    world = batch(nonprefix=True)
    noisy, levels = fixed_corruption(world, 9)
    results = []
    for network in (eager, compiled):
        result = network(world, "libero_gripper", noisy, levels)["image"]
        result.square().mean().backward()
        results.append(result.detach())
    torch.testing.assert_close(*results, rtol=1e-5, atol=1e-6)
    for name, parameter in eager.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(compiled.named_parameters())[name].grad,
            rtol=2e-5,
            atol=2e-6,
            msg=name,
        )


def test_full_image_model_has_no_additive_xy_and_keeps_relative_time_origin():
    network = spatial_model(backend="sdpa", attention_dtype="native")
    activate_conditioning(network)
    world = batch(nonprefix=True)
    noisy, levels = fixed_corruption(world, 31)
    expected = network(world, "libero_gripper", noisy, levels)["image"]
    shifted = {"image": {key: value.clone() for key, value in world["image"].items()}}
    shifted["image"]["time"] += 10
    shifted["image"]["xy"] += torch.tensor([0.2, -0.4])
    actual = network(shifted, "libero_gripper", noisy, levels)["image"]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert network.camera.weight.numel() > 0 and not hasattr(network, "xy")
