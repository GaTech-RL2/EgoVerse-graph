"""Packed image-expert parity using explicit CPU reference attention kernels.

These tests exercise the actual packing/layout/transformer/scatter path, while
replacing optional CUDA kernel loading with isolated SDPA math references. They
make no claim about actual FA3/FA4 kernels or CUDA Inductor performance.
"""

from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F

from egomimic.models.attention_execution import FlashAttentionKernels
from egomimic.models.image_proprio import ImageProprioModel
from scripts.world_model import benchmark_pixel_execution as benchmark


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def reference_kernels(monkeypatch):
    calls = []

    def dense(q, k, v, *, causal):
        assert causal is False
        return (
            F.scaled_dot_product_attention(
                q.float().transpose(1, 2),
                k.float().transpose(1, 2),
                v.float().transpose(1, 2),
                dropout_p=0.0,
            )
            .transpose(1, 2)
            .to(q.dtype)
        )

    def load(backend, device):
        assert backend in ("flash3", "flash4")
        assert torch.device(device).type == "cpu"

        def dense_reference(q, k, v, **kwargs):
            calls.append(("dense", tuple(q.shape)))
            output = dense(q, k, v, **kwargs)
            return (output, None) if backend == "flash4" else output

        def packed_reference(
            q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal
        ):
            assert cu_seqlens_q.dtype == cu_seqlens_k.dtype == torch.int32
            qs, ks = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
            assert max_seqlen_q == max(b - a for a, b in zip(qs, qs[1:]))
            assert max_seqlen_k == max(b - a for a, b in zip(ks, ks[1:]))
            assert qs[-1] == len(q) and ks[-1] == len(k)
            calls.append(("varlen", tuple(q.shape), tuple(qs), tuple(ks)))
            output = torch.cat(
                [
                    dense(
                        q[qa:qb][None], k[ka:kb][None], v[ka:kb][None], causal=causal
                    )[0]
                    for qa, qb, ka, kb in zip(qs, qs[1:], ks, ks[1:])
                ]
            )
            return (output, None) if backend == "flash4" else output

        return FlashAttentionKernels(backend, dense_reference, packed_reference)

    monkeypatch.setattr(
        "egomimic.models.image_proprio.load_flash_attention_kernels", load
    )
    monkeypatch.setattr(
        "egomimic.models.world_model.load_flash_attention_kernels", load
    )
    return calls


def model(token_execution="padded", backend="flash3", **overrides):
    arguments = dict(
        objective="visual_only",
        phase_a_condition_on_proprio=False,
        image_dim=12,
        width=16,
        depth=2,
        heads=2,
        max_tokens=32,
        mlp_ratio=2,
        attention_backend=backend,
        attention_dtype="bfloat16",
        loss_reduction="mean_per_example",
        token_execution=token_execution,
    )
    arguments.update(overrides)
    return ImageProprioModel(**arguments)


def batch(nonprefix=False, all_valid=False):
    result = benchmark.make_batch(
        (3, 3) if all_valid else (2, 3),
        seed=57,
        device=torch.device("cpu"),
        image_size=4,
        patch_size=2,
    )
    image = result["image"]
    if nonprefix:
        permutation = torch.tensor(
            [
                0,
                16,
                1,
                17,
                2,
                18,
                3,
                19,
                4,
                20,
                5,
                21,
                6,
                22,
                7,
                23,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
            ]
        )
        for key, value in image.items():
            image[key] = value.index_select(1, permutation)
    # Invalid padding must be eliminated before it can contaminate features.
    image["value"][~image["valid"]] = float("nan")
    image["xy"][~image["valid"]] = float("nan")
    image["time"][~image["valid"]] = float("nan")
    image["camera"][~image["valid"]] = -999
    return result


def relative_l2(actual, reference):
    actual, reference = actual.float(), reference.float()
    return float((actual - reference).norm() / reference.norm().clamp_min(1e-8))


def assert_gradient_parity(actual, reference):
    assert (
        dict(actual.named_parameters()).keys()
        == dict(reference.named_parameters()).keys()
    )
    errors, scales = 0.0, 0.0
    for name, parameter in actual.named_parameters():
        expected = dict(reference.named_parameters())[name]
        assert parameter.grad is not None and expected.grad is not None, name
        assert parameter.grad.dtype == expected.grad.dtype == torch.float32
        assert (
            torch.isfinite(parameter.grad).all() and torch.isfinite(expected.grad).all()
        ), name
        torch.testing.assert_close(
            parameter.grad, expected.grad, rtol=0.04, atol=0.002, msg=name
        )
        errors += float((parameter.grad - expected.grad).double().square().sum())
        scales += float(expected.grad.double().square().sum())
    assert (errors / max(scales, 1e-30)) ** 0.5 < 0.03


@pytest.mark.parametrize("backend", ["flash3", "flash4"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("time_mode", ["continuous", "discrete"])
@pytest.mark.parametrize("nonprefix", [False, True])
def test_packed_outputs_all_parameter_gradients_and_projection_work_match(
    reference_kernels, backend, mixed, nonprefix, time_mode
):
    torch.manual_seed(71)
    padded, packed = (
        model(backend=backend, time_mode=time_mode),
        model("packed_valid", backend, time_mode=time_mode),
    )
    packed.load_state_dict(padded.state_dict(), strict=True)
    world = batch(nonprefix)
    noisy, levels = benchmark.fixed_corruption(world, 19)
    shapes = []
    hooks = [
        module.register_forward_pre_hook(
            lambda _, args: shapes.append(tuple(args[0].shape))
        )
        for module in (
            packed.image_in,
            packed.image_out,
            *(block.qkv[0] for block in packed.blocks),
        )
    ]
    outputs = []
    for network in (padded, packed):
        context = (
            torch.autocast("cpu", dtype=torch.bfloat16) if mixed else nullcontext()
        )
        with context:
            predicted = network(world, "libero_gripper", noisy, levels)["image"]
            loss = benchmark.example_mean_loss(predicted, world)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        outputs.append(predicted.detach())
    for hook in hooks:
        hook.remove()
    assert len(shapes) == 4
    assert all(shape[:2] == (1, int(world["image"]["valid"].sum())) for shape in shapes)
    valid = world["image"]["valid"]
    assert relative_l2(outputs[1][valid], outputs[0][valid]) < 0.03
    assert not outputs[1][~valid].any()
    assert_gradient_parity(packed, padded)
    assert all(call[0] == "varlen" for call in reference_kernels)
    assert all(call[2] == call[3] == (0, 16, 40) for call in reference_kernels)


@pytest.mark.parametrize("backend", ["flash3", "flash4"])
def test_full_valid_batch_retains_existing_dense_path(
    reference_kernels, monkeypatch, backend
):
    padded, packed = model(backend=backend), model("packed_valid", backend)
    packed.load_state_dict(padded.state_dict(), strict=True)
    for block in packed.blocks:
        monkeypatch.setattr(
            block,
            "forward_packed",
            lambda *a, **kw: pytest.fail("unneeded packed path"),
        )
    world = batch(all_valid=True)
    noisy, levels = benchmark.fixed_corruption(world, 29)
    actual = packed(world, "libero_gripper", noisy, levels)["image"]
    expected = padded(world, "libero_gripper", noisy, levels)["image"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(reference_kernels) == 4 and all(
        call[0] == "dense" for call in reference_kernels
    )


def test_packing_does_not_change_full_shaped_noising_or_rng_order(
    reference_kernels, monkeypatch
):
    padded, packed = model(), model("packed_valid")
    packed.load_state_dict(padded.state_dict(), strict=True)
    world = batch(nonprefix=True)
    captures = []
    for network in (padded, packed):
        original = network._noisy_images

        def record(*args, _original=original, **kwargs):
            noisy, levels = _original(*args, **kwargs)
            captures.append((noisy.detach().clone(), levels.detach().clone()))
            return noisy, levels

        monkeypatch.setattr(network, "_noisy_images", record)
    results, states = [], []
    for network in (padded, packed):
        torch.manual_seed(91)
        results.append(network.training_losses(world, "libero_gripper"))
        states.append(torch.get_rng_state().clone())
    assert torch.equal(states[0], states[1])
    assert captures[0][0].shape == world["image"]["value"].shape
    assert all(torch.equal(left, right) for left, right in zip(*captures))
    torch.testing.assert_close(
        results[0]["total"], results[1]["total"], rtol=0.03, atol=1e-6
    )


def test_packed_benchmark_independent_b1_and_same_mode_isolation(reference_kernels):
    result = benchmark.check_outputs(
        model("packed_valid"),
        batch(nonprefix=True),
        backend="flash3",
        seed=27,
        tolerance=0.03,
        gradients=True,
    )
    assert result["output_relative_l2"] <= 0.03
    assert result["gradient_relative_l2"] <= 0.03
    assert result["cross_example_relative_l2"] == 0


def test_phase_a_sampler_matches_and_reuses_one_packed_layout(
    reference_kernels, monkeypatch
):
    padded, packed = model(), model("packed_valid")
    packed.load_state_dict(padded.state_dict(), strict=True)
    world = batch(nonprefix=True)
    calls = []
    original = packed._prepare_image_execution

    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(packed, "_prepare_image_execution", prepare)
    torch.manual_seed(102)
    expected = padded.sample(world, "libero_gripper", steps=3)["image"]
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(102)
    actual = packed.sample(world, "libero_gripper", steps=3)["image"]
    assert len(calls) == 1
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert relative_l2(actual, expected) < 0.03
    observed = world["image"]["observed"]
    assert torch.equal(actual[observed], world["image"]["value"][observed])
    assert not actual[~world["image"]["valid"]].any()
    assert actual.dtype == torch.float32


def stage_two_batch():
    world = batch(nonprefix=True)
    world["proprio"] = {
        "value": torch.tensor(
            [[[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]], [[-0.2, 0.3, 0.1], [0.5, 0.1, -0.4]]]
        ),
        "time": torch.tensor([[0.0, 0.1], [0.0, 0.1]]),
        "index": torch.tensor([[0, 2], [0, 2]]),
        "valid": torch.ones(2, 2, dtype=torch.bool),
        "observed": torch.tensor([[True, False], [True, False]]),
    }
    world["action"] = {
        **{
            key: value.clone()
            for key, value in world["proprio"].items()
            if key != "value"
        },
        "value": torch.zeros(2, 2, 2),
        "observed": torch.zeros(2, 2, dtype=torch.bool),
    }
    return world


def test_stage_two_keeps_phase_b_shared_expert_path(reference_kernels, monkeypatch):
    arguments = dict(
        objective="visual_state_action",
        pose_dims={"libero_gripper": 3},
        action_dims={"libero_gripper": 2},
    )
    padded, packed = model(**arguments), model("packed_valid", **arguments)
    packed.load_state_dict(padded.state_dict(), strict=True)
    world = stage_two_batch()
    noisy, levels = benchmark.fixed_corruption(world, 11)
    for block in packed.blocks:
        monkeypatch.setattr(
            block,
            "forward_packed",
            lambda *a, **kw: pytest.fail("Phase B must retain joint attention"),
        )
    before = tuple(packed.state_dict())
    actual = packed(world, "libero_gripper", noisy, levels, phase="B")
    expected = padded(world, "libero_gripper", noisy, levels, phase="B")
    assert set(actual) == {"image", "proprio", "action"}
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert tuple(packed.state_dict()) == before == tuple(padded.state_dict())
    assert all(len(block.qkv) == 2 for block in packed.blocks)


def test_stage_two_observed_proprio_phase_a_keeps_joint_path(
    reference_kernels, monkeypatch
):
    arguments = dict(
        objective="visual_state_action",
        phase_a_condition_on_proprio=True,
        pose_dims={"libero_gripper": 3},
        action_dims={"libero_gripper": 2},
    )
    padded, packed = model(**arguments), model("packed_valid", **arguments)
    packed.load_state_dict(padded.state_dict(), strict=True)
    for block in packed.blocks:
        monkeypatch.setattr(
            block,
            "forward_packed",
            lambda *a, **kw: pytest.fail("Observed-proprio Phase A is not image-only"),
        )
    world = stage_two_batch()
    noisy, levels = benchmark.fixed_corruption(world, 31)
    actual = packed(world, "libero_gripper", noisy, levels, phase="A")
    expected = padded(world, "libero_gripper", noisy, levels, phase="A")
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_state_keys_parameter_count_and_default_execution_are_unchanged():
    arguments = benchmark.model_arguments("flash3", "mixed")
    assert arguments["token_execution"] == "padded"
    with torch.device("meta"):
        padded = ImageProprioModel(**arguments)
        packed = ImageProprioModel(**{**arguments, "token_execution": "packed_valid"})
    assert (
        padded.token_execution == "padded" and packed.token_execution == "packed_valid"
    )
    assert tuple(packed.state_dict()) == tuple(padded.state_dict())
    assert (
        sum(p.numel() for p in padded.parameters())
        == sum(p.numel() for p in packed.parameters())
        == 88_178_688
    )
    assert all(parameter.dtype == torch.float32 for parameter in packed.parameters())


def test_unsupported_ragged_backend_or_chunks_fail_closed(reference_kernels):
    world = batch(nonprefix=True)
    noisy, levels = benchmark.fixed_corruption(world, 37)
    with pytest.raises(ValueError, match="flash3 or flash4"):
        model("packed_valid", backend="sdpa")(world, "libero_gripper", noisy, levels)
    with pytest.raises(ValueError, match="unchunked"):
        model("packed_valid", attention_query_chunk_size=2)(
            world, "libero_gripper", noisy, levels
        )
    # Fully valid native SDPA is still the declared dense route, never a fallback.
    dense = batch(all_valid=True)
    noisy, levels = benchmark.fixed_corruption(dense, 37)
    result = model("packed_valid", backend="sdpa")(
        dense, "libero_gripper", noisy, levels
    )
    assert torch.isfinite(result["image"]).all()


def test_packed_regional_compile_matches_eager_with_cpu_graph_backend(
    reference_kernels,
):
    torch.manual_seed(117)
    eager = model("packed_valid")
    compiled = model(
        "packed_valid",
        compile_mode="default",
        compile_dynamic=True,
        compile_backend="eager",
    )
    compiled.load_state_dict(eager.state_dict(), strict=True)
    world = batch(nonprefix=True)
    noisy, levels = benchmark.fixed_corruption(world, 71)
    actual = compiled(world, "libero_gripper", noisy, levels)["image"]
    expected = eager(world, "libero_gripper", noisy, levels)["image"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    benchmark.example_mean_loss(actual, world).backward()
    benchmark.example_mean_loss(expected, world).backward()
    assert_gradient_parity(compiled, eager)


def test_packed_optional_camera_embedding_does_not_require_camera_metadata(
    reference_kernels,
):
    padded, packed = (
        model(use_camera_embedding=False),
        model("packed_valid", use_camera_embedding=False),
    )
    packed.load_state_dict(padded.state_dict(), strict=True)
    world = batch(nonprefix=True)
    del world["image"]["camera"]
    noisy, levels = benchmark.fixed_corruption(world, 78)
    actual = packed(world, "libero_gripper", noisy, levels)["image"]
    expected = padded(world, "libero_gripper", noisy, levels)["image"]
    assert relative_l2(actual, expected) < 0.03
