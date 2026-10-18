"""CPU execution/masking contracts; real Flash4 capacity needs a GPU smoke."""

import copy

import pytest
import torch

from egomimic.models.dino_proprio import DinoProprioModel
from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.world_model import MoTBlock


def model(**kwargs):
    config = dict(
        pose_dims={"gripper": 3},
        objective="visual_only",
        image_dim=8,
        width=24,
        depth=2,
        heads=3,
        max_cameras=2,
        max_tokens=32,
    )
    config.update(kwargs)
    return ImageProprioModel(**config)


def example():
    torch.manual_seed(31)
    image = {
        "value": torch.randn(1, 6, 8),
        "time": torch.tensor([[-1., -1., 0., 0., 2., 2.]]),
        "valid": torch.tensor([[True, True, False, True, True, True]]),
        "observed": torch.tensor([[True, True, False, False, False, False]]),
        "xy": torch.randn(1, 6, 2),
        "camera": torch.tensor([[0, 1, 0, 1, 0, 1]]),
    }
    proprio = {
        "value": torch.randn(1, 4, 3),
        "time": torch.tensor([[-1., -.5, 0., 2.]]),
        "valid": torch.ones(1, 4, dtype=torch.bool),
        "observed": torch.tensor([[True, False, True, False]]),
    }
    return {"image": image, "proprio": proprio}


def forward(network, batch):
    return network(
        batch, "gripper", batch["image"]["value"],
        torch.zeros_like(batch["image"]["time"]), phase="A",
    )["image"]


def test_native_default_matches_historical_attention_and_strict_state_loading():
    torch.manual_seed(8)
    default = model()
    explicit = model(attention_backend="sdpa", attention_dtype="native", strict_fp32=True)
    historical_attention = model()
    explicit.load_state_dict(default.state_dict(), strict=True)
    historical_attention.load_state_dict(default.state_dict(), strict=True)
    # The old block called _attend directly. Bypassing only the new dispatcher
    # checks that native execution keeps precisely that arithmetic.
    for block in historical_attention.blocks:
        block._apply_attention = block._attend
    assert list(default.state_dict()) == list(explicit.state_dict())
    assert all(len(block.qkv) == 2 for block in explicit.blocks)
    assert not any("attention_backend" in key for key in explicit.state_dict())
    batch = example()
    expected = forward(historical_attention, batch)
    assert torch.equal(forward(default, batch), expected)
    assert torch.equal(forward(explicit, batch), expected)
    # Execution controls do not add or reshape checkpoint tensors.
    for settings in (
        {"attention_dtype": "bfloat16"},
        {"attention_backend": "flash4", "attention_dtype": "bfloat16"},
    ):
        alternate = model(**settings)
        alternate.load_state_dict(default.state_dict(), strict=True)
        assert list(alternate.state_dict()) == list(default.state_dict())


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"strict_fp32": "true"}, "boolean"),
        ({"strict_fp32": 1}, "boolean"),
        ({"strict_fp32": True, "attention_dtype": "bfloat16"}, "conflicts"),
        ({"attention_backend": "flash4"}, "explicit bfloat16"),
        ({"attention_backend": "unknown"}, "attention_backend"),
        ({"attention_dtype": "float16"}, "attention_dtype"),
    ],
)
def test_explicit_precision_guards(settings, match):
    with pytest.raises(ValueError, match=match):
        model(**settings)


@pytest.mark.parametrize("dtype", ["native", "bfloat16"])
def test_two_stream_holes_have_no_label_leakage_or_invalid_value_gradients(dtype):
    network = model(attention_dtype=dtype)
    batch = example()
    batch["image"]["value"].requires_grad_()
    batch["proprio"]["value"].requires_grad_()
    result = forward(network, batch)
    changed = copy.deepcopy(batch)
    hidden = ~changed["proprio"]["observed"]
    changed["proprio"]["value"] = changed["proprio"]["value"].detach()
    changed["proprio"]["value"][hidden] = float("nan")
    changed["proprio"]["time"][hidden] = 999
    assert torch.equal(forward(network, changed), result)
    result.square().sum().backward()
    pose_grad = batch["proprio"]["value"].grad
    assert pose_grad[hidden].count_nonzero() == 0
    assert pose_grad[~hidden].abs().sum() > 0
    assert batch["image"]["value"].grad[~batch["image"]["valid"]].count_nonzero() == 0
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    assert all(p.dtype == torch.float32 for p in network.parameters())

    # Removing invalid keys (including interior holes) preserves valid outputs.
    compact = copy.deepcopy(changed)
    for name, keep in (
        ("image", batch["image"]["valid"][0]),
        ("proprio", batch["proprio"]["observed"][0]),
    ):
        compact[name] = {key: value[:, keep] for key, value in compact[name].items()}
    torch.testing.assert_close(
        forward(network, compact), result[:, batch["image"]["valid"][0]],
        rtol=1e-5, atol=1e-5,
    )


def test_bf16_is_confined_to_attention_with_fp32_outputs_and_gradients(monkeypatch):
    network = model(attention_dtype="bfloat16")
    calls = []
    for block in network.blocks:
        original = block._attend

        def record(q, k, v, mask, original=original):
            calls.append((q.dtype, k.dtype, v.dtype, mask.detach().clone()))
            return original(q, k, v, mask)

        monkeypatch.setattr(block, "_attend", record)
    batch = example()
    result = forward(network, batch)
    assert result.dtype == torch.float32
    result.square().mean().backward()
    expected_keys = torch.cat((batch["image"]["valid"], batch["proprio"]["observed"]), 1)
    assert len(calls) == len(network.blocks)
    for q_dtype, k_dtype, v_dtype, mask in calls:
        assert q_dtype == k_dtype == v_dtype == torch.bfloat16
        assert torch.equal(mask[:, 0, 0], expected_keys)
    assert all(
        p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
        for p in network.parameters() if p.grad is not None
    )


def test_unchunked_noncheckpointed_model_does_not_recompute(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpointing must be disabled by both execution controls")

    monkeypatch.setattr("egomimic.models.image_proprio.checkpoint", forbidden)
    monkeypatch.setattr("egomimic.models.world_model.checkpoint", forbidden)
    network = model(
        attention_query_chunk_size=0, activation_checkpointing=False,
        attention_dtype="bfloat16",
    )
    forward(network, example()).square().mean().backward()


def test_flash4_model_does_not_silently_fallback_on_cpu():
    network = model(attention_backend="flash4", attention_dtype="bfloat16")
    with pytest.raises(RuntimeError, match="CUDA"):
        forward(network, example())


def test_legacy_dino_wrapper_keeps_native_execution_defaults():
    network = DinoProprioModel(
        pose_dims={"gripper": 3}, image_dim=8, width=16, depth=1, heads=2,
    )
    assert network.attention_backend == "sdpa"
    assert network.attention_dtype == "native"
    assert not network.strict_fp32
    assert network.use_camera_embedding and network.use_modality_embedding
    assert network.use_embodiment_embedding


def test_generic_single_modality_block_has_valid_backward():
    block = MoTBlock(24, 3, modalities=("image",), attention_dtype="bfloat16")
    values = torch.randn(1, 5, 24, requires_grad=True)
    times = torch.tensor([[-1., 0., .5, 1., 2.]])
    valid = torch.tensor([[True, False, True, True, True]])
    output, = block([values], [times], [valid])
    assert len(block.qkv) == 1
    assert output.shape == values.shape and output.dtype == torch.float32
    assert output[~valid].count_nonzero() == 0
    output.square().mean().backward()
    assert torch.isfinite(values.grad).all() and values.grad[valid].abs().sum() > 0
    assert values.grad[~valid].count_nonzero() == 0
