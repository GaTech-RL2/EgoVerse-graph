"""CPU autocast analogues; CUDA/Flash4 execution is a separate GPU smoke."""

import pytest
import torch

from tests.test_image_proprio_attention import example
from tests.test_image_proprio_image_only import image_model


def test_mixed_training_keeps_loss_timestamps_parameters_and_adamw_state_fp32():
    model = image_model(attention_dtype="bfloat16")
    batch = {"image": example()["image"]}
    # Nearby physical timestamps must remain distinct under mixed precision.
    batch["image"]["time"] = torch.tensor(
        [[30.010, 30.010, 30.020, 30.020, 30.030, 30.030]]
    )
    times = batch["image"]["time"].clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    observed_dtypes = {}

    def capture(name):
        def record(module, inputs, output):
            observed_dtypes[name] = output.dtype

        return record

    handles = [
        model.blocks[0].qkv[0].register_forward_hook(capture("qkv")),
        model.blocks[0].ffn[0][0].register_forward_hook(capture("ffn")),
        model.image_out.register_forward_hook(capture("prediction")),
    ]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses = model.training_losses(batch, "gripper")
        assert set(observed_dtypes.values()) == {torch.bfloat16}
        assert all(
            loss.dtype == torch.float32 and torch.isfinite(loss)
            for loss in losses.values()
        )
        losses["total"].backward()
        assert all(
            p.dtype == torch.float32
            and p.grad is not None
            and p.grad.dtype == torch.float32
            and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        before = model.image_out.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before, model.image_out.weight)
        for state in optimizer.state.values():
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32
        with torch.autocast("cpu", dtype=torch.bfloat16):
            predicted = model.sample(batch, "gripper", steps=20)["image"]
    finally:
        for handle in handles:
            handle.remove()
    assert torch.equal(times, batch["image"]["time"])
    assert batch["image"]["time"].dtype == torch.float32
    assert predicted.dtype == torch.float32 and torch.isfinite(predicted).all()
    observed = batch["image"]["observed"]
    assert torch.equal(predicted[observed], batch["image"]["value"][observed])


@pytest.mark.parametrize("steps", [3, 20])
def test_every_mixed_sampler_update_uses_fp32_arithmetic_before_clamping(
    monkeypatch, steps
):
    model = image_model(attention_dtype="bfloat16")
    batch = {"image": example()["image"]}
    image = batch["image"]
    valid, observed = image["valid"], image["observed"]
    fixed_prediction = (
        torch.linspace(-0.73, 0.91, image["value"].numel())
        .reshape_as(image["value"])
        .bfloat16()
    )
    actual_inputs, actual_levels = [], []

    def predict(batch, embodiment, image_values, levels, phase, **execution):
        assert phase == "A"
        actual_inputs.append(image_values.clone())
        actual_levels.append(levels.clone())
        return {"image": fixed_prediction}

    monkeypatch.setattr(model, "forward", predict)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.sample(
            batch, "gripper", steps=steps, generator=torch.Generator().manual_seed(101)
        )["image"]
    anchor = torch.where(observed[..., None], image["value"], 0)
    current = torch.where(
        observed[..., None],
        anchor,
        torch.randn(image["value"].shape, generator=torch.Generator().manual_seed(101)),
    ).masked_fill(~valid[..., None], 0)
    prediction_fp32 = fixed_prediction.float()
    rounded_path_differs = False
    for index, j in enumerate(range(steps, 0, -1)):
        assert actual_inputs[index].dtype == torch.float32
        assert actual_levels[index].dtype == torch.float32
        torch.testing.assert_close(actual_inputs[index], current, rtol=0, atol=0)
        level, next_level = j / steps, (j - 1) / steps
        expected = (1 - next_level) * prediction_fp32 + (next_level / level) * (
            current - (1 - level) * prediction_fp32
        )
        # This reference distinguishes FP32 integration from the old BF16
        # scalar products; checking only the final x0 would miss the rounding.
        rounded = (1 - next_level) * fixed_prediction + (next_level / level) * (
            current - (1 - level) * fixed_prediction
        )
        rounded_path_differs |= not torch.equal(expected, rounded)
        current = torch.where(observed[..., None], anchor, expected).masked_fill(
            ~valid[..., None], 0
        )
    assert rounded_path_differs
    torch.testing.assert_close(actual, current, rtol=0, atol=0)
    assert torch.equal(actual[observed], image["value"][observed])
