"""Batch-independent objective and regional compilation contracts."""

import copy

import pytest
import torch

from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.world_model import masked_mse


def network(**kwargs):
    return ImageProprioModel(
        objective="visual_only",
        phase_a_condition_on_proprio=False,
        image_dim=12,
        width=24,
        heads=3,
        depth=2,
        max_cameras=2,
        loss_reduction="mean_per_example",
        **kwargs,
    )


def batch():
    torch.manual_seed(19)
    valid = torch.tensor([[True] * 6, [True, True, False, True, False, False]])
    return {
        "image": {
            "value": torch.randn(2, 6, 12),
            "time": torch.tensor([[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]] * 2),
            "valid": valid,
            "observed": torch.tensor(
                [
                    [True, True, False, False, False, False],
                    [True, False, False, False, False, False],
                ]
            ),
            "xy": torch.randn(2, 6, 2),
            "camera": torch.tensor([[0, 1, 0, 1, 0, 1]] * 2),
        }
    }


def test_per_example_loss_matches_independent_clips_and_gradients():
    pred = torch.tensor(
        [[[1.0], [3.0], [5.0]], [[7.0], [0.0], [0.0]]], requires_grad=True
    )
    target = torch.zeros_like(pred)
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    actual = masked_mse(pred, target, mask, reduction="mean_per_example")
    expected = torch.stack(
        [masked_mse(p[None], t[None], m[None]) for p, t, m in zip(pred, target, mask)]
    ).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, pred, retain_graph=True)[0],
        torch.autograd.grad(expected, pred)[0],
    )
    assert actual != masked_mse(pred, target, mask)


def test_per_example_empty_clip_has_differentiable_zero_weight():
    pred = torch.ones(2, 3, 2, requires_grad=True)
    target = torch.zeros_like(pred)
    target[1] = float("nan")
    mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    loss = masked_mse(pred, target, mask, reduction="mean_per_example")
    assert loss == 0.5
    loss.backward()
    assert torch.isfinite(pred.grad).all() and not pred.grad[1].any()


def test_batched_image_model_matches_independent_outputs_and_mean_gradients():
    torch.manual_seed(3)
    net = network()
    independent = copy.deepcopy(net)
    data = batch()
    values = torch.randn_like(data["image"]["value"])
    levels = torch.full_like(data["image"]["time"], 0.4)
    prediction = net(data, "gripper", values, levels)["image"]
    expected = []
    for b in range(2):
        single = {"image": {k: v[b : b + 1] for k, v in data["image"].items()}}
        expected.append(
            independent(single, "gripper", values[b : b + 1], levels[b : b + 1])[
                "image"
            ]
        )
    torch.testing.assert_close(prediction, torch.cat(expected), rtol=1e-5, atol=1e-6)
    mask = data["image"]["valid"] & ~data["image"]["observed"]
    net._loss(prediction, data["image"]["value"], mask).backward()
    torch.stack(
        [
            independent._loss(
                expected[b], data["image"]["value"][b : b + 1], mask[b : b + 1]
            )
            for b in range(2)
        ]
    ).mean().backward()
    for a, b in zip(net.parameters(), independent.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)


def test_regional_compile_matches_eager_and_preserves_checkpoint_names():
    torch.manual_seed(3)
    eager = network()
    compiled = network(
        compile_mode="default", compile_backend="eager", compile_dynamic=True
    )
    compiled.load_state_dict(eager.state_dict(), strict=True)
    assert eager.state_dict().keys() == compiled.state_dict().keys()
    data = batch()
    values = torch.randn_like(data["image"]["value"])
    levels = torch.full_like(data["image"]["time"], 0.4)
    expected = eager(data, "gripper", values, levels)["image"]
    actual = compiled(data, "gripper", values, levels)["image"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for a, b in zip(eager.parameters(), compiled.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert not any("_orig_mod" in key for key in compiled.state_dict())


def test_sampler_validates_and_prepares_once_without_changing_observations(monkeypatch):
    net = network()
    data = batch()
    calls = {"validate": 0, "prepare": 0}
    validate, prepare = net.validate, net._prepare_execution

    def checked(*args, **kwargs):
        calls["validate"] += 1
        return validate(*args, **kwargs)

    def prepared(*args, **kwargs):
        calls["prepare"] += 1
        return prepare(*args, **kwargs)

    monkeypatch.setattr(net, "validate", checked)
    monkeypatch.setattr(net, "_prepare_execution", prepared)
    out = net.sample(data, "gripper", steps=4)["image"]
    assert calls == {"validate": 1, "prepare": 1}
    known = data["image"]["observed"]
    assert torch.equal(out[known], data["image"]["value"][known])


def test_training_validates_once(monkeypatch):
    net = network()
    original = net.validate
    calls = []

    def checked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(net, "validate", checked)
    assert torch.isfinite(net.training_losses(batch(), "gripper")["total"])
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["time", "xy"])
def test_value_validation_still_rejects_nonfinite_valid_metadata(field):
    data = batch()
    data["image"][field][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        network().training_losses(data, "gripper")
