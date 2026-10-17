"""Image x0 ablation: identical architecture, explicit target/solver semantics."""
import copy

import pytest
import torch

from egomimic.models.world_model import masked_mse
from tests.test_world_model import make_batch, make_model


@pytest.mark.parametrize("bf16", [False, True])
def test_explicit_v_preserves_implicit_default_losses_gradients_rng(bf16):
    torch.manual_seed(42)
    old = make_model()
    new = make_model(image_prediction_type="v")
    new.load_state_dict(old.state_dict(), strict=True)
    batch = make_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        torch.manual_seed(17)
        expected = old.training_losses(batch, "robot")
        state = torch.get_rng_state()
        torch.manual_seed(17)
        actual = new.training_losses(batch, "robot")
        assert torch.equal(state, torch.get_rng_state())
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
    expected["total"].backward()
    actual["total"].backward()
    for a, b in zip(old.parameters(), new.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [1, 5, 20])
def test_x0_oracle_cancels_unknown_noise_and_preserves_observations(monkeypatch, steps):
    model = make_model(image_prediction_type="x0")
    batch = make_batch()
    target = batch["image"]["value"].clone()
    batch["image"]["value"][~batch["image"]["observed"]] = float("nan")
    checks = []

    def oracle(b, embodiment, values, levels):
        observed = b["image"]["observed"]
        checks.append(torch.equal(values["image"][observed], target[observed]))
        assert torch.isfinite(values["image"]).all()
        return {"image": target, "track": torch.zeros_like(values["track"]),
                "proprio": torch.zeros_like(b["proprio"]["value"])}

    monkeypatch.setattr(model, "forward", oracle)
    result = model.sample(batch, "robot", steps=steps)
    torch.testing.assert_close(result["image"], target, rtol=0, atol=0)
    assert all(checks)


def test_x0_objective_is_clean_mse_and_track_remains_v(monkeypatch):
    model = make_model(image_prediction_type="x0")
    batch = make_batch()
    original = model.forward

    def zero_image(*args):
        output = original(*args)
        output["image"] = torch.zeros_like(output["image"])
        return output

    monkeypatch.setattr(model, "forward", zero_image)
    torch.manual_seed(7)
    losses = model.training_losses(batch, "robot", pose_weight=0)
    mask = batch["image"]["valid"] & ~batch["image"]["observed"]
    expected = masked_mse(torch.zeros_like(batch["image"]["value"]),
                          batch["image"]["value"], mask.float())
    torch.testing.assert_close(losses["image"], expected, rtol=0, atol=0)
    model.image_prediction_type = "v"
    torch.manual_seed(7)
    v_loss = model.training_losses(batch, "robot", pose_weight=0)
    torch.testing.assert_close(losses["track"], v_loss["track"], rtol=0, atol=0)


@pytest.mark.parametrize("bf16", [False, True])
def test_x0_pose_loss_reaches_both_generators(bf16):
    model = make_model(image_prediction_type="x0")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        losses = model.training_losses(make_batch(), "robot")
    losses["pose"].backward()
    for module in (model.image_out, model.track_out):
        assert module.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_x0_sampler_ignores_withheld_values_and_validity():
    model = make_model(image_prediction_type="x0")
    batch = make_batch()
    changed = copy.deepcopy(batch)
    for stream in changed.values():
        stream["value"][~stream["observed"]] = float("nan")
        stream["target_valid"].zero_()
        if "confidence" in stream:
            stream["confidence"].zero_()
    torch.manual_seed(55)
    expected = model.sample(batch, "robot", steps=3)
    torch.manual_seed(55)
    actual = model.sample(changed, "robot", steps=3)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_x0_missing_track_completion_still_uses_explicit_policy():
    model, batch = make_model(image_prediction_type="x0"), make_batch()
    batch["track"]["target_valid"][:, -1] = False
    batch["track"]["value"][:, -1] = float("nan")
    with pytest.raises(ValueError, match="zero-fill"):
        model.training_losses(batch, "robot")
    loss = model.training_losses(batch, "robot", missing_track_policy="conditional_sample", imputation_steps=2)
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_invalid_image_parameterization_rejected():
    with pytest.raises(ValueError, match="image_prediction_type"):
        make_model(image_prediction_type="epsilon")
