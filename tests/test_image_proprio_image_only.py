"""Image-only stage one and image-only phase A of the deferred joint readout."""

import copy

import pytest
import torch

from egomimic.models.image_proprio import ImageProprioModel
from tests.test_image_proprio_attention import example


def image_model(**kwargs):
    settings = dict(
        objective="visual_only",
        phase_a_condition_on_proprio=False,
        image_dim=8,
        width=24,
        heads=3,
        depth=2,
        max_cameras=2,
        max_tokens=32,
    )
    settings.update(kwargs)
    return ImageProprioModel(**settings)


def joint_model(**kwargs):
    return image_model(
        objective="visual_state_action",
        pose_dims={"gripper": 3},
        action_dims={"gripper": 2},
        **kwargs,
    )


def joint_batch():
    batch = example()
    batch["action"] = {
        "value": torch.randn(1, 4, 2),
        "time": batch["proprio"]["time"].clone(),
        "valid": torch.ones(1, 4, dtype=torch.bool),
        "observed": torch.zeros(1, 4, dtype=torch.bool),
    }
    return batch


def predict(model, batch, phase="A"):
    return model(
        batch,
        "gripper",
        batch["image"]["value"],
        torch.zeros_like(batch["image"]["time"]),
        phase=phase,
    )


def test_image_only_has_exact_parameters_and_no_proprio_modules():
    with torch.device("meta"):
        full = image_model(image_dim=768, width=768, heads=6, depth=8)
    assert sum(parameter.numel() for parameter in full.parameters()) == 58_482_432
    assert full.pose_dims == full.action_dims == {}
    assert full.modalities == ("image",)
    assert not full.has_proprio_branch
    assert len(full.norm) == 1
    assert all(len(block.qkv) == 1 for block in full.blocks)
    for name in (
        "pose_in",
        "proprio_role",
        "pose_out",
        "action_out",
        "query",
        "action_query",
    ):
        assert not hasattr(full, name)
    assert not any("pose" in key or "proprio" in key for key in full.state_dict())


def test_image_only_trains_and_samples_without_state_data():
    model = image_model()
    batch = {"image": example()["image"]}
    losses = model.training_losses(batch, "gripper")
    assert set(losses) == {"image", "total"}
    losses["total"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    query = ~batch["image"]["observed"]
    batch["image"]["value"][query] = float("nan")
    prediction = model.sample(batch, "gripper", steps=20)["image"]
    assert torch.isfinite(prediction).all()
    observed = batch["image"]["observed"]
    assert torch.equal(prediction[observed], batch["image"]["value"][observed])
    assert prediction[~batch["image"]["valid"]].count_nonzero() == 0


def test_image_only_rejects_extra_streams_or_unusable_conditioning():
    model = image_model()
    with pytest.raises(ValueError, match="exactly the image stream"):
        predict(model, example())
    batch = {"image": example()["image"]}
    batch["image"]["observed"].zero_()
    with pytest.raises(ValueError, match="observed"):
        predict(model, batch)
    with pytest.raises(ValueError, match="omit pose_dims"):
        image_model(pose_dims={"gripper": 3})
    with pytest.raises(ValueError, match="no embodiment embedding"):
        image_model(use_embodiment_embedding=True)
    with pytest.raises(ValueError, match="must be bool"):
        image_model(phase_a_condition_on_proprio=0)


def test_joint_phase_a_never_uses_even_observed_proprio_or_action_queries():
    model, batch = joint_model(), joint_batch()
    expected = predict(model, batch)
    assert set(expected) == {"image"}
    changed = copy.deepcopy(batch)
    changed["proprio"]["value"] *= 1000
    changed["proprio"]["time"] += 123
    changed["action"]["value"].fill_(float("nan"))
    assert torch.equal(expected["image"], predict(model, changed)["image"])
    expected["image"].square().mean().backward()
    assert all(p.grad is None for p in model.pose_in.parameters())
    assert model.proprio_role.weight.grad is None
    for block in model.blocks:
        assert block.qkv[0].weight.grad.abs().sum() > 0
        assert block.qkv[1].weight.grad is None


def test_joint_phase_b_uses_new_branch_but_withheld_labels_are_loss_only():
    model, batch = joint_model(), joint_batch()
    before = predict(model, batch, phase="B")
    assert set(before) == {"image", "proprio", "action"}
    changed = copy.deepcopy(batch)
    changed["proprio"]["value"][~changed["proprio"]["observed"]] = float("nan")
    changed["action"]["value"].fill_(float("nan"))
    after = predict(model, changed, phase="B")
    assert all(torch.equal(before[key], after[key]) for key in before)
    (before["proprio"].square().mean() + before["action"].square().mean()).backward()
    assert model.pose_in["gripper"].weight.grad.abs().sum() > 0
    assert model.pose_out["gripper"].weight.grad.abs().sum() > 0
    assert model.action_out["gripper"].weight.grad.abs().sum() > 0
    assert all(block.qkv[1].weight.grad.abs().sum() > 0 for block in model.blocks)


@pytest.mark.parametrize("readout_weight", [0.0, 1.0])
def test_joint_training_supports_zero_or_active_readout_after_image_only_phase_a(
    readout_weight,
):
    model = joint_model()
    losses = model.training_losses(
        joint_batch(),
        "gripper",
        pose_weight=readout_weight,
        action_weight=readout_weight,
    )
    assert set(losses) == {"image", "pose", "action", "total"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert (model.action_out["gripper"].weight.grad is None) == (readout_weight == 0)
