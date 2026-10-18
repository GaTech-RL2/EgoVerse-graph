"""Image-patch core compatibility and explicit compression contracts."""

import copy

import pytest
import torch

from egomimic.models.dino_proprio import DinoProprioModel
from egomimic.models.image_proprio import ImageProprioModel


def raw_patch_batch(direct):
    # Two 2x2 RGB patches per timestamp, scaled to [-1,1]. These are raw
    # channel values, without a pretrained feature extractor or learned encoder.
    values = torch.linspace(-1, 1, 4 * 12).reshape(1, 4, 12)
    batch = {
        "image": {
            "value": values,
            "time": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "valid": torch.ones(1, 4, dtype=torch.bool),
            "observed": torch.tensor([[True, True, False, False]]),
            "xy": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]),
            "camera": torch.zeros(1, 4, dtype=torch.long),
        },
        "proprio": {
            "value": torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
            "time": torch.tensor([[0.0, 1.0]]),
            "valid": torch.ones(1, 2, dtype=torch.bool),
            "observed": torch.tensor([[True, False]]),
        },
    }
    if direct:
        batch["action"] = {
            "value": torch.tensor([[[0.2, -0.3]]]),
            "time": torch.zeros(1, 1),
            "valid": torch.ones(1, 1, dtype=torch.bool),
            "observed": torch.zeros(1, 1, dtype=torch.bool),
        }
    return batch


def model_arguments(direct):
    return dict(
        pose_dims={"robot": 3},
        action_dims={"robot": 2} if direct else None,
        objective="direct_action" if direct else "visual_proprio",
        image_dim=12,
        depth=1,
        heads=2,
        max_cameras=1,
        max_tokens=8,
    )


def test_legacy_class_retains_identity_embeddings_and_parameter_compatibility():
    assert issubclass(DinoProprioModel, ImageProprioModel)
    previous_name = DinoProprioModel(width=16, **model_arguments(False))
    current_name = ImageProprioModel(
        width=16,
        use_modality_embedding=True,
        use_embodiment_embedding=True,
        **model_arguments(False),
    )
    assert previous_name.use_camera_embedding
    assert previous_name.use_modality_embedding
    assert previous_name.use_embodiment_embedding
    current_name.load_state_dict(previous_name.state_dict(), strict=True)
    assert not current_name.allow_patch_compression


def test_pixel_default_keeps_camera_but_removes_modality_and_embodiment_ids():
    arguments = model_arguments(False)
    arguments["max_cameras"] = 2
    model = ImageProprioModel(width=16, **arguments)
    assert model.use_camera_embedding
    assert not model.use_modality_embedding and not model.use_embodiment_embedding
    assert not any(
        hasattr(model, key) for key in ("modality", "embodiment", "embodiment_ids")
    )
    assert not any(
        key.startswith(("modality", "embodiment.")) for key in model.state_dict()
    )
    assert all(len(block.qkv) == 2 for block in model.blocks)
    batch = raw_patch_batch(False)
    images = torch.randn_like(batch["image"]["value"])
    levels = torch.zeros_like(batch["image"]["time"])
    expected = model(batch, "robot", images, levels)
    batch["image"]["camera"].fill_(1)
    changed = model(batch, "robot", images, levels)
    assert not torch.equal(expected["image"], changed["image"])
    model.training_losses(batch, "robot")["total"].backward()
    assert model.camera.weight.grad.abs().sum() > 0
    for flag in (
        "use_camera_embedding",
        "use_modality_embedding",
        "use_embodiment_embedding",
    ):
        for value in (0, 1, None, "false"):
            with pytest.raises(ValueError, match=f"{flag} must be bool"):
                ImageProprioModel(width=16, **{flag: value}, **arguments)


def test_camera_conditioning_can_be_disabled_without_hidden_slot_encoding():
    model = ImageProprioModel(
        width=16, use_camera_embedding=False, **model_arguments(False)
    )
    assert not hasattr(model, "camera")
    batch = raw_patch_batch(False)
    images = torch.randn_like(batch["image"]["value"])
    levels = torch.zeros_like(batch["image"]["time"])
    expected = model(batch, "robot", images, levels)
    batch["image"]["camera"] = torch.tensor([[100, -20, 9, 7]])
    changed = model(batch, "robot", images, levels)
    del batch["image"]["camera"]
    absent = model(batch, "robot", images, levels)
    for name in expected:
        torch.testing.assert_close(expected[name], changed[name], rtol=0, atol=0)
        torch.testing.assert_close(expected[name], absent[name], rtol=0, atol=0)


def test_pixel_has_no_embodiment_id_offset_when_adapter_weights_match():
    arguments = model_arguments(False)
    arguments["pose_dims"] = {"robot": 3, "another": 3}
    model = ImageProprioModel(width=16, **arguments)
    model.pose_in["another"].load_state_dict(model.pose_in["robot"].state_dict())
    model.pose_out["another"].load_state_dict(model.pose_out["robot"].state_dict())
    batch = raw_patch_batch(False)
    images = torch.randn_like(batch["image"]["value"])
    levels = torch.zeros_like(batch["image"]["time"])
    expected = model(batch, "robot", images, levels, phase="B")
    changed = model(batch, "another", images, levels, phase="B")
    for name in expected:
        torch.testing.assert_close(expected[name], changed[name], rtol=0, atol=0)


@pytest.mark.parametrize("direct", [False, True])
def test_patch_compression_requires_explicit_boolean_opt_in(direct):
    for kwargs in ({}, {"allow_patch_compression": False}):
        with pytest.raises(ValueError, match="allow_patch_compression=True"):
            ImageProprioModel(width=8, **model_arguments(direct), **kwargs)
    for bad_value in (1, "true", None):
        with pytest.raises(ValueError, match="must be bool"):
            ImageProprioModel(
                width=8,
                allow_patch_compression=bad_value,
                **model_arguments(direct),
            )


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("compress", [False, True])
def test_raw_rgb_patch_objectives_backward_and_sampling(direct, compress):
    model = ImageProprioModel(
        width=8 if compress else 16,
        allow_patch_compression=compress,
        **model_arguments(direct),
    )
    assert model.image_in.in_features == 12
    assert model.image_in.out_features == (8 if compress else 16)
    assert not any(isinstance(module, torch.nn.Conv2d) for module in model.modules())
    batch = raw_patch_batch(direct)
    losses = model.training_losses(batch, "robot")
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.image_in.weight.grad.abs().sum() > 0
    sampled = model.sample(batch, "robot", steps=2)
    assert all(torch.isfinite(value).all() for value in sampled.values())
    if direct:
        assert sampled["action"].shape == (1, 1, 2)
        assert not hasattr(model, "image_out")
    else:
        assert model.image_out.out_features == 12
        assert model.image_out.in_features == (8 if compress else 16)
        assert model.image_out.weight.grad.abs().sum() > 0
        assert sampled["image"].shape == (1, 4, 12)
        for name in ("image", "proprio"):
            observed = batch[name]["observed"]
            torch.testing.assert_close(
                sampled[name][observed], batch[name]["value"][observed], rtol=0, atol=0
            )


def test_visual_only_has_no_state_action_heads_queries_or_second_phase(monkeypatch):
    arguments = model_arguments(False)
    arguments["objective"] = "visual_only"
    model = ImageProprioModel(width=16, **arguments)
    assert not any(hasattr(model, name) for name in ("pose_out", "action_out", "query"))
    batch = raw_patch_batch(False)
    batch["proprio"]["value"][~batch["proprio"]["observed"]] = float("nan")
    phases = []
    original = model.forward

    def capture(*args, **kwargs):
        phases.append(kwargs.get("phase", "A"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", capture)
    losses = model.training_losses(batch, "robot", pose_weight=1)
    assert set(losses) == {"image", "total"}
    assert phases == ["A"]
    losses["total"].backward()
    assert model.pose_in["robot"].weight.grad.abs().sum() > 0
    phases.clear()
    sampled = model.sample(batch, "robot", steps=2)
    assert phases == ["A", "A"]
    assert set(sampled) == {"image"}
    assert torch.isfinite(sampled["image"]).all()
    with pytest.raises(ValueError, match="no proprio readout phase"):
        model(
            batch, "robot", batch["image"]["value"], batch["image"]["time"], phase="B"
        )


def combined_model_and_batch(**kwargs):
    arguments = model_arguments(True)
    arguments["objective"] = "visual_state_action"
    model = ImageProprioModel(width=16, **arguments, **kwargs)
    batch = raw_patch_batch(True)
    batch["action"] = {
        "value": torch.tensor([[[0.2, -0.3], [-0.4, 0.5]]]),
        "time": torch.tensor([[0.0, 1.0]]),
        "valid": torch.ones(1, 2, dtype=torch.bool),
        "observed": torch.zeros(1, 2, dtype=torch.bool),
    }
    return model, batch


def test_joint_state_action_readout_is_two_branches_and_never_embeds_labels():
    model, batch = combined_model_and_batch()
    assert all(len(block.qkv) == 2 for block in model.blocks)
    assert model.proprio_role.num_embeddings == 3
    assert model.action_query is not model.query
    images = torch.randn_like(batch["image"]["value"])
    levels = torch.zeros_like(batch["image"]["time"])
    changed = copy.deepcopy(batch)
    changed["action"]["value"].fill_(float("nan"))
    changed["proprio"]["value"][~changed["proprio"]["observed"]] = float("nan")
    expected = model(batch, "robot", images, levels, phase="B")
    actual = model(changed, "robot", images, levels, phase="B")
    assert set(actual) == {"image", "proprio", "action"}
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    # Action query metadata is absent in phase A as well as action labels.
    before = model(batch, "robot", images, levels, phase="A")
    changed["action"]["time"] += 100
    changed["action"]["valid"].fill_(False)
    after = model(changed, "robot", images, levels, phase="A")
    assert "action" not in before
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "conditioning,phases", [("low_noise", ["A", "A", "B"]), ("clean", ["A", "B"])]
)
def test_combined_readout_shares_one_aligned_image_estimate(
    monkeypatch, conditioning, phases
):
    model, batch = combined_model_and_batch(pose_conditioning=conditioning)
    calls = []
    original = model.forward

    def capture(*args, **kwargs):
        calls.append(kwargs.get("phase", "A"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", capture)
    losses = model.training_losses(
        batch, "robot", image_weight=0, pose_weight=2, action_weight=3
    )
    assert calls == phases
    assert set(losses) == {"image", "pose", "action", "total"}
    torch.testing.assert_close(
        losses["total"], 2 * losses["pose"] + 3 * losses["action"]
    )
    losses["total"].backward()
    for head in (model.pose_out["robot"], model.action_out["robot"]):
        assert head.weight.grad.abs().sum() > 0
    assert model.action_query.grad.abs().sum() > 0
    assert model.query.grad.abs().sum() > 0
    if conditioning == "low_noise":
        assert model.image_out.weight.grad.abs().sum() > 0
    calls.clear()
    sample = model.sample(batch, "robot", steps=2)
    assert calls == ["A", "A", "B"]
    assert all(torch.isfinite(value).all() for value in sample.values())
    for name in ("image", "proprio"):
        observed = batch[name]["observed"]
        torch.testing.assert_close(
            sample[name][observed], batch[name]["value"][observed], rtol=0, atol=0
        )


def test_combined_readout_zero_weights_and_missing_targets(monkeypatch):
    model, batch = combined_model_and_batch()
    calls = []
    original = model.forward

    def capture(*args, **kwargs):
        calls.append(kwargs.get("phase", "A"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", capture)
    losses = model.training_losses(batch, "robot", pose_weight=0, action_weight=0)
    assert calls == ["A"]
    assert losses["pose"] == losses["action"] == 0
    # Action supervision can train by itself when withheld state labels are absent.
    calls.clear()
    batch["proprio"]["target_valid"] = batch["proprio"]["observed"].clone()
    batch["proprio"]["value"][~batch["proprio"]["observed"]] = float("nan")
    losses = model.training_losses(batch, "robot")
    assert calls == ["A", "A", "B"]
    assert losses["pose"] == 0 and torch.isfinite(losses["total"])
    calls.clear()
    batch["action"]["target_valid"] = torch.zeros_like(batch["action"]["valid"])
    batch["action"]["value"].fill_(float("nan"))
    losses = model.training_losses(batch, "robot")
    assert calls == ["A"]
    assert torch.isfinite(losses["total"])
