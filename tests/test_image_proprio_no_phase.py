"""A/B routing without learned phase offsets, including strict readout expansion."""

import pytest
import torch

from egomimic.models.image_proprio import ImageProprioModel
from egomimic.pipeline.stages_pixel_readout import configure_readout_trainability
from scripts.world_model.initialize_pixel_readout import (
    transfer_shared_weights,
    validate_transition,
    verify_phase_a,
)
from tests.test_image_proprio_image_only import image_model, joint_batch, predict
from tests.test_image_proprio_image_only_initialization import configs


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_legacy_default_has_identical_seeded_state_and_strict_reload():
    torch.manual_seed(37)
    original = image_model()
    torch.manual_seed(37)
    explicit = image_model(use_phase_embedding=True)
    assert original.use_phase_embedding and "phase.weight" in original.state_dict()
    assert all(
        torch.equal(value, explicit.state_dict()[key])
        for key, value in original.state_dict().items()
    )
    explicit.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(ValueError, match="use_phase_embedding must be bool"):
        image_model(use_phase_embedding=0)


def test_no_phase_has_exact_parameter_delta_and_all_stage_one_gradients():
    with torch.device("meta"):
        model = image_model(
            image_dim=768, width=768, heads=6, depth=8, use_phase_embedding=False
        )
    assert sum(p.numel() for p in model.parameters()) == 58_481_664
    assert not hasattr(model, "phase")
    assert not any("phase" in key for key in model.state_dict())
    model = image_model(use_phase_embedding=False)
    batch = {"image": joint_batch()["image"]}
    model.training_losses(batch, "gripper")["total"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError, match="no proprio readout phase"):
        predict(model, batch, phase="B")


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_no_phase_readout_warm_start_preserves_images_and_routes_state_commands(mode):
    source_config, target_config = configs(mode)
    for config in (source_config, target_config):
        config.update(use_phase_embedding=False, image_dim=8)
    target_config.update(pose_dims={"gripper": 3}, action_dims={"gripper": 2})
    validate_transition(source_config, target_config)
    source, target = (
        ImageProprioModel(**source_config),
        ImageProprioModel(**target_config),
    )
    transfer = transfer_shared_weights(source, target)
    assert "phase.weight" not in transfer["expanded_embeddings"]
    assert not hasattr(source, "phase") and not hasattr(target, "phase")
    assert verify_phase_a(source, target)["bit_exact"]
    clone = ImageProprioModel(**target_config)
    clone.load_state_dict(target.state_dict(), strict=True)
    policy = configure_readout_trainability(target, "frozen_image")
    assert (
        not policy["phase_embedding_present"] and not policy["phase_embedding_frozen"]
    )
    assert sum(p.numel() for p in target.parameters() if not p.requires_grad) == sum(
        p.numel() for p in source.parameters()
    )
    batch = joint_batch()
    if mode == "discrete":
        for stream in batch.values():
            stream["index"] = torch.arange(stream["valid"].shape[1]).expand_as(
                stream["valid"]
            )
    original = source.sample(
        {"image": batch["image"]},
        "gripper",
        steps=20,
        generator=torch.Generator().manual_seed(19),
    )
    actual = target.sample(
        batch, "gripper", steps=20, generator=torch.Generator().manual_seed(19)
    )
    assert torch.equal(original["image"], actual["image"])
    assert set(actual) == {"image", "proprio", "action"}
    assert all(torch.isfinite(value).all() for value in actual.values())
    image = batch["image"]
    levels = torch.zeros_like(image["time"])
    # Control-flow labels do not introduce a hidden phase offset in either image route.
    a = target._embed_image(image, image["value"], levels, image["valid"], "A", False)
    b = target._embed_image(image, image["value"], levels, image["valid"], "B", False)
    assert torch.equal(a, b)


def test_initializer_refuses_to_drop_or_add_phase_parameters_implicitly():
    source_config, target_config = configs()
    target_config["use_phase_embedding"] = False
    with pytest.raises(ValueError, match="configuration mismatch.*use_phase_embedding"):
        validate_transition(source_config, target_config)
    source, target = (
        ImageProprioModel(**source_config),
        ImageProprioModel(**target_config),
    )
    with pytest.raises(ValueError, match="use_phase_embedding must match"):
        transfer_shared_weights(source, target)
