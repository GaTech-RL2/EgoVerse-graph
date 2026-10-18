"""CPU contracts for two-stream models; these are not model-quality tests."""

import copy

import pytest
import torch

from egomimic.models.dino_proprio import MODALITIES, DinoProprioModel
from egomimic.models.world_model import MoTBlock


def make_batch(direct=False):
    torch.manual_seed(13)
    image = {
        "value": torch.randn(2, 6, 8),
        "time": torch.tensor([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]]).expand(2, -1).clone(),
        "index": torch.tensor([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]]).expand(2, -1).clone(),
        "valid": torch.ones(2, 6, dtype=torch.bool),
        "observed": torch.tensor([[True, True, False, False, False, False]])
        .expand(2, -1)
        .clone(),
        "xy": torch.randn(2, 6, 2),
        "camera": torch.tensor([[0, 1, 0, 1, 0, 1]]).expand(2, -1).clone(),
        "target_valid": torch.ones(2, 6, dtype=torch.bool),
    }
    pose = {
        "value": torch.randn(2, 3, 3),
        "time": torch.tensor([[0.0, 1.0, 2.0]]).expand(2, -1).clone(),
        "index": torch.tensor([[0.0, 1.0, 2.0]]).expand(2, -1).clone(),
        "valid": torch.ones(2, 3, dtype=torch.bool),
        "observed": torch.tensor([[True, False, False]]).expand(2, -1).clone(),
        "target_valid": torch.ones(2, 3, dtype=torch.bool),
    }
    batch = {"image": image, "proprio": pose}
    if direct:
        batch["action"] = {
            "value": torch.randn(2, 2, 4),
            "time": torch.tensor([[0.0, 0.2]]).expand(2, -1).clone(),
            "index": torch.tensor([[0.0, 1.0]]).expand(2, -1).clone(),
            "valid": torch.ones(2, 2, dtype=torch.bool),
            "observed": torch.zeros(2, 2, dtype=torch.bool),
            "target_valid": torch.ones(2, 2, dtype=torch.bool),
        }
    return batch


def make_model(direct=False, **kwargs):
    settings = dict(
        image_dim=8, width=16, depth=2, heads=2, max_cameras=2, max_tokens=32
    )
    settings.update(kwargs)
    return DinoProprioModel(
        pose_dims={"gripper": 3},
        action_dims={"gripper": 4} if direct else None,
        objective="direct_action" if direct else "visual_proprio",
        **settings,
    )


def test_two_branches_and_width_guard_without_track_parameters():
    assert MODALITIES == ("image", "proprio")
    for direct in (False, True):
        model = make_model(direct)
        assert all(len(block.qkv) == 2 for block in model.blocks)
        assert all(
            "track" not in name and "point" not in name
            for name, _ in model.named_parameters()
        )
        assert model.image_in.in_features == 8
        assert model.image_in.out_features >= 8
        if direct:
            assert not hasattr(model, "image_out")
            assert not hasattr(model, "pose_out")
            assert not hasattr(model, "noise_level")
        else:
            assert model.noise_level[0].in_features == 1
            assert model.image_out.out_features == 8
    with pytest.raises(ValueError, match="at least image_dim"):
        make_model(width=4)
    # Existing checkpoints still construct three private branches by default.
    assert len(MoTBlock(16, 2).qkv) == 3


def test_world_phase_a_ignores_withheld_pose_values_times_and_presence():
    model, batch = make_model().eval(), make_batch()
    levels = torch.full_like(batch["image"]["time"], 0.5)
    images = torch.randn_like(batch["image"]["value"])
    before = model(batch, "gripper", images, levels, phase="A")
    changed = copy.deepcopy(batch)
    withheld = ~changed["proprio"]["observed"]
    changed["proprio"]["value"][withheld] = float("nan")
    changed["proprio"]["time"][withheld] = 999
    changed["proprio"]["target_valid"][:] = False
    after = model(changed, "gripper", images, levels, phase="A")
    torch.testing.assert_close(before["image"], after["image"], rtol=0, atol=0)
    assert torch.count_nonzero(after["proprio"][withheld]) == 0
    for key, value in changed["proprio"].items():
        changed["proprio"][key] = value[:, :1]
    without_queries = model(changed, "gripper", images, levels, phase="A")
    torch.testing.assert_close(
        before["image"], without_queries["image"], rtol=1e-6, atol=1e-6
    )


def test_world_phase_b_pose_targets_are_loss_only():
    model, batch = make_model().eval(), make_batch()
    images = torch.randn_like(batch["image"]["value"])
    levels = torch.zeros_like(batch["image"]["time"])
    before = model(batch, "gripper", images, levels, phase="B")
    batch["proprio"]["value"][~batch["proprio"]["observed"]] = float("nan")
    after = model(batch, "gripper", images, levels, phase="B")
    for name in MODALITIES:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_visual_observation_status_is_not_a_learned_condition():
    model, batch = make_model().eval(), make_batch()
    images = batch["image"]["value"].clone()
    levels = torch.zeros_like(batch["image"]["time"])
    before = model(batch, "gripper", images, levels, phase="A")["image"]
    batch["image"]["observed"][:] = True
    after = model(batch, "gripper", images, levels, phase="A")["image"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_joint_attention_couples_image_and_proprio_gradients():
    model, batch = make_model(), make_batch()
    images = batch["image"]["value"].clone().requires_grad_()
    batch["proprio"]["value"].requires_grad_()
    result = model(
        batch, "gripper", images, torch.ones_like(batch["image"]["time"]), phase="A"
    )
    result["image"].square().sum().backward()
    pose_grad = batch["proprio"]["value"].grad
    assert pose_grad[batch["proprio"]["observed"]].abs().sum() > 0
    assert pose_grad[~batch["proprio"]["observed"]].abs().sum() == 0
    assert all(block.qkv[1].weight.grad.abs().sum() > 0 for block in model.blocks)
    model.zero_grad(set_to_none=True)
    images = batch["image"]["value"].clone().requires_grad_()
    result = model(
        batch, "gripper", images, torch.zeros_like(batch["image"]["time"]), phase="B"
    )
    result["proprio"][:, 1:].square().sum().backward()
    assert images.grad[~batch["image"]["observed"]].abs().sum() > 0
    assert all(block.qkv[0].weight.grad.abs().sum() > 0 for block in model.blocks)


def test_image_loss_is_clean_feature_mse_and_pose_warmup_skips_readout(monkeypatch):
    model, batch = make_model(), make_batch()
    forwards = []
    original = model.forward

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        forwards.append((kwargs["phase"], result))
        return result

    monkeypatch.setattr(model, "forward", capture)
    losses = model.training_losses(batch, "gripper", pose_weight=0)
    assert len(forwards) == 1 and forwards[0][0] == "A"
    mask = batch["image"]["valid"] & ~batch["image"]["observed"]
    expected = (
        (forwards[0][1]["image"][mask] - batch["image"]["value"][mask]).square().mean()
    )
    torch.testing.assert_close(losses["image"], expected)
    assert losses["pose"].item() == 0
    torch.testing.assert_close(losses["total"], losses["image"])


def test_low_noise_pose_pass_and_gradient_through_image_head(monkeypatch):
    model, batch = make_model(pose_noise_max=0.15), make_batch()
    forwards = []
    original = model.forward

    def capture(*args, **kwargs):
        forwards.append((kwargs["phase"], args[3].detach().clone()))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", capture)
    losses = model.training_losses(batch, "gripper", image_weight=0)
    assert [phase for phase, _ in forwards] == ["A", "A", "B"]
    assert forwards[1][1].max() <= 0.15
    assert torch.count_nonzero(forwards[2][1]) == 0
    losses["total"].backward()
    assert model.image_out.weight.grad.abs().sum() > 0
    assert model.pose_out["gripper"].weight.grad.abs().sum() > 0


def test_clean_pose_conditioning_is_explicit_and_skips_auxiliary_denoising(monkeypatch):
    model, batch = make_model(pose_conditioning="clean"), make_batch()
    phases = []
    original = model.forward

    def capture(*args, **kwargs):
        phases.append(kwargs["phase"])
        if kwargs["phase"] == "B":
            torch.testing.assert_close(args[2], batch["image"]["value"])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", capture)
    model.training_losses(batch, "gripper")
    assert phases == ["A", "B"]


@pytest.mark.parametrize("steps", [1, 2, 20])
def test_sampling_finite_preserves_observations_and_ignores_targets(steps):
    model, batch = make_model().eval(), make_batch()
    before = model.sample(
        batch, "gripper", steps=steps, generator=torch.Generator().manual_seed(7)
    )
    for name in MODALITIES:
        assert torch.isfinite(before[name]).all()
        observed = batch[name]["observed"]
        torch.testing.assert_close(
            before[name][observed], batch[name]["value"][observed], rtol=0, atol=0
        )
        batch[name]["value"][~observed] = float("nan")
        batch[name]["target_valid"][:] = False
    after = model.sample(
        batch, "gripper", steps=steps, generator=torch.Generator().manual_seed(7)
    )
    for name in MODALITIES:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_all_queries_masked_or_absent_produces_finite_zero_losses():
    world, batch = make_model(), make_batch()
    for name in MODALITIES:
        batch[name]["observed"][:] = True
    losses = world.training_losses(batch, "gripper")
    assert all(torch.isfinite(loss) and loss.item() == 0 for loss in losses.values())
    losses["total"].backward()
    for empty in (False, True):
        direct, batch = make_model(True), make_batch(True)
        if empty:
            for key, value in batch["action"].items():
                batch["action"][key] = value[:, :0]
        else:
            batch["action"]["target_valid"][:] = False
            batch["action"]["value"][:] = float("nan")
        losses = direct.training_losses(batch, "gripper")
        assert all(
            torch.isfinite(loss) and loss.item() == 0 for loss in losses.values()
        )
        losses["total"].backward()


def test_direct_action_targets_and_unobserved_context_never_enter_model():
    model, batch = make_model(True).eval(), make_batch(True)
    before = model.sample(batch, "gripper")["action"]
    for name in MODALITIES:
        withheld = ~batch[name]["observed"]
        batch[name]["value"][withheld] = float("nan")
        batch[name]["time"][withheld] = 12345
    batch["action"]["value"][:] = float("nan")
    batch["action"]["target_valid"][:] = False
    after = model.sample(batch, "gripper")["action"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_direct_action_uses_both_observed_modalities_and_action_query_time():
    model, batch = make_model(True), make_batch(True)
    for name in MODALITIES:
        batch[name]["value"].requires_grad_()
    batch["action"]["value"].requires_grad_()
    prediction = model(batch, "gripper")["action"]
    prediction.square().sum().backward()
    assert batch["action"]["value"].grad is None
    for name in MODALITIES:
        grad = batch[name]["value"].grad
        assert grad[batch[name]["observed"]].abs().sum() > 0
        assert grad[~batch[name]["observed"]].abs().sum() == 0
    batch["action"]["time"] += 1
    assert not torch.allclose(prediction, model(batch, "gripper")["action"])
    losses = model.training_losses(batch, "gripper")
    assert set(losses) == {"action", "total"}
    expected = (
        (model(batch, "gripper")["action"] - batch["action"]["value"]).square().mean()
    )
    torch.testing.assert_close(losses["total"], expected)


def test_frame_noise_and_padding_are_consistent_and_target_masks_loss_only():
    batch = make_batch()
    levels = DinoProprioModel.frame_noise_levels(batch["image"])
    assert torch.equal(levels[:, 0::2], levels[:, 1::2])
    assert torch.count_nonzero(levels[batch["image"]["observed"]]) == 0
    for name in MODALITIES:
        batch[name]["valid"][:, -1] = False
        batch[name]["observed"][:, -1] = False
        batch[name]["value"][:, -1] = float("nan")
        batch[name]["time"][:, -1] = float("nan")
    batch["image"]["xy"][:, -1] = float("nan")
    batch["image"]["camera"][:, -1] = -999
    model = make_model()
    losses = model.training_losses(batch, "gripper")
    assert all(torch.isfinite(v) for v in losses.values())
    sampled = model.sample(batch, "gripper", steps=1)
    assert all(torch.count_nonzero(v[:, -1]) == 0 for v in sampled.values())


@pytest.mark.parametrize("direct", [False, True])
def test_checkpointed_chunked_attention_matches_full_attention(direct):
    model = make_model(direct)
    chunked = make_model(
        direct, attention_query_chunk_size=3, activation_checkpointing=True
    )
    chunked.load_state_dict(model.state_dict())
    batch = make_batch(direct)
    args = (
        (batch, "gripper")
        if direct
        else (
            batch,
            "gripper",
            batch["image"]["value"],
            torch.zeros_like(batch["image"]["time"]),
        )
    )
    first, second = model(*args), chunked(*args)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=1e-5, atol=1e-6)
    sum(v.square().sum() for v in first.values()).backward()
    sum(v.square().sum() for v in second.values()).backward()
    for first_param, second_param in zip(model.parameters(), chunked.parameters()):
        if first_param.grad is not None:
            torch.testing.assert_close(
                first_param.grad, second_param.grad, rtol=1e-4, atol=1e-5
            )


def test_rejects_tracks_observed_action_labels_missing_targets_and_invalid_controls():
    model, batch = make_model(), make_batch()
    batch["track"] = {}
    with pytest.raises(ValueError, match="track stream"):
        model.training_losses(batch, "gripper")
    del batch["track"]
    batch["image"]["target_valid"][:, -1] = False
    with pytest.raises(ValueError, match="recorded features"):
        model.training_losses(batch, "gripper")
    with pytest.raises(ValueError, match="positive integer"):
        model.sample(make_batch(), "gripper", steps=0)
    with pytest.raises(ValueError, match="0.2"):
        make_model(pose_noise_max=0.3)
    direct, batch = make_model(True), make_batch(True)
    batch["action"]["observed"][:] = True
    with pytest.raises(ValueError, match="never be marked observed"):
        direct.sample(batch, "gripper")


def test_discrete_requires_indices_and_changes_clock_not_noise():
    model, batch = make_model(time_mode="discrete").eval(), make_batch()
    images, levels = batch["image"]["value"], torch.zeros_like(batch["image"]["time"])
    before = model(batch, "gripper", images, levels, phase="B")
    for name in MODALITIES:
        batch[name]["time"] *= 100
    after = model(batch, "gripper", images, levels, phase="B")
    torch.testing.assert_close(before["proprio"], after["proprio"], rtol=0, atol=0)
    del batch["image"]["index"]
    with pytest.raises(ValueError, match="explicit grid indices"):
        model(batch, "gripper", images, levels)
