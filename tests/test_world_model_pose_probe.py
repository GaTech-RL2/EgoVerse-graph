"""Contracts for native MoT readout diagnostics; no quality claims from fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.models.world_model import MODALITIES
from egomimic.scripts.embedding_process.world_model_pose_probe import (
    fixed_partition,
    fold_data,
    make_batch,
    model_for_probe,
    predict,
    run,
    sha,
)


def inputs():
    rng = np.random.default_rng(42)
    data = {
        "features": rng.normal(size=(6, 4, 768)).astype(np.float32),
        "patch_xy": np.zeros((4, 2), np.float32),
        "patch_valid": np.ones(4, bool),
        "pairs": np.array([[0, 1], [2, 3], [4, 5]]),
        "poses": rng.normal(size=(3, 2, 126)).astype(np.float32),
        "dt": np.array([0.5, 1, 2], np.float32),
    }
    meta = {"frames": [{"episode": str(i // 2)} for i in range(6)]}
    return data, meta


def test_freeze_uses_named_track_branch_not_proprio():
    model = model_for_probe(width=16, depth=1)
    track = MODALITIES.index("track")
    pose = MODALITIES.index("proprio")
    for block in model.blocks:
        assert all(not p.requires_grad for p in block.qkv[track].parameters())
        assert all(p.requires_grad for p in block.qkv[pose].parameters())
    assert model.pose_out["human"].weight.requires_grad


def test_normalization_cannot_see_heldout_frames_or_poses():
    data, meta = inputs()
    _, _, before = fold_data(data, meta, "2")
    data["features"][4:] += 1000
    data["poses"][2] += 1000
    _, _, after = fold_data(data, meta, "2")
    for key in before:
        assert np.array_equal(before[key], after[key])


def test_fixed_split_excludes_validation_and_test_from_normalization():
    data, meta = inputs()
    for i, row in enumerate(meta["frames"]):
        row["image_sha256"] = str(i)
    split = {"train": ["0"], "validation": ["1"], "test": ["2"]}
    indices, before = fixed_partition(data, meta, split)
    assert {k: v.tolist() for k, v in indices.items()} == {
        "train": [0],
        "validation": [1],
        "test": [2],
    }
    data["features"][2:] += 1000
    data["poses"][1:] += 1000
    _, after = fixed_partition(data, meta, split)
    for key in before:
        assert np.array_equal(before[key], after[key])


def test_fixed_split_rejects_copied_images_and_cross_episode_pairs():
    data, meta = inputs()
    for i, row in enumerate(meta["frames"]):
        row["image_sha256"] = str(i)
    split = {"train": ["0"], "validation": ["1"], "test": ["2"]}
    meta["frames"][2]["image_sha256"] = "0"
    with pytest.raises(ValueError, match="Image content overlaps"):
        fixed_partition(data, meta, split)
    meta["frames"][2]["image_sha256"] = "2"
    data["pairs"][0] = [0, 2]
    with pytest.raises(ValueError, match="crosses episodes"):
        fixed_partition(data, meta, split)


def test_future_pose_is_loss_only_and_no_fake_tracks():
    data, meta = inputs()
    _, _, norm = fold_data(data, meta, "2")
    batch, target = make_batch(data, norm, [0])
    model = model_for_probe(width=16, depth=1).eval()
    assert batch["track"]["value"].shape == (1, 0, 3)
    assert torch.count_nonzero(batch["proprio"]["value"][:, 1]) == 0
    before = predict(model, batch)
    data["poses"][0, 1] += 100
    changed, target2 = make_batch(data, norm, [0])
    assert not torch.equal(target, target2)
    assert torch.equal(before, predict(model, changed))


def test_observed_only_is_image_invariant_but_proprio_gradients_flow():
    data, meta = inputs()
    _, _, norm = fold_data(data, meta, "2")
    batch, target = make_batch(data, norm, [0], visual=False)
    model = model_for_probe(width=16, depth=1)
    before = predict(model, batch)
    batch["image"]["value"] += 1000
    assert torch.equal(before, predict(model, batch))
    (before - target).square().mean().backward()
    assert model.blocks[0].qkv[MODALITIES.index("proprio")].weight.grad.norm() > 0
    assert model.blocks[0].qkv[MODALITIES.index("track")].weight.grad is None


@pytest.mark.parametrize("explicit_split", [False, True])
def test_preflight_exercises_both_backward_paths_without_optimizer_update(
    tmp_path, monkeypatch, explicit_split
):
    data, meta = inputs()
    meta["episodes"] = ["0", "1", "2"]
    for i, row in enumerate(meta["frames"]):
        row["image_sha256"] = str(i)
    split_file = tmp_path / "SPLIT.json"
    split_file.write_text(
        json.dumps({"train": ["0"], "validation": ["1"], "test": ["2"]})
    )
    np.savez(tmp_path / "INPUTS.npz", **data)
    meta["input_sha256"] = sha(tmp_path / "INPUTS.npz")
    (tmp_path / "INPUTS.json").write_text(json.dumps(meta))
    count = []
    original = torch.autograd.backward

    def backward(*args, **kwargs):
        count.append(1)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not update parameters")

    monkeypatch.setattr(torch.autograd, "backward", backward)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    run(
        SimpleNamespace(
            inputs=tmp_path,
            output=tmp_path / "output",
            threads=1,
            steps=1,
            save_every=1,
            width=16,
            depth=1,
            device="cpu",
            lr=1e-4,
            preflight_only=True,
            split_manifest=split_file if explicit_split else None,
        )
    )
    assert len(count) == 2
    assert not list((tmp_path / "output").glob("fold-*"))
    assert (
        json.loads((tmp_path / "output/RUNTIME_BACKWARD_GUARD.json").read_text())[
            "optimizer_updates"
        ]
        == 0
    )


def test_selection_uses_validation_and_seals_test_until_selection(
    tmp_path, monkeypatch
):
    """Exercise orchestration without optimizer updates or a quality claim."""
    from egomimic.scripts.embedding_process import world_model_pose_probe as probe

    data, meta = inputs()
    meta["episodes"] = ["0", "1", "2"]
    for i, row in enumerate(meta["frames"]):
        row["image_sha256"] = str(i)
    np.savez(tmp_path / "INPUTS.npz", **data)
    meta["input_sha256"] = sha(tmp_path / "INPUTS.npz")
    (tmp_path / "INPUTS.json").write_text(json.dumps(meta))
    split_file = tmp_path / "SPLIT.json"
    split_file.write_text(
        json.dumps({"train": ["0"], "validation": ["1"], "test": ["2"]})
    )
    output = tmp_path / "output"
    val_calls = {False: 0, True: 0}

    def evaluate(model, data, meta, norm, indices, device, visual):
        if indices.tolist() == [1]:
            val_calls[visual] += 1
            score = float(val_calls[visual])  # First checkpoint is best.
        else:
            condition = (
                "oracle_images_and_observed_pose" if visual else "observed_pose_only"
            )
            selection = json.loads(
                (output / "fold-0" / condition / "SELECTION.json").read_text()
            )
            assert selection["selected_step"] == 1
            assert selection["test_used_for_selection"] is False
            score = 3.0
        return [{"mpjpe_m": score, "hold_pose_mpjpe_m": 4.0}]

    monkeypatch.setattr(probe, "evaluate", evaluate)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: None)
    run(
        SimpleNamespace(
            inputs=tmp_path,
            output=output,
            threads=1,
            steps=2,
            save_every=1,
            width=16,
            depth=1,
            device="cpu",
            lr=1e-4,
            preflight_only=False,
            split_manifest=split_file,
        )
    )
    assert val_calls == {False: 2, True: 2}
    assert len(list(output.glob("fold-*/*/epoch*.pt"))) == 4
