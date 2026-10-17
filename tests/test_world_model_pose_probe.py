"""Contracts for native MoT readout diagnostics; no quality claims from fixtures."""

import json
from types import SimpleNamespace

import numpy as np
import torch

from egomimic.models.world_model import MODALITIES
from egomimic.scripts.embedding_process.world_model_pose_probe import (
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


def test_preflight_exercises_both_backward_paths_without_optimizer_update(
    tmp_path, monkeypatch
):
    data, meta = inputs()
    meta["episodes"] = ["0", "1", "2"]
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
