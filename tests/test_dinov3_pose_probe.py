"""Probe isolation, padding and leakage controls; no pretrained-quality claims."""

import pytest
import torch

from egomimic.scripts.embedding_process.dinov3_pose_probe import (
    PoseHeatmapProbe,
    episode_split,
    heatmap_targets,
    mean_pose_baseline,
    objective,
)


def test_image_only_probe_finds_informative_patch_and_excludes_padding():
    model = PoseHeatmapProbe(3, 1)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[1.0, -1.0, 0.0]]))
    features = torch.tensor([[[2.0, -2.0, 0.0], [0.0, 2.0, -2.0], [1.0, -1.0, 0.0]]])
    valid = torch.tensor([False, True, True])
    logits = model(features, valid)
    assert logits.argmax(-1).item() == 2
    assert logits.dtype == torch.float32
    assert sum(p.numel() for p in PoseHeatmapProbe(768, 42).parameters()) == 32256
    assert not features.requires_grad


def test_targets_normalize_only_over_real_image_and_ignore_unlabeled_points():
    centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    target = heatmap_targets(
        torch.tensor([[[10.0, 0.0], [0.0, 0.0]]]),
        centers,
        torch.tensor([False, True, True]),
        2.0,
    )
    assert torch.equal(target[..., 0], torch.zeros(1, 2))
    torch.testing.assert_close(target.sum(-1), torch.ones(1, 2))
    logits = torch.randn(1, 2, 3, requires_grad=True)
    loss = objective(logits, target, torch.tensor([[True, False]]))
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))


def test_empty_labels_produce_zero_loss_and_gradients():
    logits = torch.randn(1, 2, 3, requires_grad=True)
    loss = objective(
        logits,
        torch.softmax(torch.randn_like(logits), -1),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_episode_split_and_training_only_baseline_reject_image_leakage():
    metadata = [
        {"episode": e, "image_sha256": str(i)}
        for i, e in enumerate(["a", "a", "b", "c"])
    ]
    train, test = episode_split(metadata, "b")
    assert train == [0, 1, 3] and test == [2]
    xy = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[1000.0, 2000.0]], [[5.0, 6.0]]])
    baseline = mean_pose_baseline(xy[train], torch.ones(3, 1, dtype=torch.bool))
    torch.testing.assert_close(baseline, torch.tensor([[3.0, 4.0]]))
    metadata[2]["image_sha256"] = "0"
    with pytest.raises(ValueError, match="Identical image bytes"):
        episode_split(metadata, "b")


def test_missing_training_landmark_is_rejected():
    with pytest.raises(ValueError, match="training label"):
        mean_pose_baseline(torch.zeros(2, 1, 2), torch.zeros(2, 1, dtype=torch.bool))
