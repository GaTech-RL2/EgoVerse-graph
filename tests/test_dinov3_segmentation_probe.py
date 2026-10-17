import numpy as np
import pytest
import torch

from egomimic.scripts.embedding_process.dinov3_segmentation_probe import (
    boundary_f1, native_probability, objective, overlap, patch_targets, rasterize,
)


def test_letterbox_targets_ignore_padding_and_unknown():
    prep = {"frames": [{"resized_hw": [16, 32], "padding_top_left": [8, 0],
                        "original_hw": [16, 32]}],
            "processed_hw": [32, 32], "patch_grid": [2, 2]}
    mask = np.zeros((16, 32), bool)
    mask[:, :16] = True
    valid = np.ones_like(mask)
    valid[:, 16:] = False
    targets = patch_targets(mask, valid, prep)
    assert torch.equal(targets[:, 0], torch.zeros(4))
    assert torch.equal(targets[:, 1], torch.tensor([.5, 0, .5, 0]))
    # Constant log-odds must survive both letterbox removal and native resize.
    logits = torch.tensor([[0., 2.]]).repeat(4, 1)
    probability = native_probability(logits, prep)
    assert probability.shape == (16, 32)
    assert np.allclose(probability, torch.sigmoid(torch.tensor(2.)).item())


def test_mask_holes_uncertainty_and_metrics():
    frame = {"hand_polygons": [[[8, 8], [30, 8], [30, 30], [8, 30]]],
             "background_holes": [[[15, 15], [20, 15], [20, 20], [15, 20]]],
             "ignore_polygons": [[[0, 0], [3, 0], [3, 39], [0, 39]]]}
    mask, certain, valid = rasterize(frame, (40, 40), 2)
    assert mask[12, 12] and not mask[17, 17]
    assert not certain[20, 2] and not valid[8, 8]
    assert overlap(mask, mask, valid)["iou"] == 1
    assert boundary_f1(mask, mask, certain, 2) == 1
    assert overlap(np.zeros_like(mask), mask, valid)["dice"] == 0


def test_unknown_labels_have_zero_gradient_and_reject_empty():
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    targets = torch.tensor([[[1., 0.], [0., 0.]]])
    loss = objective(logits, targets, torch.ones(2))
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[0, 0]) == 2
    with pytest.raises(ValueError, match="No labeled"):
        objective(logits, torch.zeros_like(targets), torch.ones(2))
