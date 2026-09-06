import numpy as np
import pytest
import torch

from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.shared_latent_flow import (
    SyntheticDirectFlow,
    SyntheticSharedLatentFlow,
)


def test_shared_eval_exports_same_npz_contract_for_both_model_families(tmp_path):
    target = torch.randn(7, 3)
    models_and_sources = (
        (SyntheticSharedLatentFlow(latent_dim=2), torch.randn(7, 2)),
        (SyntheticDirectFlow(data_dim=3), torch.randn(7, 3)),
    )
    for index, (model, source) in enumerate(models_and_sources):
        output = tmp_path / f"trajectory-{index}.npz"
        points = SyntheticTrajectoryEval.export(model, source, target, output, steps=3)
        archive = np.load(output, allow_pickle=False)
        assert set(archive.files) == {"times", "points", "target"}
        assert points.shape == (4, 7, 3)
        assert archive["points"].shape == (4, 7, 3)
        np.testing.assert_array_equal(archive["target"], target.numpy())


def test_ground_truth_uses_same_npz_contract(tmp_path):
    source = torch.randn(5, 2)
    target = torch.randn(5, 3)
    output = tmp_path / "ground-truth.npz"
    points = SyntheticTrajectoryEval.export_linear_ground_truth(
        source, target, output, steps=2
    )
    torch.testing.assert_close(points[0, :, :2], source)
    torch.testing.assert_close(points[0, :, 2], torch.zeros(5))
    torch.testing.assert_close(points[-1], target)


def test_validation_loader_requires_exact_requested_particle_count(tmp_path):
    dataset = tmp_path / "small.npz"
    np.savez_compressed(
        dataset,
        source_latent=np.zeros((5, 2), dtype=np.float32),
        target_3d=np.zeros((5, 3), dtype=np.float32),
        split=np.array([0, 1, 1, 2, 2], dtype=np.uint8),
    )
    source, target = SyntheticTrajectoryEval.load_validation_data(
        dataset, "source_latent", 2
    )
    assert source.shape == (2, 2)
    assert target.shape == (2, 3)
    with pytest.raises(ValueError, match="requested 3 validation particles"):
        SyntheticTrajectoryEval.load_validation_data(dataset, "source_latent", 3)


def test_symmetric_nearest_neighbor_mse_checks_both_directions():
    samples = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    targets = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    loss = SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(samples, targets)
    torch.testing.assert_close(loss, torch.tensor(2.0))

    collapsed = torch.zeros(2, 2)
    collapsed_loss = SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(
        collapsed, targets
    )
    assert collapsed_loss > 0
