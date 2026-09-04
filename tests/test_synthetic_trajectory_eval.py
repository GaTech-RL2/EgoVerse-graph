import numpy as np
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
        points = SyntheticTrajectoryEval.export(
            model, source, target, output, steps=3
        )
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
