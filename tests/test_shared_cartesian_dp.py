import pytest
import torch

from egomimic.eval.yam_human_cartesian_dp_eval import YamHumanCartesianDPEval
from egomimic.pipeline.stages_matched_dp import (
    SharedNativeDPStage,
    shared_native_transformers,
)
from egomimic.rldb.zarr.action_chunk_transforms import PadGripperZeros


class IdentityNormalizer:
    def unnormalize(self, batch, embodiment_id):
        return batch


def test_shared_cartesian_head_and_optimizer_sampling_for_both_domains():
    models = shared_native_transformers(16, 144, 1, 8, 0.0, "shared_cartesian14")
    assert models["yam_bimanual"] is models["human_bimanual"]
    stage = SharedNativeDPStage(
        condition_dim=16,
        inference_steps=2,
        denoiser="transformer",
        transformer_width=144,
        transformer_depth=1,
        transformer_heads=8,
        transformer_dropout=0.0,
        action_contract="shared_cartesian14",
    )
    opt = torch.optim.AdamW(stage.parameters(), lr=1e-4)
    for domain in [7, 3]:
        opt.zero_grad(set_to_none=True)
        out = stage.execute(
            {
                "embodiment": domain,
                "condition": torch.randn(2, 16),
                "target": torch.randn(2, 100, 14),
            },
            mode="train",
        )
        loss = out["loss/diffusion_noise"]
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        stage.eval()
        with torch.no_grad():
            pred = stage.execute(
                {"embodiment": domain, "condition": torch.randn(2, 16)},
                mode="inference",
            )["pred_action"]
        assert pred.shape == (2, 100, 14) and torch.isfinite(pred).all()


def test_human_padding_preserves_pose_and_inserts_two_zero_grippers():
    pose = torch.randn(2, 100, 12)
    padded = PadGripperZeros().transform({"actions_cartesian": pose})[
        "actions_cartesian"
    ]
    assert padded.shape == (2, 100, 14)
    assert torch.equal(padded[..., :6], pose[..., :6])
    assert torch.equal(padded[..., 7:13], pose[..., 6:])
    assert padded[..., [6, 13]].count_nonzero() == 0


@pytest.mark.parametrize("label,domain", [("yam_bimanual", 7), ("human_bimanual", 3)])
def test_common14_metrics_accept_each_embodiment(label, domain):
    e = YamHumanCartesianDPEval(limit_val_batches=1.0, energy_score_enabled=False)
    e.bind_data_context(normalizer=IdentityNormalizer())
    values = e._energy_values(
        torch.zeros(32, 2, 100, 14), torch.zeros(2, 100, 14), domain, label
    )
    assert all(torch.isfinite(v).all() for v in values.values())
    assert e._native_residual(
        torch.zeros(2, 100, 14), torch.zeros(2, 100, 14), None
    ).shape == (2, 100, 14)


def test_common14_rejects_old_keypoint_contract():
    with pytest.raises(ValueError, match="Cartesian actions"):
        YamHumanCartesianDPEval(
            limit_val_batches=1.0,
            action_keys_by_embodiment={"human_bimanual": "actions_keypoints"},
        )
