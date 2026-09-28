import pytest
import torch

from egomimic.pipeline.stages_dual_flow import (
    DualFlowDenoiserStage,
    DualFlowNoisingStage,
    DualFlowVelocityLossStage,
)


class _ZeroVelocity(torch.nn.Module):
    def forward(self, sample, _time, _condition):
        return torch.zeros_like(sample)


def test_dual_flow_splits_shape_and_clock_targets():
    stage = DualFlowNoisingStage(
        waypoint_horizon=4,
        action_dim=3,
        time_dist="uniform",
    )
    result = stage({"target": torch.randn(2, 8, 3)})

    assert result["dual_flow/noisy_shape"].shape == (2, 4, 3)
    assert result["dual_flow/noisy_clock"].shape == (2, 4, 3)
    assert result["dual_flow/time"].shape == (2,)


@pytest.mark.parametrize("layout", ["rows", "channels"])
def test_dual_flow_inference_reassembles_canonical_token_layout(layout):
    stage = DualFlowDenoiserStage(
        shape_model=_ZeroVelocity(),
        clock_model=_ZeroVelocity(),
        waypoint_horizon=4,
        action_dim=3,
        condition_input_dim=5,
        num_inference_steps=2,
        token_layout=layout,
    )
    result = stage.execute({"condition": torch.randn(2, 5)}, mode="inference")

    assert result["pred_action"].shape == ((2, 4, 6) if layout == "channels" else (2, 8, 3))


def test_channel_layout_preserves_independent_flow_paths_and_training_targets():
    shape, clock = torch.randn(2, 4, 3), torch.randn(2, 4, 3) + 10
    results = []
    for layout, axis in (("rows", 1), ("channels", -1)):
        stage = DualFlowNoisingStage(4, 3, token_layout=layout)
        torch.manual_seed(47)
        results.append(stage({"target": torch.cat((shape, clock), dim=axis)}))
    for key in (
        "dual_flow/noisy_shape", "dual_flow/noisy_clock",
        "dual_flow/shape_velocity_target", "dual_flow/clock_velocity_target",
        "dual_flow/time",
    ):
        torch.testing.assert_close(results[0][key], results[1][key])


def test_channel_flow_rejects_legacy_row_targets():
    with pytest.raises(ValueError, match="Dual-flow target"):
        DualFlowNoisingStage(4, 3, token_layout="channels")({"target": torch.randn(2, 8, 3)})


def test_dual_flow_inference_supports_separate_trunk_conditions():
    stage = DualFlowDenoiserStage(
        shape_model=_ZeroVelocity(),
        clock_model=_ZeroVelocity(),
        waypoint_horizon=4,
        action_dim=3,
        condition_input_dim=5,
        num_inference_steps=2,
        shape_condition_key="condition_shape",
        clock_condition_key="condition_clock",
    )
    result = stage.execute(
        {
            "condition_shape": torch.randn(2, 5),
            "condition_clock": torch.randn(2, 5),
        },
        mode="inference",
    )

    assert result["pred_action"].shape == (2, 8, 3)


def test_dual_flow_loss_keeps_shape_and_weighted_clock_terms_separate():
    stage = DualFlowVelocityLossStage(clock_loss_weight=0.25)
    zeros = torch.zeros(2, 4, 3)
    ones = torch.ones_like(zeros)
    result = stage(
        {
            "dual_flow/predicted_shape_velocity": zeros,
            "dual_flow/predicted_clock_velocity": zeros,
            "dual_flow/shape_velocity_target": ones,
            "dual_flow/clock_velocity_target": ones,
        }
    )

    assert torch.equal(result["loss/flow_shape"], torch.tensor(1.0))
    assert torch.equal(result["loss/flow_clock"], torch.tensor(0.25))
