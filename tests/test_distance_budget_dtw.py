import json

import numpy as np
import pytest
import torch

from egomimic.eval.distance_budget_dtw import (
    METRIC_FRAME_KEY,
    METRIC_VERSION,
    distance_windows,
    fractional_waypoint_prefix,
    global_dtw,
    joint_cumulative_distance,
    score_distance_dtw_episode,
    summarize_distance_dtw,
    world_xyz,
)
from egomimic.eval.open_loop_sim import OpenLoopSimEval
from egomimic.rldb.zarr.action_chunk_transforms import StoreBimanualMetricFrame


def reference_cost(pred, gt):
    dp = np.full((len(pred) + 1, len(gt) + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, len(pred) + 1):
        for j in range(1, len(gt) + 1):
            dp[i, j] = np.square(pred[i - 1] - gt[j - 1]).mean() + min(
                dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]
            )
    return dp[-1, -1]


@pytest.mark.parametrize("n,m", [(1, 1), (1, 9), (9, 1), (7, 19), (19, 7), (40, 40)])
def test_global_dtw_matches_exact_reference_and_covers_both_sequences(n, m):
    rng = np.random.default_rng(n + m)
    for _ in range(8):
        pred, gt = rng.normal(size=(n, 6)), rng.normal(size=(m, 6))
        result = global_dtw(pred, gt, return_path=True)
        assert result["path_cost_sum"] == pytest.approx(reference_cost(pred, gt))
        np.testing.assert_array_equal(result["path"][0], [0, 0])
        np.testing.assert_array_equal(result["path"][-1], [n - 1, m - 1])
        assert result["gt_coverage"] == result["prediction_coverage"] == 1


@pytest.mark.parametrize("slower", [False, True])
def test_speed_warp_in_either_direction_has_zero_error_and_full_coverage(slower):
    # Exact same path, different dwell times. Slower has MORE pred samples.
    shape = np.arange(8, dtype=float)[:, None] * np.ones((1, 6))
    pred, gt = (
        (np.repeat(shape, 4, axis=0), shape)
        if slower
        else (shape, np.repeat(shape, 4, axis=0))
    )
    result = global_dtw(pred, gt, return_path=True)
    assert result["xyz_mse"] == 0
    assert result["gt_coverage"] == result["prediction_coverage"] == 1
    assert len(result["per_gt_cost"]) == len(gt)
    assert (result["matches_per_gt"] > 0).all()


def test_slower_prediction_averages_matches_within_each_gt_frame():
    pred = np.repeat(np.array([[0.0], [2.0], [3.0], [10.0]]), 6, axis=1)
    gt = np.repeat(np.array([[0.0], [10.0]]), 6, axis=1)
    result = global_dtw(pred, gt, return_path=True)
    np.testing.assert_array_equal(result["matches_per_gt"], [3, 1])
    assert result["xyz_mse"] == pytest.approx(((0 + 4 + 9) / 3 + 0) / 2)
    assert result["xyz_mse"] != pytest.approx(
        result["path_cost_sum"] / result["path_pairs"]
    )


def test_one_shared_alignment_does_not_hide_opposite_arm_motion():
    pred = np.zeros((9, 6))
    pred[:, 0] = np.arange(9)
    pred[:, 3] = np.arange(9)[::-1]
    gt = pred.copy()
    gt[:, 3] = np.arange(9)
    assert global_dtw(pred, gt)["xyz_mse"] > 0


def test_joint_distance_windows_and_final_partial_budget():
    xyz = np.zeros((8, 6))
    xyz[:, 0] = np.arange(8) * 0.01
    xyz[:, 3] = np.arange(8) * 0.02
    cumulative = joint_cumulative_distance(xyz)
    assert cumulative[-1] == pytest.approx(0.21)
    anchors, budgets = distance_windows(cumulative, 0.1)
    np.testing.assert_array_equal(anchors, [0, 4, 7])
    np.testing.assert_allclose(budgets, [0.1, 0.1, 0.01])
    assert len(distance_windows(np.array([0.0, 0.3]), 0.1)[0]) == 3
    np.testing.assert_array_equal(distance_windows(np.zeros(100), 0.1)[0], [0])


def evaluator(mode="arc", cap="waypoints", hybrid=False):
    ev = OpenLoopSimEval.__new__(OpenLoopSimEval)
    ev.action_mode = mode
    ev.resampled_vector_length = 100
    ev.execute_fraction = 0.3
    ev.execute_steps = 30
    ev.execute_arc_waypoints = 30
    ev.control_horizon = 100
    ev.control_dt = 1 / 30
    ev.min_distance_unit = 0.4
    ev.rotation_distance_unit = np.deg2rad(24) if hybrid else None
    ev.arc_execution_cap_mode = cap
    ev.velocity_mode = "per_waypoint"
    ev._arc_tokenizer = None
    ev.dtw_max_cells = 2_000_000
    ev.dtw_max_prediction_steps = 10_000
    ev.distance_dtw_enabled = True
    ev.require_episode_start = True
    ev.limit_val_episodes = None
    ev._metric_device = torch.device("cpu")
    return ev


def records(mode="arc", speed=0.2, stationary=False):
    result = []
    for frame in range(61):
        anchors = np.repeat(np.eye(4)[None], 2, axis=0)
        anchors[:, 0, 3] = 0 if stationary else frame * 0.004
        gt = np.zeros((100, 14))
        pred = np.zeros((200 if mode == "arc" else 100, 14))
        if mode == "arc":
            pred[:100, 0] = pred[:100, 7] = np.linspace(0, 0.2, 100)
            pred[100:, 0] = pred[100:, 7] = speed
        else:
            pred[:, 0] = pred[:, 7] = np.arange(100) * 0.004
        result.append(
            {
                "frame": frame,
                "episode": "e",
                "group": "valid",
                "source": "s",
                "label": "yam_bimanual",
                "prediction": pred,
                "ground_truth": gt,
                METRIC_FRAME_KEY: anchors,
            }
        )
    return result


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("cap", ["waypoints", "distance"])
def test_arc_chunk_count_independent_of_fast_or_slow_timing(cap, hybrid):
    ev = evaluator(cap=cap, hybrid=hybrid)
    fast = score_distance_dtw_episode(ev, records(speed=0.4))
    slow = score_distance_dtw_episode(ev, records(speed=0.015))
    expected_budget = 0.4 * (29 / 99 if cap == "waypoints" else 0.3)
    assert fast["execution_distance_budget_m"] == pytest.approx(expected_budget)
    assert (
        fast["segments"]
        == slow["segments"]
        == int(np.ceil(0.48 / expected_budget - 1e-10))
    )
    assert fast["predicted_samples"] < fast["gt_frames"]
    assert slow["predicted_samples"] > slow["gt_frames"]
    assert slow["duration_ratio"] > 1
    assert slow["gt_coverage"] == slow["prediction_coverage"] == 1
    assert fast["anchor_frames"] == slow["anchor_frames"]
    assert sum(slow["segment_distance_budgets_m"]) == pytest.approx(0.48)


def test_baseline_retains_frame_chunking_and_uses_same_global_dtw(monkeypatch):
    import egomimic.eval.distance_budget_dtw as module

    calls = []
    original = module.global_dtw

    def counted(pred, gt, **kwargs):
        calls.append((len(pred), len(gt)))
        return original(pred, gt, **kwargs)

    monkeypatch.setattr(module, "global_dtw", counted)
    result = score_distance_dtw_episode(evaluator("baseline"), records("baseline"))
    assert result["anchor_frames"] == [0, 30, 60]
    assert result["segment_control_steps"] == [30, 30, 1]
    assert calls == [(61, 61)]
    calls.clear()
    score_distance_dtw_episode(evaluator(), records())
    assert len(calls) == 1


def test_stationary_episode_still_scores_every_gt_frame_and_predicted_motion():
    result = score_distance_dtw_episode(evaluator(), records(stationary=True))
    assert result["segments"] == 1
    assert result["stationary_fallback"]
    assert result["gt_frames"] == 61
    assert result["gt_coverage"] == 1
    assert result["xyz_mse"] > 0


def test_fractional_final_prefix_preserves_velocity():
    token = records()[0]["prediction"]
    partial = fractional_waypoint_prefix(token, 0.125)
    m = len(partial) // 2
    assert partial[m - 1, 0] == pytest.approx(0.025)
    np.testing.assert_array_equal(partial[m:, 0], 0.2)


def test_hybrid_rotation_has_independent_slower_clock_after_prefix_capping():
    ev = evaluator(hybrid=True)
    samples = records(speed=0.4)
    for item in samples:
        token = item["prediction"]
        token[:100, 3] = token[:100, 10] = np.linspace(0, 0.2, 100)
        token[100:, 3] = token[100:, 10] = 0.01
    result = score_distance_dtw_episode(ev, samples)
    translation_only = score_distance_dtw_episode(ev, records(speed=0.4))
    assert result["segments"] == translation_only["segments"]
    assert result["predicted_samples"] > translation_only["predicted_samples"]
    assert result["predicted_samples"] > result["gt_frames"]
    assert result["gt_coverage"] == result["prediction_coverage"] == 1


@pytest.mark.parametrize("head_x,head_yaw", [(0.0, 0.0), (3.0, 1.2), (-4.0, -0.8)])
def test_human_world_trajectory_is_invariant_to_moving_camera(head_x, head_yaw):
    from egomimic.rldb.embodiment.human import Human

    transform_list = Human.get_transform_list(
        action_mode="cartesian_gripper_padded",
        coord_frame="eef_frame",
        rotation_mode="euler",
        stride=1,
        chunk_length=4,
    )
    left = np.array([0.3, 0.2, 0.1, 1.0, 0.0, 0.0, 0.0])
    right = np.array([0.8, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0])
    left_actions, right_actions = np.tile(left, (4, 1)), np.tile(right, (4, 1))
    left_actions[:, 0] += np.arange(4) * 0.02
    right_actions[:, 1] += np.arange(4) * 0.03
    batch = {
        "obs_head_pose": np.array(
            [head_x, 2.0, 1.0, np.cos(head_yaw / 2), 0.0, 0.0, np.sin(head_yaw / 2)]
        ),
        "left.obs_ee_pose": left,
        "right.obs_ee_pose": right,
        "left.action_ee_pose": left_actions.copy(),
        "right.action_ee_pose": right_actions.copy(),
    }
    for transform in transform_list:
        batch = transform.transform(batch)
    restored = world_xyz(batch["actions_cartesian"], batch[METRIC_FRAME_KEY])
    np.testing.assert_allclose(
        restored,
        np.concatenate((left_actions[:, :3], right_actions[:, :3]), axis=1),
        atol=1e-6,
    )


def test_world_anchors_preserved_before_wrist_frame_conversion():
    transform = StoreBimanualMetricFrame("l", "r")
    # xyzwxyz layout is xyz + quaternion wxyz. Rotation is +90 degrees about Z.
    q = np.sqrt(0.5)
    batch = {
        "l": np.array([1.0, 2.0, 3.0, q, 0.0, 0.0, q]),
        "r": np.array([-1.0, -2.0, -3.0, 1.0, 0.0, 0.0, 0.0]),
    }
    anchors = transform.transform(batch)[METRIC_FRAME_KEY]
    action = np.zeros((1, 14))
    action[0, 0] = action[0, 7] = 1
    np.testing.assert_allclose(
        world_xyz(action, anchors), [[1, 3, 3, 0, -2, -3]], atol=1e-6
    )
    batch["l"][:] = 0
    assert anchors[0, 0, 3] == 1


def test_resource_limits_and_missing_frames_fail_instead_of_silent_clipping():
    with pytest.raises(ValueError, match="limit"):
        global_dtw(np.zeros((5, 6)), np.zeros((7, 6)), max_cells=30)
    ev = evaluator()
    ev.dtw_max_prediction_steps = 1
    with pytest.raises(ValueError, match="samples"):
        score_distance_dtw_episode(ev, records(speed=0.01))
    ev.dtw_max_prediction_steps = 10000
    with pytest.raises(RuntimeError, match="missing"):
        ev._score_episode(records()[0:3] + records()[4:])
    missing = records()
    del missing[0][METRIC_FRAME_KEY]
    with pytest.raises(ValueError, match="metadata"):
        score_distance_dtw_episode(ev, missing)


def test_result_integration_keeps_legacy_metrics_and_logs_new_namespace():
    ev = evaluator()
    scored = ev._score_episode(records())
    summary = ev._summarize_episodes([scored])
    assert "mse" in summary["micro"]
    assert summary["distance_dtw"]["metric_version"] == METRIC_VERSION
    metrics = ev._metric_tensors({"per_group": {"valid": summary}})
    assert "Valid/open_loop_sim/XYZ_MSE" in metrics
    assert "Valid/open_loop_sim/Distance_DTW/XYZ_MSE" in metrics
    json.dumps(scored)


def test_episode_aggregation_is_gt_frame_weighted():
    common = dict(
        metric_version=METRIC_VERSION,
        predicted_samples=5,
        segments=1,
        gt_coverage=1.0,
        prediction_coverage=1.0,
    )
    result = summarize_distance_dtw(
        [
            {"distance_dtw": dict(common, gt_frames=10, xyz_mse=1)},
            {"distance_dtw": dict(common, gt_frames=30, xyz_mse=3)},
        ]
    )
    assert result["xyz_mse"] == 2.5
    assert result["episode_xyz_mse"] == 2
