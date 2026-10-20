import numpy as np
import pytest

from egomimic.eval.distance_budget_dtw import (
    METRIC_FRAME_KEY,
    METRIC_VERSION,
    distance_windows,
    global_dtw,
    joint_cumulative_distance,
    summarize_distance_dtw,
    world_xyz,
)
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


def wide_records(source):
    result = []
    for item in source:
        converted = dict(item)
        prediction = np.asarray(item["prediction"])
        m = prediction.shape[0] // 2
        converted["prediction"] = np.concatenate(
            (prediction[:m], prediction[m:]), axis=1
        )
        result.append(converted)
    return result


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
