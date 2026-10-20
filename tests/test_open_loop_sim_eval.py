import numpy as np
import pytest
import torch

from egomimic.eval.open_loop_sim import (
    OpenLoopSimEval,
    executed_arc_waypoints,
    executed_control_steps,
    truncate_cartesian_trajectory_by_joint_distance,
)


def test_execute_fraction_is_a_control_frequency_prefix():
    assert executed_control_steps(100, 0.01) == 1
    assert executed_control_steps(100, 0.25) == 25
    assert executed_control_steps(100, 1.0) == 100
    with pytest.raises(ValueError, match="execute_fraction"):
        executed_control_steps(100, 0.0)


def test_m_based_arc_execution_requires_an_exact_waypoint_count():
    assert executed_arc_waypoints(100, 0.30) == 30
    assert executed_arc_waypoints(20, 0.25) == 5
    with pytest.raises(ValueError, match=r"M \* execute_fraction"):
        executed_arc_waypoints(100, 0.333)


def test_video_trajectory_cap_interpolates_at_joint_cumulative_distance():
    trajectory = np.zeros((6, 14), dtype=np.float64)
    trajectory[:, 0] = np.linspace(0.0, 0.10, 6)
    trajectory[:, 7] = np.linspace(0.0, 0.20, 6)

    partial = truncate_cartesian_trajectory_by_joint_distance(trajectory, 0.12)
    joint_distance = (
        np.linalg.norm(np.diff(partial[:, 0:3], axis=0), axis=-1).sum()
        + np.linalg.norm(np.diff(partial[:, 7:10], axis=0), axis=-1).sum()
    )

    assert len(partial) == 3
    assert joint_distance == pytest.approx(0.12)
    assert partial[-1, 0] == pytest.approx(0.04)
    assert partial[-1, 7] == pytest.approx(0.08)


def test_executed_chunk_video_holds_each_chunk_until_it_ends():
    """The overlay is replanned only when the previous executed chunk ends."""
    from egomimic.eval.distance_budget_dtw import METRIC_FRAME_KEY

    evaluator = OpenLoopSimEval.__new__(OpenLoopSimEval)
    evaluator._validation_group = None
    evaluator.image_key = "image"
    evaluator.action_key = "actions"
    evaluator.obs_pose_key = "pose"
    evaluator.ground_truth_action_key = "gt"
    evaluator.dtw_max_prediction_steps = 1000
    evaluator._chunk_states = {}
    evaluator._native = lambda value, _: value
    evaluator._native_key = lambda value, key, _: value
    evaluator._native_pose = lambda value, _: value
    evaluator._revert_to_camframe = (
        lambda *, actions, obs_pose, embodiment_name: actions
    )
    evaluator._group_video_dir = lambda group, name: "unused"
    lengths = {}

    def decode(native, max_steps):
        steps = int(native[0, 0])
        return np.repeat(native[:1], steps, axis=0), steps

    evaluator._decode_prediction_with_steps = decode
    shown = []
    evaluator._buffer_per_episode = lambda key, out, frames, hashes: shown.extend(
        (int(f[0, 0, 0]), lengths[int(f[0, 0, 0])]) for f in frames
    )

    def viz(*, predictions, batch):
        pred = predictions["yam_bimanual_actions"]
        anchor = int(pred[0, 0, 1])
        lengths[anchor] = (pred.shape[1], batch["actions"].shape[1])
        return np.full((1, 2, 2, 3), anchor, dtype=np.uint8)

    eye = np.repeat(np.eye(4)[None], 2, axis=0)
    for frame in range(8):
        prediction = np.zeros((1, 4, 14))
        prediction[0, 0, 0] = 3  # every replanned chunk executes 3 frames
        prediction[0, 0, 1] = frame  # tag which frame planned it
        evaluator._log_executed_chunk_frame(
            source_id="yam_bimanual",
            source_batch={
                "episode_hash": ["episode"],
                "frame_index": torch.tensor([frame]),
                "image": torch.zeros(1, 2, 2, 3),
                "gt": torch.zeros(1, 10, 14),
                "pose": torch.zeros(1, 14),
                "embodiment": torch.tensor([7]),
                METRIC_FRAME_KEY: torch.as_tensor(eye[None]),
            },
            prediction=torch.as_tensor(prediction),
            embodiment_id=7,
            embodiment_name="yam_bimanual",
            viz_partial=viz,
        )
    evaluator._flush_all_executed_chunks()
    # Frames 0-2 show the chunk planned at 0, 3-5 the one at 3, 6-7 the one at 6
    # (the episode ends after two of its three frames).
    assert [anchor for anchor, _ in shown] == [0, 0, 0, 3, 3, 3, 6, 6]
    # Both overlays stop at the episode boundary, just like the metric.
    assert lengths[0] == (3, 3) and lengths[3] == (3, 3) and lengths[6] == (2, 2)
