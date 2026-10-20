import math

import numpy as np
import pytest

from egomimic.eval.open_loop_sim import truncate_cartesian_trajectory_by_joint_clocks
from egomimic.rldb.zarr.arc_length_tokenizer import (
    cumulative_arc_length,
    cumulative_rotation_length,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def _two_clock_chunk(steps: int = 121) -> np.ndarray:
    """Joint translation and each arm's rotation reach their caps in 2 s."""
    time = np.arange(steps, dtype=np.float64) / 30.0
    chunk = np.zeros((steps, 14), dtype=np.float64)
    for offset in (0, 7):
        # Joint translation sums both 0.10 m/s arm increments and reaches
        # D=0.40 m in 2 s. Each arm rotates at 12 deg/s and reaches its own
        # R=24 deg cap in 2 s.
        chunk[:, offset] = 0.1 * time
        chunk[:, offset + 3] = math.radians(12.0) * time
    return chunk


def test_hybrid_video_cap_holds_each_clock_at_its_execution_fraction():
    time = np.arange(61, dtype=np.float64) / 30.0
    trajectory = np.zeros((61, 14), dtype=np.float64)
    for offset in (0, 7):
        # Translation reaches 30% D in 0.3 s; rotation reaches 30% R in
        # 0.6 s. The displayed prefix must keep translation fixed while the
        # slower rotation clock finishes.
        trajectory[:, offset] = 0.2 * time
        trajectory[:, offset + 3] = math.radians(6.0) * time

    capped = truncate_cartesian_trajectory_by_joint_clocks(
        trajectory,
        max_translation_distance=0.30 * 0.40,
        max_rotation_distance=0.30 * math.radians(24.0),
    )

    translation = sum(
        cumulative_arc_length(capped[:, offset : offset + 3])[-1] for offset in (0, 7)
    )
    rotation = sum(
        cumulative_rotation_length(capped[:, offset + 3 : offset + 6])[-1]
        for offset in (0, 7)
    )
    assert translation == pytest.approx(0.12, abs=1e-8)
    assert rotation == pytest.approx(2 * 0.30 * math.radians(24.0), abs=1e-8)
    np.testing.assert_allclose(
        capped[9:, [0, 7]],
        np.repeat(capped[9:10, [0, 7]], len(capped) - 9, axis=0),
    )


def test_hybrid_source_horizon_waits_for_both_clocks_and_both_arms():
    time = np.arange(121, dtype=np.float64) / 30.0
    pose = np.zeros((121, 7), dtype=np.float64)
    pose[:, 0] = 0.1 * time
    yaw = math.radians(12.0) * time
    pose[:, 3] = np.cos(yaw / 2.0)  # WXYZ quaternion
    pose[:, 6] = np.sin(yaw / 2.0)

    class Reader:
        def read(self, ranges):
            return {key: pose[start:end] for key, (start, end) in ranges.items()}

    dataset = ZarrDataset.__new__(ZarrDataset)
    dataset.total_frames = len(pose)
    dataset.episode_reader = Reader()
    horizon = dataset._resolve_dynamic_horizon(
        0,
        {
            "type": "arc_hybrid",
            "distance": 0.40,
            "rotation_distance": math.radians(24.0),
            "source_buffer_frames": 121,
            "pose_zarr_keys": ["left.cmd_ee_pose", "right.cmd_ee_pose"],
            "require_all_arms": True,
        },
    )
    assert horizon == 61
