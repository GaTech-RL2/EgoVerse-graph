from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from egomimic.robot.hdf5_tempo import (
    buckets,
    build_report,
    compute_cut_sets,
    discover_hdf5,
    read_episode_tempo,
)


def _write_episode(path: Path, *, speed_mps: float, complete: bool = True) -> None:
    fps = 30.0
    frames = 90
    pose = np.zeros((frames, 14), dtype=np.float32)
    pose[:, 0] = np.arange(frames, dtype=np.float32) * speed_mps / fps
    pose[:, 7] = np.arange(frames, dtype=np.float32) * speed_mps * 0.5 / fps
    with h5py.File(path, "w") as episode:
        episode.attrs["complete"] = complete
        episode.create_dataset("observations/eepose", data=pose)


def test_bucket_boundaries_and_cut_schemes() -> None:
    cuts = compute_cut_sets([0.0, 1.0, 2.0, 3.0])
    assert cuts["equal_width"] == pytest.approx([1.0, 2.0])
    assert cuts["equal_width_robust"] == pytest.approx([1.05, 1.95])
    assert cuts["tertile"] == pytest.approx([1.0, 2.0])
    assert buckets(0.99, [1.0, 2.0]) == "slow"
    assert buckets(1.0, [1.0, 2.0]) == "medium"
    assert buckets(2.0, [1.0, 2.0]) == "fast"


def test_hdf5_tempo_uses_observed_bimanual_pose(tmp_path: Path) -> None:
    path = tmp_path / "demo_0.hdf5"
    _write_episode(path, speed_mps=0.3)
    result = read_episode_tempo(
        path,
        fps=30.0,
        lowpass_hz=3.0,
        hold_threshold=0.05,
    )
    assert result is not None
    assert result.frames == 90
    # sosfiltfilt has a small finite-record edge transient even for a ramp.
    assert result.tempo_mps == pytest.approx(0.3, abs=1e-4)
    assert result.moving_fraction == pytest.approx(1.0)


def test_incomplete_episode_is_skipped_by_default(tmp_path: Path) -> None:
    path = tmp_path / "demo_1.hdf5"
    _write_episode(path, speed_mps=0.2, complete=False)
    assert read_episode_tempo(path, fps=30.0) is None


def test_discovery_and_reference_cuts(tmp_path: Path) -> None:
    first = tmp_path / "demo_0.hdf5"
    second = tmp_path / "demo_1.hdf5"
    _write_episode(first, speed_mps=0.15)
    _write_episode(second, speed_mps=0.45)
    assert discover_hdf5([tmp_path]) == [first, second]

    episodes = [read_episode_tempo(path, fps=30.0) for path in (first, second)]
    report = build_report(
        [episode for episode in episodes if episode is not None],
        cut_sets={
            "equal_width": [0.2, 0.4],
            "equal_width_robust": [0.2, 0.4],
            "tertile": [0.2, 0.4],
        },
        pose_source="observations",
        lowpass_hz=3.0,
        hold_threshold=0.05,
    )
    assert report["cut_source"] == "reference_file"
    assert [item["buckets"]["tertile"] for item in report["episodes"]] == [
        "slow",
        "fast",
    ]
