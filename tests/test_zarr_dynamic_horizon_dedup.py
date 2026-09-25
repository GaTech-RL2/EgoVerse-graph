from __future__ import annotations

import numpy as np
import pytest

from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def _dataset_with_horizons(specs):
    dataset = ZarrDataset.__new__(ZarrDataset)
    dataset.key_map = {
        f"action_{index}": {"zarr_key": f"pose_{index}", "horizon": spec}
        for index, spec in enumerate(specs)
    }
    return dataset


def test_equivalent_yam_horizons_are_resolved_once_per_sample():
    base = {
        "type": "arc_hybrid",
        "distance": np.float64(0.4),
        "rotation_distance": np.float64(0.42),
        "source_buffer_frames": np.int64(600),
        "pose_zarr_keys": ["left.cmd_ee_pose", "right.cmd_ee_pose"],
        "translation_horizon_mode": "race",
    }
    specs = [
        dict(base),
        {**base, "pose_zarr_keys": tuple(base["pose_zarr_keys"])},
        dict(reversed(list(base.items()))),
        dict(base),
    ]
    dataset = _dataset_with_horizons(specs)
    calls = []

    def resolve(start_idx, spec):
        calls.append((start_idx, spec))
        return 47

    dataset._resolve_dynamic_horizon = resolve

    resolved = dataset._resolve_dynamic_horizons_for_sample(11)

    assert len(calls) == 1
    assert set(resolved.values()) == {47}


def test_different_dynamic_horizons_preserve_conflict_error():
    common = {
        "type": "arc_distance",
        "source_buffer_frames": 600,
        "pose_zarr_keys": ["left.cmd_ee_pose", "right.cmd_ee_pose"],
    }
    dataset = _dataset_with_horizons(
        [{**common, "distance": 0.4}, {**common, "distance": 0.8}]
    )
    dataset._resolve_dynamic_horizon = lambda _idx, spec: (
        40 if spec["distance"] == 0.4 else 80
    )

    with pytest.raises(
        ValueError,
        match=r"multiple dynamic horizon specs resolved to different lengths \(40 vs 80\)",
    ):
        dataset._resolve_dynamic_horizons_for_sample(0)
