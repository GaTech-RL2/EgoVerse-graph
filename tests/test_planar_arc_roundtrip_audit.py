"""The round-trip audit must exercise native grabber grip as well as pose."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

_PATH = Path(__file__).parents[1] / "scripts/eval/audit_planar_arc_roundtrip.py"
_SPEC = importlib.util.spec_from_file_location("planar_roundtrip_audit", _PATH)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


@pytest.mark.parametrize("allocation", ["uniform", "curvature"])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_roundtrip_audit_infers_native_layout_and_reports_allocation(
    tmp_path, allocation, width
):
    root = tmp_path / "data"
    group = zarr.open_group(str(root / "episode_0.zarr"), mode="w")
    actions = np.zeros((61, width), dtype=np.float32)
    actions[:, :2] = [100.0, 100.0]
    actions[:, -1] += np.linspace(0.0, 0.4, 61)
    group.create_array("actions", data=actions)
    group.attrs["observation_alignment"] = "pre_step"
    domain = "test_native_layout"
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "domains": {
                    domain: {
                        "valid_ids": ["episode_0"],
                        "train_ids": ["episode_1"],
                        "valid_count": 1,
                        "valid_names_sha256": "fixture",
                    }
                }
            }
        )
    )
    output = tmp_path / "audit.json"
    args = [
        "--dataset-root",
        str(root),
        "--split-manifest",
        str(split),
        "--output",
        str(output),
        "--waypoint-sampling",
        allocation,
    ]
    audit.main(args)
    report = json.loads(output.read_text())
    assert report["dataset"]["domain"] == domain
    assert report["contract"]["native_action_dim"] == width
    assert report["contract"]["waypoint_sampling"] == allocation
    assert report["summary"]["anchor_error_max_abs"] < 1e-5
    assert report["episodes"][0]["decoded_shape"] == [40, width]
    assert report["episodes"][0]["stored_observation_alignment"] == "pre_step"
    if width == 4:
        assert report["summary"]["grip_rmse_mean"] < 1e-6
    # Evidence is append-only: accidentally reusing an output must fail.
    with pytest.raises(FileExistsError):
        audit.main(args)
