"""Native geometry tests, independent of original dataset availability."""

import copy
import json
from argparse import Namespace

import h5py
import mujoco
import numpy as np
import pytest

from egomimic.benchmarks.libero.state_tracks import (
    build_arrays,
    digest,
    prepare,
    restore_state,
    validate_arrays,
    validate_cache,
)

XML = """<mujoco><worldbody>
<camera name="view" pos="0 0 3"/><camera name="second" pos=".1 0 3"/>
<geom type="plane" size="3 3 .1" group="1"/>
<body pos="0 0 .2"><joint name="slide" type="slide" axis="1 0 0"/>
<geom type="sphere" size=".2" mass="1" group="1" rgba="1 0 0 1"/>
<site name="eef"/></body></worldbody></mujoco>"""


def fixture():
    model = mujoco.MjModel.from_xml_string(XML)
    states = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [0.2, 0.4, 0.0]])
    return model, states


def test_actual_state_alignment_and_persistent_motion():
    model, states = fixture()
    arrays = build_arrays(model, states, ["view", "second"], "eef", 32, 32, [16, 16])
    assert arrays["rgb"].shape == (3, 2, 32, 32, 3)
    np.testing.assert_allclose(arrays["pose"][:, 0], states[:, 1])
    moving = arrays["point_body_id"] != 0
    assert moving.any()
    np.testing.assert_allclose(
        arrays["xyz"][2, moving, 0] - arrays["xyz"][0, moving, 0], 0.4
    )
    damaged = copy.deepcopy(arrays)
    damaged["source_row"] = np.roll(damaged["source_row"], 1)
    with pytest.raises(ValueError, match="row"):
        validate_arrays(damaged)
    damaged = copy.deepcopy(arrays)
    damaged["xyz"][1] = arrays["xyz"][0]
    with pytest.raises(AssertionError):
        validate_arrays(damaged)
    arrays["visibility"][:] = False
    validate_arrays(arrays)  # occlusion must not erase exact coordinates
    arrays["coordinate_available"][0, 0] = False
    with pytest.raises(ValueError, match="survive occlusion"):
        validate_arrays(arrays)


def test_reject_missing_state_and_time_contracts():
    model, states = fixture()
    with pytest.raises(ValueError, match="strictly increasing"):
        build_arrays(model, states[[1, 0, 2]], ["view"], "eef", 32, 32, [8, 8])
    with pytest.raises(ValueError, match="Invalid recorded state"):
        restore_state(model, mujoco.MjData(model), np.zeros(2))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body mocap="true"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="mocap"):
        restore_state(model, mujoco.MjData(model), np.zeros(1))


def test_cache_rejects_original_observation_substitution_and_tampering(tmp_path):
    _, states = fixture()
    original = tmp_path / "demo.hdf5"
    with h5py.File(original, "w") as f:
        d = f.create_group("data/demo_0")
        d.attrs["model_file"] = XML
        d["states"] = states
        d["obs/ee_pos"] = np.full((3, 3), 1000.0)  # deliberately wrong originals
        d["obs/agentview_rgb"] = np.full((3, 32, 32, 3), 255, np.uint8)
    args = Namespace(
        demo=str(original),
        episode="demo_0",
        robot_assets=tmp_path,
        scene_assets=tmp_path,
        output=tmp_path / "aligned",
        pose_site="eef",
        cameras=["view"],
        height=32,
        width=32,
        grid=[8, 8],
    )
    result = prepare(args)
    assert result["all_channels_native_replay_exact"]
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["original_cache_max_eef_error_m"] > 100
    assert manifest["original_cache_repaired"] is False
    assert manifest["training_ready"] is False
    with pytest.raises(FileExistsError):
        prepare(args)
    with np.load(args.output / "aligned.npz") as z:
        arrays = dict(z)
    arrays["rgb"] = np.roll(arrays["rgb"], 1, axis=0)
    np.savez_compressed(args.output / "aligned.npz", **arrays)
    with pytest.raises(ValueError, match="Cache hash mismatch"):
        validate_cache(args.output)

    # Even a re-hashed accidental row shift is rejected by actual native replay.
    manifest["artifacts"]["aligned.npz"] = digest(args.output / "aligned.npz")
    (args.output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        validate_cache(args.output, replay=True)
