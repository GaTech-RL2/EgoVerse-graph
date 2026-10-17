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
    exclude_visualization_mocap,
    _prepare_corpus_episode,
    prepare_corpus,
    compare_rgb,
    export_rgb,
    _export_rgb_episode,
    export_rgb_corpus,
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


@pytest.mark.parametrize("pose_joints", [[], ["slide"]])
def test_cache_rejects_original_observation_substitution_and_tampering(tmp_path, pose_joints):
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
        pose_joints=pose_joints,
        cameras=["view"],
        height=32,
        width=32,
        grid=[8, 8],
    )
    result = prepare(args)
    assert result["all_channels_native_replay_exact"]
    options = {**vars(args), "input_schema": "libero", "asset_remap": [],
               "exclude_visualization_body": []}
    reused = _prepare_corpus_episode(options)
    assert reused["reused"]
    assert reused["all_channels_native_replay_exact"]
    with pytest.raises(ValueError, match="request mismatch"):
        _prepare_corpus_episode({**options, "pose_site": "changed"})
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["original_cache_max_eef_error_m"] > 100
    assert manifest["original_cache_repaired"] is False
    assert manifest["training_ready"] is False
    highres = args.output / "rgb64.h5"
    receipt = export_rgb(Namespace(output=args.output, rgb_cache=highres,
                                   height=64, width=64, rgb_policy="exact"))
    assert receipt["native_render"] and receipt["native_geometry_replay_exact"]
    assert receipt["rgb_cache_sha256"] == digest(highres)
    with h5py.File(highres) as cache:
        assert cache["rgb"].shape == (3, 1, 64, 64, 3)
        np.testing.assert_array_equal(cache["time"][:], states[:, 0])
        assert cache["camera_K"][0, 0, 0] > 0
    reused_rgb = _export_rgb_episode(dict(output=str(args.output), rgb_cache=str(highres),
                                          height=64, width=64, rgb_policy="exact"))
    assert reused_rgb["status"] == "PASS"
    with pytest.raises(ValueError, match="mismatch"):
        _export_rgb_episode(dict(output=str(args.output), rgb_cache=str(highres),
                                  height=128, width=128, rgb_policy="exact"))
    with pytest.raises(FileExistsError):
        export_rgb(Namespace(output=args.output, rgb_cache=highres,
                             height=64, width=64, rgb_policy="exact"))
    with pytest.raises(FileExistsError):
        prepare(args)
    with np.load(args.output / "aligned.npz") as z:
        arrays = dict(z)
    assert arrays["pose"].shape[1] == 7 + len(pose_joints)
    if pose_joints:
        np.testing.assert_array_equal(arrays["pose"][:, 7], states[:, 1])
    arrays["rgb"] = np.roll(arrays["rgb"], 1, axis=0)
    np.savez_compressed(args.output / "aligned.npz", **arrays)
    with pytest.raises(ValueError, match="Cache hash mismatch"):
        validate_cache(args.output)

    # Even a re-hashed accidental row shift is rejected by actual native replay.
    manifest["artifacts"]["aligned.npz"] = digest(args.output / "aligned.npz")
    (args.output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        validate_cache(args.output, replay=True)

    # Failed replay leaves a hash-valid manifest and arrays on disk. A corpus
    # restart must not convert this failed native replay into a successful hit.
    assert validate_cache(args.output)["status"] == "PASS"
    with pytest.raises(AssertionError):
        _prepare_corpus_episode(options)

    # If RGB and geometry both differ, the geometry failure must be exposed
    # first; an RGB-only assertion should imply all non-RGB channels matched.
    arrays["pose"][0, 0] += 0.01
    np.savez_compressed(args.output / "aligned.npz", **arrays)
    manifest["artifacts"]["aligned.npz"] = digest(args.output / "aligned.npz")
    (args.output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="pose"):
        validate_cache(args.output, replay=True)
    with pytest.raises(AssertionError, match="pose"):
        validate_cache(args.output, replay=True, rgb_policy="uint8-roundoff-v1")


def test_rgb_roundoff_policy_is_explicit_bounded_and_never_modifies_pixels():
    original = np.full((100000, 3), 100, dtype=np.uint8)
    repeated = original.copy()
    repeated[0, 0] += 2
    before = repeated.copy()
    with pytest.raises(AssertionError):
        compare_rgb(original, repeated)
    proof = compare_rgb(original, repeated, "uint8-roundoff-v1")
    assert proof["max_abs"] == 2 and not proof["exact"]
    np.testing.assert_array_equal(repeated, before)
    repeated[0, 0] = 103
    with pytest.raises(ValueError, match="exceeds"):
        compare_rgb(original, repeated, "uint8-roundoff-v1")
    with pytest.raises(ValueError, match="exceeds"):
        compare_rgb(original, repeated, "uint8-roundoff-v2")
    repeated[0, 0] = 101
    repeated[1:5, 0] += 1
    with pytest.raises(ValueError, match="exceeds"):
        compare_rgb(original, repeated, "uint8-roundoff-v1")
    accepted = compare_rgb(original, repeated, "uint8-roundoff-v2")
    assert accepted["changed_fraction_limit"] == 0.0001
    assert not accepted["exact"]
    repeated[5:35, 0] += 1
    with pytest.raises(ValueError, match="exceeds"):
        compare_rgb(original, repeated, "uint8-roundoff-v2")
    with pytest.raises(ValueError, match="policy"):
        compare_rgb(original, repeated, "anything")


def test_rgb_corpus_rejects_duplicate_output_before_processing(tmp_path):
    spec = {"common": {"height": 64, "width": 64, "rgb_policy": "exact"}, "episodes": [
        {"embodiment": "one", "episode_id": "first", "output": "source", "rgb_cache": str(tmp_path / "rgb.h5")},
        {"embodiment": "two", "episode_id": "second", "output": "source2", "rgb_cache": str(tmp_path / "rgb.h5")}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Duplicate"):
        export_rgb_corpus(Namespace(spec=path, status=tmp_path / "status.json", workers=2))


def test_reject_unknown_or_duplicate_pose_joint():
    model, states = fixture()
    for names in (["missing"], ["slide", "slide"]):
        with pytest.raises(ValueError, match="joint"):
            build_arrays(model, states, ["view"], "eef", 32, 32, [8, 8], names)


def test_corpus_rejects_duplicate_source_and_unsafe_identity(tmp_path):
    spec = {"common": {}, "embodiments": {"gripper": {}}, "episodes": [
        {"embodiment": "gripper", "episode_id": "one", "episode": "demo_0", "demo": str(tmp_path / "source.h5")},
        {"embodiment": "gripper", "episode_id": "two", "episode": "demo_0", "demo": str(tmp_path / "source.h5")}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    args = Namespace(spec=path, output=tmp_path / "output", workers=1)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_corpus(args)
    spec["episodes"] = [spec["episodes"][0]]
    spec["episodes"][0]["episode_id"] = "../outside"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_corpus(args)


def test_quest_recording_and_explicit_nonphysical_marker_exclusion(tmp_path):
    _, states = fixture()
    marker = '<body name="marker" mocap="true"><geom name="dot" type="sphere" size=".02" contype="0" conaffinity="0"/></body>'
    xml = XML.replace('</worldbody>', marker + '</worldbody>')
    source = tmp_path / 'quest.hdf5'
    with h5py.File(source, 'w') as f:
        f['state'] = states
        f['destination_model_xml'] = xml
        f['numeric_observations/robot0_eef_pos'] = np.zeros((3, 3))
    args = Namespace(demo=str(source), episode='root', input_schema='quest',
                     robot_assets=tmp_path, scene_assets=tmp_path, output=tmp_path/'quest',
                     pose_site='eef', cameras=['view'], height=32, width=32, grid=[8, 8],
                     exclude_visualization_body=['marker'], asset_remap=[])
    assert prepare(args)['all_channels_native_replay_exact']
    manifest = json.loads((args.output/'manifest.json').read_text())
    assert manifest['excluded_visualization_bodies'] == ['marker']
    assert manifest['input_schema'] == 'quest'
    assert mujoco.MjModel.from_xml_string(exclude_visualization_mocap(xml, ['marker'])).nmocap == 0
    with pytest.raises(ValueError, match='physical'):
        exclude_visualization_mocap(xml.replace('contype="0"', 'contype="1"'), ['marker'])
    with pytest.raises(ValueError, match='referenced'):
        exclude_visualization_mocap(xml.replace('</mujoco>', '<sensor><framepos objtype="geom" objname="dot"/></sensor></mujoco>'), ['marker'])


def test_explicit_asset_prefix_remap_rejects_ambiguity(tmp_path, monkeypatch):
    from egomimic.benchmarks.libero.replay import rebase_demo_xml
    (tmp_path/'hand.stl').write_bytes(b'fixture')
    xml = '<mujoco><asset><mesh file="/recorded/hand/hand.stl"/></asset></mujoco>'
    result = rebase_demo_xml(xml, tmp_path, tmp_path, [('/recorded/hand', tmp_path)])
    assert str(tmp_path/'hand.stl') in result
    monkeypatch.chdir(tmp_path.parent)
    relative = rebase_demo_xml(xml, tmp_path, tmp_path,
                               [('/recorded/hand', tmp_path.name)])
    assert relative == result
    with pytest.raises(ValueError, match='Ambiguous'):
        rebase_demo_xml(xml, tmp_path, tmp_path,
                        [('/recorded/hand', tmp_path), ('/recorded', tmp_path)])
