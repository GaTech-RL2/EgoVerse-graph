import math
from pathlib import Path

import numpy as np
import pytest

from egomimic.eval.open_loop_sim import (
    arc_prefix_control_steps,
    truncate_cartesian_trajectory_by_joint_clocks,
)
from egomimic.rldb.embodiment.yam import Yam
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    cumulative_arc_length,
    cumulative_rotation_length,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def _two_clock_chunk(steps: int = 121) -> np.ndarray:
    """Translation reaches D in 2 s; rotation reaches R in 1 s."""
    time = np.arange(steps, dtype=np.float64) / 30.0
    chunk = np.zeros((steps, 14), dtype=np.float64)
    for offset in (0, 7):
        # Each joint clock is the sum of both arm increments: 0.10 + 0.10
        # m/s reaches D=0.40 m in 2 s, while 12 + 12 deg/s reaches R=24 deg
        # in 1 s.
        chunk[:, offset] = 0.1 * time
        chunk[:, offset + 3] = math.radians(12.0) * time
    return chunk


def _hybrid_tokenizer(**overrides) -> TokenizeBimanualArcLengthCartesian:
    kwargs = {
        "action_key": "actions",
        "output_action_key": "actions",
        "min_distance_unit": 0.40,
        "rotation_distance_unit": math.radians(24.0),
        "resampled_vector_length": 25,
        "velocity_mode": "per_waypoint",
        "dt": 1.0 / 30.0,
    }
    kwargs.update(overrides)
    return TokenizeBimanualArcLengthCartesian(**kwargs)


def test_hybrid_token_has_independently_capped_translation_and_rotation_paths():
    token = _hybrid_tokenizer().transform({"actions": _two_clock_chunk()})["actions"]
    waypoints = token[:25]
    assert token.shape == (50, 14)
    translations = [
        cumulative_arc_length(waypoints[:, offset : offset + 3])[-1]
        for offset in (0, 7)
    ]
    rotations = [
        cumulative_rotation_length(waypoints[:, offset + 3 : offset + 6])[-1]
        for offset in (0, 7)
    ]
    assert sum(translations) == pytest.approx(0.40, abs=1e-8)
    assert sum(rotations) == pytest.approx(math.radians(24.0), abs=1e-8)


def test_hybrid_detokenize_runs_rotation_and_translation_on_separate_clocks():
    tokenizer = _hybrid_tokenizer()
    token = tokenizer.transform({"actions": _two_clock_chunk()})["actions"]
    decoded = tokenizer.detokenize(token, action_horizon=61)

    # At one second the rotation cap is already complete while translation is
    # halfway to its own cap. Translation continues for another second.
    for offset in (0, 7):
        assert decoded[30, offset] == pytest.approx(0.10, abs=2e-3)
        assert decoded[30, offset + 3] == pytest.approx(math.radians(12.0), abs=2e-3)
        assert decoded[60, offset] == pytest.approx(0.20, abs=2e-3)
        assert decoded[60, offset + 3] == pytest.approx(math.radians(12.0), abs=2e-3)


def test_open_loop_replan_boundary_waits_for_slower_rotation_clock():
    raw = _two_clock_chunk()
    raw[:, [0, 7]] *= 2.0  # D in 1 s; R still takes 1 s in the base fixture.
    raw[:, [3, 10]] *= 0.5  # R now takes 2 s.
    tokenizer = _hybrid_tokenizer()
    token = tokenizer.transform({"actions": raw})["actions"]
    steps = arc_prefix_control_steps(
        token,
        1.0,
        "per_waypoint",
        1.0 / 30.0,
        0.40,
        rotation_distance_unit=math.radians(24.0),
    )
    assert steps == 60


def test_hybrid_rotation_clock_advances_during_in_place_rotation():
    raw = _two_clock_chunk()
    raw[:, [0, 7]] = 0.0
    tokenizer = _hybrid_tokenizer()
    token = tokenizer.transform({"actions": raw})["actions"]
    decoded = tokenizer.detokenize(token, action_horizon=31)
    np.testing.assert_allclose(decoded[:, [0, 1, 2, 7, 8, 9]], 0.0, atol=1e-9)
    assert decoded[-1, 3] == pytest.approx(math.radians(12.0), abs=2e-3)
    assert decoded[-1, 10] == pytest.approx(math.radians(12.0), abs=2e-3)


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
    assert rotation == pytest.approx(0.30 * math.radians(24.0), abs=1e-8)
    np.testing.assert_allclose(
        capped[9:, [0, 7]],
        np.repeat(capped[9:10, [0, 7]], len(capped) - 9, axis=0),
    )


def test_hybrid_replan_timing_does_not_ignore_a_moving_arm_with_zero_rate():
    tokenizer = _hybrid_tokenizer()
    token = tokenizer.transform({"actions": _two_clock_chunk()})["actions"]
    token[25:, 0:3] = 0.0
    steps = arc_prefix_control_steps(
        token,
        1.0,
        "per_waypoint",
        1.0 / 30.0,
        0.40,
        rotation_distance_unit=math.radians(24.0),
        max_steps=77,
    )
    assert steps == 77


def test_hybrid_mode_requires_per_waypoint_velocity():
    with pytest.raises(ValueError, match="velocity_mode='per_waypoint'"):
        _hybrid_tokenizer(velocity_mode="mean")


def test_yam_hybrid_keymap_requests_both_distance_caps():
    keymap = Yam.get_keymap(keymap_mode="hybrid_arc_tokenizer_cartesian")
    specs = [
        spec["horizon"]
        for spec in keymap.values()
        if spec.get("key_type") == "action_keys"
    ]
    assert specs
    assert {spec["type"] for spec in specs} == {"arc_hybrid"}
    assert {spec["distance"] for spec in specs} == {0.40}
    assert all(
        spec["rotation_distance"] == pytest.approx(math.radians(24.0)) for spec in specs
    )


def test_yam_hybrid_transform_wires_rotation_cap_to_tokenizer():
    transforms = Yam.get_transform_list(
        action_mode="hybrid_arc_tokenizer_cartesian",
        coord_frame="eef_frame",
        rotation_mode="euler",
        min_distance_unit=0.40,
        rotation_distance_unit=math.radians(24.0),
        resampled_vector_length=100,
        velocity_mode="per_waypoint",
    )
    tokenizer = next(
        transform
        for transform in transforms
        if isinstance(transform, TokenizeBimanualArcLengthCartesian)
    )
    assert tokenizer.rotation_distance_unit == pytest.approx(math.radians(24.0))


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


def test_hybrid_hpt_experiment_wires_one_rotation_cap_everywhere():
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=abc_arc/robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop"
            ],
        )
    expected = math.radians(24.0)
    assert cfg.abc.arc_rotation_distance == pytest.approx(expected)
    assert cfg.evaluator.rotation_distance_unit == pytest.approx(expected)
    assert (
        cfg.data.train_datasets.yam_bimanual.resolver.transform_list.rotation_distance_unit
        == pytest.approx(expected)
    )
    assert cfg.abc.arc_velocity_mode == "per_waypoint"
    assert cfg.hpt.action_horizon == 200
