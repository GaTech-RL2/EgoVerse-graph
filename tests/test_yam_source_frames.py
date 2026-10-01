"""The YAM cartesian source window is 100 frames unless yam_source_frames overrides it."""

import numpy as np
import pytest

from egomimic.rldb.embodiment.bimanual_arc import get_keymap, get_transform_list


def _horizons(key_map):
    return {s["horizon"] for s in key_map.values() if isinstance(s.get("horizon"), int)}


def test_default_yam_window_ignores_horizon():
    assert _horizons(
        get_keymap(horizon=200, embodiment="yam", drop_wrist_images=False)
    ) == {100}


def test_yam_source_frames_overrides_window():
    key_map = get_keymap(
        horizon=200, embodiment="yam", drop_wrist_images=False, yam_source_frames=400
    )
    assert _horizons(key_map) == {400}


def test_human_keeps_caller_horizon():
    assert _horizons(get_keymap(horizon=45, embodiment="human")) == {45}


def fixed_index_yam_sample(window, index, variant):
    """Slice deterministic raw poses using the keymap, then run real transforms."""
    keymap = get_keymap(horizon=200, embodiment="yam", yam_source_frames=window)
    horizon = keymap["left.cmd_ee_pose"]["horizon"]
    data = {}
    for side, step in (("left", 0.003), ("right", 0.001)):
        poses = np.zeros((700, 7))
        poses[:, 0] = np.arange(700) * step
        poses[:, 3] = 1  # WXYZ identity
        data[f"{side}.obs_ee_pose"] = poses[index].copy()
        data[f"{side}.cmd_ee_pose"] = poses[index : index + horizon].copy()
        data[f"{side}.obs_gripper"] = np.array([0.25])
        data[f"{side}.cmd_gripper"] = np.full((horizon, 1), 0.25)
    for transform in get_transform_list(variant, chunk_length=200, embodiment="yam"):
        data = transform.transform(data)
    return np.asarray(data["actions_cartesian"]), np.asarray(data["actions_time"])


@pytest.mark.parametrize("window", [100, 400])
@pytest.mark.parametrize("index", [17, 203])
@pytest.mark.parametrize("variant", ["arcdur", "arcvel"])
def test_fixed_index_windows_preserve_expected_geometry(window, index, variant):
    token, time = fixed_index_yam_sample(window, index, variant)
    assert token.shape == (100, 16)
    expected = [min(0.4, (window - 1) * 0.003), min(0.4, (window - 1) * 0.001)]
    # E1 retains the legacy 200-step zero-translation rule. The slower arm
    # needs 399 steps in w400, so it holds its anchor while the faster arm
    # reaches D in 133 steps. This differs deliberately from hybrid ARC holds.
    if window == 400:
        expected[1] = 0.0
    np.testing.assert_allclose(token[-1, [0, 7]], expected, atol=1e-6)
    np.testing.assert_allclose(token[0, [0, 7]], 0, atol=1e-7)
    np.testing.assert_allclose(time[:, 0], np.arange(100) * 0.003, atol=1e-6)
    np.testing.assert_allclose(time[:, 7], np.arange(100) * 0.001, atol=1e-6)


def test_time_baseline_is_exactly_the_first_100_native_samples():
    actions, time = fixed_index_yam_sample(100, 17, "time")
    np.testing.assert_array_equal(actions, time)
    np.testing.assert_allclose(actions[:, 0], np.arange(100) * 0.003, atol=1e-6)
