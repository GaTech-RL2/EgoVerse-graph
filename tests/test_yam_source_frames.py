"""The YAM cartesian source window is 100 frames unless yam_source_frames overrides it."""

from egomimic.rldb.embodiment.bimanual_arc import get_keymap


def _horizons(key_map):
    return {s["horizon"] for s in key_map.values() if isinstance(s.get("horizon"), int)}


def test_default_yam_window_ignores_horizon():
    assert _horizons(get_keymap(horizon=200, embodiment="yam", drop_wrist_images=False)) == {100}


def test_yam_source_frames_overrides_window():
    key_map = get_keymap(
        horizon=200, embodiment="yam", drop_wrist_images=False, yam_source_frames=400
    )
    assert _horizons(key_map) == {400}


def test_human_keeps_caller_horizon():
    assert _horizons(get_keymap(horizon=45, embodiment="human")) == {45}
