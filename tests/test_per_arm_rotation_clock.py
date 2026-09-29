"""The rotation clock is per-arm, and race is a four-way stopping time.

Two invariants:

1. Rotation is per-arm in every chunking mode. Each arm spends R on its own
   SO(3) travel. No mode sums left and right rotation into one clock.
2. ``race`` is a stopping time, not a budget, so the chunk has an end frame. All
   four streams race for it -- each arm's translation for D and each arm's
   rotation for R -- and each is interpolated on its own cumulative length
   through that frame.

Before the fix, rotation was a single summed left-plus-right clock resampled
over the whole raw window. Under ``race`` that meant a waypoint's orientation
could come from frames after the chunk had already ended. Round-trip and golden
hash tests cannot catch it: detokenize mirrored the same shared clock, so the
token round-tripped a value that was wrong at construction.
"""

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    cumulative_rotation_length,
)

MODES = ("race", "multistream", "joint_distance")
BUDGET_MODES = ("multistream", "joint_distance")

T = 100
D = 1.0
ROTATION = 0.5  # radians


def window(
    translation_per_frame: float,
    left_rotation_per_frame: float,
    right_rotation_per_frame: float = 0.0,
) -> np.ndarray:
    """One synthetic (T, 14) bimanual cartesian window.

    Both arms translate along +x at the same speed. Each arm rotates about a
    single euler axis, so its geodesic rotation length over a frame is exactly
    its ``*_rotation_per_frame`` under any euler convention.
    """
    raw = np.zeros((T, 14), dtype=np.float64)
    frames = np.arange(T, dtype=np.float64)
    raw[:, 0] = frames * translation_per_frame  # left x
    raw[:, 7] = frames * translation_per_frame  # right x
    raw[:, 3] = frames * left_rotation_per_frame  # left first euler angle
    raw[:, 10] = frames * right_rotation_per_frame  # right first euler angle
    return raw


def codec(mode: str) -> TokenizeBimanualArcLengthCartesian:
    return TokenizeBimanualArcLengthCartesian(
        min_distance_unit=D,
        rotation_distance_unit=ROTATION,
        resampled_vector_length=10,
        velocity_mode="per_waypoint",
        arc_chunking_mode=mode,
    )


def ends(mode: str, raw: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-arm ``(translation, rotation)`` source distance the token covers.

    A translation stream is ``(cumulative, targets, hold)``; a rotation stream is
    ``(cumulative, targets)``, because rotation never holds.
    """
    translation, rotation = codec(mode)._hybrid_source_coordinates(raw)
    return (
        [float(stream[1][-1]) for stream in translation],
        [float(stream[1][-1]) for stream in rotation],
    )


def test_race_ends_when_a_rotation_spends_r_first():
    # Neither arm travels D inside the window; the left arm spends R at frame
    # 50. The chunk ends there, so translation stops well short of D.
    raw = window(translation_per_frame=0.001, left_rotation_per_frame=0.01)
    assert cumulative_rotation_length(raw[:, 3:6])[-1] > ROTATION
    translation_ends, rotation_ends = ends("race", raw)
    assert np.allclose(translation_ends, 0.001 * 50.0)
    assert all(end < D for end in translation_ends)
    assert rotation_ends[0] == pytest.approx(ROTATION)


def test_race_rotation_budget_is_per_arm_not_summed():
    # Each arm turns 0.006 rad/frame, so each spends R at frame 83.33. A summed
    # clock would cross R at frame 41.67 and cut every chunk in half.
    raw = window(
        translation_per_frame=0.001,
        left_rotation_per_frame=0.006,
        right_rotation_per_frame=0.006,
    )
    translation_ends, rotation_ends = ends("race", raw)
    assert np.allclose(translation_ends, 0.001 * (ROTATION / 0.006))
    assert np.allclose(rotation_ends, ROTATION)


def test_race_slower_arm_keeps_only_the_rotation_it_travelled():
    # Right spends R at frame 50; left would need frame 250. The chunk ends at
    # 50 and the left arm keeps 0.002 * 50 rad.
    raw = window(
        translation_per_frame=0.001,
        left_rotation_per_frame=0.002,
        right_rotation_per_frame=0.01,
    )
    translation_ends, rotation_ends = ends("race", raw)
    assert np.allclose(translation_ends, 0.001 * 50.0)
    assert rotation_ends[0] == pytest.approx(0.002 * 50.0)
    assert rotation_ends[1] == pytest.approx(ROTATION)


def test_race_translation_win_truncates_rotation():
    # Both arms cross D at frame 20, before either rotation spends R at frame
    # 50. Rotation must be cut at frame 20 rather than run to R. This is the
    # assertion the pre-fix tokenizer fails: it returned R.
    raw = window(
        translation_per_frame=D / 20.0,
        left_rotation_per_frame=0.01,
        right_rotation_per_frame=0.01,
    )
    translation_ends, rotation_ends = ends("race", raw)
    assert np.allclose(translation_ends, D)
    assert all(end < ROTATION for end in rotation_ends)
    assert np.allclose(rotation_ends, 0.01 * 20.0)


def test_race_with_no_budget_spent_uses_the_whole_window():
    raw = window(
        translation_per_frame=0.001,
        left_rotation_per_frame=0.001,
        right_rotation_per_frame=0.001,
    )
    translation_ends, rotation_ends = ends("race", raw)
    assert np.allclose(translation_ends, 0.001 * (T - 1))
    assert np.allclose(rotation_ends, 0.001 * (T - 1))


@pytest.mark.parametrize("mode", BUDGET_MODES)
def test_budget_modes_spend_both_budgets_untruncated(mode):
    # multistream and joint_distance define no end frame, so each stream spends
    # its own budget over the whole window.
    raw = window(
        translation_per_frame=D / 20.0,
        left_rotation_per_frame=0.01,
        right_rotation_per_frame=0.01,
    )
    translation_ends, rotation_ends = ends(mode, raw)
    assert np.allclose(translation_ends, D)
    assert np.allclose(rotation_ends, ROTATION)


@pytest.mark.parametrize("mode", BUDGET_MODES)
def test_budget_modes_rotation_is_per_arm_not_summed(mode):
    # Each arm turns 0.3 rad over the window, under R. A summed clock reaches
    # R=0.5 at 0.25 rad apiece and would report 0.5 for both arms.
    raw = window(
        translation_per_frame=D / 20.0,
        left_rotation_per_frame=0.3 / (T - 1),
        right_rotation_per_frame=0.3 / (T - 1),
    )
    _, rotation_ends = ends(mode, raw)
    assert np.allclose(rotation_ends, 0.3)


@pytest.mark.parametrize("mode", MODES)
def test_still_arm_never_inherits_the_other_arms_rotation(mode):
    raw = window(
        translation_per_frame=0.001,
        left_rotation_per_frame=0.01,
        right_rotation_per_frame=0.0,
    )
    _, rotation_ends = ends(mode, raw)
    assert rotation_ends[0] > 0.0
    assert rotation_ends[1] == pytest.approx(0.0)


@pytest.mark.parametrize("mode", MODES)
def test_round_trip_shapes_and_finiteness(mode):
    raw = window(
        translation_per_frame=D / 20.0,
        left_rotation_per_frame=0.002,
        right_rotation_per_frame=0.01,
    )
    batch = codec(mode).transform({"actions_cartesian": raw.copy()})
    assert batch["actions_cartesian"].shape == (20, 14)
    chunk = codec(mode).detokenize(batch["actions_cartesian"], 30)
    assert chunk.shape == (30, 14)
    assert np.isfinite(chunk).all()


def test_no_summed_rotation_clock_remains():
    """No API can produce a joint left-plus-right rotation clock."""
    import egomimic.rldb.zarr.arc_length_tokenizer as module

    assert not hasattr(module, "cumulative_bimanual_rotation_length")
    with pytest.raises(TypeError):
        codec("race")._hybrid_clock_durations(
            np.zeros((10, 14)), np.zeros((10, 14)), rotation=True, action_horizon=1
        )


@pytest.mark.parametrize("mode", MODES)
def test_rotation_streams_carry_no_hold_index(mode):
    """A hold is a translation-distance rule, so rotation never carries one."""
    raw = window(
        translation_per_frame=0.0,
        left_rotation_per_frame=0.01,
        right_rotation_per_frame=0.002,
    )
    translation, rotation = codec(mode)._hybrid_source_coordinates(raw)
    assert all(len(stream) == 3 for stream in translation)
    assert all(len(stream) == 2 for stream in rotation)
    # Neither arm translates, so both translation streams hold, and both arms
    # still keep the rotation they actually performed.
    assert all(stream[2] is not None for stream in translation)
    _, rotation_ends = ends(mode, raw)
    assert rotation_ends[0] > rotation_ends[1] > 0.0
