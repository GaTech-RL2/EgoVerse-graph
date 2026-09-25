"""Numeric contracts for the public bimanual translation modes and hybrid R."""

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian


def source_chunk():
    time = np.arange(241, dtype=np.float64) / 30.0
    raw = np.zeros((len(time), 14))
    raw[:, 0] = 0.2 * time
    raw[:, 7] = 0.1 * time
    raw[:, [3, 10]] = 0.05 * time[:, None]
    raw[:, [6, 13]] = 0.1 * time[:, None]
    return raw


def codec(mode, **overrides):
    options = dict(
        min_distance_unit=0.4, rotation_distance_unit=0.6,
        resampled_vector_length=31, velocity_mode="per_waypoint",
        arc_chunking_mode=mode,
    )
    options.update(overrides)
    return TokenizeBimanualArcLengthCartesian(**options)


@pytest.mark.parametrize("mode,end", [
    ("race", (0.4, 0.2)), ("multistream", (0.4, 0.4)),
    ("joint_distance", (0.4 * 2 / 3, 0.4 / 3)),
])
def test_translation_mode_caps_and_independent_rotation_target(mode, end):
    tokenizer = codec(mode)
    token = tokenizer.transform({"actions_cartesian": source_chunk()})["actions_cartesian"]
    assert token.shape == (62, 14)
    np.testing.assert_allclose(token[30, [0, 7]], end, atol=1e-10)
    # R=0.6 needs SIX seconds, after every translation mode finishes.
    # Truncating rotation at a translation D crossing must fail this test.
    np.testing.assert_allclose(token[30, [3, 10]], [0.3, 0.3], atol=1e-10)
    decoded = tokenizer.detokenize(token, 211)
    np.testing.assert_allclose(decoded[30, [0, 7]], [0.2, 0.1], atol=1e-9)
    np.testing.assert_allclose(decoded[180, [3, 10]], [0.3, 0.3], atol=1e-9)
    np.testing.assert_allclose(decoded[210, [0, 7]], end, atol=1e-9)


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
def test_rotation_finishes_early_and_holds_independent_of_translation(mode):
    tokenizer = codec(mode, rotation_distance_unit=0.05)
    token = tokenizer.transform({"actions_cartesian": source_chunk()})["actions_cartesian"]
    decoded = tokenizer.detokenize(token, 121)
    np.testing.assert_allclose(decoded[15:, 3], 0.025, atol=1e-9)
    np.testing.assert_allclose(decoded[15:, 10], 0.025, atol=1e-9)
    assert decoded[30, 0] > decoded[15, 0]


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
def test_stationary_arm_holds_while_rotation_uses_its_own_clock(mode):
    raw = source_chunk()
    raw[:, 7:10] = [0.2, 0.3, 0.4]
    tokenizer = codec(mode)
    token = tokenizer.transform({"actions_cartesian": raw})["actions_cartesian"]
    np.testing.assert_allclose(token[:31, 7:10], np.tile([0.2, 0.3, 0.4], (31, 1)))
    np.testing.assert_allclose(token[31:, 7:10], 0.0)
    decoded = tokenizer.detokenize(token, 211)
    assert np.isfinite(decoded).all()
    np.testing.assert_allclose(decoded[:, 7:10], np.tile([0.2, 0.3, 0.4], (211, 1)))
    np.testing.assert_allclose(decoded[180, [3, 10]], [0.3, 0.3], atol=1e-9)


def test_existing_hybrid_default_is_joint_distance_bitwise():
    options = dict(min_distance_unit=0.4, rotation_distance_unit=0.6,
                   resampled_vector_length=31, velocity_mode="per_waypoint")
    implicit = TokenizeBimanualArcLengthCartesian(**options)
    explicit = codec("joint_distance")
    old_token = implicit.transform({"actions_cartesian": source_chunk()})["actions_cartesian"]
    new_token = explicit.transform({"actions_cartesian": source_chunk()})["actions_cartesian"]
    np.testing.assert_array_equal(old_token, new_token)
    np.testing.assert_array_equal(implicit.detokenize(old_token, 211), explicit.detokenize(new_token, 211))


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="arc_chunking_mode"):
        codec("not_a_mode")


def test_race_uses_first_crossing_time_but_independent_arc_coordinates():
    raw = np.zeros((5, 14))
    raw[:, 0] = [0.0, 0.8, 0.8, 1.6, 1.6]
    raw[:, 7] = [0.0, 0.1, 0.6, 0.7, 1.2]
    raw[:, [3, 10]] = np.arange(5)[:, None] * 0.1
    tokenizer = codec("race", min_distance_unit=1.0,
                      resampled_vector_length=5, dt=1.0)
    token = tokenizer.transform({"actions_cartesian": raw})["actions_cartesian"]
    # Left reaches 1 at source frame 2.25; right is then at 0.625.
    np.testing.assert_allclose(token[:5, 0], [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(token[:5, 7], [0, 0.15625, 0.3125, 0.46875, 0.625])
    # Rotation reaches its independent cap at source frame 3, not 2.25.
    np.testing.assert_allclose(token[4, [3, 10]], [0.3, 0.3])


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
def test_new_shared_translation_modes_require_hybrid_rotation(mode):
    # Otherwise these names would silently use the legacy per-arm codec path.
    with pytest.raises(ValueError, match="rotation_distance_unit"):
        codec(mode, rotation_distance_unit=None)


def test_race_exact_distance_plateau_stops_at_first_crossing():
    raw = np.zeros((4, 14))
    raw[:, 0] = [0, 1, 1, 1]
    raw[:, 7] = [0, 0.25, 0.5, 0.75]
    raw[:, [6, 13]] = np.arange(4)[:, None]
    tokenizer = codec("race", min_distance_unit=1.0)
    token = tokenizer.transform({"actions_cartesian": raw})["actions_cartesian"]
    np.testing.assert_allclose(token[30, [0, 7]], [1, 0.25])
    np.testing.assert_allclose(token[30, [6, 13]], [1, 1])


@pytest.mark.parametrize("mode", ["race", "multistream"])
def test_short_translation_finishes_then_holds_without_late_gripper_leak(mode):
    raw = np.zeros((5, 14))
    raw[:, 0] = [0, 0.25, 0.5, 0.5, 0.5]
    raw[:, 7] = [0, 0.25, 0.5, 0.75, 1]
    raw[:, [6, 13]] = np.arange(5)[:, None]
    tokenizer = codec(mode, min_distance_unit=1.0, resampled_vector_length=3, dt=1.0)
    token = tokenizer.transform({"actions_cartesian": raw})["actions_cartesian"]
    decoded = tokenizer.detokenize(token, 5)
    np.testing.assert_allclose(decoded[:, 0], [0, 0.25, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(decoded[:, 6], [0, 1, 2, 2, 2])


def test_multistream_short_arm_resamples_all_100_waypoints_then_holds_on_decode():
    """D40/actual10 uses 100 motion waypoints, NOT 25 motion + 75 padding."""
    time = np.arange(600, dtype=np.float64) / 30.0
    raw = np.zeros((len(time), 14))
    raw[:, 0] = np.minimum(0.1 * time, 0.1)
    raw[:, 7] = np.minimum(0.1 * time, 0.4)
    raw[:, [3, 10]] = 0.01 * time[:, None]
    tokenizer = codec("multistream", resampled_vector_length=100,
                      rotation_distance_unit=0.2)
    token = tokenizer.transform({"actions_cartesian": raw})["actions_cartesian"]
    assert token.shape == (200, 14)
    np.testing.assert_allclose(token[:100, 0], np.linspace(0.0, 0.1, 100))
    assert np.all(np.diff(token[:100, 0]) > 0.0)
    np.testing.assert_allclose(token[:100, 7], np.linspace(0.0, 0.4, 100))
    decoded = tokenizer.detokenize(token, 600)
    np.testing.assert_allclose(decoded[:, 0], np.minimum(0.1 * time, 0.1), atol=1e-9)
    np.testing.assert_allclose(decoded[:, 7], np.minimum(0.1 * time, 0.4), atol=1e-9)
    np.testing.assert_allclose(decoded[300:, 3], 0.1, atol=1e-9)
    assert decoded[120, 3] < decoded[300, 3]
