"""Versioned non-E1 six-stream clocks with delay in row zero."""

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    clock_token_seconds,
)

FIRST_DELAY_LAYOUTS = ("velocity", "duration")


def make_codec(layout, **kwargs):
    options = dict(
        action_key="actions",
        output_action_key="actions",
        min_distance_unit=0.03,
        rotation_distance_unit=0.3,
        gripper_stream_mode="events",
        resampled_vector_length=4,
        dt=0.1,
        clock_mode=layout,
        streams="full_separation",
    )
    options.update(kwargs)
    return TokenizeBimanualArcLengthCartesian(**options)


def delayed_translation(delay=6, moving=3):
    raw = np.zeros((delay + moving + 1, 14))
    raw[:, 0] = 0.2 + np.maximum(np.arange(len(raw)) - delay, 0) * 0.01
    return raw


def encode(codec, raw):
    return codec.transform({"actions": raw.copy()})["actions"]


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_leading_hold_is_not_smeared_into_first_moving_interval(layout):
    raw = delayed_translation()
    codec = make_codec(layout)
    token = encode(codec, raw)
    assert token.shape == (4, 20)
    native = clock_token_seconds(token, layout)
    np.testing.assert_allclose(native[:, 14], [0.6, 0.1, 0.1, 0.1], atol=1e-12)
    # Deliberately equal speed and interval values here (0.01m / 0.1s).
    # The first entry is nevertheless a different quantity: 0.6 seconds.
    decoded = codec.detokenize(token, len(raw))
    np.testing.assert_allclose(decoded, raw, atol=1e-12)
    np.testing.assert_allclose(decoded[:7, 0], 0.2, atol=1e-12)


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_no_delay_has_zero_first_clock_and_incoming_intervals(layout):
    codec = make_codec(layout)
    token = encode(codec, delayed_translation(delay=0))
    native = clock_token_seconds(token, layout)
    np.testing.assert_allclose(native[:, 14], [0, 0.1, 0.1, 0.1], atol=1e-12)
    if layout == "duration":  # absolute seconds from the anchor
        np.testing.assert_allclose(token[:, 14], [0, 0.1, 0.2, 0.3], atol=1e-12)


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
@pytest.mark.parametrize("gripper_mode", ["events", "distance"])
def test_staggered_delays_are_independent_for_all_six_streams(layout, gripper_mode):
    dt = 0.1
    frames = np.arange(15)
    delays = (2, 4, 6, 3, 5, 7)  # canonical T_L,T_R,R_L,R_R,G_L,G_R
    columns = (0, 7, 3, 10, 6, 13)
    raw = np.zeros((len(frames), 14))
    for col, delay in zip(columns, delays):
        raw[:, col] = np.clip(frames - delay, 0, 3) * 0.01
    codec = make_codec(
        layout,
        rotation_distance_unit=0.03,
        gripper_distance_unit=0.03,
        gripper_stream_mode=gripper_mode,
        resampled_vector_length=8,
    )
    token = encode(codec, raw)
    native = clock_token_seconds(token, layout)
    np.testing.assert_allclose(native[0, 14:], np.array(delays) * dt, atol=1e-12)
    decoded = codec.detokenize(token, len(frames))
    # Distance sampling is linear here; events retain command-change times.
    np.testing.assert_allclose(decoded, raw, atol=1e-10)


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_full_hold_zero_delay_and_constant_geometry(layout):
    codec = make_codec(layout, gripper_stream_mode="distance", gripper_distance_unit=1)
    raw = np.tile(np.arange(14) * 0.01, (400, 1))
    token = encode(codec, raw)
    native = clock_token_seconds(token, layout)
    np.testing.assert_allclose(native[0, 14:], 0, atol=1e-12)
    if layout == "velocity":
        np.testing.assert_array_equal(native[:, 14:], 0)
    else:
        np.testing.assert_allclose(native[1:, 14:], 400 * 0.1 / 3)
    np.testing.assert_allclose(codec.detokenize(token, 400), raw, atol=1e-12)


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_partial_waypoint_prefix_retains_delay_and_incoming_interval(layout):
    codec = make_codec(layout)
    token = encode(codec, delayed_translation())
    prefix = token[:2]
    decoded = codec.detokenize(prefix, 10)
    expected = delayed_translation()
    expected[:, 0] = np.minimum(expected[:, 0], 0.21)
    np.testing.assert_allclose(decoded, expected, atol=1e-12)


def test_absolute_clock_is_decoded_exactly_once():
    codec = make_codec("duration")
    token = encode(codec, delayed_translation())
    seconds = clock_token_seconds(token, codec.clock_mode)
    np.testing.assert_allclose(
        codec.detokenize(token, 10),
        codec.detokenize(seconds, 10, clocks_in_seconds=True),
        atol=1e-12,
    )


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_nonfinite_tokens_fail_closed(layout):
    codec = make_codec(layout)
    token = encode(codec, delayed_translation())
    token[0, 14] = np.nan
    with pytest.raises(ValueError, match="finite"):
        codec.detokenize(token, 10)


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_replanning_includes_delay_and_excludes_constant_streams(layout):
    codec = make_codec(layout)
    token = encode(codec, delayed_translation())
    assert codec.control_steps(token) == 9
    assert codec.control_steps(token[:2]) == 7
    assert codec.control_steps(token[:2], max_steps=5) == 5
    # Only left translation moves: right-arm holds must not win the race.
    assert codec.control_steps(encode(codec, np.zeros((400, 14)))) == 400


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_gripper_can_win_the_six_stream_race(layout):
    raw = delayed_translation()
    raw[5:, 6] = 1
    codec = make_codec(layout)
    token = encode(codec, raw)
    assert codec.control_steps(token[:2]) == 5
    decoded = codec.detokenize(token, len(raw))
    np.testing.assert_array_equal(decoded[:5, 6], 0)
    np.testing.assert_array_equal(decoded[5:, 6], 1)


def test_only_invalid_moving_stream_requires_a_bound():
    codec = make_codec("velocity")
    token = encode(codec, delayed_translation())
    token[1:, 14] = 0
    with pytest.raises(ValueError, match="finite replan"):
        codec.control_steps(token)
    assert codec.control_steps(token, max_steps=20) == 20


def test_collapsed_absolute_clock_still_moves_at_bounded_speed():
    codec = make_codec("duration")
    token = encode(codec, delayed_translation())
    token[1:, 14] = 0  # every waypoint predicted at the anchor
    seconds = clock_token_seconds(token, "duration")[:, 14]
    assert np.all(seconds >= 0)
    assert np.all(seconds[1:] >= 0.01 / 5.0 - 1e-12)  # 1 cm segments, <= 5 m/s
    assert 1 <= codec.control_steps(token) < 20


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_validation_and_robot_use_identical_prefix_clocks(layout, fraction):
    import torch

    from egomimic.eval.open_loop_sim import OpenLoopSimEval
    from egomimic.robot.arc_decoder import FirstStreamArcDecoder

    codec = make_codec(layout)
    raw = delayed_translation()
    token = encode(codec, raw)
    evaluator = OpenLoopSimEval(
        action_mode="arc",
        arc_chunking_mode="multistream",
        execute_fraction=fraction,
        min_distance_unit=0.03,
        rotation_distance_unit=0.3,
        gripper_stream_mode="events",
        resampled_vector_length=4,
        control_dt=0.1,
        clock_mode=layout,
    )
    native = evaluator._clock_seconds(torch.from_numpy(token)).numpy()
    decoded, steps = evaluator._decode_prediction_with_steps(native, max_steps=20)
    expected_steps = 7 if fraction == 0.5 else 9
    assert steps == expected_steps
    np.testing.assert_allclose(decoded, raw[:steps], atol=1e-12)
    decoder = FirstStreamArcDecoder(
        clock_mode=layout,
        min_distance_unit=0.03,
        rotation_distance_unit=0.3,
        resampled_vector_length=4,
        dt=0.1,
        action_horizon=20,
        execute_percent=int(fraction * 100),
        gripper_stream_mode="events",
    )
    rollout = decoder(token)[0]
    assert decoder.execution_steps() == steps
    np.testing.assert_allclose(rollout[:steps], decoded, atol=1e-12)
    np.testing.assert_allclose(
        rollout[steps:], np.tile(rollout[steps - 1], (20 - steps, 1))
    )
    decoder.first_stream = 0
    np.testing.assert_allclose(decoder(token)[0], codec.detokenize(token, 20))
    # Mode 0 defers the replan to the policy's fixed replan_every.
    from egomimic.pipeline.action_adapter import execution_prefix

    assert decoder.execution_steps() is None
    assert (
        len(execution_prefix(decoder(token)[0], decoder=decoder, replan_every=5)) == 5
    )


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_whole_token_rollout_decoder_preserves_first_delay(layout):
    from egomimic.robot.arc_decoder import BimanualIntervalArcDecoder

    codec = make_codec(layout)
    token = encode(codec, delayed_translation())
    decoder = BimanualIntervalArcDecoder(
        clock_mode=layout,
        min_distance_unit=0.03,
        resampled_vector_length=4,
        dt=0.1,
        action_horizon=20,
        rotation_distance_unit=0.3,
        gripper_stream_mode="events",
    )
    np.testing.assert_allclose(decoder(token[None])[0], codec.detokenize(token, 20))


@pytest.mark.parametrize("layout", FIRST_DELAY_LAYOUTS)
def test_distance_prefix_fails_closed_instead_of_misreading_first_delay(layout):
    from egomimic.eval.open_loop_sim import OpenLoopSimEval

    with pytest.raises(ValueError, match="waypoint-prefix"):
        OpenLoopSimEval(
            action_mode="arc", clock_mode=layout, arc_execution_cap_mode="distance"
        )
