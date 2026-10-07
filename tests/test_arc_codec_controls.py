"""Explicit contracts, per-arm clocks, and hardware-free control validation."""

import numpy as np
import pytest

from egomimic.rldb.zarr import arc_length_tokenizer as canonical
from egomimic.rldb.zarr import arc_length_tokenizer_m28 as m28
from egomimic.rldb.zarr import arc_length_tokenizer_pr193 as pr193
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1
from egomimic.robot.arc_decoder import (
    BimanualArcDecoder,
    BimanualIntervalArcDecoder,
    FirstStreamArcDecoder,
    TimeChunkRetimer,
)

M, H, DT = 100, 100, 1 / 30


def moving_chunk():
    t = np.arange(200)[:, None] * DT
    out = np.zeros((200, 14))
    out[:, 0:3] = t * np.array([0.1, 0.03, 0.0])
    out[:, 7:10] = t * np.array([0.0, 0.06, 0.0])
    out[:, 3:6] = t * np.array([0.12, 0.0, 0.0])
    out[:, 10:13] = t * np.array([0.3, 0.0, 0.0])
    out[:, 6] = 0.8 - 0.5 * t[:, 0] / t[-1, 0]
    out[:, 13] = 0.7
    return out


def test_canonical_additions_preserve_per_arm_rate_codec_and_frozen_parity():
    kwargs = dict(
        min_distance_unit=0.4,
        rotation_distance_unit=1.0,
        resampled_vector_length=M,
        dt=DT,
        velocity_mode="per_waypoint",
        velocity_layout="wide",
        arc_chunking_mode="multistream",
    )
    current = canonical.TokenizeBimanualArcLengthCartesian(**kwargs)
    frozen = m28.TokenizeBimanualArcLengthCartesian(**kwargs)
    token = current.transform({"actions_cartesian": moving_chunk()})[
        "actions_cartesian"
    ]
    old_token = frozen.transform({"actions_cartesian": moving_chunk()})[
        "actions_cartesian"
    ]
    np.testing.assert_array_equal(token, old_token)
    expected = frozen.detokenize(token, H)
    np.testing.assert_array_equal(current.detokenize(token, H), expected)
    interval = BimanualIntervalArcDecoder(
        "per_waypoint",
        0.4,
        M,
        DT,
        H,
        velocity_layout="wide",
        rotation_distance_unit=1.0,
        arc_chunking_mode="multistream",
    )
    np.testing.assert_array_equal(interval(token[None])[0], expected)
    assert interval.execution_steps() is None


@pytest.mark.parametrize("layout", ["wide", "stacked", "clock"])
def test_canonical_duration_layouts_have_four_equivalent_clocks(layout):
    kwargs = dict(
        min_distance_unit=0.4,
        rotation_distance_unit=1.0,
        resampled_vector_length=M,
        dt=DT,
        velocity_mode="duration",
        velocity_layout=layout,
        arc_chunking_mode="multistream",
    )
    codec = canonical.TokenizeBimanualArcLengthCartesian(**kwargs)
    frozen = m28.TokenizeBimanualArcLengthCartesian(**kwargs)
    token = codec.transform({"actions_cartesian": moving_chunk()})["actions_cartesian"]
    np.testing.assert_array_equal(
        token,
        frozen.transform({"actions_cartesian": moving_chunk()})["actions_cartesian"],
    )
    np.testing.assert_array_equal(
        codec.detokenize(token, H), frozen.detokenize(token, H)
    )
    decoder = BimanualIntervalArcDecoder(
        "duration",
        0.4,
        M,
        DT,
        H,
        velocity_layout=layout,
        rotation_distance_unit=1.0,
        arc_chunking_mode="multistream",
        first_stream=1,
        execute_percent=50,
    )
    out = decoder(token[None])[0]
    n = decoder.execution_steps()
    assert 1 <= n <= H
    np.testing.assert_allclose(out[:n], codec.detokenize(token, H)[:n], atol=1e-12)


def test_identical_18d_shapes_do_not_imply_identical_timing_layout():
    token = np.zeros((M, 18))
    token[:, 0] = np.linspace(0, 0.2, M)
    token[:, 3] = np.linspace(0, 0.5, M)
    token[:-1, 14] = 1.0 / (M - 1)  # E1 left translation; also M28 left translation
    token[:-1, 15] = 3.0 / (M - 1)  # E1 right translation; M28 left rotation
    token[:-1, 16] = 2.0 / (M - 1)  # E1 left rotation; M28 right translation
    e1 = BimanualArcDecoder("e1_durhyb", 0.4, M, DT, H, rotation_distance_unit=1.0)(
        token
    )[0]
    native_m28 = FirstStreamArcDecoder(
        "duration", "clock", 0.4, 1.0, M, DT, H, first_stream=0
    )(token)[0]
    assert np.max(np.abs(e1[:, 3] - native_m28[:, 3])) > 0.1
    with pytest.raises(ValueError, match="per_waypoint"):
        pr193.TokenizeBimanualArcLengthCartesian(
            velocity_mode="duration", velocity_layout="clock"
        )


def test_custom_rotation_budget_and_legacy_defaults_remain_explicit():
    decoder = BimanualArcDecoder("e1_durhyb", rotation_distance_unit=0.6)
    assert decoder.codec.rotation_distance_unit == 0.6
    assert decoder.first_stream == 0 and decoder.execute_percent == 100
    assert decoder.execution_steps() is None
    with pytest.raises(TypeError):
        BimanualArcDecoder()


@pytest.mark.parametrize(
    "attribute,bad",
    [
        ("execute_percent", 33),
        ("first_stream", True),
        ("speed_percent", 500),
        ("hold_speed_percent", 100.0),
    ],
)
def test_control_preflight_is_nonmutating_and_rejects_invalid_values(attribute, bad):
    decoder = BimanualArcDecoder("e1_dur", resampled_vector_length=50)
    before = (
        decoder.first_stream,
        decoder.execute_percent,
        decoder.speed_percent,
        decoder.hold_speed_percent,
    )
    with pytest.raises(ValueError):
        decoder.validate_inference_control(attribute, bad)
    assert before == (
        decoder.first_stream,
        decoder.execute_percent,
        decoder.speed_percent,
        decoder.hold_speed_percent,
    )
    for name, value in [
        ("first_stream", 1),
        ("execute_percent", 50),
        ("speed_percent", 200),
        ("hold_speed_percent", 100),
    ]:
        assert decoder.validate_inference_control(name, value) == value
    assert before == (
        decoder.first_stream,
        decoder.execute_percent,
        decoder.speed_percent,
        decoder.hold_speed_percent,
    )


def test_time_retimer_has_only_tempo_controls_and_validates_unit_tempo_input():
    retimer = TimeChunkRetimer(action_horizon=H, dt=DT)
    row = moving_chunk()[:H][None]
    assert retimer(row) is row
    for name, value in [("speed_percent", 200), ("hold_speed_percent", 100)]:
        assert retimer.validate_inference_control(name, value) == value
    with pytest.raises(ValueError):
        retimer.validate_inference_control("first_stream", 1)
    with pytest.raises(ValueError):
        retimer(np.full((1, H, 14), np.nan))
    with pytest.raises(ValueError):
        retimer(np.zeros((1, H, 18)))


@pytest.mark.parametrize("mode,width", [("durhyb", 18), ("durtri", 20)])
def test_zero_time_geometric_motion_cannot_teleport(mode, width):
    codec = TokenizeBimanualArcLengthE1(velocity_mode=mode, resampled_vector_length=M)
    token = np.zeros((M, width))
    token[:, 0] = np.linspace(0, 0.2, M)
    with pytest.raises(ValueError, match="positive durations"):
        codec.detokenize(token, H)


def test_tri_is_decode_only_while_trained_hybrid_transform_stays_available():
    hybrid = TokenizeBimanualArcLengthE1(
        velocity_mode="durhyb", resampled_vector_length=M
    )
    assert hybrid.transform({"actions_cartesian": moving_chunk()})[
        "actions_cartesian"
    ].shape == (M, 18)
    tri = TokenizeBimanualArcLengthE1(velocity_mode="durtri", resampled_vector_length=M)
    with pytest.raises(NotImplementedError, match="decode-only"):
        tri.transform({"actions_cartesian": moving_chunk()})
