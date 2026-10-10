"""Station decode of the Elmo+Aidan token shapes: hybrid arcdur (100, 18) and the
PR #193 lab per-waypoint wide (100, 28) / stacked (200, 14) layouts."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.robot.arc_codecs import pr193
from egomimic.robot.arc_codecs.e1 import TokenizeBimanualArcLengthE1
from egomimic.robot.arc_decoder import BimanualArcDecoder

DT = 1 / 30
M = 100


def _chunk(frames=200, seed=0):
    """A smooth bimanual (T, 14) chunk: both arms travel and turn, grippers close."""
    t = np.linspace(0.0, 1.0, frames)[:, None]
    rng = np.random.default_rng(seed)
    out = []
    for arm in range(2):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        xyz = 0.3 * direction * (t**2 * (3 - 2 * t)) + np.array([0.3, 0.2 * arm, 0.2])
        ypr = np.concatenate([0.6 * t, 0.2 * np.sin(2 * t), -0.1 * t], axis=1)
        grip = 1.0 - t
        out.append(np.concatenate([xyz, ypr, grip], axis=1))
    return np.concatenate(out, axis=1)


@pytest.mark.parametrize("layout,shape", [("wide", (M, 28)), ("stacked", (2 * M, 14))])
def test_lab_pw_layouts_decode_with_the_pr193_codec(layout, shape):
    codec = pr193.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        resampled_vector_length=M,
        velocity_mode="per_waypoint",
        velocity_layout=layout,
    )
    token = codec.transform({"actions_cartesian": _chunk()})["actions_cartesian"]
    assert token.shape == shape
    decoder = BimanualArcDecoder(f"lab_pw_{layout}", 0.4, M, DT, 100)
    assert decoder.shape == shape
    np.testing.assert_array_equal(
        decoder(token)[0], codec.detokenize(token, action_horizon=100)
    )


def test_lab_pw_wide_and_stacked_carry_the_same_motion():
    chunk = _chunk(seed=3)
    tokens = {}
    for layout in ("wide", "stacked"):
        codec = pr193.TokenizeBimanualArcLengthCartesian(
            min_distance_unit=0.4,
            resampled_vector_length=M,
            velocity_mode="per_waypoint",
            velocity_layout=layout,
        )
        tokens[layout] = codec.transform({"actions_cartesian": chunk.copy()})[
            "actions_cartesian"
        ]
    np.testing.assert_allclose(pr193.stack_arc_token(tokens["wide"]), tokens["stacked"])
    wide = BimanualArcDecoder("lab_pw_wide", 0.4, M, DT, 100)(tokens["wide"])
    stacked = BimanualArcDecoder("lab_pw_stacked", 0.4, M, DT, 100)(tokens["stacked"])
    np.testing.assert_allclose(wide, stacked, atol=1e-12)


def test_lab_pw_rejects_the_other_layout_and_scales_uniformly():
    codec = pr193.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        resampled_vector_length=M,
        velocity_mode="per_waypoint",
        velocity_layout="wide",
    )
    token = codec.transform({"actions_cartesian": _chunk()})["actions_cartesian"]
    with pytest.raises(ValueError):
        BimanualArcDecoder("lab_pw_stacked", 0.4, M, DT, 100)(token)
    decoder = BimanualArcDecoder("lab_pw_wide", 0.4, M, DT, 100)
    decoder.set_speed(2.0)
    fast = pr193.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        resampled_vector_length=M,
        dt=2 * DT,
        velocity_mode="per_waypoint",
        velocity_layout="wide",
    )
    np.testing.assert_array_equal(
        decoder(token)[0], fast.detokenize(token, action_horizon=100)
    )


def _hybrid_codec():
    return TokenizeBimanualArcLengthE1(
        min_distance_unit=0.4,
        resampled_vector_length=M,
        velocity_norm="path",
        velocity_mode="durhyb",
    )


def test_hybrid_matches_dur_when_both_streams_share_the_clock():
    """With no start delay and rotation on the translation clock, durhyb is dur."""
    dur = TokenizeBimanualArcLengthE1(
        min_distance_unit=0.4,
        resampled_vector_length=M,
        velocity_norm="path",
        velocity_mode="dur",
    )
    token16 = dur.transform({"actions_cartesian": _chunk(seed=5)})["actions_cartesian"]
    token18 = np.concatenate([token16, token16[:, 14:16]], axis=1)
    token18[M - 1, 14:18] = 0.0  # start delays
    np.testing.assert_allclose(
        _hybrid_codec().detokenize(token18, 100),
        dur.detokenize(token16, 100),
        atol=1e-9,
    )


def test_hybrid_delayed_wrist_turn_and_held_arm_gripper():
    token = np.zeros((M, 18))
    # Left arm: 0.2 m along x over 1 s, then a 60 deg yaw turn that starts at 1.5 s
    # and takes 1 s. Gripper stays open.
    token[:, 0] = np.linspace(0.0, 0.2, M)
    token[:, 6] = 1.0
    token[: M - 1, 14] = 1.0 / (M - 1)
    yaw = np.linspace(0.0, np.pi / 3, M)
    token[:, 3] = yaw
    token[: M - 1, 16] = 1.0 / (M - 1)
    token[M - 1, 16] = 1.5
    # Right arm: held still, gripper closes uniformly over the 3.3 s window.
    token[:, 7:10] = [0.3, 0.6, 0.2]
    token[:, 13] = np.linspace(1.0, 0.0, M)
    token[: M - 1, 15] = 3.3 / (M - 1)
    out = _hybrid_codec().detokenize(token, 100)
    t = DT * np.arange(100)
    np.testing.assert_allclose(
        out[:, 0], np.interp(t, np.linspace(0, 1, M), token[:, 0]), atol=1e-12
    )
    np.testing.assert_allclose(out[t < 1.5, 3], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[t >= 2.5, 3], np.pi / 3, atol=1e-9)
    mid = np.argmin(np.abs(t - 2.0))
    assert abs(out[mid, 3] - np.pi / 6) < np.deg2rad(1.0)
    np.testing.assert_allclose(out[:, 7:10], np.tile([0.3, 0.6, 0.2], (100, 1)))
    np.testing.assert_allclose(
        out[:, 13], np.interp(t, np.linspace(0, 3.3, M), token[:, 13]), atol=1e-12
    )
    # Rotation decodes on the geodesic: the turn is about yaw alone.
    rel = Rotation.from_euler("ZYX", out[:, 3:6])
    np.testing.assert_allclose(rel.as_euler("ZYX")[:, 1:], 0.0, atol=1e-9)


def test_hybrid_decoder_shape_and_valid_steps_cover_the_late_turn():
    token = np.zeros((M, 18))
    token[:, 0] = np.linspace(0.0, 0.1, M)
    token[: M - 1, 14] = 0.5 / (M - 1)
    token[:, 3] = np.linspace(0.0, 1.0, M)
    token[: M - 1, 16] = 1.0 / (M - 1)
    token[M - 1, 16] = 1.0  # turn runs 1.0 .. 2.0 s, translation ends at 0.5 s
    decoder = BimanualArcDecoder("e1_durhyb", 0.4, M, DT, 100)
    assert decoder.shape == (M, 18)
    out = decoder(token)
    assert out.shape == (1, 100, 14)
    assert decoder.last_stats["valid_steps"] >= int(2.0 / DT)
    decoder.set_speed(2.0)
    fast = decoder(token)[0]
    t = DT * np.arange(100)
    # At 2x the turn finishes by 1.0 s of wall time.
    np.testing.assert_allclose(fast[t >= 1.0 + 1e-9, 3], 1.0, atol=1e-9)
    with pytest.raises(ValueError):
        decoder(np.zeros((M, 16)))
