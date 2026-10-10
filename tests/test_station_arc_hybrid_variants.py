"""Decode-only E1 multi-clock variants on the station: arcvelhyb (profhyb), arcdurtri / arcveltri (tri)."""

import numpy as np
import pytest

from egomimic.robot.arc_codecs.e1 import (
    cumulative_arc_length,
    gripper_arc_length,
    rotation_arc_length,
)
from egomimic.robot.arc_decoder import BimanualArcDecoder

M, DT, H = 100, 1 / 30, 100


def _durations(total, delay=0.0):
    col = np.full(M, total / (M - 1))
    col[-1] = delay
    return col


def _hyb_token(tri=False, grip_delay=1.0):
    """Left arm: 0.2 m in x over 1.0 s, a 40 deg wrist turn over 2.0 s, gripper
    closing 1 -> 0 (on its own stream for tri: starting at ``grip_delay``, over
    0.5 s). Right arm: still, gripper open. Timing columns are durations."""
    s = np.linspace(0.0, 1.0, M)
    token = np.zeros((M, 20 if tri else 18))
    token[:, 0] = 0.2 * s
    token[:, 3] = np.deg2rad(40) * s
    token[:, 6] = 1.0 - s
    token[:, 13] = 1.0
    token[:, 14] = _durations(1.0)
    token[:, 16] = _durations(2.0)
    if tri:
        token[:, 18] = _durations(0.5, delay=grip_delay)
    return token


def _as_speeds(token, tri):
    """The profhyb / proftri token encoding a duration token (training's _dur_col_to_speed)."""
    out = token.copy()
    for k, (xyz, ypr, grip) in enumerate(((0, 3, 6), (7, 10, 13))):
        streams = [
            (14 + k, np.diff(cumulative_arc_length(token[:, xyz : xyz + 3])), 1e-6),
            (16 + k, np.diff(rotation_arc_length(token[:, ypr : ypr + 3])), 1e-6),
        ]
        if tri:
            streams.append((18 + k, np.diff(gripper_arc_length(token[:, grip])), 1e-3))
        for column, seg, eps in streams:
            col = np.zeros(M)
            if seg.sum() >= eps:
                col[:-1] = seg / np.maximum(token[:-1, column], 1e-4)
                col[-1] = token[-1, column]
            out[:, column] = col
    return out


@pytest.mark.parametrize("tri", [False, True])
@pytest.mark.parametrize("first_stream", [0, 1])
def test_speed_columns_decode_exactly_like_the_durations_they_encode(tri, first_stream):
    durations = _hyb_token(tri)
    dur_layout, speed_layout = (
        ("e1_durtri", "e1_proftri") if tri else ("e1_durhyb", "e1_profhyb")
    )
    by_duration = BimanualArcDecoder(
        dur_layout, 0.4, M, DT, H, first_stream=first_stream
    )
    by_speed = BimanualArcDecoder(
        speed_layout, 0.4, M, DT, H, first_stream=first_stream
    )
    assert by_speed.shape == (M, 20 if tri else 18)
    np.testing.assert_allclose(
        by_speed(_as_speeds(durations, tri)), by_duration(durations), atol=1e-9
    )
    assert by_speed.last_stats[
        "replan_steps" if first_stream else "valid_steps"
    ] == pytest.approx(
        by_duration.last_stats["replan_steps" if first_stream else "valid_steps"]
    )


def test_tri_gripper_keeps_its_own_clock_and_start_delay():
    decoded = BimanualArcDecoder("e1_durtri", 0.4, M, DT, H, first_stream=0)(
        _hyb_token(tri=True)
    )[0]
    t = DT * np.arange(H)
    grip = decoded[:, 6]
    assert np.all(
        grip[t < 0.99] == pytest.approx(1.0)
    )  # still open during its 1.0 s delay
    assert grip[np.searchsorted(t, 1.25)] == pytest.approx(
        0.5, abs=0.05
    )  # half closed mid-motion
    assert np.all(grip[t > 1.51] == pytest.approx(0.0, abs=1e-9))  # closed by 1.5 s
    # durhyb rides translation: the same gripper closes over the 1.0 s translation instead
    hyb = BimanualArcDecoder("e1_durhyb", 0.4, M, DT, H, first_stream=0)(_hyb_token())[
        0
    ]
    assert hyb[np.searchsorted(t, 0.5), 6] == pytest.approx(0.5, abs=0.05)


def test_an_idle_tri_gripper_does_not_end_a_fastest_stream_chunk():
    token = _hyb_token(tri=True)
    rng = np.random.default_rng(0)
    token[:, 6] = 0.8 + rng.normal(
        0, 0.002, M
    )  # a held gripper, as a prediction would show it
    token[:, 18] = np.abs(rng.normal(0, 1e-4, M))  # its near-zero (hold-coded) column
    decoder = BimanualArcDecoder("e1_durtri", 0.4, M, DT, H, first_stream=1)
    decoder(token)
    stats = decoder.last_stats
    assert len(stats["stream_ends_s"]) == 2  # left translation + left rotation only
    assert (
        stats["replan_steps"] >= 9
    )  # translation reaches its 30 % cap at ~0.3 s, not ~0


def test_tri_speed_up_scales_every_stream():
    token = _hyb_token(tri=True)
    slow = BimanualArcDecoder("e1_durtri", 0.4, M, DT, 2 * H, first_stream=0)(token)[0]
    fast_decoder = BimanualArcDecoder("e1_durtri", 0.4, M, DT, H, first_stream=0)
    fast_decoder.set_speed(2.0)
    np.testing.assert_allclose(fast_decoder(token)[0], slow[::2][:H], atol=1e-9)
