"""M28 multistream rollout: first N % of the waypoints, replan at the first stream."""

from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.rldb.zarr import arc_length_tokenizer_m28 as m28
from egomimic.robot.arc_decoder import BimanualArcDecoder, FirstStreamArcDecoder
from egomimic.robot.graph_policy import GraphRobotPolicy

D, R, M, H, DT = 0.7143, 2.0997, 100, 100, 1 / 30


def token(mode="per_waypoint", layout="wide", right_speed=0.1):
    """Left arm translates fast, right arm slowly (or not at all); both turn."""
    t = np.arange(400)[:, None] * DT
    traj = np.zeros((400, 14))
    traj[:, 0:3] = t * [0.5, 0.1, 0.0]
    traj[:, 7:10] = t * [0.0, right_speed, 0.0]
    traj[:, 3:6] = traj[:, 10:13] = t * [0.2, 0.0, 0.0]
    traj[:, [6, 13]] = 0.5
    codec = m28.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=D, rotation_distance_unit=R, resampled_vector_length=M,
        dt=DT, velocity_mode=mode, velocity_layout=layout, arc_chunking_mode="multistream",
    )
    return codec.transform({"actions_cartesian": traj})["actions_cartesian"]


@pytest.mark.parametrize(("mode", "layout"), [
    ("per_waypoint", "wide"), ("per_waypoint", "stacked"), ("duration", "clock"),
])
def test_replans_when_the_first_moving_stream_ends(mode, layout):
    decoder = FirstStreamArcDecoder(mode, layout, D, R, M, DT, H, execute_percent=30)
    out = decoder(token(mode, layout))
    stats = decoder.last_stats
    moving = [d for d in stats["stream_durations_s"] if d > 1e-9]
    n = stats["replan_steps"]
    assert stats["execute_waypoints"] == 30
    assert n == min(H, int(np.ceil(min(moving) / DT - 1e-9)))
    assert n < int(np.ceil(max(moving) / DT - 1e-9))  # the slow arm is cut, not waited for
    assert out.shape == (1, H, 14)
    assert np.all(out[0, n:] == out[0, n - 1])


def test_a_still_arm_cannot_end_the_chunk():
    decoder = FirstStreamArcDecoder("per_waypoint", "wide", D, R, M, DT, H)
    decoder(token(right_speed=0.0))
    left_translation, _, right_translation, _ = decoder.last_stats["stream_durations_s"]
    assert right_translation == 0.0
    assert decoder.last_stats["replan_steps"] > 1


def test_execute_percent_needs_a_whole_two_waypoint_prefix():
    decoder = FirstStreamArcDecoder("per_waypoint", "wide", D, R, M, DT, H)
    for bad in (0, 1, 101, 30.0):
        with pytest.raises(ValueError):
            decoder.execute_percent = bad
    half = FirstStreamArcDecoder("per_waypoint", "wide", D, R, 50, DT, H)
    with pytest.raises(ValueError):
        half.execute_percent = 33
    half.execute_percent = 4


def test_policy_executes_exactly_the_first_stream_prefix():
    decoder = FirstStreamArcDecoder("per_waypoint", "wide", D, R, M, DT, H)
    decoder(token())
    policy = SimpleNamespace(adapter=SimpleNamespace(decoder=decoder), _replan_every=30)
    plan = GraphRobotPolicy.execution_plan(policy, np.zeros((H, 14)))
    assert len(plan) == decoder.last_stats["replan_steps"]


@pytest.mark.parametrize("make", [
    lambda: FirstStreamArcDecoder("per_waypoint", "wide", D, R, M, DT, H),
    lambda: BimanualArcDecoder("e1_dur", 0.4, M, DT, H),
])
def test_dashboard_toggle_switches_between_fastest_stream_and_fixed_repredict(make):
    from egomimic.robot.graph_policy import chunk_termination_controls

    decoder = make()
    controls = {c.name: c for c in chunk_termination_controls(decoder)}
    toggle, cap = controls["multistream_fastest_stream"], controls["execute_waypoint_percent"]
    assert (toggle.minimum, toggle.maximum, toggle.step, toggle.value) == (0, 1, 1, 1)
    assert (cap.minimum, cap.maximum, cap.value) == (2, 100, 50)  # ARC default cap, 2026-10-06
    policy = SimpleNamespace(adapter=SimpleNamespace(decoder=decoder), _replan_every=7,
                             _inference_controls=controls, inference_controls=lambda: {})
    row = token() if isinstance(decoder, FirstStreamArcDecoder) else np.c_[
        np.linspace(0, 0.4, M)[:, None] * np.r_[1, 0, 0, 0, 0, 0, 0.5, 0, 1, 0, 0, 0, 0, 0.5],
        np.full((M, 2), 0.02)]

    decoder(row)
    on = GraphRobotPolicy.execution_plan(policy, np.zeros((H, 14)))
    assert len(on) == decoder.last_stats["replan_steps"]

    GraphRobotPolicy.apply_inference_overrides(policy, {"multistream_fastest_stream": 0})
    assert decoder.first_stream == 0
    decoder(row)
    assert "replan_steps" not in decoder.last_stats  # original method: whole token
    assert len(GraphRobotPolicy.execution_plan(policy, np.zeros((H, 14)))) == 7

    with pytest.raises(ValueError):
        GraphRobotPolicy.apply_inference_overrides(policy, {"multistream_fastest_stream": 2})
    assert chunk_termination_controls(None) == ()


def test_fixed_repredict_mode_decodes_the_whole_token():
    decoder = BimanualArcDecoder("e1_dur", 0.4, M, DT, H, first_stream=0)
    row = np.c_[np.linspace(0, 0.4, M)[:, None] * np.r_[1, 0, 0, 0, 0, 0, 0.5, 0, 1, 0, 0, 0, 0, 0.5],
                np.full((M, 2), 0.02)]
    assert np.array_equal(decoder(row)[0], decoder.codec.detokenize(row, action_horizon=H))


def _station_tokens(layout, rng):
    from egomimic.rldb.zarr import arc_length_tokenizer_pr193 as pr193
    from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
    from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1

    t = np.arange(400)[:, None] * DT
    traj = np.zeros((400, 14))
    for off in (0, 7):
        traj[:, off:off + 3] = rng.uniform(0, 0.5, 3) * t + 0.03 * np.sin(rng.uniform(1, 6, 3) * t)
        traj[:, off + 3:off + 6] = rng.uniform(-0.6, 0.6, 3) * t
        traj[:, off + 6] = 0.5 + 0.4 * np.sin(rng.uniform(0.5, 3) * t[:, 0])
    common = dict(min_distance_unit=0.4, resampled_vector_length=M, dt=DT)
    if layout == "e1_durhyb":  # decode-only on the station
        token = np.zeros((M, 18))
        token[:, :14] = traj[::3][:M]
        token[:-1, 14:] = rng.uniform(0.0, 0.05, (M - 1, 4))
        token[-1, 14:] = rng.uniform(0.0, 0.2, 4)
        return token
    if layout.startswith("e1_"):
        codec = TokenizeBimanualArcLengthE1(**common, velocity_norm="path",
                                            velocity_mode=layout.removeprefix("e1_"))
    elif layout.startswith("lab_pw_"):
        codec = pr193.TokenizeBimanualArcLengthCartesian(
            **common, velocity_mode="per_waypoint", velocity_layout=layout.removeprefix("lab_pw_"))
    elif layout == "lab":
        codec = TokenizeBimanualArcLengthCartesian(**common)
    else:
        mode = layout.removeprefix("cartesian_")
        extra = dict(rotation_distance_unit=0.6, arc_chunking_mode="multistream") if mode == "per_waypoint" else {}
        codec = TokenizeBimanualArcLengthCartesian(**common, velocity_mode=mode, **extra)
    return codec.transform({"actions_cartesian": traj})["actions_cartesian"]


@pytest.mark.parametrize("layout", [
    "lab", "cartesian_per_waypoint", "cartesian_duration", "lab_pw_wide", "lab_pw_stacked",
    "e1_dur", "e1_logdur", "e1_profile", "e1_durhyb",
])
@pytest.mark.parametrize("percent", [10, 30, 100])
def test_every_layout_caps_then_ends_at_the_first_stream(layout, percent):
    """Cap, detokenize the capped token, end at the fastest moving stream's cap.
    The executed rows are exactly the full token's rows: capping changes no clock."""
    kwargs = {"rotation_distance_unit": 0.6, "arc_chunking_mode": "multistream"} if layout == "cartesian_per_waypoint" else {}
    row = np.asarray(_station_tokens(layout, np.random.default_rng(percent)), dtype=np.float64)
    decoder = BimanualArcDecoder(layout, 0.4, M, DT, 200, execute_percent=percent, **kwargs)
    out = decoder(row)[0]
    stats = decoder.last_stats
    n = stats["replan_steps"]
    assert stats["execute_waypoints"] == M * percent // 100
    assert n == min(200, max(1, int(np.ceil(min(stats["stream_ends_s"]) / DT - 1e-9))))
    full = decoder.codec.detokenize(row, action_horizon=200)
    np.testing.assert_allclose(out[:n], full[:n], rtol=0, atol=1e-12)
