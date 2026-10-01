"""Hardware-free checks for the ARC decoder's replay tempo (speed / hold_speed)."""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.rldb.zarr.e1_arc_tokenizer import ARM_LAYOUT
from egomimic.robot.arc_decoder import BimanualArcDecoder
from egomimic.robot.arc_speed import ARC_SPEED_RANGE

M, H, DT = 100, 100, 1 / 30
STEP_M = 0.4 / (M - 1)  # waypoint spacing
MOVE_S = 0.02  # s per waypoint interval while moving -> 0.20 m/s
HOLD_S = 0.20  # s per interval inside a hold        -> 0.02 m/s, under the threshold
HOLD = slice(40, 45)  # five slow intervals = a 1.0 s hold in the middle of the path


def dur_token(left_hold=True, right_hold=True):
    """A straight 0.4 m path per arm as an e1_dur (M, 16) token, optionally holding."""
    token = np.zeros((M, 16))
    for k, (xyz, _ypr, grip, _vsl) in enumerate(ARM_LAYOUT):
        token[:, xyz + k] = STEP_M * np.arange(M)  # left along x, right along y
        token[:, grip] = np.linspace(0.0, 1.0, M)
        token[:, 14 + k] = MOVE_S
        if (left_hold, right_hold)[k]:
            token[HOLD, 14 + k] = HOLD_S
    return token


def decoder(**kwargs):
    return BimanualArcDecoder("e1_dur", 0.4, M, DT, H, **kwargs)


def progress(plan, k=0):
    """Path length reached by arm k at each decoded step."""
    xyz = ARM_LAYOUT[k][0]
    return np.linalg.norm(plan[:, xyz : xyz + 3] - plan[0, xyz : xyz + 3], axis=1)


def steps_inside(plan, lo, hi):
    reached = progress(plan)
    return int(((reached > lo) & (reached < hi)).sum())


def test_demonstrated_tempo_is_the_unmodified_decode():
    token = dur_token()
    unit = decoder()
    expected = unit.codec.detokenize(token, action_horizon=H)
    np.testing.assert_array_equal(unit(token)[0], expected)
    # ... and the warp itself is the identity there, not merely bypassed.
    cums = [STEP_M * np.arange(M)] * 2
    times, _ = unit._warp(unit.codec.clock_at_waypoints(token), cums)
    np.testing.assert_allclose(times, DT * np.arange(H), atol=1e-12)


def test_uniform_speedup_replays_the_same_path_twice_as_fast():
    token = dur_token()
    slow = decoder()(token)[0]
    fast_decoder = decoder(speed=2.0, hold_speed=2.0)
    fast = fast_decoder(token)[0]
    # Step j at 2x is step 2j at 1x, for both arms: geometry, rotation and
    # gripper untouched, and the two arms still in step with each other.
    np.testing.assert_allclose(fast[: H // 2], slow[::2], atol=1e-9)
    assert fast_decoder.last_stats["valid_steps"] == pytest.approx(
        decoder_valid_steps(token) / 2, abs=1
    )


def decoder_valid_steps(token):
    unit = decoder()
    unit(token)
    return unit.last_stats["valid_steps"]


def test_hold_keeps_its_duration_when_only_moving_phases_speed_up():
    token = dur_token()
    hold_at = STEP_M * HOLD.start
    lo, hi = hold_at + 1e-4, STEP_M * HOLD.stop - 1e-4
    baseline = steps_inside(decoder()(token)[0], lo, hi)
    assert baseline == pytest.approx(1.0 / DT, abs=2)  # the 1.0 s hold, at 30 Hz

    selective = decoder(speed=2.0, hold_speed=1.0)
    plan = selective(token)[0]
    uniform = decoder(speed=2.0, hold_speed=2.0)(token)[0]
    # Tempo eases over RATE_RAMP_S at each edge, so allow a few steps either way.
    assert steps_inside(plan, lo, hi) == pytest.approx(baseline, abs=5)
    assert steps_inside(uniform, lo, hi) == pytest.approx(baseline / 2, abs=2)
    # The approach before the hold did get faster.
    assert np.argmax(progress(plan) > lo) < 0.65 * np.argmax(progress(decoder()(token)[0]) > lo)
    assert 0.2 < selective.last_stats["hold_fraction"] < 0.6


def test_one_arm_holding_while_the_other_moves_is_not_a_hold():
    # Bimanual timing comes first: a lone arm's hold rides the pair's clock, so
    # the arms stay exactly as coordinated as the token encoded.
    unit = decoder(speed=2.0, hold_speed=1.0)
    plan = unit(dur_token(left_hold=True, right_hold=False))[0]
    assert unit.last_stats["hold_fraction"] < 0.35  # only the span after the right arm's path ends
    reference = decoder()(dur_token(left_hold=True, right_hold=False))[0]
    np.testing.assert_allclose(plan[:20], reference[:40:2], atol=1e-6)


def test_path_is_never_left_at_any_speed():
    token = dur_token()
    for speed, hold in [(0.5, 0.5), (1.5, 1.0), (3.0, 1.0), (4.0, 4.0)]:
        plan = decoder(speed=speed, hold_speed=hold)(token)[0]
        assert np.isfinite(plan).all()
        for k, (xyz, *_rest) in enumerate(ARM_LAYOUT):
            reached = progress(plan, k)
            assert (np.diff(reached) >= -1e-12).all() and reached[-1] <= 0.4 + 1e-9
            off_axis = np.delete(plan[:, xyz : xyz + 3], k, axis=1)
            np.testing.assert_allclose(off_axis, 0.0, atol=1e-12)


def test_speed_is_validated_and_a_rejected_change_leaves_the_tempo_alone():
    unit = decoder(speed=1.5)
    assert (unit.speed, unit.hold_speed) == (1.5, 1.0)
    for bad in (True, "2", float("nan"), float("inf"), ARC_SPEED_RANGE[0] / 2, ARC_SPEED_RANGE[1] * 2):
        with pytest.raises(ValueError):
            unit.set_speed(bad)
        with pytest.raises(ValueError):
            unit.set_speed(2.0, bad)
        assert (unit.speed, unit.hold_speed) == (1.5, 1.0)
    unit.set_speed(2.0)  # hold_speed=None -> a uniform speed-up
    assert (unit.speed, unit.hold_speed) == (2.0, 2.0)
    with pytest.raises(ValueError):
        decoder(hold_threshold=-0.1)


def test_lab_tokens_take_a_uniform_speed():
    token = np.zeros((M + 1, 14))
    for k, (xyz, _ypr, grip, vsl) in enumerate(ARM_LAYOUT):
        token[:M, xyz + k] = STEP_M * np.arange(M)
        token[:M, grip] = np.linspace(0.0, 1.0, M)
        token[M, vsl] = np.eye(3)[k] * 0.2  # 0.2 m/s along the path
    slow = BimanualArcDecoder("lab", 0.4, M, DT, H)(token)[0]
    unit = BimanualArcDecoder("lab", 0.4, M, DT, H, speed=2.0)
    np.testing.assert_allclose(unit(token)[0][: H // 2], slow[::2], atol=1e-9)
    assert unit.last_stats["hold_speed"] is None


def test_cartesian_per_waypoint_tokens_take_a_uniform_speed():
    # The (2M, 14) per-waypoint velocity token has no per-waypoint clock to warp,
    # so it gets the lab treatment: the whole replay runs at one faster tempo.
    t = np.arange(H) / H
    chunk = np.zeros((H, 14))
    for k, (xyz, _ypr, grip, _vsl) in enumerate(ARM_LAYOUT):
        chunk[:, xyz + k] = 0.3 * t  # 0.3 m over the chunk, left along x, right along y
        chunk[:, grip] = t
    unit = BimanualArcDecoder("cartesian_per_waypoint", 0.4, M, DT, H)
    token = unit.codec.transform({"actions_cartesian": chunk.copy()})["actions_cartesian"]
    slow = unit(token)[0]
    unit.set_speed(2.0)
    fast = unit(token)[0]
    np.testing.assert_allclose(fast[: H // 2 - 1], slow[: H - 2 : 2], atol=2e-3)
    assert unit.last_stats["hold_speed"] is None


def test_policy_exposes_arc_speed_as_typed_controls_that_drive_the_decoder():
    from types import SimpleNamespace

    from egomimic.robot.graph_policy import GraphRobotPolicy, arc_speed_controls

    assert arc_speed_controls(None) == ()
    lab = arc_speed_controls(BimanualArcDecoder("lab", 0.4, M, DT, H))
    assert [control.name for control in lab] == ["arc_speed_percent"]

    unit = decoder()
    controls = {control.name: control for control in arc_speed_controls(unit)}
    assert set(controls) == {"arc_speed_percent", "arc_hold_speed_percent"}
    public = controls["arc_speed_percent"].public()
    assert (public["min"], public["max"], public["step"], public["value"]) == (25, 400, 5, 100)

    policy = SimpleNamespace(_inference_controls=controls, _replan_every=None)
    policy.inference_controls = lambda: GraphRobotPolicy.inference_controls(policy)
    GraphRobotPolicy.apply_inference_overrides(policy, {"arc_speed_percent": 200})
    assert (unit.speed, unit.hold_speed) == (2.0, 1.0)
    GraphRobotPolicy.apply_inference_overrides(policy, {"arc_hold_speed_percent": 150})
    assert (unit.speed, unit.hold_speed) == (2.0, 1.5)
    for bad in (5, 401, 102, 2.0):
        with pytest.raises(ValueError):
            GraphRobotPolicy.apply_inference_overrides(policy, {"arc_speed_percent": bad})
    assert (unit.speed, unit.hold_speed) == (2.0, 1.5)
