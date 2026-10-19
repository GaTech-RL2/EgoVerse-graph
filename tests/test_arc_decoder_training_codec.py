"""The station decoder reads this repository's own tokens: ``BimanualArcDecoder``
decodes every E1 layout with the training tokenizer, warps its tempo through the
codec's own clocks, and ends a chunk at the first moving stream. Hardware-free."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.pipeline.action_adapter import execution_prefix
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian as TrainingCartesianCodec,
)
from egomimic.rldb.zarr.e1_arc_tokenizer import (
    ARM_LAYOUT,
)
from egomimic.robot.arc_decoder import (
    ARC_TOKEN_LAYOUTS,
    E1_VELOCITY_MODE,
    BimanualArcDecoder,
    FirstStreamArcDecoder,
)

M, H, DT = 100, 100, 1 / 30


def _chunk(frames=200, seed=0, turn=0.6):
    """A smooth bimanual (T, 14) chunk: both arms travel and turn, grippers close."""
    t = np.linspace(0.0, 1.0, frames)[:, None]
    rng = np.random.default_rng(seed)
    out = []
    for arm in range(2):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        xyz = 0.3 * direction * (t**2 * (3 - 2 * t)) + np.array([0.3, 0.2 * arm, 0.2])
        ypr = np.concatenate([turn * t, 0.2 * np.sin(2 * t), -0.1 * t], axis=1)
        grip = 1.0 - t
        out.append(np.concatenate([xyz, ypr, grip], axis=1))
    return np.concatenate(out, axis=1)


def _training_token(layout, chunk, **codec_kwargs):
    decoder = BimanualArcDecoder(layout, 0.4, M, DT, H, **codec_kwargs)
    token = decoder.codec.transform({"actions_cartesian": chunk.copy()})[
        "actions_cartesian"
    ]
    assert token.shape == decoder.shape
    return decoder, token


@pytest.mark.parametrize("layout", [*E1_VELOCITY_MODE, "lab"])
def test_demonstrated_tempo_without_termination_is_the_training_decode(layout):
    decoder, token = _training_token(layout, _chunk())
    expected = decoder.codec.detokenize(token, action_horizon=H)
    np.testing.assert_array_equal(decoder(token)[0], expected)
    assert (
        decoder.last_stats["speed"] == 1.0 and "replan_steps" not in decoder.last_stats
    )


@pytest.mark.parametrize("layout", list(E1_VELOCITY_MODE))
def test_uniform_speedup_reads_the_same_path_twice_as_fast(layout):
    decoder, token = _training_token(layout, _chunk(seed=2))
    slow = decoder(token)[0]
    decoder.set_speed(2.0)
    fast = decoder(token)[0]
    # Row k of the 2x replay is row 2k of the demonstrated one, as long as the path lasts.
    valid = min(decoder.last_stats["valid_steps"], H // 2)
    assert valid >= 10
    np.testing.assert_allclose(fast[:valid], slow[: 2 * valid : 2], atol=1e-7)
    assert decoder.last_stats["speed"] == 2.0


def test_hold_speed_keeps_the_hold_and_the_waypoints():
    # Left arm: a 0.4 m straight path at 0.2 m/s with a 1.0 s hold in the middle.
    token = np.zeros((M, 16))
    step = 0.4 / (M - 1)
    for k, (xyz, _ypr, grip, _vsl) in enumerate(ARM_LAYOUT):
        token[:, xyz + k] = step * np.arange(M)
        token[:, grip] = np.linspace(0.0, 1.0, M)
        token[:, 14 + k] = 0.02
        token[40:45, 14 + k] = 0.20
    decoder = BimanualArcDecoder("e1_dur", 0.4, M, DT, 2 * H, execute_percent=100)
    selective = BimanualArcDecoder(
        "e1_dur", 0.4, M, DT, 2 * H, speed=2.0, hold_speed=1.0, execute_percent=100
    )
    uniform = BimanualArcDecoder(
        "e1_dur", 0.4, M, DT, 2 * H, speed=2.0, hold_speed=2.0, execute_percent=100
    )

    def hold_ticks(plan):
        return int(((plan[:, 0] > 0.39 * 0.4) & (plan[:, 0] < 0.46 * 0.4)).sum())

    base = decoder(token)[0]
    held = hold_ticks(base)
    assert held >= 25  # the 1.0 s hold is ~30 ticks of the demonstrated replay
    assert hold_ticks(selective(token)[0]) == pytest.approx(held, abs=6)
    assert hold_ticks(uniform(token)[0]) == pytest.approx(held / 2, abs=4)
    # The hold share is a property of the token's clock, not of the replay tempo.
    assert selective.last_stats["hold_fraction"] == pytest.approx(
        uniform.last_stats["hold_fraction"]
    )
    assert 0.2 < selective.last_stats["hold_fraction"] < 0.5


@pytest.mark.parametrize(
    "layout", ["e1_durhyb", "e1_profhyb", "e1_durtri", "e1_proftri"]
)
def test_first_stream_ends_the_chunk_where_the_fastest_stream_reaches_its_cap(layout):
    decoder, token = _training_token(layout, _chunk(seed=4), first_stream=1)
    plan = decoder(token)
    stats = decoder.last_stats
    assert stats["execute_waypoints"] == M // 2
    ends = stats["stream_ends_s"]
    assert len(ends) >= 2 and all(e > 0 for e in ends)
    assert stats["replan_steps"] == min(H, int(np.ceil(min(ends) / DT - 1e-9)))
    assert (
        len(execution_prefix(plan[0], decoder=decoder, replan_every=7))
        == stats["replan_steps"]
    )
    # The capped token holds every stream at its own cap: the half-way waypoint.
    xyz = ARM_LAYOUT[0][0]
    half = token[M // 2 - 1, xyz : xyz + 3]
    assert np.linalg.norm(plan[0, -1, xyz : xyz + 3] - half) < 1e-6
    decoder.first_stream = 0
    whole = decoder(token)
    assert len(execution_prefix(whole[0], decoder=decoder, replan_every=7)) == 7
    np.testing.assert_allclose(whole[0], decoder.codec.detokenize(token, H), atol=1e-12)


def test_tri_gripper_stream_can_end_the_chunk_and_an_idle_one_cannot():
    decoder, token = _training_token("e1_durtri", _chunk(seed=5), first_stream=1)
    decoder(token)
    with_gripper = len(decoder.last_stats["stream_ends_s"])
    still = token.copy()
    still[:, 6] = 0.8  # left gripper held: its stream must not end the chunk
    still[:, 18] = 0.0
    decoder(still)
    assert len(decoder.last_stats["stream_ends_s"]) == with_gripper - 1


def test_absolute_duration_timing_decodes_and_terminates_with_the_training_codec():
    decoder, token = _training_token(
        "e1_dur", _chunk(seed=6), duration_timing="absolute", first_stream=1
    )
    assert decoder.codec.duration_timing == "absolute"
    plan = decoder(token)
    assert plan.shape == (1, H, 14) and np.isfinite(plan).all()
    assert decoder.last_stats["replan_steps"] >= 1
    with pytest.raises(ValueError, match="e1_dur"):
        BimanualArcDecoder("e1_profile", duration_timing="absolute")


@pytest.mark.parametrize("layout", ["e1_durevt", "e1_stkevt", "e1_durref", "e1_stkref"])
def test_event_and_reference_layouts_terminate_at_their_translation_or_rotation_stream(
    layout,
):
    decoder, token = _training_token(layout, _chunk(seed=7), first_stream=1)
    plan = decoder(token)
    stats = decoder.last_stats
    ends = stats["stream_ends_s"]
    assert len(ends) >= 2
    assert 1 <= stats["replan_steps"] <= H
    assert plan.shape == (1, H, 14) and np.isfinite(plan).all()
    # Past every stream's end the decoded rows repeat the capped pose.
    last = int(np.ceil(max(ends) / DT)) + 1
    if last < H - 1:
        np.testing.assert_allclose(plan[0, last:], plan[0, -1], atol=1e-9)


@pytest.mark.parametrize("layout", ["e1_projection_dur", "e1_projection_vel"])
def test_projection_layouts_decode_with_the_projection_codec(layout):
    decoder, token = _training_token(layout, _chunk(seed=8, turn=1.2), first_stream=1)
    plan = decoder(token)
    assert plan.shape == (1, H, 14) and decoder.last_stats["replan_steps"] >= 1
    decoder.set_speed(2.0)
    fast = decoder(token)[0]
    valid = min(decoder.last_stats["valid_steps"], H // 2)
    decoder.set_speed(1.0)
    np.testing.assert_allclose(
        fast[:valid], decoder(token)[0][: 2 * valid : 2], atol=1e-7
    )


def test_hybrid_rotation_budget_and_speed_floors_reach_the_training_codec():
    decoder = BimanualArcDecoder(
        "e1_profhyb", rotation_distance_unit=1.2566, speed_floors=(0.02, 0.1, 0.05)
    )
    assert decoder.codec.rotation_distance_unit == pytest.approx(1.2566)
    assert decoder.codec.speed_floors == (0.02, 0.1, 0.05)
    with pytest.raises(ValueError, match="E1 layouts"):
        BimanualArcDecoder("lab", speed_floors=(0.02, 0.1, 0.05))
    with pytest.raises(ValueError, match="hybrid token layout"):
        BimanualArcDecoder("e1_dur", rotation_distance_unit=1.0)


def test_first_stream_decoder_uses_the_training_multistream_codec():
    dt, waypoints = 1 / 30, 100
    t = np.arange(400)[:, None] * dt
    actions = np.zeros((400, 14))
    actions[:, 0] = t[:, 0] * 0.5
    actions[:, 8] = t[:, 0] * 0.1
    actions[:, 3] = actions[:, 10] = t[:, 0] * 0.2
    actions[:, [6, 13]] = 0.5
    decoder = FirstStreamArcDecoder("duration", "clock", 0.7, 2.0, waypoints, dt, H)
    assert type(decoder.codec) is TrainingCartesianCodec
    wide = decoder._lib.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.7,
        rotation_distance_unit=2.0,
        resampled_vector_length=waypoints,
        dt=dt,
        velocity_mode="duration",
        velocity_layout="clock",
        arc_chunking_mode="multistream",
    )
    native = wide.transform({"actions_cartesian": actions.copy()})["actions_cartesian"]
    assert native.shape == (waypoints, 18) == decoder.shape
    prediction = decoder(native)
    n = decoder.last_stats["replan_steps"]
    assert 1 <= n <= H and prediction.shape == (1, H, 14)
    assert np.all(prediction[0, n:] == prediction[0, n - 1])


def test_every_declared_layout_is_constructible_with_its_natural_codec():
    for layout in ARC_TOKEN_LAYOUTS:
        decoder = BimanualArcDecoder(layout, 0.4, 20, DT, 40)
        assert decoder.shape[0] in (20, 21, 40) and decoder.M == 20
        rows = np.zeros(decoder.shape)
        rows[..., [6, 13]] = 0.5
        if layout == "lab":
            rows[0, :] = 0.01
        plan = decoder(rows[None] if rows.ndim == 2 else rows)
        assert plan.shape == (1, 40, 14) and np.isfinite(plan).all()


def test_rotation_only_turn_is_held_by_single_clock_layouts_and_moves_under_projection():
    chunk = _chunk(seed=10, turn=1.0)
    for arm in range(2):
        off = ARM_LAYOUT[arm][0]
        chunk[:, off : off + 3] = chunk[0, off : off + 3]  # no translation at all
    plain, token = _training_token("e1_dur", chunk)
    assert (
        plain.codec.stream_ends(token, H) == []
    )  # nothing moves on a translation clock
    proj, token = _training_token("e1_projection_dur", chunk, first_stream=1)
    assert len(proj.codec.stream_ends(token, H)) == 2  # the projected point turns
    plan = proj(token)[0]
    turned = (
        Rotation.from_euler("ZYX", plan[-1, 3:6])
        * Rotation.from_euler("ZYX", plan[0, 3:6]).inv()
    )
    assert turned.magnitude() > 0.2
