"""Six-clock ARC token layout, hold semantics, and cotraining recipes."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from egomimic.eval import distance_budget_dtw as dtw
from egomimic.eval.open_loop_sim import (
    OpenLoopSimEval,
    truncate_cartesian_trajectory_by_arc_mode,
)
from egomimic.rldb.zarr.arc_length_tokenizer import (
    CLOCK_COLUMNS,
    SIX_CLOCK_COLUMNS,
    TokenizeBimanualArcLengthCartesian,
    bimanual_arc_token_shapes,
    clock_token_seconds,
    stack_arc_token,
)

DT = 1.0 / 30.0
OPTIONS = dict(
    action_key="actions",
    output_action_key="actions",
    min_distance_unit=0.42,
    rotation_distance_unit=0.4188790204786391,
    resampled_vector_length=100,
    dt=DT,
)


def chunk(frames=200, right_still=False):
    """Arms moving at different rates, rotating, and operating grippers."""
    time = np.arange(frames) * DT
    raw = np.zeros((frames, 14))
    raw[:, 0], raw[:, 1] = 0.25 * np.sin(time), 0.1 * time
    raw[:, 3], raw[:, 6] = 0.3 * time, np.clip(time - 2, 0, 1)
    if not right_still:
        raw[:, 7] = 0.05 * np.minimum(time, 1.5)
        raw[:, 10], raw[:, 13] = 0.05 * np.sin(3 * time), np.clip(4 - time, 0, 1)
    return raw


def codec(mode, velocity_mode="duration", velocity_layout="clock"):
    return TokenizeBimanualArcLengthCartesian(
        **OPTIONS,
        gripper_distance_unit=0.3,
        arc_chunking_mode=mode,
        velocity_mode=velocity_mode,
        velocity_layout=velocity_layout,
    )


def six_codec(velocity_mode="duration", velocity_layout="log_clock6"):
    return codec("multistream", velocity_mode, velocity_layout)


def test_six_independent_clocks_use_the_canonical_twenty_dimensional_layout():
    raw = chunk(frames=400)
    seconds = six_codec("duration", "log_clock6").transform({"actions": raw.copy()})[
        "actions"
    ]
    assert seconds.shape == (100, 20)
    assert np.isfinite(seconds).all()
    decoded_seconds = clock_token_seconds(seconds, "log_clock6")
    stacked = stack_arc_token(decoded_seconds)
    assert stacked.shape == (200, 14)
    np.testing.assert_allclose(
        stacked[100:, list(SIX_CLOCK_COLUMNS)], decoded_seconds[:, 14:], atol=1e-10
    )
    rates = six_codec("per_waypoint", "rate_clock").transform({"actions": raw.copy()})[
        "actions"
    ]
    assert rates.shape == (100, 20)
    assert (100, 20) in bimanual_arc_token_shapes(100, "duration")
    assert (100, 20) in bimanual_arc_token_shapes(100, "per_waypoint")


def test_six_clock_hold_waypoints_are_constant_and_rate_clocks_are_zero():
    raw = np.zeros((400, 14), dtype=np.float64)
    logged = six_codec("duration", "log_clock6").transform({"actions": raw.copy()})[
        "actions"
    ]
    seconds = clock_token_seconds(logged, "log_clock6")
    np.testing.assert_array_equal(logged[:, :14], np.zeros((100, 14)))
    np.testing.assert_allclose(seconds[:-1, 14:], (400 * DT) / 99.0, atol=1e-10)
    rates = six_codec("per_waypoint", "rate_clock").transform({"actions": raw.copy()})[
        "actions"
    ]
    np.testing.assert_array_equal(rates[:, 14:], 0.0)
    np.testing.assert_allclose(
        six_codec("duration", "log_clock6").detokenize(logged, 400), raw
    )


def test_six_log_evaluator_decodes_physical_seconds_exactly_once():
    raw = np.zeros((400, 14))
    raw[:, 0] = raw[:, 7] = np.arange(400) * DT * 0.1
    raw[:, 3] = raw[:, 10] = np.arange(400) * DT * 0.2
    cod = six_codec()
    logged = cod.transform({"actions": raw})["actions"]
    seconds = clock_token_seconds(logged, "log_clock6")
    ev = evaluator("duration", velocity_layout="log_clock6", gripper_distance_unit=0.3)
    actual, steps = ev._decode_prediction_with_steps(seconds, max_steps=400)
    expected = cod.detokenize(logged[:30], steps)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(
        cod.detokenize(seconds[:30], steps, clocks_in_seconds=True), expected, atol=1e-9
    )
    assert actual[-1, 0] == pytest.approx(0.06)


def test_six_rate_targets_use_emitted_waypoint_distances():
    time = np.arange(400) * DT
    raw = np.zeros((400, 14))
    for offset in (0, 7):
        raw[:, offset] = 0.03 * np.sin(time * 8)
        raw[:, offset + 1] = 0.02 * np.cos(time * 7)
        raw[:, offset + 3] = 0.1 * np.sin(time * 9)
        raw[:, offset + 4] = 0.08 * np.cos(time * 6)
        raw[:, offset + 6] = 0.5 + 0.1 * np.sin(time * 4)
    rate_codec = six_codec("per_waypoint", "rate_clock")
    rates = rate_codec.transform({"actions": raw.copy()})["actions"]
    seconds = clock_token_seconds(
        six_codec().transform({"actions": raw})["actions"], "log_clock6"
    )
    stacked = stack_arc_token(rates)
    for col, (kind, offset) in enumerate(
        (kind, offset)
        for kind in ("translation", "rotation", "gripper")
        for offset in (0, 7)
    ):
        decoded = rate_codec._six_stream_duration(
            stacked[:100], stacked[100:], offset, kind, 400
        )
        np.testing.assert_allclose(decoded, seconds[:-1, 14 + col], atol=1e-9)


@pytest.mark.parametrize(
    "mode,layout", [("duration", "log_clock6"), ("per_waypoint", "rate_clock")]
)
def test_event_grippers_preserve_commands_and_event_times(mode, layout):
    cod = TokenizeBimanualArcLengthCartesian(
        **OPTIONS,
        velocity_mode=mode,
        velocity_layout=layout,
        gripper_stream_mode="events",
        arc_chunking_mode="multistream",
    )
    raw = np.zeros((400, 14))
    raw[30:70, 6] = 1.0
    raw[70:, 6] = -1.0
    raw[90:, 13] = 0.75
    token = cod.transform({"actions": raw})["actions"]
    # PR #202: initial support, actual command changes, final support, padding.
    np.testing.assert_array_equal(token[:4, 6], [0, 1, -1, -1])
    decoded = cod.detokenize(token, 400)
    np.testing.assert_array_equal(decoded[:, [6, 13]], raw[:, [6, 13]])
    assert set(decoded[:, 6]) == {-1.0, 0.0, 1.0}


@pytest.mark.parametrize(
    "mode,layout", [("duration", "log_clock6"), ("per_waypoint", "rate_clock")]
)
def test_all_six_event_streams_hold_for_400_frames(mode, layout):
    cod = TokenizeBimanualArcLengthCartesian(
        **OPTIONS,
        velocity_mode=mode,
        velocity_layout=layout,
        gripper_stream_mode="events",
        arc_chunking_mode="multistream",
    )
    token = cod.transform({"actions": np.zeros((400, 14))})["actions"]
    native = clock_token_seconds(token, layout)
    ev = evaluator(mode, velocity_layout=layout, gripper_stream_mode="events")
    decoded, steps = ev._decode_prediction_with_steps(native, max_steps=1000)
    assert steps == 400
    np.testing.assert_array_equal(decoded, np.zeros((400, 14)))


def test_event_keymaps_read_full_window_after_motion_caps():
    from egomimic.rldb.embodiment.human import Human
    from egomimic.rldb.embodiment.yam import Yam

    for embodiment in (Human, Yam):
        keys = embodiment.get_keymap(
            "hybrid_arc_tokenizer_cartesian",
            source_buffer_frames=400,
            min_distance_unit=0.42,
            rotation_distance_unit=0.42,
            arc_chunking_mode="multistream",
            gripper_stream_mode="events",
        )
        for spec in keys.values():
            if spec.get("key_type") == "action_keys":
                assert spec["horizon"] == 400


def test_gripper_stream_can_end_the_executed_chunk_first():
    raw = np.zeros((400, 14), dtype=np.float64)
    raw[:6, 6] = np.linspace(0.0, 0.3, 6)
    raw[6:, 6] = 0.3
    token = six_codec("duration", "log_clock6").transform({"actions": raw.copy()})[
        "actions"
    ]
    steps = evaluator(
        "duration", velocity_layout="log_clock6", gripper_distance_unit=0.3
    )
    _, predicted_steps = steps._decode_prediction_with_steps(
        clock_token_seconds(token, "log_clock6"), max_steps=400
    )
    assert predicted_steps <= 3


def test_multistream_video_prefix_stops_at_first_gripper_distance_cap():
    trajectory = np.zeros((10, 14), dtype=np.float64)
    trajectory[:, 6] = np.linspace(0.0, 0.9, 10)
    trajectory[:, 13] = np.linspace(0.0, 0.6, 10)
    capped = truncate_cartesian_trajectory_by_arc_mode(
        trajectory,
        1.0,
        "multistream",
        max_rotation_distance=0.5,
        max_gripper_distance=0.3,
    )
    assert len(capped) == 4
    assert capped[-1, 6] == pytest.approx(0.3)
    assert capped[-1, 13] == pytest.approx(0.2)


@pytest.mark.parametrize("mode", ["multistream", "race"])
def test_clock_token_decodes_exactly_like_per_waypoint_rates(mode):
    raw = chunk()
    clock = codec(mode).transform({"actions": raw.copy()})["actions"]
    rates = codec(mode, "per_waypoint", "wide").transform({"actions": raw.copy()})[
        "actions"
    ]
    assert clock.shape == (100, 18)
    np.testing.assert_array_equal(clock[:, :14], rates[:, :14])
    np.testing.assert_allclose(
        codec(mode).detokenize(clock, 200),
        codec(mode, "per_waypoint", "wide").detokenize(rates, 200),
        atol=1e-9,
    )


def test_clock_columns_are_per_interval_seconds_and_restack_losslessly():
    clock = codec("multistream").transform({"actions": chunk()})["actions"]
    seconds = clock[:, 14:]
    assert np.all(seconds >= 0)
    # Each clock's intervals sum to at most the 200-frame source window.
    assert np.all(seconds[:-1].sum(axis=0) <= 200 * DT + 1e-9)
    stacked = stack_arc_token(clock)
    assert stacked.shape == (200, 14)
    np.testing.assert_array_equal(stacked[100:, list(CLOCK_COLUMNS)], seconds)
    other = [c for c in range(14) if c not in CLOCK_COLUMNS]
    assert not stacked[100:, other].any()
    stacked_codec = codec("multistream", velocity_layout="stacked")
    np.testing.assert_allclose(
        stacked_codec.detokenize(stacked, 200),
        codec("multistream").detokenize(clock, 200),
    )
    assert (100, 18) in bimanual_arc_token_shapes(100, "duration")
    assert (100, 18) not in bimanual_arc_token_shapes(100, "per_waypoint")


def test_held_arm_has_constant_waypoints_and_a_full_window_duration():
    raw = chunk(frames=400, right_still=True)
    clock = codec("multistream").transform({"actions": raw})["actions"]
    np.testing.assert_array_equal(clock[:, 7:14], np.repeat(raw[:1, 7:14], 100, 0))
    right_translation_seconds = clock[:-1, 14 + CLOCK_COLUMNS.index(7)]
    assert right_translation_seconds.sum() == pytest.approx(400 * DT)
    right_rotation_seconds = clock[:-1, 14 + CLOCK_COLUMNS.index(10)]
    assert right_rotation_seconds.sum() == pytest.approx(400 * DT)
    decoded = codec("multistream").detokenize(clock, 400)
    np.testing.assert_allclose(decoded[:, 7:14], np.repeat(raw[:1, 7:14], 400, 0))


def test_zero_rate_hold_decodes_for_the_complete_source_window():
    raw = chunk(frames=400, right_still=True)
    rates_codec = codec("multistream", "per_waypoint", "wide")
    rates = rates_codec.transform({"actions": raw.copy()})["actions"]
    np.testing.assert_array_equal(rates[:, 21:28], 0.0)
    decoded = rates_codec.detokenize(rates, 400)
    np.testing.assert_allclose(decoded[:, 7:14], np.repeat(raw[:1, 7:14], 400, 0))


def test_log_clock_stores_log_seconds_and_decodes_like_clock():
    import torch

    raw = chunk(right_still=True)
    clock = codec("multistream").transform({"actions": raw.copy()})["actions"]
    logged = codec("multistream", velocity_layout="log_clock").transform(
        {"actions": raw.copy()}
    )["actions"]
    assert logged.shape == clock.shape == (100, 18)
    np.testing.assert_array_equal(logged[:, :14], clock[:, :14])
    assert np.isfinite(logged).all()
    # A zero (hold) interval encodes to a finite log and decodes back to zero.
    held = np.zeros((2, 18))
    held[:, 14:] = np.log(1e-3)
    np.testing.assert_allclose(
        clock_token_seconds(held, "log_clock")[:, 14:], 0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        clock_token_seconds(logged, "log_clock"), clock, atol=1e-12
    )
    np.testing.assert_array_equal(clock_token_seconds(logged, "clock"), logged)
    np.testing.assert_allclose(
        codec("multistream", velocity_layout="log_clock").detokenize(logged, 200),
        codec("multistream").detokenize(clock, 200),
        atol=1e-9,
    )
    # The evaluator undoes the log once, at unnormalization.
    ev = OpenLoopSimEval.__new__(OpenLoopSimEval)
    ev.velocity_layout = "log_clock"
    np.testing.assert_allclose(
        ev._clock_seconds(torch.from_numpy(logged)).numpy(), clock, atol=1e-12
    )
    ev.velocity_layout = "clock"
    assert ev._clock_seconds(torch.from_numpy(clock)).numpy() is not None
    with pytest.raises(ValueError, match="velocity_mode='duration'"):
        codec("multistream", velocity_mode="per_waypoint", velocity_layout="log_clock")


def test_gripper_twitch_does_not_stretch_a_moving_arms_clock():
    raw = chunk()
    rates_codec = codec("multistream", "per_waypoint", "wide")
    token = rates_codec.transform({"actions": raw.copy()})["actions"]
    clean = rates_codec.detokenize(token, 200)
    # A model-style twitch: tiny gripper changes with near-zero gripper rates
    # on intervals where the arm is moving.
    noisy = token.copy()
    noisy[:, 6] += 1e-3 * (np.arange(100) % 2)
    noisy[:, 14 + 6] = 1e-4
    decoded = rates_codec.detokenize(noisy, 200)
    np.testing.assert_allclose(decoded[:, :3], clean[:, :3], atol=1e-9)


def test_joint_distance_cannot_use_per_arm_duration_clocks():
    with pytest.raises(ValueError, match="joint_distance"):
        codec("joint_distance")


def evaluator(velocity_mode, **extra):
    return OpenLoopSimEval(
        action_mode="arc",
        arc_chunking_mode="multistream",
        execute_fraction=0.3,
        min_distance_unit=OPTIONS["min_distance_unit"],
        rotation_distance_unit=OPTIONS["rotation_distance_unit"],
        resampled_vector_length=100,
        control_dt=DT,
        velocity_mode=velocity_mode,
        **extra,
    )


@pytest.mark.parametrize("cap", ["waypoints", "distance"])
def test_evaluator_executes_the_same_prefix_from_either_timing_form(cap):
    raw = chunk()
    clock = codec("multistream").transform({"actions": raw.copy()})["actions"]
    rates = codec("multistream", "per_waypoint", "wide").transform(
        {"actions": raw.copy()}
    )["actions"]
    by_clock, clock_steps = evaluator(
        "duration", arc_execution_cap_mode=cap
    )._decode_prediction_with_steps(clock, max_steps=100)
    by_rate, rate_steps = evaluator(
        "per_waypoint", arc_execution_cap_mode=cap
    )._decode_prediction_with_steps(rates, max_steps=100)
    assert clock_steps == rate_steps
    np.testing.assert_allclose(by_clock, by_rate, atol=1e-9)


def test_dtw_scores_oracle_clock_tokens_near_zero():
    time = np.arange(90 + 200) * DT
    raw = np.zeros((len(time), 14))
    raw[:, 0], raw[:, 7] = 0.30 * time, 0.08 * time
    raw[:, 3] = raw[:, 10] = 0.2 * time
    tokenize = codec("multistream")
    records = [
        dict(
            frame=frame,
            ground_truth=raw[frame : frame + 200],
            prediction=tokenize.transform({"actions": raw[frame : frame + 200].copy()})[
                "actions"
            ],
            **{dtw.METRIC_FRAME_KEY: np.repeat(np.eye(4)[None], 2, axis=0)},
        )
        for frame in range(90)
    ]
    result = dtw.score_distance_dtw_episode(evaluator("duration"), records)
    assert result["xyz_mse"] < 1e-4
    # Multistream replays the episode once, replanning where each chunk ends,
    # so the rollout spans exactly the recorded episode.
    assert result["duration_ratio"] == pytest.approx(1.0)


def test_fractional_prefix_scales_stored_seconds_with_the_waypoint_fraction():
    raw = chunk()
    clock = stack_arc_token(
        codec("multistream").transform({"actions": raw.copy()})["actions"]
    )
    rates = stack_arc_token(
        codec("multistream", "per_waypoint", "wide").transform({"actions": raw.copy()})[
            "actions"
        ]
    )
    partial_clock = dtw.fractional_waypoint_prefix(clock, 0.335, "duration")
    partial_rate = dtw.fractional_waypoint_prefix(rates, 0.335)
    stacked = dict(velocity_layout="stacked")
    np.testing.assert_allclose(
        codec("multistream", **stacked).detokenize(partial_clock, 120),
        codec("multistream", "per_waypoint", **stacked).detokenize(partial_rate, 120),
        atol=1e-9,
    )


_CONFIGS = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"


def _cfg(name, monkeypatch):
    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/tmp/arc_abc")
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=abc_arc/cotrain/{name}", "++paths.root_dir=."],
        )


@pytest.mark.parametrize(
    "name,dim,evaluator_mode",
    [
        ("organize_rl2_elmo_hpt300_cotrain_baseline", 14, "baseline"),
        ("organize_rl2_elmo_hpt300_cotrain_multistream", 20, "arc"),
        ("organize_rl2_elmo_hpt300_cotrain_multistream_logclock", 20, "arc"),
        ("organize_rl2_elmo_hpt300_cotrain_multistream_perwaypoint", 20, "arc"),
    ],
)
def test_cotrain_recipes_compose(name, dim, evaluator_mode, monkeypatch):
    cfg = _cfg(name, monkeypatch)
    stages = cfg.model.pipeline.stages
    stem = stages[0].stems["observations.images.front_img_1"]._target_
    assert stem.endswith("ResNetMLPImageStem")
    assert list(stages[1].domains) == ["human_bimanual", "yam_bimanual"]
    assert stages[-3].action_horizon == 100 and stages[-3].action_dim == dim
    assert (cfg.hpt.embed_dim, cfg.hpt.num_blocks) == (840, 19)
    assert cfg.evaluator.action_mode == evaluator_mode
    assert cfg.abc.arc_distance == 0.42
    assert cfg.abc.arc_rotation_distance == pytest.approx(np.radians(24))
    train = cfg.data.train_datasets
    assert set(train) == {"yam_bimanual", "human_bimanual"}
    human = train.human_bimanual.resolver.key_map
    assert (human.action_horizon, human.source_buffer_frames) == (
        100,
        200 if evaluator_mode == "baseline" else 400,
    )
    robot_keymap = train.yam_bimanual.resolver.key_map
    assert robot_keymap.source_buffer_frames == 400
    assert train.human_bimanual.resolver.transform_list.stride == 1
    row = dict(
        lab="rl2",
        task="organize_stationary",
        operator="Elmo",
        zarr_processed_path="x",
        is_deleted=False,
    )
    for embodiment, dataset in train.items():
        (rule,) = dataset.filters.filter_lambdas
        keep = eval(rule)
        assert keep({**row, "embodiment": embodiment})
        assert not keep({**row, "embodiment": embodiment, "operator": "Aidan"})
        assert not keep({**row, "embodiment": embodiment, "lab": "abc"})
        assert dataset.valid_ratio == 0.05
    if name.endswith("_perwaypoint"):
        assert cfg.evaluator.velocity_mode == "per_waypoint"
        assert cfg.abc.arc_velocity_layout == "rate_clock"
        for dataset in train.values():
            transforms = dataset.resolver.transform_list
            assert (transforms.velocity_mode, transforms.velocity_layout) == (
                "per_waypoint",
                "rate_clock",
            )
    if not name.endswith("_perwaypoint") and dim == 20:
        assert cfg.evaluator.velocity_mode == "duration"
        assert cfg.evaluator.arc_chunking_mode == "multistream"
        robot = train.yam_bimanual.resolver.transform_list
        layout = "log_clock6"
        assert (robot.velocity_mode, robot.velocity_layout) == ("duration", layout)
        human = train.human_bimanual.resolver.transform_list
        assert human.velocity_layout == layout
        assert cfg.evaluator.velocity_layout == layout
        assert robot.preserve_action_rows == 400
        assert human.preserve_action_rows == 400


def test_towel_logclock_recipe_tokenizes_and_evaluates_as_log_clock(monkeypatch):
    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/tmp/arc_abc")
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=abc_arc/robot_bc/abc_rl2_towels_abcdit_multistream_logclock_visual_openloop",
                "++paths.root_dir=.",
            ],
        )
    robot = cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    assert (robot.velocity_mode, robot.velocity_layout) == ("duration", "log_clock")
    assert (cfg.evaluator.velocity_mode, cfg.evaluator.velocity_layout) == (
        "duration",
        "log_clock",
    )
    assert cfg.abc_dit.token_dim == 18


def test_dtw_segment_cap_clips_and_reports_instead_of_raising():
    """A moving interval with near-zero predicted time must not kill validation."""
    time = np.arange(60 + 200) * DT
    raw = np.zeros((len(time), 14))
    raw[:, 0], raw[:, 7] = 0.30 * time, 0.08 * time
    raw[:, 3] = raw[:, 10] = 0.2 * time
    tokenize = codec("multistream")
    records = []
    for frame in range(60):
        token = tokenize.transform({"actions": raw[frame : frame + 200].copy()})[
            "actions"
        ]
        if frame == 0:
            token[:, 14:] = 50.0  # a nonsense 50 s per interval: huge decode
        if frame == 1:
            token[:, 14:] = 0.0  # no usable time on moving intervals: unbounded
        records.append(
            dict(
                frame=frame,
                ground_truth=raw[frame : frame + 200],
                prediction=token,
                **{dtw.METRIC_FRAME_KEY: np.repeat(np.eye(4)[None], 2, axis=0)},
            )
        )
    # Race keeps per-window decodes, where one stalled token is unbounded.
    tokenize = codec("race")
    records = [
        dict(
            r,
            prediction=tokenize.transform(
                {"actions": raw[r["frame"] : r["frame"] + 200].copy()}
            )["actions"],
        )
        for r in records
    ]
    for frame, value in ((0, 50.0), (1, 0.0)):
        records[frame]["prediction"][:, 14:] = value

    def evaluator_race(**extra):
        return OpenLoopSimEval(
            action_mode="arc",
            arc_chunking_mode="race",
            execute_fraction=0.3,
            min_distance_unit=OPTIONS["min_distance_unit"],
            rotation_distance_unit=OPTIONS["rotation_distance_unit"],
            resampled_vector_length=100,
            control_dt=DT,
            velocity_mode="duration",
            **extra,
        )

    strict = evaluator_race(dtw_max_prediction_steps=1000)
    with pytest.raises(ValueError, match="dtw_max_prediction_steps"):
        dtw.score_distance_dtw_episode(strict, records[:1] + records[2:])
    capped = evaluator_race(dtw_max_prediction_steps=1000, dtw_max_segment_steps=300)
    result = dtw.score_distance_dtw_episode(capped, records)
    assert result["clipped_segments"] >= 1  # frame 0 is always a race anchor
    assert max(result["segment_control_steps"]) <= 300
    assert np.isfinite(result["xyz_mse"])


def test_yam_keymap_source_buffer_override():
    from egomimic.rldb.embodiment.yam import Yam

    keymap = Yam.get_keymap(
        "hybrid_arc_tokenizer_cartesian",
        min_distance_unit=0.42,
        rotation_distance_unit=0.42,
        arc_chunking_mode="multistream",
        source_buffer_frames=400,
    )
    horizons = [
        v["horizon"] for v in keymap.values() if isinstance(v.get("horizon"), dict)
    ]
    assert horizons and all(h["source_buffer_frames"] == 400 for h in horizons)
    default = Yam.get_keymap(
        "hybrid_arc_tokenizer_cartesian",
        min_distance_unit=0.42,
        rotation_distance_unit=0.42,
        arc_chunking_mode="multistream",
    )
    assert all(
        v["horizon"]["source_buffer_frames"] == Yam.ARC_SOURCE_BUFFER_FRAMES
        for v in default.values()
        if isinstance(v.get("horizon"), dict)
    )


def test_gt_chunk_in_anchor_frame_preserves_world_pose():
    from scipy.spatial.transform import Rotation

    from egomimic.eval.open_loop_sim import gt_chunk_in_anchor_frame

    rng = np.random.default_rng(0)

    def pose():
        value = np.eye(4)
        value[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        value[:3, 3] = rng.normal(size=3)
        return value

    frames = np.stack([[pose(), pose()] for _ in range(6)])
    rows = rng.normal(size=(6, 14)) * 0.2
    chunk = gt_chunk_in_anchor_frame(rows, frames, frames[2])
    for i in range(6):
        for arm, offset in enumerate((0, 7)):
            own = frames[i, arm] @ np.r_[rows[i, offset : offset + 3], 1]
            via_anchor = frames[2, arm] @ np.r_[chunk[i, offset : offset + 3], 1]
            np.testing.assert_allclose(own, via_anchor, atol=1e-9)
            own_rot = (
                frames[i, arm, :3, :3]
                @ Rotation.from_euler(
                    "ZYX", rows[i, offset + 3 : offset + 6]
                ).as_matrix()
            )
            anchor_rot = (
                frames[2, arm, :3, :3]
                @ Rotation.from_euler(
                    "ZYX", chunk[i, offset + 3 : offset + 6]
                ).as_matrix()
            )
            np.testing.assert_allclose(own_rot, anchor_rot, atol=1e-9)
        np.testing.assert_allclose(chunk[i, [6, 13]], rows[i, [6, 13]])


def test_open_loop_executes_long_chunks_against_episode_ground_truth():
    """Chunks are not cut at the 100-row preserved GT copy (here only 20 rows)."""
    time = np.arange(300 + 400) * DT
    raw = np.zeros((len(time), 14))
    raw[:, 0], raw[:, 7] = 0.05 * time, 0.02 * time  # slow: chunks span >20 frames
    tokenize = codec("multistream")
    ev = evaluator("duration")
    ev.require_episode_start = True
    eye = np.repeat(np.eye(4)[None], 2, axis=0)
    records = []
    for frame in range(300):
        anchors = eye.copy()
        anchors[:, :3, 3] = raw[frame, [0, 1, 2]], raw[frame, [7, 8, 9]]
        window = raw[frame : frame + 400] - raw[frame]
        records.append(
            dict(
                group="valid",
                source="yam_bimanual",
                label="yam_bimanual",
                episode="e",
                frame=frame,
                ground_truth=window[:20],
                prediction=tokenize.transform({"actions": window.copy()})["actions"],
                **{dtw.METRIC_FRAME_KEY: anchors},
            )
        )
    result = ev._score_episode(records)
    assert max(result["segment_control_steps"]) > 20
    assert result["executed_steps"] == 300
    assert result["metrics"]["xyz_mse"] < 1e-8
