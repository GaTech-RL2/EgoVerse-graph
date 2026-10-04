"""Four-clock duration ARC token, (M, 18), and the cotraining recipes that use it."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from egomimic.eval import distance_budget_dtw as dtw
from egomimic.eval.open_loop_sim import OpenLoopSimEval
from egomimic.rldb.zarr.arc_length_tokenizer import (
    CLOCK_COLUMNS,
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
        arc_chunking_mode=mode,
        velocity_mode=velocity_mode,
        velocity_layout=velocity_layout,
    )


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


def test_held_arm_parks_xyz_but_times_its_gripper_on_the_translation_clock():
    raw = chunk(right_still=True)
    clock = codec("multistream").transform({"actions": raw})["actions"]
    np.testing.assert_array_equal(clock[:, 7:10], np.repeat(raw[:1, 7:10], 100, 0))
    right_translation_seconds = clock[:-1, 14 + CLOCK_COLUMNS.index(7)]
    assert right_translation_seconds.sum() > 0
    decoded = codec("multistream").detokenize(clock, 200)
    np.testing.assert_allclose(decoded[:, 7:10], raw[:1, 7:10].repeat(200, 0))
    # The hold walks the gripper to its state at the end of the covered span
    # over that span's duration, instead of jumping at frame 0.
    assert decoded[0, 13] == pytest.approx(1.0)
    assert np.all(np.diff(decoded[:, 13]) <= 1e-12)
    assert decoded[-1, 13] == pytest.approx(raw[-1, 13])


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
    np.testing.assert_allclose(clock_token_seconds(held, "log_clock")[:, 14:], 0.0, atol=1e-15)
    np.testing.assert_allclose(clock_token_seconds(logged, "log_clock"), clock, atol=1e-12)
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
        ("organize_rl2_elmo_hpt300_cotrain_multistream", 18, "arc"),
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
    assert (human.action_horizon, human.source_buffer_frames) == (100, 200)
    robot_keymap = train.yam_bimanual.resolver.key_map
    assert robot_keymap.source_buffer_frames == 200
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
    if dim == 18:
        assert cfg.evaluator.velocity_mode == "duration"
        assert cfg.evaluator.arc_chunking_mode == "multistream"
        robot = train.yam_bimanual.resolver.transform_list
        assert (robot.velocity_mode, robot.velocity_layout) == ("duration", "clock")


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
