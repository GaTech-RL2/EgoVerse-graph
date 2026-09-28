"""Mode-aware evaluator budgets, clocks, overlays, and sweep provenance."""

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from egomimic.eval import distance_budget_dtw as dtw
from egomimic.eval.open_loop_sim import (
    OpenLoopSimEval,
    arc_execution_prefix,
    arc_prefix_control_steps,
    truncate_cartesian_trajectory_by_arc_mode,
)
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    cumulative_rotation_length,
)
from scripts.eval import validate_checkpoint_sweep as sweep


def xyz(left, right):
    values = np.zeros((len(left), 6))
    values[:, 0], values[:, 3] = left, right
    return values


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("joint_distance", [0, 1, 2, 3, 4]),
        ("race", [0, 1, 1, 2, 2]),
        ("multistream", [0, 0, 1, 1, 2]),
    ],
)
def test_episode_progress_accumulates_each_arm_before_reducing(mode, expected):
    values = xyz([0, 1, 1, 2, 2], [0, 0, 1, 1, 2])
    np.testing.assert_allclose(dtw.translation_progress(values, mode), expected)


def test_race_windows_reset_when_winning_arm_alternates():
    values = xyz([0, 1, 1, 2, 2], [0, 0, 1, 1, 2])
    anchors, budgets = dtw.chunk_distance_windows(values, 1, "race")
    np.testing.assert_array_equal(anchors, [0, 1, 2, 3])
    np.testing.assert_allclose(budgets, 1)


def test_multistream_windows_reset_both_arms_after_slow_arm_catches_up():
    # The fast arm's excess in the first chunk must not fund the next chunk.
    values = xyz([0, 2, 2, 2, 3, 3.25], [0, 0, 1, 2, 2, 2.25])
    anchors, budgets = dtw.chunk_distance_windows(values, 1, "multistream")
    np.testing.assert_array_equal(anchors, [0, 2, 4])
    np.testing.assert_allclose(budgets, [1, 1, 0.25])


def test_multistream_remaining_tail_gets_fallback_after_completed_window():
    values = xyz([0, 1, 1, 1], [0, 1, 2, 3])
    anchors, budgets = dtw.chunk_distance_windows(values, 1, "multistream")
    np.testing.assert_array_equal(anchors, [0, 1])
    np.testing.assert_allclose(budgets, [1, 1])
    assert dtw.translation_progress(values, "multistream")[-1] == 1


@pytest.mark.parametrize("mode", ["multistream", "race"])
def test_completed_translation_does_not_add_fallback_for_static_tail(mode):
    values = xyz([0, 1, 1, 1], [0, 1, 1, 1])
    anchors, budgets = dtw.chunk_distance_windows(values, 1, mode)
    np.testing.assert_array_equal(anchors, [0])
    np.testing.assert_allclose(budgets, [1])


def test_race_windows_keep_fractional_crossings_and_repeated_frame_anchors():
    values = xyz([0, 2.5], [0, 0])
    anchors, budgets = dtw.chunk_distance_windows(values, 1, "race")
    np.testing.assert_array_equal(anchors, [0, 1, 1])
    np.testing.assert_allclose(budgets, [1, 1, 0.5])


@pytest.mark.parametrize("mode", dtw.ARC_CHUNKING_MODES)
def test_stationary_budget_has_one_full_prefix(mode):
    anchors, budgets = dtw.chunk_distance_windows(np.zeros((10, 6)), 0.4, mode)
    np.testing.assert_array_equal(anchors, [0])
    np.testing.assert_allclose(budgets, [0.4])


def test_joint_windows_are_unchanged():
    values = xyz([0, 0.09, 0.21, 0.25], [0, 0.02, 0.04, 0.05])
    actual = dtw.chunk_distance_windows(values, 0.1, "joint_distance")
    expected = dtw.distance_windows(dtw.joint_cumulative_distance(values), 0.1)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)


def test_joint_progress_preserves_legacy_floating_point_accumulation():
    values = np.random.default_rng(13).normal(size=(200, 6)).cumsum(axis=0)
    legacy = np.concatenate(
        (
            [0.0],
            np.linalg.norm(np.diff(values, axis=0).reshape(-1, 2, 3), axis=2)
            .sum(axis=1)
            .cumsum(),
        )
    )
    np.testing.assert_array_equal(dtw.joint_cumulative_distance(values), legacy)


def token(m=6):
    result = np.zeros((2 * m, 14))
    result[:m, 0] = np.linspace(0, 1, m)
    result[:m, 7] = np.linspace(0, 0.5, m)
    result[:m, 3] = result[:m, 10] = np.linspace(0, 0.4, m)
    result[m:, 0], result[m:, 7] = 1, 0.25
    result[m:, 3] = result[m:, 10] = 0.1
    return result


@pytest.mark.parametrize(
    "mode,endpoints",
    [
        ("joint_distance", [1 / 3, 1 / 6]),
        ("race", [0.5, 0.125]),
        ("multistream", [0.5, 0.5]),
    ],
)
def test_distance_prefix_uses_translation_mode_and_independent_rotation(
    mode, endpoints
):
    partial = arc_execution_prefix(
        token(),
        0.5,
        "per_waypoint",
        1,
        "distance",
        arc_chunking_mode=mode,
        rotation_distance_unit=0.8,
        control_dt=0.1,
    )
    m = len(partial) // 2
    np.testing.assert_allclose(partial[m - 1, [0, 7]], endpoints)
    total_rotation = sum(
        cumulative_rotation_length(partial[:m, off : off + 3])[-1] for off in (3, 10)
    )
    assert total_rotation == pytest.approx(0.4)
    # R needs 2 s even when the translation race ends at 0.5 s.
    assert (
        arc_prefix_control_steps(
            partial,
            1,
            "per_waypoint",
            0.1,
            1,
            arc_chunking_mode=mode,
            rotation_distance_unit=0.8,
        )
        == 20
    )


@pytest.mark.parametrize("mode", dtw.ARC_CHUNKING_MODES)
def test_waypoint_execution_keeps_exact_rows_for_every_clock(mode):
    value = token()
    partial = arc_execution_prefix(
        value,
        0.5,
        "per_waypoint",
        1,
        "waypoints",
        arc_chunking_mode=mode,
        rotation_distance_unit=0.8,
    )
    np.testing.assert_array_equal(partial, np.concatenate((value[:3], value[6:9])))


@pytest.mark.parametrize(
    "mode,steps", [("joint_distance", 40), ("race", 30), ("multistream", 30)]
)
def test_translation_clocks_do_not_share_interval_waits_in_per_arm_modes(mode, steps):
    value = np.zeros((6, 14))
    value[:3, 0] = value[:3, 7] = [0, 0.1, 0.2]
    value[:3, 3] = value[:3, 10] = [0, 0.05, 0.1]
    value[3:, 0] = [0.05, 0.1, 0.1]
    value[3:, 7] = [0.1, 0.05, 0.05]
    value[3:, 3] = value[3:, 10] = 1
    assert (
        arc_prefix_control_steps(
            value,
            1,
            "per_waypoint",
            0.1,
            0.4,
            rotation_distance_unit=0.2,
            arc_chunking_mode=mode,
        )
        == steps
    )


def test_multistream_one_stationary_arm_uses_one_fallback_prefix():
    anchors, budgets = dtw.chunk_distance_windows(
        xyz([0, 1, 2], [0, 0, 0]), 0.4, "multistream"
    )
    np.testing.assert_array_equal(anchors, [0])
    np.testing.assert_allclose(budgets, [0.4])


@pytest.mark.parametrize("mode", dtw.ARC_CHUNKING_MODES)
def test_unbounded_missing_translation_rate_fails_instead_of_silently_retiming(mode):
    value = token()
    value[6:, 0] = 0
    with pytest.raises(ValueError, match="finite replan boundary"):
        arc_prefix_control_steps(
            value,
            1,
            "per_waypoint",
            0.1,
            1,
            rotation_distance_unit=0.8,
            arc_chunking_mode=mode,
        )
    assert (
        arc_prefix_control_steps(
            value,
            1,
            "per_waypoint",
            0.1,
            1,
            rotation_distance_unit=0.8,
            arc_chunking_mode=mode,
            max_steps=7,
        )
        == 7
    )


@pytest.mark.parametrize(
    "mode,endpoints",
    [
        ("joint_distance", [1 / 3, 1 / 6]),
        ("race", [0.5, 0.25]),
        ("multistream", [0.5, 0.5]),
    ],
)
def test_video_caps_follow_mode_and_keep_later_rotation(mode, endpoints):
    trajectory = token()[:6]
    partial = truncate_cartesian_trajectory_by_arc_mode(trajectory, 0.5, mode, 0.8)
    np.testing.assert_allclose(partial[-1, [0, 7]], endpoints)
    np.testing.assert_allclose(partial[-1, [3, 10]], [0.4, 0.4])


@pytest.mark.parametrize("mode", dtw.ARC_CHUNKING_MODES)
def test_evaluator_routes_mode_and_uses_separate_rotation_clock(mode):
    evaluator = OpenLoopSimEval(
        action_mode="arc",
        arc_chunking_mode=mode,
        execute_fraction=0.5,
        resampled_vector_length=6,
        min_distance_unit=1,
        rotation_distance_unit=0.8,
        control_dt=0.1,
    )
    decoded, steps = evaluator._decode_prediction_with_steps(token())
    assert steps == 16  # Exact three-waypoint prefix rotates 0.16 rad per arm.
    assert decoded.shape == (steps, 14)
    assert evaluator._arc_tokenizer.arc_chunking_mode == mode


@pytest.mark.parametrize(
    "rotation,expected", [(None, "multistream"), (0.8, "joint_distance")]
)
def test_evaluator_default_mode_matches_codec(rotation, expected):
    evaluator = OpenLoopSimEval(rotation_distance_unit=rotation)
    assert evaluator.arc_chunking_mode == expected


@pytest.mark.parametrize("mode", dtw.ARC_CHUNKING_MODES)
def test_explicit_evaluator_mode_requires_rotation(mode):
    with pytest.raises(ValueError, match="explicit arc_chunking_mode requires"):
        OpenLoopSimEval(action_mode="arc", arc_chunking_mode=mode)


def test_legacy_no_rotation_default_decodes_and_scores_with_resolved_metadata():
    evaluator = OpenLoopSimEval(
        action_mode="arc",
        execute_fraction=0.5,
        resampled_vector_length=6,
        min_distance_unit=1,
        control_dt=0.1,
    )
    decoded, steps = evaluator._decode_prediction_with_steps(token())
    assert evaluator.arc_chunking_mode == "multistream"
    assert steps == 8
    assert np.isfinite(decoded).all()
    records = [
        dict(
            frame=frame,
            prediction=token(),
            ground_truth=np.zeros((10, 14)),
            **{
                dtw.METRIC_FRAME_KEY: np.repeat(np.eye(4)[None], 2, axis=0),
            },
        )
        for frame in range(2)
    ]
    result = dtw.score_distance_dtw_episode(evaluator, records)
    assert result["arc_chunking_mode"] == "multistream"


def test_multistream_short_arm_holds_absolute_endpoint_through_evaluator_horizon():
    time = np.arange(61) * 0.1
    raw = np.zeros((len(time), 14))
    raw[:, 0] = 0.7 + 0.1 * np.minimum(time, 1)
    raw[:, 7] = 0.1 * time
    raw[:, 3] = raw[:, 10] = 0.1 * time
    codec = TokenizeBimanualArcLengthCartesian(
        action_key="actions",
        output_action_key="actions",
        min_distance_unit=0.4,
        rotation_distance_unit=0.8,
        resampled_vector_length=100,
        velocity_mode="per_waypoint",
        arc_chunking_mode="multistream",
        dt=0.1,
    )
    value = codec.transform({"actions": raw})["actions"]
    assert np.all(np.diff(value[:100, 0]) > 0)
    np.testing.assert_allclose(value[:100, 0], np.linspace(0.7, 0.8, 100))
    evaluator = OpenLoopSimEval(
        action_mode="arc",
        arc_chunking_mode="multistream",
        execute_fraction=1,
        min_distance_unit=0.4,
        rotation_distance_unit=0.8,
        resampled_vector_length=100,
        control_dt=0.1,
    )
    decoded, steps = evaluator._decode_prediction_with_steps(value, max_steps=50)
    assert steps == 40
    np.testing.assert_allclose(decoded[10:, 0], 0.8)
    assert decoded[30, 3] > decoded[10, 3]
    bounded, bounded_steps = evaluator._decode_prediction_with_steps(
        value, max_steps=20
    )
    assert bounded_steps == 20
    np.testing.assert_allclose(bounded[10:, 0], 0.8)
    np.testing.assert_allclose(bounded, decoded[:20])


def test_evaluator_yaml_routes_abc_mode():
    config = OmegaConf.load("egomimic/hydra_configs/evaluator/eval_open_loop_sim.yaml")
    full = OmegaConf.create({"abc": {"arc_chunking_mode": "race"}, "evaluator": config})
    assert full.evaluator.arc_chunking_mode == "race"


def test_sweep_mode_is_in_command_and_changes_completion_signature(tmp_path):
    checkpoint_path = tmp_path / "step=10.ckpt"
    checkpoint_path.write_bytes(b"fixture")
    checkpoint = sweep.Checkpoint(str(checkpoint_path), 10, 7)
    options = dict(
        python="python",
        experiment="fixture",
        checkpoint=checkpoint,
        output_dir=tmp_path,
        action_mode="arc",
        execute_fraction=0.5,
        limit_val_episodes=1,
        wandb_run_id="fixture",
        wandb_name="fixture",
        wandb_group="fixture",
        extra_overrides=[],
    )
    race = sweep.validation_command(**options, arc_chunking_mode="race")
    multi = sweep.validation_command(**options, arc_chunking_mode="multistream")
    assert "evaluator.arc_chunking_mode=race" in race
    assert sweep.completion_signature(race, checkpoint) != sweep.completion_signature(
        multi, checkpoint
    )


def test_sweep_manifest_records_mode_and_metric_version(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step=10.ckpt").write_bytes(b"fixture")
    output = tmp_path / "validation"
    assert (
        sweep.main(
            [
                "--checkpoint-root",
                str(checkpoints),
                "--output-root",
                str(output),
                "--experiment",
                "fixture",
                "--wandb-run-id",
                "fixture",
                "--wandb-name",
                "fixture",
                "--arc-chunking-mode",
                "race",
                "--action-mode",
                "arc",
                "--execute-fraction",
                "0.5",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "checkpoint_sweep_manifest.json").read_text())
    assert manifest["arc_chunking_mode"] == "race"
    assert manifest["distance_dtw_metric_version"] == dtw.METRIC_VERSION


@pytest.mark.parametrize(
    "mode,expected_anchors", [("race", [0, 1, 2, 3]), ("multistream", [0, 2])]
)
def test_dtw_rollout_uses_chunk_local_windows_and_mode_metadata(mode, expected_anchors):
    evaluator = OpenLoopSimEval(
        action_mode="arc",
        arc_chunking_mode=mode,
        execute_fraction=1,
        arc_execution_cap_mode="distance",
        resampled_vector_length=6,
        min_distance_unit=1,
        rotation_distance_unit=0.8,
        control_dt=0.1,
    )
    positions = xyz([0, 1, 1, 2, 2], [0, 0, 1, 1, 2])
    records = []
    for frame, position in enumerate(positions):
        anchors = np.repeat(np.eye(4)[None], 2, axis=0)
        anchors[:, :3, 3] = position.reshape(2, 3)
        records.append(
            dict(
                frame=frame,
                prediction=token(),
                ground_truth=np.zeros((10, 14)),
                **{dtw.METRIC_FRAME_KEY: anchors},
            )
        )
    result = dtw.score_distance_dtw_episode(evaluator, records)
    assert result["anchor_frames"] == expected_anchors
    assert result["arc_chunking_mode"] == mode
    assert result["distance_window_semantics"] == "chunk_local_per_arm_reset"
    assert result["gt_joint_distance_m"] == 4
    assert result["gt_mode_progress_m"] == 2
    assert result["gt_coverage"] == result["prediction_coverage"] == 1
    json.dumps(result)
