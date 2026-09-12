"""Arc-matched, DTW and time-domain metric families."""

import numpy as np
import pytest

from egomimic.eval.arc_metrics import (
    ARM_XYZ,
    PAIRED_COLS,
    arcmatch_metrics,
    arm_travel,
    chunk_metrics,
    clip_to_distance,
    dtw_metrics,
    dtw_path,
    geodesic_deg,
    match_spans,
    mse,
    pose_err_m,
    shared_spans,
    tokenize_span,
)

_DT = 1.0 / 30.0
_M = 32
_LEVER = 0.1


def _path(n: int = 200, span: float = 0.6) -> np.ndarray:
    """One fixed bimanual curve: forward sweep plus a lateral wiggle."""
    t = np.linspace(0.0, 1.0, n)
    chunk = np.zeros((n, 14))
    for base in (0, 7):
        chunk[:, base + 0] = t * span
        chunk[:, base + 1] = 0.05 * np.sin(2 * np.pi * t)
        chunk[:, base + 3] = t * 0.4
        chunk[:, base + 6] = t
    return chunk


# -- travel and spans -------------------------------------------------------


def test_arm_travel_is_per_arm_and_in_metres():
    travel = arm_travel(_path())
    assert travel.shape == (2,)
    # Both arms follow the same curve here, so they must agree.
    assert travel[0] == pytest.approx(travel[1])
    assert travel[0] > 0.6  # the sweep plus the wiggle exceeds the x extent


def test_match_spans_takes_the_shorter_travel_per_arm():
    full = _path()
    short = full[: len(full) // 2]
    spans = match_spans(short, full)
    np.testing.assert_allclose(spans, arm_travel(short), rtol=1e-9)


def test_shared_spans_also_clamps_by_D():
    full = _path(span=0.6)
    spans = shared_spans(full, full, min_distance_unit=0.4)
    np.testing.assert_allclose(spans, np.full(2, 0.4), rtol=1e-9)


def test_arcmatch_with_D_caps_reported_span():
    full = _path(span=0.6)
    metrics = arcmatch_metrics(
        [full], [full], num_points=_M, dt=_DT, lever_m=_LEVER, min_distance_unit=0.4
    )
    assert metrics["arcmatch_span_m"] == pytest.approx(0.4, abs=1e-6)


def test_arcmatch_keeps_samples_with_one_idle_arm():
    """A held arm (span 0) must not drop the whole sample from arcmatch."""
    moving = _path(80, span=0.5)
    idle_right = moving.copy()
    idle_right[:, 7:14] = idle_right[0:1, 7:14]  # right arm stationary
    metrics = arcmatch_metrics(
        [idle_right],
        [idle_right],
        num_points=_M,
        dt=_DT,
        lever_m=_LEVER,
        min_distance_unit=0.4,
    )
    assert metrics, "idle-arm sample was dropped"
    assert metrics["arcmatch_xyz_mse"] == pytest.approx(0.0, abs=1e-12)


def test_arm_travel_rejects_a_bad_shape():
    with pytest.raises(ValueError, match=r"\(T, 14\)"):
        arm_travel(np.zeros((10, 7)))


def test_arm_travel_rejects_a_single_timestep():
    with pytest.raises(ValueError, match="at least two"):
        arm_travel(np.zeros((1, 14)))


# -- the defining property: arcmatch divides travel out ---------------------


@pytest.mark.parametrize("fraction", [0.5, 0.75, 1.0])
def test_arcmatch_is_travel_invariant_along_the_same_path(fraction):
    """Same path, less of it travelled: shape error must stay zero.

    This is the whole point of the matched span. Without the cut, a prediction
    that stops short would be scored against ground truth it never reached.
    """
    full = _path()
    pred = full[: max(2, int(len(full) * fraction))]
    metrics = arcmatch_metrics([pred], [full], num_points=_M, dt=_DT, lever_m=_LEVER)
    assert metrics["arcmatch_xyz_mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["arcmatch_travel_ratio"] == pytest.approx(fraction, abs=0.02)


def test_dtw_does_see_a_travel_mismatch_where_arcmatch_does_not():
    """The reason both families exist."""
    full = _path()
    short = full[: len(full) // 2]
    arc = arcmatch_metrics([short], [full], num_points=_M, dt=_DT, lever_m=_LEVER)
    warped = dtw_metrics([short], [full])
    assert arc["arcmatch_xyz_mse"] == pytest.approx(0.0, abs=1e-12)
    assert warped["dtw_xyz_l2_m"] > 0.01


def test_a_perfect_prediction_scores_zero_on_every_family():
    full = _path()
    arc = arcmatch_metrics([full], [full], num_points=_M, dt=_DT, lever_m=_LEVER)
    warped = dtw_metrics([full], [full])
    plain = chunk_metrics(full[None], full[None], lever_m=_LEVER)
    for name, value in {**arc, **warped, **plain}.items():
        if "span" in name or "ratio" in name:
            continue
        assert value == pytest.approx(0.0, abs=1e-12), name


def test_arcmatch_reports_span_and_ratio_for_reading_alongside():
    full = _path()
    metrics = arcmatch_metrics(
        [full[: len(full) // 2]], [full], num_points=_M, dt=_DT, lever_m=_LEVER
    )
    assert metrics["arcmatch_span_m"] > 0
    assert 0.4 < metrics["arcmatch_travel_ratio"] < 0.6


def test_arcmatch_withvel_differs_once_timing_differs():
    """The gap between the two variants is the timing error."""
    full = _path()
    # Same path, traversed at half the rate: identical waypoints, different
    # velocity row.
    slow = np.repeat(full, 2, axis=0)
    metrics = arcmatch_metrics([slow], [full], num_points=_M, dt=_DT, lever_m=_LEVER)
    assert metrics["arcmatch_paired_mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["arcmatch_withvel_paired_mse"] > metrics["arcmatch_paired_mse"]


def test_arcmatch_returns_nothing_for_an_empty_batch():
    assert arcmatch_metrics([], [], num_points=_M, dt=_DT, lever_m=_LEVER) == {}


# -- tokenize_span ----------------------------------------------------------


def test_tokenize_span_emits_the_requested_points_and_a_velocity_row():
    waypoints, velocity = tokenize_span(_path(), arm_travel(_path()), _M, _DT)
    assert waypoints.shape == (_M, 14)
    assert velocity.shape == (14,)


def test_tokenize_span_starts_at_the_chunk_origin():
    full = _path()
    waypoints, _ = tokenize_span(full, arm_travel(full), _M, _DT)
    np.testing.assert_allclose(waypoints[0, :3], full[0, :3], atol=1e-9)


def test_tokenize_span_velocity_is_zero_for_a_stationary_arm():
    still = np.zeros((50, 14))
    _, velocity = tokenize_span(still, np.zeros(2), _M, _DT)
    assert np.isfinite(velocity).all()
    np.testing.assert_allclose(velocity, 0.0)


@pytest.mark.parametrize("bad", [{"num_points": 1}, {"dt": 0.0}])
def test_tokenize_span_rejects_degenerate_settings(bad):
    full = _path()
    kwargs = {"num_points": _M, "dt": _DT, **bad}
    with pytest.raises(ValueError):
        tokenize_span(full, arm_travel(full), kwargs["num_points"], kwargs["dt"])


# -- dtw --------------------------------------------------------------------


def test_dtw_path_aligns_identical_sequences_one_to_one():
    seq = _path(40)[:, :3]
    ia, ib = dtw_path(seq, seq)
    np.testing.assert_array_equal(ia, ib)


def test_dtw_path_absorbs_a_pure_time_stretch():
    seq = _path(30)[:, :3]
    stretched = np.repeat(seq, 2, axis=0)
    ia, ib = dtw_path(stretched, seq)
    # Warping should match every stretched frame to its original.
    residual = np.linalg.norm(stretched[ia] - seq[ib], axis=-1).mean()
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_dtw_path_endpoints_are_anchored():
    a, b = _path(20)[:, :3], _path(25)[:, :3]
    ia, ib = dtw_path(a, b)
    assert (ia[0], ib[0]) == (0, 0)
    assert (ia[-1], ib[-1]) == (len(a) - 1, len(b) - 1)


def test_dtw_path_rejects_mismatched_widths():
    with pytest.raises(ValueError, match="equal width"):
        dtw_path(np.zeros((5, 3)), np.zeros((5, 2)))


def test_dtw_metrics_caps_the_samples_it_warps():
    full = _path(60)
    many = [full] * 20
    # Should not raise or hang; the cap keeps it affordable in a val loop.
    assert dtw_metrics(many, many, max_samples=2)


def test_clip_to_distance_truncates_and_keeps_at_least_two_rows():
    full = _path()
    clipped = clip_to_distance(full, 0.1)
    assert 2 <= len(clipped) < len(full)
    assert clip_to_distance(full, 1e-9).shape[0] >= 2


# -- rotation and the combined pose error -----------------------------------


def test_geodesic_is_zero_for_equal_rotations_and_positive_otherwise():
    full = _path()
    assert geodesic_deg(full, full) == pytest.approx(0.0, abs=1e-9)
    rotated = full.copy()
    rotated[:, 3] += 0.1
    rotated[:, 10] += 0.1
    assert geodesic_deg(rotated, full) > 0


def test_geodesic_is_invariant_to_a_shared_frame_change():
    """Conjugation preserves the geodesic angle; a per-axis ypr diff would not."""
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(0)
    pred, gt = np.zeros((4, 14)), np.zeros((4, 14))
    pred[:, 3:6] = rng.normal(size=(4, 3)) * 0.2
    gt[:, 3:6] = rng.normal(size=(4, 3)) * 0.2
    pred[:, 10:13], gt[:, 10:13] = pred[:, 3:6], gt[:, 3:6]
    before = geodesic_deg(pred, gt)

    shift = R.from_euler("ZYX", [0.3, -0.2, 0.5])
    for cols in ((3, 4, 5), (10, 11, 12)):
        for chunk in (pred, gt):
            rot = R.from_euler("ZYX", chunk[:, list(cols)])
            chunk[:, list(cols)] = (shift * rot).as_euler("ZYX")
    assert geodesic_deg(pred, gt) == pytest.approx(before, rel=1e-6)


def test_pose_err_matches_its_closed_form_for_a_pure_rotation():
    """theta radians at lever_m costs exactly lever_m * theta metres."""
    theta, lever = 0.2, 0.25
    pred, gt = np.zeros((4, 14)), np.zeros((4, 14))
    pred[:, 3] = theta
    pred[:, 10] = theta
    assert pose_err_m(pred, gt, lever) == pytest.approx(lever * theta, rel=1e-9)


def test_pose_err_matches_its_closed_form_for_a_pure_translation():
    pred, gt = np.zeros((4, 14)), np.zeros((4, 14))
    pred[:, 0] = 0.03
    pred[:, 7] = 0.03
    assert pose_err_m(pred, gt, 0.1) == pytest.approx(0.03, rel=1e-9)


def test_pose_err_lever_zero_ignores_rotation_entirely():
    pred, gt = np.zeros((4, 14)), np.zeros((4, 14))
    pred[:, 3] = 0.5
    pred[:, 10] = 0.5
    assert pose_err_m(pred, gt, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_pose_err_rejects_a_negative_lever():
    with pytest.raises(ValueError, match="lever_m"):
        pose_err_m(np.zeros((2, 14)), np.zeros((2, 14)), -0.1)


# -- chunk family -----------------------------------------------------------


def test_chunk_metrics_splits_by_component():
    full = _path()
    shifted = full.copy()
    shifted[:, 6] += 0.2  # left gripper only
    metrics = chunk_metrics(shifted[None], full[None], lever_m=_LEVER)
    assert metrics["chunk_grip_mse"] > 0
    assert metrics["chunk_xyz_mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["chunk_ypr_mse"] == pytest.approx(0.0, abs=1e-12)


def test_chunk_metrics_requires_matching_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        chunk_metrics(np.zeros((1, 10, 14)), np.zeros((1, 12, 14)), lever_m=_LEVER)


def test_paired_columns_exclude_rotation():
    """Rotation must never be summed with metres in a paired MSE."""
    ypr = {c for arm in ((3, 4, 5), (10, 11, 12)) for c in arm}
    assert not (set(PAIRED_COLS) & ypr)


def test_mse_respects_a_column_subset():
    a = np.zeros((2, 4, 14))
    b = np.zeros((2, 4, 14))
    b[..., list(ARM_XYZ[0])] = 1.0
    assert mse(a, b) > 0
    assert mse(a, b, [6, 13]) == pytest.approx(0.0)


# -- metrics must be logged on the compute device ---------------------------


def test_extra_metrics_land_on_the_prediction_device():
    """log_dict(sync_dist=True) all-reduces these over NCCL.

    NCCL has no CPU support, so a CPU tensor raises "No backend type
    associated with device type cpu" the moment a GPU run tries to sync. The
    metrics are computed in numpy on the host, so the cast back to a tensor is
    the one place device placement matters.
    """
    import torch

    from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    evaluator = ArcBimanualCartesianEval.__new__(ArcBimanualCartesianEval)
    evaluator.action_key = "actions_cartesian"
    evaluator.untokenized_action_key = "actions_cartesian_untokenized"
    evaluator.min_distance_unit = 0.4
    evaluator.resampled_vector_length = _M
    evaluator.action_horizon = 32
    evaluator.velocity_mode = "mean"
    evaluator.arc_metrics = True
    evaluator.include_reconstruction_loss = False
    evaluator.arcmatch_points = 8
    evaluator.rot_lever_m = _LEVER
    evaluator.dtw_max_samples = 2
    evaluator.metric_dt = _DT
    evaluator.arcmatch_distance = 0.4
    evaluator.arc_chunk_rows = 45
    evaluator._tokenizer = TokenizeBimanualArcLengthCartesian(
        action_key="actions_cartesian",
        output_action_key="actions_cartesian",
        min_distance_unit=0.4,
        resampled_vector_length=_M,
        preserve_action_key=None,
    )

    class _Passthrough:
        def unnormalize(self, mapping, embodiment_id):
            del embodiment_id
            return mapping

    evaluator.normalizer = _Passthrough()

    gt = torch.from_numpy(np.stack([_path(60)]))
    token = torch.zeros(1, _M + 1, 14, dtype=torch.float64)
    token[0, :_M, 0] = torch.linspace(0.0, 0.4, _M)
    token[0, :_M, 7] = torch.linspace(0.0, 0.4, _M)
    token[0, _M, 0] = 0.3
    token[0, _M, 7] = 0.3

    metrics = evaluator._extra_metrics(
        label="yam_bimanual",
        embodiment_id=7,
        source_batch={"actions_cartesian_untokenized": gt},
        prediction=token,
        target=None,
    )
    assert metrics, "no arc metrics were produced"
    for name, value in metrics.items():
        assert value.device == token.device, name
        assert torch.isfinite(value), name
    assert any("arcmatch" in k for k in metrics)


def test_extra_metrics_are_skipped_without_the_preserved_ground_truth():
    """No untokenized chunk means no scoring against a reconstruction."""
    from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval

    evaluator = ArcBimanualCartesianEval.__new__(ArcBimanualCartesianEval)
    evaluator.arc_metrics = True
    evaluator.untokenized_action_key = "actions_cartesian_untokenized"
    evaluator.action_key = "actions_cartesian"
    assert (
        evaluator._extra_metrics(
            label="x",
            embodiment_id=7,
            source_batch={},
            prediction=None,
            target=None,
        )
        == {}
    )


def test_extra_metrics_can_be_turned_off():
    from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval

    evaluator = ArcBimanualCartesianEval.__new__(ArcBimanualCartesianEval)
    evaluator.arc_metrics = False
    assert (
        evaluator._extra_metrics(
            label="x", embodiment_id=7, source_batch={}, prediction=None, target=None
        )
        == {}
    )


# -- one evaluator, both arms of the ablation -------------------------------
#
# The baseline and arc runs must land on the SAME arcmatch charts or the
# comparison is meaningless. Shared-D prep: baseline keeps tokenizer-resolution
# poses; ARC uses token waypoints unless L_gt < D (then detok). Overlay still
# always detokenizes ARC for viz.


def _arc_evaluator(velocity_mode="mean", action_horizon=45):
    from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    ev = ArcBimanualCartesianEval.__new__(ArcBimanualCartesianEval)
    ev.action_key = "actions_cartesian"
    ev.untokenized_action_key = "actions_cartesian_untokenized"
    ev.min_distance_unit = 0.4
    ev.resampled_vector_length = _M
    ev.action_horizon = action_horizon
    ev.velocity_mode = velocity_mode
    ev.arc_metrics = True
    ev.include_reconstruction_loss = False
    ev.arcmatch_points = 8
    ev.arcmatch_distance = 0.4
    ev.arc_chunk_rows = 45
    ev.rot_lever_m = _LEVER
    ev.dtw_max_samples = 2
    ev.metric_dt = _DT
    ev._tokenizer = TokenizeBimanualArcLengthCartesian(
        action_key="actions_cartesian",
        output_action_key="actions_cartesian",
        min_distance_unit=0.4,
        resampled_vector_length=_M,
        preserve_action_key=None,
        velocity_mode=velocity_mode,
    )

    class _Passthrough:
        def unnormalize(self, mapping, embodiment_id):
            del embodiment_id
            return mapping

    ev.normalizer = _Passthrough()
    return ev


def _arc_token(velocity_mode="mean"):
    import torch

    rows = _M + 1 if velocity_mode == "mean" else 2 * _M
    token = torch.zeros(1, rows, 14, dtype=torch.float64)
    token[0, :_M, 0] = torch.linspace(0.0, 0.4, _M)
    token[0, :_M, 7] = torch.linspace(0.0, 0.4, _M)
    if velocity_mode == "mean":
        token[0, _M, 0] = token[0, _M, 7] = 0.3
    else:
        token[0, _M:, 0] = token[0, _M:, 7] = 0.3
    return token


@pytest.mark.parametrize("velocity_mode", ["mean", "per_waypoint"])
def test_the_predicate_recognises_an_arc_token_of_either_layout(velocity_mode):
    evaluator = _arc_evaluator(velocity_mode)
    assert evaluator._is_arc(_arc_token(velocity_mode))


def test_the_predicate_rejects_a_baseline_pose_chunk():
    import torch

    evaluator = _arc_evaluator()
    assert not evaluator._is_arc(torch.zeros(1, 100, 14))
    assert not evaluator._is_arc(torch.zeros(1, 45, 14))


def test_the_predicate_rejects_the_other_velocity_modes_row_count():
    evaluator = _arc_evaluator("mean")
    assert not evaluator._is_arc(_arc_token("per_waypoint"))


def test_the_same_evaluator_scores_a_baseline_chunk_too():
    """A baseline prediction goes down the pose path, not the detokenizer."""
    import torch

    evaluator = _arc_evaluator()
    gt = torch.from_numpy(np.stack([_path(100)]))
    pred = torch.from_numpy(np.stack([_path(100, span=0.5)]))  # pose chunk
    metrics = evaluator._extra_metrics(
        label="yam_bimanual",
        embodiment_id=7,
        source_batch={"actions_cartesian_untokenized": gt},
        prediction=pred,
        target=None,
    )
    assert metrics, "the baseline path produced no metrics"
    assert any("arcmatch" in k for k in metrics)


def test_both_run_types_emit_the_same_metric_names():
    """Same keys means the two arms overlay on one chart."""
    import torch

    evaluator = _arc_evaluator()
    gt = torch.from_numpy(np.stack([_path(100)]))
    arc = evaluator._extra_metrics(
        label="x",
        embodiment_id=7,
        source_batch={"actions_cartesian_untokenized": gt},
        prediction=_arc_token(),
        target=None,
    )
    baseline = evaluator._extra_metrics(
        label="x",
        embodiment_id=7,
        source_batch={"actions_cartesian_untokenized": gt},
        prediction=torch.from_numpy(np.stack([_path(100, span=0.5)])),
        target=None,
    )
    assert set(arc) == set(baseline)


def test_arcmatch_uses_arc_waypoints_when_gt_travel_reaches_D():
    """L_gt >= D: keep token waypoints; do not detok for arcmatch."""
    evaluator = _arc_evaluator(action_horizon=100)
    gt = np.stack([_path(100, span=0.6)])  # travel > D=0.4
    pred = _arc_token()
    out = evaluator._arc_pred_for_arcmatch(pred, gt, 7)
    assert isinstance(out, list)
    assert out[0].shape == (_M, 14)


def test_include_reconstruction_loss_always_detoks_arc_for_arcmatch():
    """Flag on: detok even when L_gt >= D (reconstruction baked in)."""
    evaluator = _arc_evaluator(action_horizon=100)
    evaluator.include_reconstruction_loss = True
    gt = np.stack([_path(100, span=0.6)])
    out = evaluator._arc_pred_for_arcmatch(_arc_token(), gt, 7)
    assert out.shape == (1, 100, 14)


def test_arcmatch_detoks_arc_when_gt_travel_is_shorter_than_D():
    """L_gt < D: detok so both sides re-tokenize over the shorter shared span."""
    evaluator = _arc_evaluator(action_horizon=100)
    gt = np.stack([_path(100, span=0.2)])  # travel < D=0.4
    pred = _arc_token()
    out = evaluator._arc_pred_for_arcmatch(pred, gt, 7)
    assert isinstance(out, list)
    assert out[0].shape == (100, 14)


def test_arcmatch_detoks_when_only_one_arm_is_shorter_than_D():
    """Mixed arms: any arm with L_gt < D forces detok (intentional)."""
    evaluator = _arc_evaluator(action_horizon=100)
    gt = _path(100, span=0.6)
    gt[:, 7:14] = gt[0:1, 7:14]  # right idle → L_right ≈ 0 < D
    # Give right a short non-zero travel still < D
    gt[:, 7] = np.linspace(0.0, 0.15, 100)
    out = evaluator._arc_pred_for_arcmatch(_arc_token(), gt[None], 7)
    assert out[0].shape == (100, 14)


def test_arc_evaluator_rejects_mismatched_arcmatch_and_codec_D():
    from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval

    with pytest.raises(ValueError, match="arcmatch_distance"):
        ArcBimanualCartesianEval(
            min_distance_unit=0.40,
            resampled_vector_length=_M,
            arcmatch_distance=0.50,
        )


def test_gt_and_baseline_arcmatch_prep_are_not_deinterpolated():
    """Shared-D uses tokenizer resolution, not arc_chunk_rows=45."""
    import torch

    evaluator = _arc_evaluator()
    gt_t = torch.from_numpy(np.stack([_path(100, span=0.5)]))
    gt = evaluator._arc_gt_time_indexed(
        {"actions_cartesian_untokenized": gt_t}, embodiment_id=7
    )
    pred = evaluator._arc_pred_time_indexed(gt_t, embodiment_id=7)
    assert gt.shape == (1, 100, 14)
    assert pred.shape == (1, 100, 14)


def test_deinterpolation_reduces_rows_without_inventing_samples():
    from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval

    chunk = _path(100)[None]
    reduced = BimanualCartesianEval._deinterpolate(chunk, 45)
    assert reduced.shape == (1, 45, 14)
    # Every retained row must be one of the originals, not an interpolant.
    for row in reduced[0]:
        assert np.any(np.all(np.isclose(chunk[0], row), axis=-1))


def test_deinterpolation_is_a_noop_when_already_short_enough():
    from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval

    chunk = _path(30)[None]
    assert BimanualCartesianEval._deinterpolate(chunk, 45) is chunk


def test_deinterpolation_matters_for_arc_length():
    """Why it exists: arc length on an interpolated path reads differently."""
    from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval

    dense = _path(400)
    reduced = BimanualCartesianEval._deinterpolate(dense[None], 45)[0]
    # A coarser sampling chord-cuts the wiggle, so it measures shorter.
    assert arm_travel(reduced)[0] < arm_travel(dense)[0]


# -- the overlay path needs the same predicate as the metrics path ----------
#
# Regression: _is_arc was wired into _arc_pred_time_indexed but not into
# _viz_source, so a baseline run reached an arc-only shape check and died with
# "expects (B, 200, D) arc tokens ... got (32, 100, 14)". Both paths route on
# the run type, so both need the predicate.


def test_viz_source_passes_a_baseline_chunk_through():
    import torch

    evaluator = _arc_evaluator("per_waypoint")
    chunk = torch.zeros(4, 100, 14)
    out = evaluator._viz_source(chunk, 7)
    assert out.shape == chunk.shape


def test_viz_source_still_detokenizes_an_arc_token():
    evaluator = _arc_evaluator("per_waypoint")
    out = evaluator._viz_source(_arc_token("per_waypoint"), 7)
    assert out.shape == (1, evaluator.action_horizon, 14)


def test_viz_source_rejects_a_non_bimanual_width():
    """A wrong width IS a misconfiguration, unlike a baseline row count."""
    import torch

    evaluator = _arc_evaluator("per_waypoint")
    with pytest.raises(ValueError, match=r"\(B, T, 14\)"):
        evaluator._viz_source(torch.zeros(4, 100, 7), 7)


def test_both_paths_agree_on_the_run_type():
    """metrics and overlay must never disagree about what they were handed."""
    import torch

    evaluator = _arc_evaluator("per_waypoint")
    for actions, is_arc in (
        (_arc_token("per_waypoint"), True),
        (torch.zeros(1, 100, 14), False),
    ):
        assert evaluator._is_arc(actions) is is_arc
        # The overlay path returns detokenized rows only for an arc token.
        out = evaluator._viz_source(actions, 7)
        assert (out.shape[1] == evaluator.action_horizon) or not is_arc
