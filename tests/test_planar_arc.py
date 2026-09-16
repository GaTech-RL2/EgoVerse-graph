import math

import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import (
    PlanarArcTrajectoryNativeDecoder,
    PlanarArcWaypointZeroNativeDecoder,
    PlanarCommon5NativeDecoder,
)
from egomimic.rldb.embodiment.pushshapes import get_planar_arc_length_transform_list
from egomimic.rldb.zarr.planar_arc import (
    PadPlanarAction,
    TokenizePlanarArcLength,
    curvature_adaptive_curve_samples,
    lambda_for_radius,
    planar_step_distance,
)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_pad_planar_action_has_common_layout(width):
    native = np.arange(3 * width, dtype=np.float32).reshape(3, width) / 10
    output = PadPlanarAction().transform({"actions": native.copy()})["actions"]
    assert output.shape == (3, 5)
    np.testing.assert_allclose(output[:, :2], native[:, :2])
    theta = native[:, 2] if width >= 3 else np.zeros(3)
    grip = native[:, 3] if width == 4 else np.zeros(3)
    np.testing.assert_allclose(output[:, 2], np.cos(theta))
    np.testing.assert_allclose(output[:, 3], np.sin(theta))
    np.testing.assert_allclose(output[:, 4], grip)


def test_planar_arc_shape_anchor_and_timing():
    action = np.column_stack(
        (
            np.arange(6, dtype=np.float32),
            np.zeros(6, dtype=np.float32),
            np.linspace(0, math.pi / 2, 6, dtype=np.float32),
            np.linspace(0, 1, 6, dtype=np.float32),
        )
    )
    transform = TokenizePlanarArcLength(
        min_distance_unit=3,
        resampled_vector_length=4,
        dt=0.5,
        rotation_radius=0,
    )
    token = transform.transform({"actions": action})["actions"]
    assert token.shape == (8, 5)
    np.testing.assert_allclose(token[0, :2], action[0, :2])
    np.testing.assert_allclose(token[0, 2:4], [1, 0], atol=1e-6)
    assert token[0, 4] == action[0, 3]
    np.testing.assert_allclose(token[4:, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(token[4:, 1:], 0.0, atol=1e-6)


def test_rotation_radius_adds_metric_distance():
    xy = np.zeros((3, 2))
    theta = np.array([0, math.pi / 2, math.pi])
    assert planar_step_distance(xy).sum() == 0
    weighted = planar_step_distance(xy, theta, lambda_for_radius(30))
    assert np.all(weighted > 0)


def test_hybrid_rotation_budget_caps_a_shared_cartesian_window():
    """A small angular budget shortens the common arc window, not a stream."""
    action = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, math.pi], [20.0, 0.0, math.pi]])
    token = TokenizePlanarArcLength(
        min_distance_unit=20,
        resampled_vector_length=3,
        rotation_radius=0,
        hybrid_rotation_unit=0.25,
    ).transform({"actions": action})["actions"]

    # The first pi rotation has unit chordal metric, so its 0.25 budget keeps
    # only the first quarter of the 20-unit Cartesian future: x=5.
    np.testing.assert_allclose(token[2, :2], [5.0, 0.0], atol=1e-6)


def test_hybrid_rotation_budget_does_not_shorten_translation_only_motion():
    action = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    legacy = TokenizePlanarArcLength(
        min_distance_unit=12, resampled_vector_length=3
    ).transform({"actions": action})["actions"]
    hybrid = TokenizePlanarArcLength(
        min_distance_unit=12,
        resampled_vector_length=3,
        hybrid_rotation_unit=0.25,
    ).transform({"actions": action})["actions"]
    np.testing.assert_allclose(hybrid, legacy)


@pytest.mark.parametrize("budget", [0.0, -1.0, float("nan")])
def test_hybrid_rotation_budget_must_be_positive_and_finite(budget):
    with pytest.raises(ValueError, match="hybrid_rotation_unit"):
        TokenizePlanarArcLength(hybrid_rotation_unit=budget)


def test_zero_motion_holds_pose_and_grip():
    action = np.repeat(np.array([[4.0, 7.0, 0.5, 0.75]]), 5, axis=0)
    token = TokenizePlanarArcLength(resampled_vector_length=3).transform(
        {"actions": action}
    )["actions"]
    np.testing.assert_allclose(token[:3, :2], [[4, 7]] * 3)
    np.testing.assert_allclose(token[:3, 4], 0.75)
    np.testing.assert_allclose(token[3:, 0], 2.0 / 30.0)


@pytest.mark.parametrize("allocation", ["uniform", "curvature"])
@pytest.mark.parametrize("channel", [2, 3])
def test_duration_preserves_stationary_rotation_and_grip(allocation, channel):
    raw = np.repeat(np.array([[100.0, 100.0, 0.0, 0.0]]), 40, axis=0)
    raw[:, channel] = np.linspace(0.0, 0.4 if channel == 2 else 1.0, 40)
    token = TokenizePlanarArcLength(
        min_distance_unit=40.0,
        resampled_vector_length=16,
        rotation_radius=0.0,
        hybrid_rotation_unit=0.14776,
        waypoint_sampling=allocation,
    ).tokenize(raw)
    decoded = PlanarArcTrajectoryNativeDecoder(16, 4, 40).decode(token)[0]
    np.testing.assert_allclose(decoded, raw, atol=5e-7)
    assert np.sum(token[16:-1, 0]) == pytest.approx(39.0 / 30.0)


@pytest.mark.parametrize("allocation", ["uniform", "curvature"])
@pytest.mark.parametrize("hold", ["initial", "middle", "trailing", "multiple"])
def test_duration_roundtrip_preserves_hold_boundaries_and_grip(allocation, hold):
    raw = np.zeros((40, 4))
    if hold == "initial":
        raw[:, 0] = np.r_[np.zeros(10), np.arange(30)]
        start, end = 0, 10
    elif hold == "middle":
        raw[:, 0] = np.r_[np.arange(11), np.full(9, 10), np.arange(11, 31)]
        start, end = 10, 19
    elif hold == "trailing":
        raw[:, 0] = np.r_[np.arange(30), np.full(10, 29)]
        start, end = 29, 39
    else:
        raw[:, 0] = np.r_[
            np.zeros(4), np.arange(1, 11), np.full(6, 10), np.arange(11, 31)
        ]
        start, end = 13, 19
    raw[:, 3] = np.clip((np.arange(40) - start) / (end - start), 0.0, 1.0)
    token = TokenizePlanarArcLength(
        min_distance_unit=40.0,
        resampled_vector_length=16,
        rotation_radius=0.0,
        hybrid_rotation_unit=0.14776,
        waypoint_sampling=allocation,
    ).tokenize(raw)
    decoded = PlanarArcTrajectoryNativeDecoder(16, 4, 40).decode(token)[0]
    np.testing.assert_allclose(decoded, raw, atol=1e-5)
    assert token.shape == (32, 5)
    assert np.sum(token[16:-1, 0]) == pytest.approx(39.0 / 30.0)


@pytest.mark.parametrize("allocation", ["uniform", "curvature"])
def test_duration_short_pauses_fit_budget_without_deleting_elapsed_time(allocation):
    raw = np.column_stack((np.repeat(np.arange(4.0), 2), np.zeros((8, 3))))
    tokenizer = TokenizePlanarArcLength(
        resampled_vector_length=4, waypoint_sampling=allocation
    )
    token = tokenizer.tokenize(raw)
    decoded = PlanarArcTrajectoryNativeDecoder(4, 4, 8).decode(token)[0]
    assert token.shape == (8, 5)
    assert np.sum(token[4:-1, 0]) == pytest.approx(7.0 / 30.0)
    assert np.all(token[4:, 0] >= 0)
    np.testing.assert_allclose(decoded[[0, -1]], raw[[0, -1]], atol=1e-6)
    assert np.max(np.abs(decoded - raw)) < 1.0


@pytest.mark.parametrize("allocation", ["uniform", "curvature"])
def test_duration_keeps_long_hold_when_quantized_pauses_exceed_budget(allocation):
    raw = np.zeros((40, 4))
    raw[:, 0] = np.r_[np.zeros(10), np.repeat(np.arange(1.0, 16.0), 2)]
    raw[:, 3] = np.clip(np.arange(40) / 9.0, 0, 1)
    token = TokenizePlanarArcLength(
        min_distance_unit=40, resampled_vector_length=16,
        waypoint_sampling=allocation,
    ).tokenize(raw)
    decoded = PlanarArcTrajectoryNativeDecoder(16, 4, 40).decode(token)[0]
    assert token.shape == (32, 5)
    assert np.sum(token[16:-1, 0]) == pytest.approx(39.0 / 30.0)
    np.testing.assert_allclose(decoded[:10], raw[:10], atol=1e-6)
    np.testing.assert_allclose(decoded[-1], raw[-1], atol=1e-6)
    assert np.max(np.abs(decoded - raw)) < 1.0


def test_curvature_sampling_allocates_denser_support_on_a_bend():
    straight_x = np.linspace(0.0, 10.0, 61)
    straight = np.column_stack((straight_x, np.zeros_like(straight_x)))
    angle = np.linspace(-math.pi / 2, 0.0, 41)[1:]
    bend = np.column_stack((10.0 + 2.0 * np.cos(angle), 2.0 + 2.0 * np.sin(angle)))
    xy = np.concatenate((straight, bend), axis=0)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(planar_step_distance(xy))))

    _, targets = curvature_adaptive_curve_samples(
        xy, cumulative, float(cumulative[-1]), 16, dense_samples=513
    )
    straight_end = cumulative[len(straight) - 1]
    bend_gaps = np.diff(targets[targets >= straight_end])
    straight_gaps = np.diff(targets[targets <= straight_end])
    assert len(bend_gaps) >= 2
    assert np.median(bend_gaps) < np.median(straight_gaps)


def test_curvature_allocation_samples_the_same_geometry_as_uniform():
    # This axis-aligned path has an analytic location at every arc coordinate.
    # A spline fitted through the corners overshoots the path; it may estimate
    # curvature for allocation, but must not replace the uniform arm's geometry.
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [5.0, 3.0]])
    cumulative = np.array([0.0, 2.0, 5.0, 8.0])
    sampled, targets = curvature_adaptive_curve_samples(xy, cumulative, 8.0, 16)
    expected_x = np.where(
        targets < 2.0, targets, np.where(targets <= 5.0, 2.0, targets - 3.0)
    )
    expected_y = np.clip(targets - 2.0, 0.0, 3.0)
    np.testing.assert_allclose(sampled, np.column_stack((expected_x, expected_y)))


def test_duration_decoder_restores_the_full_control_rate_trajectory():
    raw = np.column_stack((np.arange(40, dtype=np.float32), np.zeros(40), np.zeros(40)))
    token = TokenizePlanarArcLength(
        min_distance_unit=40.0,
        resampled_vector_length=16,
        dt=1.0 / 30.0,
    ).tokenize(raw)
    decoder = PlanarArcTrajectoryNativeDecoder(
        resampled_vector_length=16,
        native_action_dim=3,
        raw_action_horizon=40,
    )
    trajectory = decoder.decode(token)
    assert trajectory.shape == (1, 40, 3)
    np.testing.assert_allclose(trajectory[0], raw, atol=1e-5)
    with pytest.raises(ValueError, match="requires per-point timing"):
        PlanarArcTrajectoryNativeDecoder(16, 3, 40, velocity_mode="mean")


def test_arc_transform_aligns_the_native_target_before_tokenization():
    """Hobs=2 must encode a_(t+1)..a_(t+40), never the 41-row loader window."""
    loader_window = np.column_stack(
        (np.arange(41, dtype=np.float32), np.zeros(41), np.zeros(41))
    )
    transforms = get_planar_arc_length_transform_list(
        action_horizon=32,
        raw_action_horizon=40,
        action_target_offset=1,
        min_distance_unit=40.0,
        resampled_vector_length=16,
        dt=1.0 / 30.0,
        velocity_mode="duration",
    )
    batch = {"actions": loader_window.copy()}
    for transform in transforms:
        batch = transform.transform(batch)

    assert batch["actions"].shape == (32, 5)
    decoded = PlanarArcTrajectoryNativeDecoder(
        resampled_vector_length=16,
        native_action_dim=3,
        raw_action_horizon=40,
    ).decode(batch["actions"])
    np.testing.assert_allclose(decoded[0], loader_window[1:], atol=1e-5)


@pytest.mark.parametrize("native_dim", [2, 3, 4])
def test_common_and_arc_adapters_decode_same_anchor(native_dim):
    token = torch.tensor([[[2.0, 3.0, 0.0, 1.0, 0.4], [9.0, 8.0, 1.0, 0.0, 0.0]]])
    dense = PlanarCommon5NativeDecoder(2, native_dim).decode(token)
    arc_input = torch.cat((token, torch.zeros(1, 1, 5)), dim=1)
    arc = PlanarArcWaypointZeroNativeDecoder(
        2, native_dim, velocity_mode="mean"
    ).decode(arc_input)
    assert dense.shape == (1, 2, native_dim)
    torch.testing.assert_close(arc, dense[:, :1])


def test_arc_rejects_nonfinite_or_short_input():
    transform = TokenizePlanarArcLength()
    with pytest.raises(ValueError):
        transform.transform({"actions": np.zeros((1, 3))})
    with pytest.raises(ValueError):
        transform.transform({"actions": np.array([[0, 0], [np.nan, 1]])})
