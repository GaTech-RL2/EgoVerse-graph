import math

import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import (
    PlanarArcWaypointZeroNativeDecoder,
    PlanarCommon5NativeDecoder,
)
from egomimic.rldb.embodiment.pushshapes import (
    get_planar_arc_length_transform_list,
)
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
    assert token.shape == (5, 5)
    np.testing.assert_allclose(token[0, :2], action[0, :2])
    np.testing.assert_allclose(token[0, 2:4], [1, 0], atol=1e-6)
    assert token[0, 4] == action[0, 3]
    np.testing.assert_allclose(token[-1], [2, 0, 0, 0, 0], atol=1e-6)


def test_rotation_radius_adds_metric_distance():
    xy = np.zeros((3, 2))
    theta = np.array([0, math.pi / 2, math.pi])
    assert planar_step_distance(xy).sum() == 0
    weighted = planar_step_distance(xy, theta, lambda_for_radius(30))
    assert np.all(weighted > 0)


def test_hybrid_rotation_budget_caps_a_shared_cartesian_window():
    """A small angular budget shortens the common arc window, not a stream."""
    action = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, math.pi], [20.0, 0.0, math.pi]]
    )
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
    assert token[-1, 0] == 0


def test_curvature_sampling_reduces_spacing_on_a_bend():
    straight_x = np.linspace(0.0, 10.0, 61)
    straight = np.column_stack((straight_x, np.zeros_like(straight_x)))
    angle = np.linspace(-math.pi / 2, 0.0, 41)[1:]
    bend = np.column_stack((10.0 + 2.0 * np.cos(angle), 2.0 + 2.0 * np.sin(angle)))
    xy = np.concatenate((straight, bend), axis=0)
    cumulative = np.concatenate((np.zeros(1), np.cumsum(planar_step_distance(xy))))

    _, targets = curvature_adaptive_curve_samples(
        xy,
        cumulative,
        float(cumulative[-1]),
        16,
        dense_samples=513,
    )
    straight_end = cumulative[len(straight) - 1]
    bend_gaps = np.diff(targets[targets >= straight_end])
    straight_gaps = np.diff(targets[targets <= straight_end])
    assert len(bend_gaps) >= 2
    assert np.median(bend_gaps) < np.median(straight_gaps)


def test_curvature_sampling_is_uniform_on_a_straight_curve():
    xy = np.column_stack((np.linspace(0.0, 12.0, 31), np.zeros(31)))
    cumulative = np.concatenate((np.zeros(1), np.cumsum(planar_step_distance(xy))))
    sampled, targets = curvature_adaptive_curve_samples(
        xy,
        cumulative,
        float(cumulative[-1]),
        7,
    )
    np.testing.assert_allclose(targets, np.linspace(0.0, 12.0, 7), atol=1e-8)
    np.testing.assert_allclose(sampled[:, 0], targets, atol=1e-8)
    np.testing.assert_allclose(sampled[:, 1], 0.0, atol=1e-8)


def test_curvature_token_preserves_anchor_shape_and_timing():
    t = np.linspace(0.0, 1.0, 40)
    action = np.column_stack((20.0 * t, 8.0 * t**2, 0.4 * t))
    token = TokenizePlanarArcLength(
        min_distance_unit=30.0,
        resampled_vector_length=16,
        waypoint_sampling="curvature",
    ).transform({"actions": action})["actions"]
    assert token.shape == (17, 5)
    np.testing.assert_allclose(token[0, :2], action[0, :2])
    assert token[-1, 0] > 0.0
    np.testing.assert_allclose(token[-1, 1:], 0.0)


def test_paper_dp_arc_alignment_slices_offset_before_tokenization():
    actions = np.column_stack(
        (
            np.arange(41, dtype=np.float64),
            np.zeros(41),
            np.zeros(41),
        )
    )
    transforms = get_planar_arc_length_transform_list(
        raw_action_horizon=40,
        action_target_offset=1,
        min_distance_unit=40.0,
        resampled_vector_length=16,
    )
    batch = {"actions": actions}
    for transform in transforms:
        batch = transform.transform(batch)
    assert batch["actions"].shape == (17, 5)
    np.testing.assert_allclose(batch["actions"][0, :2], actions[1, :2])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"waypoint_sampling": "unknown"}, "waypoint_sampling"),
        ({"curvature_dense_samples": 8}, "curvature_dense_samples"),
        ({"curvature_floor": 0.0}, "curvature_floor"),
        ({"curvature_floor": float("nan")}, "curvature_floor"),
    ],
)
def test_curvature_sampling_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TokenizePlanarArcLength(resampled_vector_length=16, **kwargs)


@pytest.mark.parametrize("native_dim", [2, 3, 4])
def test_common_and_arc_adapters_decode_same_anchor(native_dim):
    token = torch.tensor([[[2.0, 3.0, 0.0, 1.0, 0.4], [9.0, 8.0, 1.0, 0.0, 0.0]]])
    dense = PlanarCommon5NativeDecoder(2, native_dim).decode(token)
    arc_input = torch.cat((token, torch.zeros(1, 1, 5)), dim=1)
    arc = PlanarArcWaypointZeroNativeDecoder(2, native_dim).decode(arc_input)
    assert dense.shape == (1, 2, native_dim)
    torch.testing.assert_close(arc, dense[:, :1])


def test_native_decoders_preserve_energy_score_sample_and_batch_dimensions():
    token = torch.zeros(32, 2, 17, 5)
    token[..., 0, :2] = torch.tensor([2.0, 3.0])
    token[..., 0, 3] = 1.0
    arc = PlanarArcWaypointZeroNativeDecoder(16, 3).decode(token)
    assert arc.shape == (32, 2, 1, 3)
    torch.testing.assert_close(arc[..., 0, :2], token[..., 0, :2])

    dense = PlanarCommon5NativeDecoder(17, 3).decode(token)
    assert dense.shape == (32, 2, 17, 3)


def test_arc_rejects_nonfinite_or_short_input():
    transform = TokenizePlanarArcLength()
    with pytest.raises(ValueError):
        transform.transform({"actions": np.zeros((1, 3))})
    with pytest.raises(ValueError):
        transform.transform({"actions": np.array([[0, 0], [np.nan, 1]])})
