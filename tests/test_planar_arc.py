import math

import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import (
    PlanarArcWaypointZeroRolloutAdapter,
    PlanarCommon5RolloutAdapter,
)
from egomimic.rldb.zarr.planar_arc import (
    PadPlanarAction,
    TokenizePlanarArcLength,
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


def test_zero_motion_holds_pose_and_grip():
    action = np.repeat(np.array([[4.0, 7.0, 0.5, 0.75]]), 5, axis=0)
    token = TokenizePlanarArcLength(resampled_vector_length=3).transform(
        {"actions": action}
    )["actions"]
    np.testing.assert_allclose(token[:3, :2], [[4, 7]] * 3)
    np.testing.assert_allclose(token[:3, 4], 0.75)
    assert token[-1, 0] == 0


@pytest.mark.parametrize("native_dim", [2, 3, 4])
def test_common_and_arc_adapters_decode_same_anchor(native_dim):
    token = torch.tensor([[[2.0, 3.0, 0.0, 1.0, 0.4], [9.0, 8.0, 1.0, 0.0, 0.0]]])
    dense = PlanarCommon5RolloutAdapter(2, native_dim).decode(token)
    arc_input = torch.cat((token, torch.zeros(1, 1, 5)), dim=1)
    arc = PlanarArcWaypointZeroRolloutAdapter(2, native_dim).decode(arc_input)
    assert dense.shape == (1, 2, native_dim)
    torch.testing.assert_close(arc, dense[:, :1])


def test_arc_rejects_nonfinite_or_short_input():
    transform = TokenizePlanarArcLength()
    with pytest.raises(ValueError):
        transform.transform({"actions": np.zeros((1, 3))})
    with pytest.raises(ValueError):
        transform.transform({"actions": np.array([[0, 0], [np.nan, 1]])})
