"""ChainGripper six-point action space: FK/IK, load-time transforms, native decoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from egomimic.pipeline.pushshapes import ChainGripperPointsNativeDecoder
from egomimic.rldb.embodiment.pushshapes import (
    get_chain_gripper_paper_points_transform_list,
    get_chain_gripper_points_action_state_transform_list,
)
from egomimic.rldb.zarr import chain_gripper_points as cg
from egomimic.rldb.zarr.action_chunk_transforms import (
    ChainGripperNative4ToPoints6,
    ChainGripperPoints6ToNative4,
)


def _controls(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(0.0, 512.0, (n, 2)),
            rng.uniform(-np.pi, np.pi, n),
            rng.uniform(0.0, 1.0, n),
        ]
    )


def test_fk_layout_and_geometry():
    control = np.array([[256.0, 128.0, 0.0, 0.0]])
    points = cg.pose_control_to_points(control)
    assert points.shape == (1, 6)
    left, center, right = points[0, 0:2], points[0, 2:4], points[0, 4:6]
    np.testing.assert_allclose(center, control[0, :2])
    radius = 2.0 * cg.CHAIN_GRIPPER_LINK_LEN * np.cos(cg.CHAIN_GRIPPER_OPEN_ANGLE / 2)
    np.testing.assert_allclose(np.linalg.norm(left - center), radius)
    np.testing.assert_allclose(np.linalg.norm(right - center), radius)
    # left tip lies at theta - phi, right at theta + phi: positive chirality.
    left_ray, right_ray = center - left, right - center
    assert left_ray[0] * right_ray[1] - left_ray[1] * right_ray[0] > 0


def test_exact_inverse_roundtrip():
    control = _controls(500)
    points = cg.pose_control_to_points(control)
    projection = cg.project_points_to_pose_control(points)
    assert projection.used_exact_inverse.all()
    np.testing.assert_allclose(
        projection.control, cg.canonicalize_pose_control(control), atol=1e-9
    )


def test_projection_recovers_noisy_points_approximately():
    control = _controls(200, seed=1)
    points = cg.pose_control_to_points(control) + np.random.default_rng(2).normal(
        0.0, 1.0, (200, 6)
    )
    projection = cg.project_points_to_pose_control(points, previous_control=control)
    assert not projection.used_exact_inverse.any()
    assert float(np.max(projection.point_rmse)) < 5.0
    canonical = cg.canonicalize_pose_control(control)
    assert np.abs(projection.control[:, :2] - canonical[:, :2]).max() < 5.0
    theta_err = np.angle(np.exp(1j * (projection.control[:, 2] - canonical[:, 2])))
    assert np.abs(theta_err).max() < 0.35


def test_native4_to_points6_transform_matches_fk_and_keeps_dtype():
    control = _controls(16, seed=3)
    batch = {"actions": torch.from_numpy(control).float()}
    out = ChainGripperNative4ToPoints6(keys=["actions"]).transform(batch)["actions"]
    assert out.shape == (16, 6) and out.dtype == torch.float32
    np.testing.assert_allclose(
        out.numpy(), cg.pose_control_to_points(control), rtol=0, atol=1e-3
    )
    with pytest.raises(ValueError):
        ChainGripperNative4ToPoints6(keys=["actions"]).transform(
            {"actions": torch.zeros(16, 3)}
        )


def test_points6_to_native4_uses_context_state_for_seed():
    control = _controls(2 * 16, seed=4).reshape(2, 16, 4)
    points = cg.pose_control_to_points(control)
    # Duplicate the middle point onto the left tip so the pair is degenerate at t=0;
    # the previous control (from the context state) must carry the orientation.
    degenerate = points.copy()
    degenerate[:, 0, 0:2] = degenerate[:, 0, 2:4]
    state = np.zeros((2, 6))
    state[:, :3] = control[:, 0, :3]
    transform = ChainGripperPoints6ToNative4(keys=["actions"])
    out = transform.transform({"actions": degenerate, "state_agent_obj": state})["actions"]
    assert out.shape == (2, 16, 4)
    diagnostics = transform.last_projection_diagnostics
    assert diagnostics["degenerate_count"] == 2
    np.testing.assert_allclose(out[:, 0, 2], cg.wrap_angle(control[:, 0, 2]), atol=1e-9)
    np.testing.assert_allclose(
        out[:, 1:], cg.canonicalize_pose_control(control)[:, 1:], atol=1e-9
    )


def test_decoder_and_embodiment_transform_lists():
    control = _controls(16, seed=5)
    state = np.zeros((16, 6))
    state[:, :3] = control[:, :3]
    batch = {
        "actions": torch.from_numpy(control).float(),
        "state_agent_model": torch.from_numpy(state).float(),
    }
    for transform in get_chain_gripper_points_action_state_transform_list():
        batch = transform.transform(batch)
    assert tuple(batch["actions"].shape) == (16, 6)
    assert tuple(batch["state_agent_model"].shape) == (16, 4)
    decoder = ChainGripperPointsNativeDecoder()
    native = decoder.decode(batch["actions"][None])
    assert tuple(native.shape) == (1, 16, 4)
    np.testing.assert_allclose(
        native[0].numpy(), cg.canonicalize_pose_control(control), atol=1e-3
    )
    paper = get_chain_gripper_paper_points_transform_list(
        action_horizon=16, action_target_offset=1
    )
    aligned = {"actions": np.asarray(_controls(17, seed=6))}
    for transform in paper:
        aligned = transform.transform(aligned)
    assert aligned["actions"].shape == (16, 6)
