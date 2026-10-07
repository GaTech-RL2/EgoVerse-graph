"""Explicit native-action conversions; no hardware or checkpoint discovery."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


class CartesianRotationDecoder:
    def __init__(self, has_gripper: bool):
        if type(has_gripper) is not bool:
            raise TypeError("has_gripper must be declared as a boolean")
        self.has_gripper = has_gripper

    def __call__(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        half = 9 + int(self.has_gripper)
        if (
            actions.ndim != 3
            or actions.shape[-1] != half * 2
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Expected finite (B,H,{half * 2}) Cartesian 6D actions")
        arms = actions.reshape(*actions.shape[:-1], 2, half)
        first, second = arms[..., 3:6].copy(), arms[..., 6:9].copy()
        norm = np.linalg.norm(first, axis=-1, keepdims=True)
        if (norm < 1e-8).any():
            raise ValueError("Degenerate first rotation basis")
        first /= norm
        second -= (first * second).sum(axis=-1, keepdims=True) * first
        norm = np.linalg.norm(second, axis=-1, keepdims=True)
        if (norm < 1e-8).any():
            raise ValueError("Degenerate second rotation basis")
        second /= norm
        matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
        angles = (
            Rotation.from_matrix(matrix.reshape(-1, 3, 3))
            .as_euler("ZYX")
            .reshape(*arms.shape[:-1], 3)
        )
        pieces = [arms[..., :3], angles]
        if self.has_gripper:
            pieces.append(arms[..., -1:])
        return np.concatenate(pieces, axis=-1).reshape(*actions.shape[:-1], -1)
