"""Explicit graph observation/action boundary for the Yam joint protocol."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _matrix(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1])
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Frame calibration must be a finite rigid 4x4 transform")
    return value


class YamCartesianAdapter:
    def __init__(
        self,
        kinematics,
        base_T_model,
        camera_keys,
        embodiment_id,
        rotation_mode="euler",
        action_frame="eef_frame",
        proprio_key="observations.state.ee_pose",
        action_key="actions_cartesian",
        image_hw=(480, 640),
        prompt="",
        decoder=None,
    ):
        if rotation_mode not in ("euler", "6D") or action_frame not in (
            "eef_frame",
            "model_frame",
        ):
            raise ValueError(
                "Select euler/6D rotation and eef_frame/model_frame actions"
            )
        if set(base_T_model) != {"left", "right"}:
            raise ValueError("Both arms require an explicit base_T_model calibration")
        if not camera_keys or not set(camera_keys) <= {
            "top",
            "left_wrist",
            "right_wrist",
        }:
            raise ValueError("Camera mappings must select Yam camera names")
        if len(set(camera_keys.values())) != len(camera_keys):
            raise ValueError("Each camera requires a distinct graph key")
        self.kinematics = kinematics
        self.base_T_model = {side: _matrix(t) for side, t in base_T_model.items()}
        self.camera_keys = dict(camera_keys)
        self.embodiment_id = int(embodiment_id)
        self.rotation_mode, self.action_frame = rotation_mode, action_frame
        self.proprio_key, self.action_key = proprio_key, action_key
        self.image_hw = tuple(int(x) for x in image_hw)
        if len(self.image_hw) != 2 or min(self.image_hw) <= 0:
            raise ValueError("image_hw must contain two positive dimensions")
        self.prompt, self.decoder = prompt, decoder

    def observation(self, samples):
        # Current graph Cartesian recipes consume one observation; the Yam
        # transport still supplies the required two-frame history in order.
        sample = samples[-1]
        proprio = []
        for side in ("left", "right"):
            q = sample[side]
            pose = np.linalg.inv(self.base_T_model[side]) @ self.kinematics.fk(q[:6])
            rot = (
                Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX")
                if self.rotation_mode == "euler"
                else np.r_[pose[:3, 0], pose[:3, 1]]
            )
            proprio.extend(np.r_[pose[:3, 3], rot, q[6]])
        values = {
            self.proprio_key: torch.tensor([proprio], dtype=torch.float32),
            "embodiment": torch.tensor([self.embodiment_id]),
            "annotations": [[self.prompt]],
        }
        for camera, key in self.camera_keys.items():
            # Match the dataset's CHW [0,1] representation before model transforms.
            rgb = np.ascontiguousarray(sample["images"][camera])
            image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
            values[key] = torch.nn.functional.interpolate(
                image, self.image_hw, mode="bilinear", align_corners=False
            )
        return values

    def actions(self, native, samples):
        if self.decoder is not None:
            native = self.decoder(native)
        if torch.is_tensor(native):
            native = native.detach().cpu().numpy()
        native = np.asarray(native)
        if native.ndim != 3 or native.shape[0] != 1 or native.shape[1] < 24:
            raise ValueError(
                "The deployed decoder must provide at least 24 action rows for one observation"
            )
        width = native.shape[-1]
        if width not in (14, 20) or not np.isfinite(native).all():
            raise ValueError(
                "Native Cartesian actions must be finite bimanual 14D Euler or 20D 6D poses"
            )
        # Decoded ARC is Euler even when other model recipes use 6D rotations.
        if self.decoder is None and width != (
            14 if self.rotation_mode == "euler" else 20
        ):
            raise ValueError(
                "Action layout differs from the configured rotation convention"
            )
        sample, half = samples[-1], width // 2
        current = {side: sample[side][:6].copy() for side in ("left", "right")}
        anchor = {side: self.kinematics.fk(q) for side, q in current.items()}
        actions = []
        for row in native[0, :24]:
            command = []
            for i, side in enumerate(("left", "right")):
                pose = row[i * half : (i + 1) * half]
                if not 0 <= pose[-1] <= 1:
                    raise ValueError("Predicted gripper opening is outside [0, 1]")
                matrix = np.eye(4)
                matrix[:3, 3] = pose[:3]
                if width == 14:
                    matrix[:3, :3] = Rotation.from_euler("ZYX", pose[3:6]).as_matrix()
                else:
                    a, b = pose[3:6].astype(float), pose[6:9].astype(float)
                    if np.linalg.norm(a) < 1e-8:
                        raise ValueError("Degenerate predicted 6D rotation")
                    a /= np.linalg.norm(a)
                    b -= a * np.dot(a, b)
                    if np.linalg.norm(b) < 1e-8:
                        raise ValueError("Degenerate predicted 6D rotation")
                    b /= np.linalg.norm(b)
                    matrix[:3, :3] = np.column_stack([a, b, np.cross(a, b)])
                base_T_target = (
                    anchor[side]
                    if self.action_frame == "eef_frame"
                    else self.base_T_model[side]
                ) @ matrix
                current[side] = np.asarray(
                    self.kinematics.ik(base_T_target, current[side])
                )
                if current[side].shape != (6,) or not np.isfinite(current[side]).all():
                    raise ValueError("IK returned an invalid joint target")
                command.extend(np.r_[current[side], pose[-1]])
            actions.append(command)
        return np.asarray(actions, dtype=np.float32)
