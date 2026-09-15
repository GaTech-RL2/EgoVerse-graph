"""Local graph inference for robot embodiments using the training data contract."""

import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.robot.interface import ARM_OFFSET, pose_matrix, pose_vector
from egomimic.robot.teleop import rigid_transform


def load_normalizer(path):
    payload = json.loads(Path(path).read_text())
    state = payload.get("normalizer_state", payload)
    required = {
        "norm_mode",
        "embodiments",
        "key_types",
        "zarr_keys",
        "shapes",
        "norm_stats",
    }
    if not required <= state.keys():
        raise ValueError(
            "Export a full normalizer_state with trainHydra norm_stats_only=true and the training recipe"
        )
    for name in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        state[name] = {int(key): value for key, value in state[name].items()}
    state["embodiments"] = [int(key) for key in state["embodiments"]]
    normalizer = MultiDataset.from_state(state)
    for embodiment in normalizer.embodiments:
        for key, stats in normalizer.norm_stats[embodiment].items():
            for values in stats.values():
                tensor = torch.as_tensor(values)
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Nonfinite normalization statistics: {embodiment}/{key}"
                    )
                torch.broadcast_to(tensor, tuple(normalizer.key_shape(key, embodiment)))
    return normalizer


def validate_graph_device(device: str) -> torch.device:
    """Fail before robot construction when PyTorch cannot execute on a GPU."""
    try:
        target = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid graph policy device: {device!r}") from error
    if target.type != "cuda":
        return target
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Requested a CUDA graph policy device, but PyTorch reports no CUDA device"
        )
    index = 0 if target.index is None else target.index
    if not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device {index}, but it is unavailable")
    capability = torch.cuda.get_device_capability(index)
    architecture = f"sm_{capability[0]}{capability[1]}"
    supported = tuple(torch.cuda.get_arch_list())
    if (
        architecture not in supported
        and f"compute_{capability[0]}{capability[1]}" not in supported
    ):
        supported_text = ", ".join(supported) or "none"
        raise RuntimeError(
            f"PyTorch cannot execute CUDA capability {architecture}; this build supports "
            f"{supported_text}. Install a compatible PyTorch build before opening a robot."
        )
    return target


class CartesianGraphAdapter:
    """Explicit camera keys, calibration and rotation layout from deployment YAML.

    base_T_model maps the training proprio/action frame into each arm base.
    Wrist-relative actions instead anchor at the observation's measured EEF pose.
    """

    def __init__(
        self,
        base_T_model,
        camera_keys,
        embodiment_id,
        rotation_mode,
        action_frame,
        image_hw,
        proprio_key="observations.state.ee_pose",
        action_key="actions_cartesian",
        prompt="",
        decoder=None,
    ):
        if rotation_mode not in ("euler", "6D") or action_frame not in (
            "eef_frame",
            "model_frame",
        ):
            raise ValueError("Select euler/6D and eef_frame/model_frame explicitly")
        if set(base_T_model) != set(ARM_OFFSET):
            raise ValueError("Both arms require an explicit base_T_model calibration")
        self.base_T_model = {
            arm: rigid_transform(value) for arm, value in base_T_model.items()
        }
        if not camera_keys or len(set(camera_keys.values())) != len(camera_keys):
            raise ValueError("Map each camera to a distinct graph key")
        self.camera_keys, self.embodiment_id = dict(camera_keys), int(embodiment_id)
        self.rotation_mode, self.action_frame = rotation_mode, action_frame
        self.proprio_key, self.action_key = proprio_key, action_key
        self.image_hw = tuple(int(x) for x in image_hw)
        if len(self.image_hw) != 2 or min(self.image_hw) <= 0:
            raise ValueError("image_hw must be two positive dimensions")
        self.prompt, self.decoder = prompt, decoder

    def observation(self, obs):
        proprio = []
        for arm, offset in ARM_OFFSET.items():
            pose = np.linalg.inv(self.base_T_model[arm]) @ pose_matrix(
                obs["ee_poses"][offset : offset + 6]
            )
            rot = (
                Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX")
                if self.rotation_mode == "euler"
                else np.r_[pose[:3, 0], pose[:3, 1]]
            )
            proprio.extend(np.r_[pose[:3, 3], rot, obs["joint_positions"][offset + 6]])
        values = {
            self.proprio_key: torch.tensor([proprio], dtype=torch.float32),
            "embodiment": torch.tensor([self.embodiment_id]),
            "annotations": [[self.prompt]],
        }
        for camera, key in self.camera_keys.items():
            bgr = obs.get(camera)
            if (
                bgr is None
                or bgr.ndim != 3
                or bgr.shape[-1] != 3
                or bgr.dtype != np.uint8
            ):
                raise ValueError(f"Missing uint8 BGR observation: {camera}")
            image = (
                torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))
                .permute(2, 0, 1)[None]
                .float()
                / 255.0
            )
            values[key] = torch.nn.functional.interpolate(
                image, self.image_hw, mode="bilinear", align_corners=False
            )
        return values

    def actions(self, native, obs):
        if self.decoder is not None:
            native = self.decoder(native)
        if torch.is_tensor(native):
            native = native.detach().cpu().numpy()
        native = np.asarray(native)
        width = 14 if self.decoder is not None or self.rotation_mode == "euler" else 20
        if (
            native.ndim != 3
            or native.shape[0] != 1
            or native.shape[1] < 1
            or native.shape[-1] != width
            or not np.isfinite(native).all()
        ):
            raise ValueError(f"Expected finite Cartesian graph output (1, H, {width})")
        anchors = {
            arm: pose_matrix(obs["ee_poses"][offset : offset + 6])
            for arm, offset in ARM_OFFSET.items()
        }
        actions = []
        for row in native[0]:
            command = []
            for index, arm in enumerate(ARM_OFFSET):
                half = width // 2
                values = row[index * half : (index + 1) * half]
                if not 0 <= values[-1] <= 1:
                    raise ValueError("Predicted gripper opening is outside [0, 1]")
                if width == 14:
                    target = pose_matrix(values[:6])
                else:
                    a, b = values[3:6].astype(float), values[6:9].astype(float)
                    if np.linalg.norm(a) < 1e-8:
                        raise ValueError("Degenerate 6D rotation")
                    a /= np.linalg.norm(a)
                    b -= a * np.dot(a, b)
                    if np.linalg.norm(b) < 1e-8:
                        raise ValueError("Degenerate 6D rotation")
                    b /= np.linalg.norm(b)
                    target = np.eye(4)
                    target[:3, 3] = values[:3]
                    target[:3, :3] = np.column_stack([a, b, np.cross(a, b)])
                base = (
                    anchors[arm]
                    if self.action_frame == "eef_frame"
                    else self.base_T_model[arm]
                )
                command.extend(np.r_[pose_vector(base @ target), values[-1]])
            actions.append(command)
        return np.asarray(actions, dtype=np.float64)


class GraphRobotPolicy:
    action_type = "cartesian"

    def __init__(self, graph, normalizer, adapter):
        if not isinstance(graph, PipelineAlgo):
            raise TypeError("Robot inference requires PipelineAlgo")
        self.graph, self.normalizer, self.adapter = graph, normalizer, adapter
        embodiment = adapter.embodiment_id
        if embodiment not in normalizer.embodiments:
            raise ValueError("Embodiment is absent from the training normalizer")
        for key in (adapter.proprio_key, adapter.action_key):
            if (
                normalizer.zarr_keys[embodiment].get(key) != key
                or normalizer.key_types[embodiment].get(key)
                not in normalizer.NORMALIZE_KEY_TYPES
            ):
                raise ValueError(
                    f"Live graph key must match the normalized training keymap: {key}"
                )
            if key not in normalizer.shapes[embodiment]:
                raise ValueError(f"Key is absent from the training schema: {key}")
            if (
                normalizer.norm_mode != "none"
                and key not in normalizer.norm_stats[embodiment]
            ):
                raise ValueError(f"Missing normalization statistics: {key}")

    @torch.no_grad()
    def predict(self, obs):
        adapter = self.adapter
        values = self.normalizer.normalize(
            adapter.observation(obs), adapter.embodiment_id
        )
        batch = self.graph.process_batch_for_training({"robot": values})
        prediction = self.graph.forward_eval(batch)["robot"]["pred_action"]
        native = self.normalizer.unnormalize(
            {adapter.action_key: prediction}, adapter.embodiment_id
        )[adapter.action_key]
        return adapter.actions(native, obs)


def load_graph_policy(config):
    normalizer = load_normalizer(config["normalizer_path"])
    training = OmegaConf.load(config["training_config"])
    device = validate_graph_device(str(config["device"]))
    graph = instantiate(training.model.pipeline, device=str(device))
    if not isinstance(graph, PipelineAlgo):
        raise TypeError("Robot inference requires a graph PipelineAlgo")
    graph.bind_data_context(normalizer=normalizer)
    checkpoint = torch.load(
        config["checkpoint"], map_location="cpu", weights_only=False
    )
    strict_load_pipeline_checkpoint(
        graph, checkpoint, use_ema=bool(config.get("use_ema", False))
    )
    graph.nets.eval()
    return GraphRobotPolicy(graph, normalizer, instantiate(config["adapter"]))
