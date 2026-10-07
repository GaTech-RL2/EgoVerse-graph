"""Local graph inference for robot embodiments using the training data contract."""

import copy
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.inference_config import (
    build_inference_config,
    find_inference_config,
    load_inference_config,
    validate_inference_config,
    validate_input_constants,
)
from egomimic.pipeline.inference_controls import (
    validate_control,
    validate_control_value,
)
from egomimic.robot.interface import ARM_OFFSET, pose_matrix, pose_vector
from egomimic.robot.teleop import rigid_transform


def resolve_inference_profile(training, inference_profiles):
    """Resolve an explicitly declared stable stage identifier."""
    if not isinstance(inference_profiles, Mapping) or len(inference_profiles) != 1:
        raise ValueError("Declare exactly one resolved model-owned inference profile")
    name, profile = next(iter(inference_profiles.items()))
    if not isinstance(profile, Mapping):
        raise ValueError("Inference profile must be a mapping")
    stage_id = profile.get("stage_id")
    stage_ids = OmegaConf.select(training, "model.pipeline.stage_ids", default={})
    if not isinstance(stage_id, str) or stage_id not in stage_ids:
        raise ValueError(
            "Inference profile requires a declared stable stage_id. Legacy class-matched "
            "profiles must be migrated explicitly to a model-owned inference contract."
        )
    index = stage_ids[stage_id]
    stages = training.model.pipeline.stages
    if type(index) is not int or not 0 <= index < len(stages):
        raise ValueError(f"Invalid stage position for {stage_id!r}")
    return name, profile, stages[index]


def configure_adapter_for_training(adapter_config, training, inference_profiles=None):
    """Merge declared model semantics into the station's calibration adapter."""
    if not isinstance(adapter_config, Mapping):
        raise TypeError("policy.adapter must be a mapping")
    result = dict(adapter_config)
    if inference_profiles is None:
        return result
    name, profile, _ = resolve_inference_profile(training, inference_profiles)
    shape = profile.get("native_shape")
    if OmegaConf.is_list(shape):
        shape = list(shape)
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(type(x) is not int or x <= 0 for x in shape)
    ):
        raise ValueError(
            f"Inference profile {name!r} native_shape must be [horizon, width]"
        )
    override = profile.get("adapter", {})
    declared_shape = OmegaConf.select(training, "model.inference.native_output.shape")
    if declared_shape is None or list(declared_shape) != list(shape):
        raise ValueError(
            f"Inference profile {name!r} expects a different native shape than the model declaration"
        )
    if not isinstance(override, Mapping):
        raise ValueError(f"Inference profile {name!r} adapter must be a mapping")
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass
class _InferenceControlBinding:
    name: str
    spec: dict
    value: object
    target_kind: str
    attribute_path: str
    owner: object | None = None
    attribute: str | None = None

    def validate(self, value):
        return validate_control_value(self.name, self.spec, value)

    def public(self):
        return {
            **{
                key: value
                for key, value in self.spec.items()
                if key not in {"target", "default"}
            },
            "value": self.value,
        }


def configure_profile_controls(graph, training, inference_profiles):
    """Bind declared settings by stable stage ID, with atomic preflight."""
    _, profile, _ = resolve_inference_profile(training, inference_profiles)
    stage_ids = OmegaConf.select(training, "model.pipeline.stage_ids", default={})
    bindings = []
    for name, spec in profile.get("overrides", {}).items():
        validate_control(name, spec, stage_ids=stage_ids)
        target = spec["target"]
        kind, path = target["kind"], target["attribute_path"]
        owner = attribute = None
        if kind == "stage_attribute":
            owner = graph.pipeline.stage_by_id(target["stage_id"])
            parts = path.split(".")
            for part in parts[:-1]:
                if not hasattr(owner, part):
                    raise ValueError(
                        f"Inference override target {path!r} does not exist"
                    )
                owner = getattr(owner, part)
            attribute = parts[-1]
            if not hasattr(owner, attribute):
                raise ValueError(f"Inference override target {path!r} does not exist")
        bindings.append(
            _InferenceControlBinding(
                name, dict(spec), spec["default"], kind, path, owner, attribute
            )
        )
    # No mutation occurs until every target and every default is valid.
    for binding in bindings:
        if binding.owner is not None:
            setattr(binding.owner, binding.attribute, binding.value)
    return tuple(bindings)


def load_normalizer(path):
    """Compatibility alias owned by graph integration; migrate to data_context_loader."""
    import warnings

    from egomimic.rldb.zarr.data_module import load_normalizer as restore

    warnings.warn(
        "Import load_normalizer from rldb.zarr.data_module; alias owner: graph integration",
        FutureWarning,
        stacklevel=2,
    )
    return restore(path)


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


def configure_flow_inference_steps(
    graph: PipelineAlgo, num_inference_steps: int
) -> None:
    """Deprecated setting alias; requires an explicit sampler ID, never class discovery."""
    import warnings

    warnings.warn(
        "Use model.inference runtime controls; alias owner: graph integration",
        DeprecationWarning,
        stacklevel=2,
    )
    if type(num_inference_steps) is not int or num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer")
    stage = graph.pipeline.stage_by_id("sampler")
    if not hasattr(stage, "num_inference_steps"):
        raise ValueError(
            "sampler does not expose num_inference_steps; configure its declared control"
        )
    stage.num_inference_steps = num_inference_steps


class InvalidGraphActionSample(ValueError):
    """A stochastic graph sample cannot be converted to a safe robot action."""


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
        gripper_clip_tolerance=0.0,
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
        if not isinstance(gripper_clip_tolerance, (float, int)) or not np.isfinite(
            gripper_clip_tolerance
        ):
            raise ValueError("gripper_clip_tolerance must be a finite number")
        self.gripper_clip_tolerance = float(gripper_clip_tolerance)
        if not 0 <= self.gripper_clip_tolerance <= 0.5:
            raise ValueError("gripper_clip_tolerance must be in [0, 0.5]")
        self.prompt, self.decoder = prompt, decoder

    def input_constants(self):
        return {"embodiment": self.embodiment_id}

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
                if (
                    not -self.gripper_clip_tolerance
                    <= values[-1]
                    <= 1 + self.gripper_clip_tolerance
                ):
                    raise InvalidGraphActionSample(
                        "Predicted gripper opening is outside [0, 1]"
                    )
                gripper = float(np.clip(values[-1], 0, 1))
                if width == 14:
                    target = pose_matrix(values[:6])
                else:
                    a, b = values[3:6].astype(float), values[6:9].astype(float)
                    if np.linalg.norm(a) < 1e-8:
                        raise InvalidGraphActionSample("Degenerate 6D rotation")
                    a /= np.linalg.norm(a)
                    b -= a * np.dot(a, b)
                    if np.linalg.norm(b) < 1e-8:
                        raise InvalidGraphActionSample("Degenerate 6D rotation")
                    b /= np.linalg.norm(b)
                    target = np.eye(4)
                    target[:3, 3] = values[:3]
                    target[:3, :3] = np.column_stack([a, b, np.cross(a, b)])
                base = (
                    anchors[arm]
                    if self.action_frame == "eef_frame"
                    else self.base_T_model[arm]
                )
                command.extend(np.r_[pose_vector(base @ target), gripper])
            actions.append(command)
        return np.asarray(actions, dtype=np.float64)


class GraphRobotPolicy:
    action_type = "cartesian"

    def __init__(
        self,
        graph,
        normalizer,
        adapter,
        max_valid_samples=1,
        inference_graph=None,
        inference_controls=(),
    ):
        if not isinstance(graph, PipelineAlgo):
            raise TypeError("Robot inference requires PipelineAlgo")
        if type(max_valid_samples) is not int or not 1 <= max_valid_samples <= 16:
            raise ValueError("max_valid_samples must be an integer in [1, 16]")
        self.graph, self.normalizer, self.adapter = graph, normalizer, adapter
        self.max_valid_samples = max_valid_samples
        if inference_graph is None:
            inference_graph = {}
        if not isinstance(inference_graph, Mapping):
            raise TypeError("policy.inference_graph must be a mapping")
        validate_input_constants(inference_graph, adapter.input_constants())
        input_contract = inference_graph.get("input", {})
        output_contract = inference_graph.get("output", {})
        if not isinstance(input_contract, Mapping) or not isinstance(
            output_contract, Mapping
        ):
            raise ValueError("Inference graph input and output must be mappings")
        self.history_length = input_contract.get("history_length", 1)
        if type(self.history_length) is not int or self.history_length <= 0:
            raise ValueError("Inference graph history_length must be positive")
        history_keys = input_contract.get(
            "history_keys", [adapter.proprio_key, *adapter.camera_keys.values()]
        )
        if (
            not isinstance(history_keys, (list, tuple))
            or not history_keys
            or any(not isinstance(key, str) or not key for key in history_keys)
        ):
            raise ValueError("Inference graph history_keys must be nonempty strings")
        self.history_keys = tuple(history_keys)
        self._observation_history = deque(maxlen=self.history_length)
        representation = output_contract.get("representation", "cartesian")
        if representation != self.action_type:
            raise ValueError(
                f"Inference graph output must be {self.action_type!r}, got "
                f"{representation!r}"
            )
        output_shape = output_contract.get("shape")
        if output_shape is None:
            self.output_shape = None
        elif (
            not isinstance(output_shape, (list, tuple))
            or len(output_shape) != 2
            or any(type(value) is not int or value <= 0 for value in output_shape)
        ):
            raise ValueError("Inference graph output shape must be [H, D]")
        else:
            self.output_shape = tuple(output_shape)
        inference_controls = tuple(inference_controls)
        self._inference_controls = {
            control.name: control for control in inference_controls
        }
        if len(self._inference_controls) != len(inference_controls):
            raise ValueError("Inference override names must be unique")
        self.replan_every = None
        self.native_output = dict(inference_graph.get("native_output", {}))
        for control in self._inference_controls.values():
            if control.target_kind == "policy_attribute":
                # This robot policy explicitly exposes only these runtime settings.
                if control.attribute_path not in {"replan_every", "max_valid_samples"}:
                    raise ValueError(
                        f"Robot policy does not expose {control.attribute_path!r}"
                    )
                setattr(self, control.attribute_path, control.value)
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

    def reset(self):
        """Clear inference-owned observation history at an episode boundary."""
        self._observation_history.clear()

    def inference_controls(self):
        """Return the profile's typed, operator-visible inference overrides."""
        return {
            name: control.public() for name, control in self._inference_controls.items()
        }

    def apply_inference_overrides(self, overrides):
        """Atomically validate and apply explicitly exposed inference values."""
        if not isinstance(overrides, Mapping) or not overrides:
            raise ValueError("Inference overrides must be a nonempty mapping")
        unknown = set(overrides) - set(self._inference_controls)
        if unknown:
            raise ValueError(
                "Inference override is not exposed by this model profile: "
                + ", ".join(sorted(unknown))
            )
        validated = {
            name: self._inference_controls[name].validate(value)
            for name, value in overrides.items()
        }
        for name, value in validated.items():
            control = self._inference_controls[name]
            if control.target_kind == "stage_attribute":
                setattr(control.owner, control.attribute, value)
            else:
                setattr(self, control.attribute_path, value)
            control.value = value
        return self.inference_controls()

    def execution_plan(self, prediction):
        """Choose the executable prefix; the full prediction remains visualizable."""
        actions = np.asarray(prediction)
        if self.replan_every is None:
            return actions
        return actions[: min(self.replan_every, len(actions))]

    def _observation(self, obs):
        values = self.adapter.observation(obs)
        snapshot = {}
        for key in self.history_keys:
            value = values.get(key)
            if not torch.is_tensor(value) or value.shape[0] != 1:
                raise ValueError(
                    f"Inference history key {key!r} must have leading batch size 1"
                )
            snapshot[key] = value.detach().clone()
        self._observation_history.append(snapshot)
        if self.history_length == 1:
            return values
        samples = list(self._observation_history)
        samples = [samples[0]] * (self.history_length - len(samples)) + samples
        for key in self.history_keys:
            values[key] = torch.stack([sample[key][0] for sample in samples], dim=0)[
                None
            ]
        return values

    @torch.no_grad()
    def predict(self, obs):
        adapter = self.adapter
        values = self.normalizer.normalize(
            self._observation(obs), adapter.embodiment_id
        )
        batch = self.graph.process_batch_for_training({"robot": values})
        error = None
        for _ in range(self.max_valid_samples):
            output_key = self.native_output.get("key", "pred_action")
            prediction = self.graph.forward_eval(batch)["robot"][output_key]
            expected = self.native_output.get("shape")
            if expected is not None and tuple(prediction.shape) != (1, *expected):
                raise ValueError(
                    f"Graph native output violates declared shape {expected}: {tuple(prediction.shape)}"
                )
            native = self.normalizer.unnormalize(
                {adapter.action_key: prediction}, adapter.embodiment_id
            )[adapter.action_key]
            try:
                actions = adapter.actions(native, obs)
                if self.output_shape is not None and actions.shape != self.output_shape:
                    raise ValueError(
                        "Inference graph violated its canonical output contract: "
                        f"expected {self.output_shape}, got {actions.shape}"
                    )
                return actions
            except InvalidGraphActionSample as caught:
                # HPT-Flow is stochastic. Reject the entire sampled plan rather
                # than clamping one actuator, then ask the graph for a new plan.
                error = caught
        raise ValueError(
            "Graph policy rejected all "
            f"{self.max_valid_samples} sampled plan(s): {error}. "
            "No robot command was issued."
        )


def load_graph_policy(config):
    training = OmegaConf.load(config["training_config"])
    device = validate_graph_device(str(config["device"]))
    inference_config_path = config.get("inference_config")
    if inference_config_path is None:
        inference_config_path = find_inference_config(config["checkpoint"])
    inference_graph = (
        load_inference_config(inference_config_path, training)
        if inference_config_path is not None
        else validate_inference_config(build_inference_config(training), training)
    )
    if inference_graph is not None and not isinstance(inference_graph, Mapping):
        raise TypeError("policy.inference_graph must be a mapping")
    inference_profiles = (
        None if inference_graph is None else inference_graph.get("profiles")
    )
    adapter_config = configure_adapter_for_training(
        config["adapter"], training, inference_profiles
    )
    adapter = instantiate(adapter_config)
    validate_input_constants(inference_graph, adapter.input_constants())
    supplied_keys = {
        adapter.proprio_key,
        "embodiment",
        "annotations",
        *adapter.camera_keys.values(),
    }
    missing = set(inference_graph["input"]["keys"]) - supplied_keys
    if missing:
        raise ValueError(
            f"Station adapter cannot supply required model observations: {sorted(missing)}"
        )
    loader = training.get("data_context_loader")
    if loader is None:
        raise ValueError(
            "Saved training config must declare data_context_loader for its immutable normalizer export"
        )
    normalizer = instantiate(loader, path=config["normalizer_path"])
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
    inference_controls = ()
    if inference_profiles is not None:
        inference_controls = configure_profile_controls(
            graph, training, inference_profiles
        )
    elif "num_inference_steps" in config:
        raise ValueError(
            "Move num_inference_steps into the model-owned typed inference controls"
        )
    graph.nets.eval()
    return GraphRobotPolicy(
        graph,
        normalizer,
        adapter,
        max_valid_samples=config.get("max_valid_samples", 1),
        inference_graph=inference_graph,
        inference_controls=inference_controls,
    )
