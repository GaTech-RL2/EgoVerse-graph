"""Local graph inference for robot embodiments using the training data contract."""

import copy
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
)
from egomimic.pipeline.stages_flow import FlowDenoiserStage
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.robot.arc_decoder import TimeChunkRetimer
from egomimic.robot.arc_speed import ARC_SPEED_RANGE
from egomimic.robot.interface import ARM_OFFSET, pose_matrix, pose_vector
from egomimic.robot.teleop import rigid_transform


def resolve_inference_profile(training, inference_profiles):
    """Select one YAML profile from checkpoint-declared graph structure."""
    if not isinstance(inference_profiles, Mapping):
        raise TypeError("Inference graph profiles must be a mapping")
    stages = OmegaConf.select(training, "model.pipeline.stages")
    if stages is None:
        raise ValueError("Selected model must declare model.pipeline.stages")
    stages = OmegaConf.to_container(stages, resolve=True)
    if not isinstance(stages, list) or not all(
        isinstance(stage, dict) for stage in stages
    ):
        raise ValueError("Selected model pipeline stages must be a list of mappings")
    variant = OmegaConf.select(training, "e1.variant")
    matches = []
    for name, profile in inference_profiles.items():
        if not isinstance(name, str) or not isinstance(profile, Mapping):
            raise ValueError("Every policy inference profile must be a named mapping")
        match = profile.get("match")
        if not isinstance(match, Mapping):
            raise ValueError(
                f"Inference graph profile {name!r} match must be a mapping"
            )
        target = match.get("stage_target")
        if not isinstance(target, str) or not target:
            raise ValueError(
                f"Inference graph profile {name!r} stage_target must be a string"
            )
        matching_stages = [stage for stage in stages if stage.get("_target_") == target]
        if not matching_stages:
            continue
        if len(matching_stages) != 1:
            raise ValueError(
                f"Selected model contains {len(matching_stages)} stages matching {target!r}"
            )
        required_horizon = match.get("action_horizon")
        if (
            "action_horizon" in match
            and required_horizon != matching_stages[0].get("action_horizon")
        ):
            continue
        # A missing variant remains a wildcard for backwards compatibility.
        # An explicitly declared null variant is an exact match for baseline
        # Flow models whose training config has no e1.variant field/value.
        if "variant" in match and match.get("variant") != variant:
            continue
        matches.append((name, profile, matching_stages[0]))
    if len(matches) != 1:
        targets = sorted(
            stage.get("_target_") for stage in stages if stage.get("_target_")
        )
        raise ValueError(
            "Selected model must match exactly one policy inference profile; "
            f"matched {len(matches)} for variant={variant!r}, stages={targets}"
        )
    return matches[0]


def configure_adapter_for_training(adapter_config, training, inference_profiles=None):
    """Apply the model-matched YAML adapter and validate native output shape."""
    if not isinstance(adapter_config, Mapping):
        raise TypeError("policy.adapter must be a mapping")
    result = dict(adapter_config)
    if inference_profiles is None:
        return result
    name, profile, stage = resolve_inference_profile(training, inference_profiles)
    expected_shape = profile.get("native_shape")
    if OmegaConf.is_list(expected_shape):
        expected_shape = tuple(expected_shape)
    if (
        not isinstance(expected_shape, (list, tuple))
        or len(expected_shape) != 2
        or any(type(value) is not int or value <= 0 for value in expected_shape)
    ):
        raise ValueError(
            f"Inference graph profile {name!r} native_shape must be [H, D]"
        )
    action_dim = stage.get("action_dim")
    action_horizon = stage.get("action_horizon")
    if (
        type(action_horizon) is not int
        or type(action_dim) is not int
        or action_horizon <= 0
        or action_dim <= 0
    ):
        raise ValueError(
            "Selected model hpt.action_horizon and hpt.action_dim must be positive integers"
        )
    actual_shape = (action_horizon, action_dim)
    if tuple(expected_shape) != actual_shape:
        raise ValueError(
            f"Selected model emits {actual_shape}, but YAML profile {name!r} "
            f"expects {tuple(expected_shape)}"
        )
    override = profile.get("adapter", {})
    if not isinstance(override, Mapping):
        raise ValueError(f"Inference graph profile {name!r} adapter must be a mapping")
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        else:
            if key == "decoder":
                result.pop("roundtrip_arc", None)
            result[key] = copy.deepcopy(value)
    return result


@dataclass
class _InferenceControlBinding:
    """Validated YAML control plus its private runtime mutation target."""

    name: str
    label: str
    description: str
    minimum: int
    maximum: int
    step: int
    value: int
    target_kind: str
    attribute_path: str
    owner: object | None = None
    attribute: str | None = None

    def validate(self, value):
        if type(value) is not int or not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"Inference override {self.name!r} must be an integer in "
                f"[{self.minimum}, {self.maximum}]"
            )
        if (value - self.minimum) % self.step:
            raise ValueError(
                f"Inference override {self.name!r} must use step {self.step}"
            )
        return value

    def public(self):
        return {
            "label": self.label,
            "description": self.description,
            "type": "integer",
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "value": self.value,
        }


def configure_profile_controls(graph, training, inference_profiles):
    """Bind only the runtime controls explicitly exposed by the matched profile."""
    name, profile, _ = resolve_inference_profile(training, inference_profiles)
    controls = profile.get("overrides", {})
    if not isinstance(controls, Mapping):
        raise ValueError(
            f"Inference graph profile {name!r} overrides must be a mapping"
        )
    target = profile["match"]["stage_target"]
    stages = [
        stage
        for stage in graph.pipeline.stages
        if f"{type(stage).__module__}.{type(stage).__qualname__}" == target
    ]
    bindings = []
    for control_name, spec in controls.items():
        if (
            not isinstance(control_name, str)
            or not control_name.isidentifier()
            or control_name.startswith("_")
            or not isinstance(spec, Mapping)
        ):
            raise ValueError(
                f"Inference graph profile {name!r} has an invalid override declaration"
            )
        if spec.get("type") != "integer":
            raise ValueError(
                f"Inference override {control_name!r} must declare type: integer"
            )
        label = spec.get("label")
        description = spec.get("description", "")
        if not isinstance(label, str) or not label or not isinstance(description, str):
            raise ValueError(
                f"Inference override {control_name!r} needs a label and description"
            )
        minimum, maximum = spec.get("min"), spec.get("max")
        step, value = spec.get("step", 1), spec.get("default")
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or type(step) is not int
            or minimum > maximum
            or step <= 0
        ):
            raise ValueError(
                f"Inference override {control_name!r} has invalid integer bounds"
            )
        target_spec = spec.get("target")
        if not isinstance(target_spec, Mapping):
            raise ValueError(f"Inference override {control_name!r} needs a target")
        target_kind = target_spec.get("kind")
        path = target_spec.get("attribute_path")
        if (
            target_kind not in {"stage_attribute", "policy_attribute"}
            or not isinstance(path, str)
            or not path
            or any(
                not part.isidentifier() or part.startswith("_")
                for part in path.split(".")
            )
        ):
            raise ValueError(f"Inference override {control_name!r} target is invalid")
        owner = attribute = None
        if target_kind == "stage_attribute":
            if len(stages) != 1:
                raise ValueError(
                    f"YAML profile {name!r} requires exactly one {target}, "
                    f"found {len(stages)}"
                )
            owner = stages[0]
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
        elif path != "replan_every":
            raise ValueError(
                "The only supported policy inference override is replan_every"
            )
        binding = _InferenceControlBinding(
            name=control_name,
            label=label,
            description=description,
            minimum=minimum,
            maximum=maximum,
            step=step,
            value=value,
            target_kind=target_kind,
            attribute_path=path,
            owner=owner,
            attribute=attribute,
        )
        binding.validate(value)
        if target_kind == "stage_attribute":
            setattr(owner, attribute, value)
        bindings.append(binding)
    return tuple(bindings)


def arc_speed_controls(decoder):
    """Replay-tempo controls for a tempo-capable decoder (ARC decoder or the
    time-chunk retimer); none otherwise.

    Tempo belongs to the decoder, not to the model profile, so every such policy
    exposes it whatever produced its profile (rollout YAML, checkpoint sidecar or
    a config-derived graph). Integer percents fit the typed integer controls.
    """
    if not callable(getattr(decoder, "set_speed", None)):
        return ()
    low, high = (int(round(value * 100)) for value in ARC_SPEED_RANGE)
    specs = [(
        "arc_speed_percent", "Replay speed (%)", "speed_percent",
        "100 = demonstrated tempo. Scales moving phases from the next prediction; "
        "set Hold speed to the same value for a uniform (naive) speed-up. "
        "A faster replay ends sooner: keep Repredict every at or under the path "
        "length the rollout log reports, or the arms stand still for the rest.",
    )]
    if decoder.holds:
        specs.append((
            "arc_hold_speed_percent", "Hold speed (%)", "hold_speed_percent",
            "Tempo of holds, grasps and slow placement (both arms under "
            f"{decoder.hold_threshold:g} m/s). Leave at 100 until gripper closing "
            "time has been checked against the shorter hold.",
        ))
    controls = []
    for name, label, attribute, description in specs:
        control = _InferenceControlBinding(
            name=name, label=label, description=description, minimum=low,
            maximum=high, step=5, value=getattr(decoder, attribute),
            target_kind="decoder_attribute", attribute_path=attribute,
            owner=decoder, attribute=attribute,
        )
        control.validate(control.value)
        controls.append(control)
    return tuple(controls)


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


def configure_flow_inference_steps(
    graph: PipelineAlgo, num_inference_steps: int
) -> None:
    """Override the rollout-only Euler solver budget for one FlowDenoiser."""
    if type(num_inference_steps) is not int or num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer")
    stages = [
        stage for stage in graph.pipeline.stages if isinstance(stage, FlowDenoiserStage)
    ]
    if len(stages) != 1:
        raise ValueError(
            "A rollout num_inference_steps override requires exactly one "
            f"FlowDenoiserStage, found {len(stages)}"
        )
    stages[0].num_inference_steps = num_inference_steps


class InvalidGraphActionSample(ValueError):
    """A stochastic graph sample cannot be converted to a safe robot action."""


def _plain_config_node(value):
    """Convert one OmegaConf node to ordinary Python containers."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def validate_training_input_contract(adapter_config, training):
    """Fail closed when rollout inputs differ from the selected Yam recipe.

    The robot boundary is intentionally explicit: the live adapter must feed
    the same three RGB modalities, image dimensions, EEF-relative Cartesian
    transform, Euler layout, and 14D proprio vector used by the checkpoint.
    Older minimal test configs do not contain a data resolver, so those remain
    valid; a real Yam training config is checked strictly.
    """
    datasets = _plain_config_node(
        OmegaConf.select(training, "data.train_datasets", default=None)
    )
    if not isinstance(datasets, Mapping):
        return
    dataset = datasets.get("yam_bimanual")
    if not isinstance(dataset, Mapping):
        return
    resolver = dataset.get("resolver")
    resolver = _plain_config_node(resolver)
    if not isinstance(resolver, Mapping):
        return

    configured_hw = resolver.get("image_hw")
    adapter_hw = tuple(int(value) for value in adapter_config.get("image_hw", ()))
    if configured_hw is not None and tuple(map(int, configured_hw)) != adapter_hw:
        raise ValueError(
            "Rollout image_hw does not match the Yam training resolver: "
            f"training={tuple(configured_hw)}, rollout={adapter_hw}"
        )

    key_map = _plain_config_node(resolver.get("key_map"))
    training_action_mode = None
    if isinstance(key_map, Mapping) and str(key_map.get("_target_", "")).endswith(
        "Yam.get_keymap"
    ):
        training_action_mode = key_map.get("keymap_mode")
        if training_action_mode not in {
            "cartesian",
            "arc_tokenizer_cartesian",
            "hybrid_arc_tokenizer_cartesian",
        }:
            raise ValueError(
                "Yam rollout requires a Cartesian or supported ARC training "
                f"keymap, got {training_action_mode!r}"
            )

    transform = _plain_config_node(resolver.get("transform_list"))
    if isinstance(transform, Mapping) and str(
        transform.get("_target_", "")
    ).endswith("Yam.get_transform_list"):
        expected = {
            "coord_frame": "eef_frame",
            "rotation_mode": adapter_config.get("rotation_mode"),
        }
        if (
            training_action_mode is not None
            and transform.get("action_mode") != training_action_mode
        ):
            raise ValueError(
                "Yam keymap and transform action modes disagree: "
                f"keymap={training_action_mode!r}, "
                f"transform={transform.get('action_mode')!r}"
            )
        mismatches = {
            key: (expected[key], transform.get(key))
            for key in expected
            if transform.get(key) != expected[key]
        }
        if mismatches:
            raise ValueError(
                "Rollout pose/action transform differs from the Yam training "
                f"contract: {mismatches}"
            )

    stages = _plain_config_node(
        OmegaConf.select(training, "model.pipeline.stages", default=None)
    )
    if not isinstance(stages, list):
        return
    image_stems = {}
    all_selected_stems = {}
    for stage in stages:
        if not isinstance(stage, Mapping) or not str(stage.get("_target_", "")).endswith(
            "HPTStemStage"
        ):
            continue
        stems = stage.get("stems", {})
        if isinstance(stems, Mapping):
            all_selected_stems.update(stems)
        aliases = stage.get("selector_aliases", {})
        alias = aliases.get(7, aliases.get("7")) if isinstance(aliases, Mapping) else None
        domains = stage.get("domain_stems", {})
        selected = domains.get(alias, {}) if isinstance(domains, Mapping) else {}
        if isinstance(selected, Mapping):
            all_selected_stems.update(selected)

    for key, stem in all_selected_stems.items():
        if not isinstance(stem, Mapping):
            continue
        if str(stem.get("_target", stem.get("_target_", ""))).endswith(
            "hpt_stems.ResNet"
        ):
            image_stems[key] = stem

    if image_stems:
        expected_camera_keys = set(image_stems)
        actual_camera_keys = set(
            dict(adapter_config.get("camera_keys", {})).values()
        )
        if actual_camera_keys != expected_camera_keys:
            raise ValueError(
                "Rollout camera graph keys do not match the selected HPT model: "
                f"training={sorted(expected_camera_keys)}, "
                f"rollout={sorted(actual_camera_keys)}"
            )
        for key, stem in image_stems.items():
            weights = stem.get("weights")
            if weights not in (None, "null") and stem.get("imagenet_normalize") is not True:
                raise ValueError(
                    f"Pretrained image stem {key!r} is missing imagenet_normalize=true"
                )

    proprio_key = adapter_config.get("proprio_key")
    if all_selected_stems and proprio_key not in all_selected_stems:
        raise ValueError(
            f"Rollout proprio key {proprio_key!r} is absent from the selected HPT stems"
        )
    action_targets = [
        stage
        for stage in stages
        if isinstance(stage, Mapping)
        and stage.get("_target_") == "egomimic.pipeline.stages_io.ActionTargetBuilder"
    ]
    if action_targets and any(
        stage.get("action_key") != adapter_config.get("action_key")
        for stage in action_targets
    ):
        raise ValueError(
            "Rollout action_key does not match the selected training graph"
        )


def validate_native_action_contract(normalizer, adapter_config, training, profiles):
    """Ensure the normalizer and selected profile agree on native model tokens."""
    if profiles is None:
        return
    name, profile, _ = resolve_inference_profile(training, profiles)
    expected_shape = profile.get("native_shape")
    if OmegaConf.is_list(expected_shape):
        expected_shape = tuple(expected_shape)
    if not isinstance(expected_shape, (list, tuple)):
        raise ValueError(f"Inference graph profile {name!r} has no native_shape")
    embodiment = int(adapter_config.get("embodiment_id"))
    action_key = adapter_config.get("action_key", "actions_cartesian")
    actual_shape = tuple(normalizer.key_shape(action_key, embodiment))
    if actual_shape != tuple(expected_shape):
        raise ValueError(
            f"Selected model normalizer has native action shape {actual_shape}, "
            f"but inference profile {name!r} declares {tuple(expected_shape)}"
        )


def validate_temporary_arc_roundtrip_contract(
    normalizer, adapter_config, training
):
    """Keep the temporary round trip limited to direct baseline HPT output."""
    if adapter_config.get("roundtrip_arc") is None:
        return
    action_key = adapter_config.get("action_key", "actions_cartesian")
    embodiment = int(adapter_config.get("embodiment_id"))
    shape = tuple(normalizer.key_shape(action_key, embodiment))
    action_mode = OmegaConf.select(training, "abc.action_mode", default=None)
    if shape != (100, 14) or action_mode != "cartesian":
        raise ValueError(
            "Temporary ARC round trip is only valid for direct 100x14 "
            f"Cartesian baseline output; got shape={shape}, action_mode={action_mode!r}"
        )


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
        roundtrip_arc=None,
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
        if decoder is not None and roundtrip_arc is not None:
            raise ValueError("ARC decoder and temporary ARC round trip are exclusive")
        self.prompt, self.decoder, self.roundtrip_arc = prompt, decoder, roundtrip_arc

    def observation(self, obs):
        ee_poses = np.asarray(obs.get("ee_poses"), dtype=float)
        joint_positions = np.asarray(obs.get("joint_positions"), dtype=float)
        if (
            ee_poses.shape != (14,)
            or joint_positions.shape != (14,)
            or not np.isfinite(ee_poses).all()
            or not np.isfinite(joint_positions).all()
        ):
            raise ValueError(
                "Yam observation must contain finite 14D ee_poses and joint_positions"
            )
        proprio = []
        for arm, offset in ARM_OFFSET.items():
            pose = np.linalg.inv(self.base_T_model[arm]) @ pose_matrix(
                ee_poses[offset : offset + 6]
            )
            rot = (
                Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX")
                if self.rotation_mode == "euler"
                else np.r_[pose[:3, 0], pose[:3, 1]]
            )
            proprio.extend(np.r_[pose[:3, 3], rot, joint_positions[offset + 6]])
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
        if self.roundtrip_arc is not None:
            native = self.roundtrip_arc(native)
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
        self._replan_every = None
        for control in self._inference_controls.values():
            if control.target_kind == "policy_attribute":
                self._replan_every = control.value
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
        expected_proprio_dim = 14 if adapter.rotation_mode == "euler" else 20
        proprio_shape = tuple(normalizer.key_shape(adapter.proprio_key, embodiment))
        if proprio_shape != (expected_proprio_dim,):
            raise ValueError(
                "Training proprio shape does not match the rollout rotation mode: "
                f"expected {(expected_proprio_dim,)}, got {proprio_shape}"
            )
        if adapter.decoder is None:
            action_shape = tuple(normalizer.key_shape(adapter.action_key, embodiment))
            if len(action_shape) != 2 or action_shape[-1] != expected_proprio_dim:
                raise ValueError(
                    "Training Cartesian action shape does not match the rollout "
                    f"rotation mode: expected [..., {expected_proprio_dim}], "
                    f"got {action_shape}"
                )
        else:
            decoder_shape = getattr(adapter.decoder, "shape", None)
            action_shape = tuple(normalizer.key_shape(adapter.action_key, embodiment))
            if decoder_shape is not None and tuple(decoder_shape) != action_shape:
                raise ValueError(
                    "Selected tokenizer decoder expects native action shape "
                    f"{tuple(decoder_shape)}, but training normalizer provides "
                    f"{action_shape}"
                )

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
            if control.owner is not None:
                setattr(control.owner, control.attribute, value)
            else:
                self._replan_every = value
            control.value = value
        return self.inference_controls()

    def execution_plan(self, prediction):
        """Choose the executable prefix; the full prediction remains visualizable."""
        actions = np.asarray(prediction)
        if self._replan_every is None:
            return actions
        stats = getattr(getattr(self.adapter, "decoder", None), "last_stats", None) or {}
        valid = stats.get("valid_steps")
        if valid is not None and valid < self._replan_every and stats.get("speed") != 1.0:
            print(
                f"ARC {stats['speed']:.2f}x: the decoded path lasts {valid} steps, "
                f"under Repredict every {self._replan_every}; the arms hold still for "
                "the rest, so lower Repredict every to keep the speed-up"
            )
        return actions[: min(self._replan_every, len(actions))]

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
            prediction = self.graph.forward_eval(batch)["robot"]["pred_action"]
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


def resolve_inference_graph(config, training):
    """Load a checkpoint sidecar, migrating stale codec hashes in memory."""
    inference_config_path = config.get("inference_config")
    if inference_config_path is None:
        inference_config_path = find_inference_config(config["checkpoint"])
    if inference_config_path is not None:
        try:
            return load_inference_config(inference_config_path, training)
        except ValueError as error:
            # Older artifacts can have the same model pipeline but a contract
            # hash from before a decoder was added or an inference field was
            # included. Re-derive only that stale-contract case; pipeline and
            # graph-content mismatches remain fail-closed.
            if (
                not config.get("auto_inference_config", False)
                or "codec and inference defaults" not in str(error)
            ):
                raise
            artifact = build_inference_config(training)
            if artifact.get("status") != "ready":
                reason = artifact.get("reason", "model has no rollout contract")
                raise ValueError(
                    f"Stale inference config could not be migrated: {reason}"
                ) from error
            print(
                f"Using a config-derived inference graph for stale sidecar "
                f"{inference_config_path}: {error}"
            )
            return artifact["inference_graph"]
    if config.get("auto_inference_config", False):
        artifact = build_inference_config(training)
        if artifact.get("status") != "ready":
            reason = artifact.get("reason", "model has no rollout contract")
            raise ValueError(f"Selected model is not rollout-ready: {reason}")
        return artifact["inference_graph"]
    return config.get("inference_graph")


def load_graph_policy(config):
    normalizer = load_normalizer(config["normalizer_path"])
    training = OmegaConf.load(config["training_config"])
    device = validate_graph_device(str(config["device"]))
    execution_whitelist = config.get("execution_whitelist")
    instantiate_kwargs = (
        {"_execution_whitelist_": execution_whitelist}
        if execution_whitelist is not None
        else {}
    )
    graph = instantiate(
        training.model.pipeline,
        device=str(device),
        **instantiate_kwargs,
    )
    if not isinstance(graph, PipelineAlgo):
        raise TypeError("Robot inference requires a graph PipelineAlgo")
    graph.bind_data_context(normalizer=normalizer)
    checkpoint = torch.load(
        config["checkpoint"], map_location="cpu", weights_only=False
    )
    strict_load_pipeline_checkpoint(
        graph, checkpoint, use_ema=bool(config.get("use_ema", False))
    )
    inference_graph = resolve_inference_graph(config, training)
    if inference_graph is not None and not isinstance(inference_graph, Mapping):
        raise TypeError("policy.inference_graph must be a mapping")
    inference_profiles = (
        None if inference_graph is None else inference_graph.get("profiles")
    )
    if inference_profiles is None:
        inference_profiles = config.get("inference_profiles")
    inference_controls = ()
    if inference_profiles is not None:
        inference_controls = configure_profile_controls(
            graph, training, inference_profiles
        )
    elif "num_inference_steps" in config:
        # Backward compatibility for older Flow-only rollout profiles.
        configure_flow_inference_steps(graph, config["num_inference_steps"])
    graph.nets.eval()
    adapter_config = configure_adapter_for_training(
        config["adapter"], training, inference_profiles
    )
    validate_training_input_contract(adapter_config, training)
    validate_native_action_contract(
        normalizer, adapter_config, training, inference_profiles
    )
    validate_temporary_arc_roundtrip_contract(normalizer, adapter_config, training)
    adapter = instantiate(adapter_config, **instantiate_kwargs)
    if adapter.decoder is None and adapter.roundtrip_arc is None and adapter.rotation_mode == "euler":
        # A time-indexed policy gets the naive speed-up: its own chunk replayed faster.
        # Identity at 100 %, so the default rollout is unchanged.
        horizon, width = normalizer.key_shape(adapter.action_key, adapter.embodiment_id)
        if width == 14:
            adapter.decoder = TimeChunkRetimer(int(horizon))
    inference_controls = (*inference_controls, *arc_speed_controls(adapter.decoder))
    if adapter.roundtrip_arc is not None:
        print(
            "TEMPORARY ARC round trip enabled: baseline predictions are "
            "re-tokenized with M=100 and prediction-span D before execution"
        )
    return GraphRobotPolicy(
        graph,
        normalizer,
        adapter,
        max_valid_samples=config.get("max_valid_samples", 1),
        inference_graph=inference_graph,
        inference_controls=inference_controls,
    )
