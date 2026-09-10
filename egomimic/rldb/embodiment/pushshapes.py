"""Dataset schema and transforms for the two Planar PushShapes embodiments."""

from __future__ import annotations

from egomimic.rldb.zarr.action_chunk_transforms import (
    ChainGripperNative4ToPoints6,
    PadActionWidth,
    PlanarAgentStateToRotVec4,
    ThetaToRotVec,
)
from egomimic.rldb.zarr.planar_arc import PadPlanarAction, TokenizePlanarArcLength


def get_planar_keymap(
    action_horizon: int = 16,
    observation_horizon: int = 1,
    action_target_offset: int = 0,
    norm_mode: bool = False,
    action_zarr_key: str = "actions",
    **_kwargs,
):
    """Map observations and a future native-action chunk from one episode."""
    observation_horizon = int(observation_horizon)
    action_target_offset = int(action_target_offset)
    if observation_horizon <= 0 or action_target_offset < 0:
        raise ValueError("observation_horizon must be positive and offset non-negative")
    keymap = {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
        },
        "state_agent_obj": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state",
        },
        "actions": {
            "key_type": "action_keys",
            "zarr_key": str(action_zarr_key),
            "horizon": int(action_horizon) + action_target_offset,
        },
    }
    if observation_horizon > 1:
        keymap["front_img_1"]["horizon"] = observation_horizon
        keymap["state_agent_obj"]["horizon"] = observation_horizon
    if norm_mode:
        keymap.pop("front_img_1")
    return keymap


def get_planar_keymap_per_source_proprio(
    action_horizon: int = 16,
    norm_mode: bool = False,
    action_zarr_key: str = "actions",
    **kwargs,
):
    """Keep native simulator state as metadata and add model-only proprio."""
    keymap = get_planar_keymap(
        action_horizon=action_horizon,
        norm_mode=norm_mode,
        action_zarr_key=action_zarr_key,
        **kwargs,
    )
    keymap["state_agent_obj"]["key_type"] = "metadata_keys"
    keymap["state_agent_model"] = {
        "key_type": "proprio_keys",
        "zarr_key": "observations.state",
    }
    # The model-only proprio key replaces state_agent_obj as the observation, so
    # it needs the same multi-frame horizon that get_planar_keymap applied there.
    observation_horizon = int(kwargs.get("observation_horizon", 1))
    if observation_horizon > 1:
        keymap["state_agent_model"]["horizon"] = observation_horizon
    return keymap


def get_usocket_rotvec_obs2_transform_list(
    action_key: str = "actions",
    state_key: str = "state_agent_model",
    action_horizon: int = 16,
    action_target_offset: int = 1,
    **_kwargs,
):
    """Multi-frame U-Socket variant: align the target with the final observation.

    Mirrors the Paper-DP contract (observation_horizon=2, offset=1), so the
    keymap requests action_horizon + offset raw steps and the target is sliced
    to the chunk that follows the last observed frame.
    """
    return [
        SliceActionTarget([action_key], start=action_target_offset, horizon=action_horizon),
        ThetaToRotVec(keys=[action_key], angle_col=2),
        PlanarAgentStateToRotVec4(keys=[state_key], angle_col=2),
    ]


class SliceActionTarget:
    """Align the target to the final observation in a multi-frame window."""

    def __init__(self, keys: list[str], start: int, horizon: int):
        self.keys = list(keys)
        self.start = int(start)
        self.horizon = int(horizon)
        if self.start < 0 or self.horizon <= 0:
            raise ValueError("start must be non-negative and horizon positive")

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = batch[key][self.start : self.start + self.horizon]
            if len(value) != self.horizon:
                raise ValueError(
                    f"{key!r} has {len(value)} aligned steps, expected {self.horizon}"
                )
            batch[key] = value
        return batch


def get_planar_dense_transform_list(keys: list[str] | None = None, **_kwargs):
    """Convert either native Planar action layout into common five-space."""
    return [PadPlanarAction(keys=keys or ["actions"])]


def get_planar_paper_transform_list(
    keys: list[str] | None = None,
    action_horizon: int = 16,
    action_target_offset: int = 1,
    **_kwargs,
):
    """Align Paper-DP targets then convert them into common five-space."""
    keys = keys or ["actions"]
    return [
        SliceActionTarget(keys, start=action_target_offset, horizon=action_horizon),
        PadPlanarAction(keys),
    ]


def get_planar_paper_padded_transform_list(
    keys: list[str] | None = None,
    action_horizon: int = 16,
    action_target_offset: int = 1,
    width: int = 6,
    **_kwargs,
):
    """Paper-DP alignment, common five-space, then zero-pad to ``width``.

    Used by the Paper-DP cotrain row so the U-Socket target shares the six-wide
    head with the ChainGripper six-point target.
    """
    keys = keys or ["actions"]
    return [
        SliceActionTarget(keys, start=action_target_offset, horizon=action_horizon),
        PadPlanarAction(keys),
        PadActionWidth(keys, width=width),
    ]


def get_planar_arc_length_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 200.0,
    resampled_vector_length: int = 100,
    dt: float = 1.0 / 30.0,
    rotation_radius: float = 0.0,
    hybrid_rotation_unit: float | None = None,
    waypoint_sampling: str = "uniform",
    curvature_dense_samples: int = 257,
    curvature_floor: float | None = None,
    **_kwargs,
):
    """Create the active Planar SE(2) arc transform."""
    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("Planar arc tokenization requires exactly one action key")
    return [
        TokenizePlanarArcLength(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
            hybrid_rotation_unit=hybrid_rotation_unit,
            waypoint_sampling=waypoint_sampling,
            curvature_dense_samples=curvature_dense_samples,
            curvature_floor=curvature_floor,
        )
    ]


def get_usocket_rotvec_action_state_transform_list(
    action_key: str = "actions",
    state_key: str = "state_agent_model",
):
    """Encode action theta and the observed U-Socket agent pose."""
    return [
        ThetaToRotVec(keys=[action_key], angle_col=2),
        PlanarAgentStateToRotVec4(keys=[state_key], angle_col=2),
    ]


def get_chain_gripper_points_action_state_transform_list(
    action_key: str = "actions",
    state_key: str = "state_agent_model",
    world_size: float = 512.0,
):
    """ChainGripper twin of the U-Socket UNITE transform: six-point actions.

    Native ``[x, y, theta, grip]`` targets become the ordered
    ``[left, center, right]`` command points (FK, see
    ``egomimic.rldb.zarr.chain_gripper_points``) so the two embodiments share
    nothing in action space; the model proprio stays the 4-D
    ``[x, y, cos, sin]`` agent pose in both, as the dataset state carries no
    grip opening.
    """
    return [
        ChainGripperNative4ToPoints6(keys=[action_key], world_size=world_size),
        PlanarAgentStateToRotVec4(keys=[state_key], angle_col=2),
    ]


def get_chain_common5_action_rotvec_state_transform_list(
    action_key: str = "actions",
    state_key: str = "state_agent_model",
):
    """Encode ChainGripper actions in common-five space and pose as rotvec4."""
    return [
        PadPlanarAction(keys=[action_key]),
        PlanarAgentStateToRotVec4(keys=[state_key], angle_col=2),
    ]


def get_chain_gripper_paper_points_transform_list(
    keys: list[str] | None = None,
    action_horizon: int = 16,
    action_target_offset: int = 1,
    world_size: float = 512.0,
    **_kwargs,
):
    """Paper-DP alignment, then native ChainGripper controls -> six points."""
    keys = keys or ["actions"]
    return [
        SliceActionTarget(keys, start=action_target_offset, horizon=action_horizon),
        ChainGripperNative4ToPoints6(keys=keys, world_size=world_size),
    ]


def get_usocket_rotvec_action_transform_list(action_key: str = "actions"):
    """Encode only the U-Socket action angle as cosine/sine.

    This adapter is for models whose observation pathway consumes the native
    three-coordinate agent state while their target pathway uses the smooth
    four-coordinate ``[x, y, cos(theta), sin(theta)]`` representation.  Keeping
    this conversion at the dataset boundary prevents generic Pipeline stages
    from acquiring robot- or embodiment-specific behavior.
    """

    return [ThetaToRotVec(keys=[action_key], angle_col=2)]
