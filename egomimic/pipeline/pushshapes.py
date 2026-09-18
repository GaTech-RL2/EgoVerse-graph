"""Planar action decoders shared by metrics and downstream consumers."""

from __future__ import annotations

import numpy as np
import torch

from egomimic.pipeline.stages_arc import ArcDetokenizeStage
from egomimic.rldb.zarr.action_chunk_transforms import (
    ChainGripperPoints6ToNative4,
    PlanarAgentStateToRotVec4,
)
from egomimic.rldb.zarr.planar_arc import (
    PLANAR_ACTION_DIM,
    PLANAR_ARC_TIMED_DIM,
    arc_token_rows,
    validate_velocity_mode,
)


def _common5_to_native(actions, native_action_dim: int):
    if native_action_dim not in (2, 3, 4):
        raise ValueError("native_action_dim must be 2, 3, or 4")
    if actions.shape[-1] != PLANAR_ACTION_DIM:
        raise ValueError(
            f"expected common Planar width {PLANAR_ACTION_DIM}, got {actions.shape}"
        )
    if torch.is_tensor(actions):
        theta = torch.atan2(actions[..., 3], actions[..., 2]).unsqueeze(-1)
        native = torch.cat((actions[..., :2], theta, actions[..., 4:5]), dim=-1)
    else:
        value = np.asarray(actions)
        theta = np.arctan2(value[..., 3], value[..., 2])[..., None]
        native = np.concatenate((value[..., :2], theta, value[..., 4:5]), axis=-1)
    return native[..., :native_action_dim]


class PlanarCommon5NativeDecoder:
    """Decode a fixed-rate common-five chunk without changing its timing."""

    preserves_decoded_timing = True

    def __init__(self, action_horizon: int, native_action_dim: int):
        self.action_horizon = int(action_horizon)
        self.native_action_dim = int(native_action_dim)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

    def decode(self, actions, context: dict | None = None):
        del context
        value = (
            actions.unsqueeze(0)
            if torch.is_tensor(actions) and actions.ndim == 2
            else actions
        )
        value = (
            np.asarray(actions)[None]
            if not torch.is_tensor(actions) and np.asarray(actions).ndim == 2
            else value
        )
        expected = (self.action_horizon, PLANAR_ACTION_DIM)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"expected (B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        return _common5_to_native(value, self.native_action_dim)

    __call__ = decode


class PaddedPlanarCommon5NativeDecoder(PlanarCommon5NativeDecoder):
    """Drop zero-padding beyond the common five, then decode as common five."""

    def __init__(self, action_horizon: int, native_action_dim: int, padded_dim: int = 6):
        super().__init__(action_horizon=action_horizon, native_action_dim=native_action_dim)
        self.padded_dim = int(padded_dim)
        if self.padded_dim < PLANAR_ACTION_DIM:
            raise ValueError("padded_dim must be at least the common Planar width")

    def decode(self, actions, context: dict | None = None):
        if actions.shape[-1] != self.padded_dim:
            raise ValueError(
                f"expected padded width {self.padded_dim}, got {actions.shape}"
            )
        return super().decode(actions[..., :PLANAR_ACTION_DIM], context)

    __call__ = decode


class PlanarArcWaypointZeroNativeDecoder:
    """Decode the anchored first waypoint from a Planar arc token.

    ``velocity_mode`` only changes how many rows the token has -- waypoint zero
    is row zero either way -- but it has to be declared so the shape check
    stays meaningful. It must match the tokenizer's mode; sizing comes from
    ``arc_token_rows`` so the two cannot disagree about the layout.
    """

    preserves_decoded_timing = True
    action_horizon = 1

    def __init__(
        self,
        resampled_vector_length: int,
        native_action_dim: int,
        velocity_mode: str = "duration",
    ):
        self.num_waypoints = int(resampled_vector_length)
        self.native_action_dim = int(native_action_dim)
        self.velocity_mode = validate_velocity_mode(velocity_mode)
        if self.num_waypoints < 2:
            raise ValueError("resampled_vector_length must be at least two")

    def decode(self, actions, context: dict | None = None):
        del context
        value = (
            actions.unsqueeze(0)
            if torch.is_tensor(actions) and actions.ndim == 2
            else actions
        )
        value = (
            np.asarray(actions)[None]
            if not torch.is_tensor(actions) and np.asarray(actions).ndim == 2
            else value
        )
        expected = (
            arc_token_rows(self.num_waypoints, self.velocity_mode),
            PLANAR_ACTION_DIM,
        )
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"expected (B, {expected[0]}, {expected[1]}) for velocity_mode="
                f"{self.velocity_mode!r}, got {value.shape}"
            )
        return _common5_to_native(value[:, :1], self.native_action_dim)

    __call__ = decode


class PlanarArcTrajectoryNativeDecoder:
    """Decode all ARC supports into a smooth, fixed-rate native trajectory."""

    preserves_decoded_timing = True
    timing_semantics = "per_waypoint_duration_cubic_v1"

    def __init__(
        self,
        resampled_vector_length: int,
        native_action_dim: int,
        raw_action_horizon: int,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 0.0,
        velocity_mode: str = "duration",
    ):
        self.num_waypoints = int(resampled_vector_length)
        self.native_action_dim = int(native_action_dim)
        self.action_horizon = int(raw_action_horizon)
        self.velocity_mode = validate_velocity_mode(velocity_mode)
        if self.velocity_mode == "mean":
            raise ValueError(
                "full ARC trajectory reconstruction requires per-point timing"
            )
        self.detokenizer = ArcDetokenizeStage(
            resampled_vector_length=self.num_waypoints,
            action_horizon=self.action_horizon,
            dt=dt,
            native_action_dim=self.native_action_dim,
            rotation_radius=rotation_radius,
            velocity_mode=self.velocity_mode,
        )

    def decode(self, actions, context: dict | None = None):
        del context
        is_torch = torch.is_tensor(actions)
        value = actions if is_torch else torch.as_tensor(np.asarray(actions))
        if value.ndim == 2:
            value = value.unsqueeze(0)
        expected = (
            arc_token_rows(self.num_waypoints, self.velocity_mode),
            PLANAR_ACTION_DIM,
        )
        if value.ndim < 2 or tuple(value.shape[-2:]) != expected:
            raise ValueError(
                f"expected (..., {expected[0]}, {expected[1]}) for velocity_mode="
                f"{self.velocity_mode!r}, got {tuple(value.shape)}"
            )
        leading = value.shape[:-2]
        flat = value.reshape(-1, *expected)
        decoded = self.detokenizer.forward({"pred_action": flat})[
            "pred_action_native"
        ].reshape(*leading, self.action_horizon, self.native_action_dim)
        return decoded if is_torch else decoded.cpu().numpy()

    def waypoint_schedule(self, actions):
        """Return the same distance and duration clocks used by the decoder.

        Accept one unnormalized token. The final timing row is padding and
        does not create an extra waypoint interval.
        """
        value = torch.as_tensor(actions)
        expected = (
            arc_token_rows(self.num_waypoints, self.velocity_mode), PLANAR_ACTION_DIM
        )
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError(f"Expected one finite ARC token of shape {expected}")
        tokens = value.unsqueeze(0)
        distance = self.detokenizer._arc_positions(tokens[:, : self.num_waypoints])
        elapsed = self.detokenizer.duration_waypoint_times(tokens)
        return distance[0], elapsed[0]

    __call__ = decode


class USocketModelStateObservationAdapter:
    """Add rotvec4 model proprio while retaining native simulator context."""

    def __init__(
        self,
        raw_state_key: str = "state_agent_obj",
        model_state_key: str = "state_agent_model",
    ):
        self.raw_state_key = str(raw_state_key)
        self.model_state_key = str(model_state_key)
        self.transform = PlanarAgentStateToRotVec4(keys=[self.model_state_key])

    def encode(self, batch: dict) -> dict:
        output = dict(batch)
        if self.raw_state_key not in output:
            raise KeyError(f"Missing raw U-Socket state {self.raw_state_key!r}")
        output[self.model_state_key] = output[self.raw_state_key]
        return self.transform.transform(output)


class USocketRotVecNativeDecoder:
    """Decode ``[x, y, cos(theta), sin(theta)]`` into simulator actions."""

    preserves_decoded_timing = True

    def decode(self, actions, context: dict | None = None):
        del context
        if actions.ndim < 2 or actions.shape[-1] != 4:
            raise ValueError(
                f"USocketRotVecNativeDecoder expects (..., 4), got {actions.shape}"
            )
        if torch.is_tensor(actions):
            theta = torch.atan2(actions[..., 3], actions[..., 2])
            return torch.cat((actions[..., :2], theta.unsqueeze(-1)), dim=-1)
        value = np.asarray(actions)
        theta = np.arctan2(value[..., 3], value[..., 2])
        return np.concatenate((value[..., :2], theta[..., None]), axis=-1)

    __call__ = decode


class ChainGripperModelStateObservationAdapter(USocketModelStateObservationAdapter):
    """ChainGripper proprio: the same rotvec4 agent pose as the U-Socket adapter."""


class ChainGripperPointsNativeDecoder:
    """Decode ``[L, C, R]`` six-point chunks into native ``[x, y, theta, grip]``.

    Sequential constrained IK (``ChainGripperPoints6ToNative4``); the previous
    native control, or the rollout state's ``x, y, theta``, seeds orientation
    continuity. Timing is unchanged, so replan-every-k execution is identical
    to the U-Socket rotvec decoder's.
    """

    preserves_decoded_timing = True

    def __init__(
        self,
        world_size: float = 512.0,
        grid_size: int = 33,
        refinements: int = 6,
        context_state_key: str = "state_agent_obj",
        previous_control_key: str = "previous_control",
    ):
        self.transform = ChainGripperPoints6ToNative4(
            keys=["actions"],
            world_size=world_size,
            grid_size=grid_size,
            refinements=refinements,
            context_state_key=context_state_key,
            previous_control_key=previous_control_key,
        )

    @property
    def last_projection_diagnostics(self):
        return self.transform.last_projection_diagnostics

    def decode(self, actions, context: dict | None = None):
        if actions.ndim < 2 or actions.shape[-1] != 6:
            raise ValueError(
                f"ChainGripperPointsNativeDecoder expects (..., 6), got {actions.shape}"
            )
        batch = dict(context or {})
        batch["actions"] = actions
        return self.transform.transform(batch)["actions"]

    __call__ = decode


class PlanarArcTimedNativeDecoder:
    """Decode seven-wide timed ARC tokens to fixed-rate native controls."""

    preserves_decoded_timing = True

    def __init__(
        self,
        resampled_vector_length: int,
        action_horizon: int,
        native_action_dim: int,
        timing_mode: str,
        dt: float = 1.0 / 30.0,
        zero_dist_epsilon: float = 1e-9,
    ):
        self.num_waypoints = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.native_action_dim = int(native_action_dim)
        self.timing_mode = str(timing_mode)
        self.dt = float(dt)
        self.epsilon = float(zero_dist_epsilon)
        if self.num_waypoints < 2 or self.action_horizon <= 0 or self.dt <= 0:
            raise ValueError("waypoints, action_horizon, and dt must be positive")
        if self.native_action_dim not in (2, 3, 4):
            raise ValueError("native_action_dim must be 2, 3, or 4")
        if self.timing_mode not in {"velocity", "duration"}:
            raise ValueError("timing_mode must be 'velocity' or 'duration'")

    def _durations(self, geometry, timing):
        distance = torch.linalg.vector_norm(geometry[:, 1:] - geometry[:, :-1], dim=-1)
        active = distance > self.epsilon
        stop = torch.full_like(timing, self.dt * (self.action_horizon + 1))
        if self.timing_mode == "duration":
            return torch.where(active & (timing <= self.epsilon), stop, timing.clamp_min(0.0))
        usable = timing.abs() > self.epsilon
        duration = torch.where(
            active & usable,
            distance / timing.abs().clamp_min(self.epsilon),
            torch.zeros_like(distance),
        )
        return torch.where(active & ~usable, stop, duration)

    def _sample(self, geometry, duration):
        cumulative = torch.cat(
            (torch.zeros_like(duration[:, :1]), torch.cumsum(duration, dim=1)), dim=1
        )
        targets = self.dt * torch.arange(
            self.action_horizon, device=geometry.device, dtype=geometry.dtype
        )[None]
        upper = torch.searchsorted(
            cumulative.contiguous(), targets.expand(len(geometry), -1).contiguous(), right=True
        ).clamp(1, self.num_waypoints - 1)
        lower = upper - 1
        lo_t = torch.gather(cumulative, 1, lower)
        hi_t = torch.gather(cumulative, 1, upper)
        alpha = ((targets - lo_t) / (hi_t - lo_t).clamp_min(self.epsilon)).clamp(0, 1)
        width = geometry.shape[-1]
        lo = torch.gather(geometry, 1, lower[..., None].expand(-1, -1, width))
        hi = torch.gather(geometry, 1, upper[..., None].expand(-1, -1, width))
        return (1 - alpha[..., None]) * lo + alpha[..., None] * hi

    def decode(self, actions, context: dict | None = None):
        del context
        is_tensor = torch.is_tensor(actions)
        value = actions if is_tensor else torch.as_tensor(np.asarray(actions))
        squeeze = value.ndim == 2
        if squeeze:
            value = value.unsqueeze(0)
        expected = (self.num_waypoints, PLANAR_ARC_TIMED_DIM)
        if value.ndim < 3 or tuple(value.shape[-2:]) != expected:
            raise ValueError(f"expected (..., {expected[0]}, {expected[1]}), got {value.shape}")
        leading = tuple(value.shape[:-2])
        value = value.reshape(-1, *expected)
        xy_duration = self._durations(value[..., :2], value[:, :-1, 2])
        heading_geometry = value[..., 3:5]
        if self.timing_mode == "duration":
            heading_duration = self._durations(heading_geometry, value[:, :-1, 5])
        else:
            dot = (heading_geometry[:, 1:] * heading_geometry[:, :-1]).sum(-1)
            cross = (
                heading_geometry[:, 1:, 1] * heading_geometry[:, :-1, 0]
                - heading_geometry[:, 1:, 0] * heading_geometry[:, :-1, 1]
            )
            angle = torch.atan2(cross, dot).abs()
            rate = value[:, :-1, 5]
            heading_duration = torch.where(
                (angle > self.epsilon) & (rate.abs() > self.epsilon),
                angle / rate.abs().clamp_min(self.epsilon),
                torch.zeros_like(angle),
            )
            heading_duration = torch.where(
                (angle > self.epsilon) & (rate.abs() <= self.epsilon),
                torch.full_like(angle, self.dt * (self.action_horizon + 1)),
                heading_duration,
            )
        xy = self._sample(value[..., :2], xy_duration)
        heading = self._sample(heading_geometry, heading_duration)
        heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1e-8)
        theta = torch.atan2(heading[..., 1], heading[..., 0])[..., None]
        grip = self._sample(value[..., 6:7], xy_duration)
        native = torch.cat((xy, theta, grip), dim=-1)[..., : self.native_action_dim]
        # Every prediction begins at its causal anchor, including degenerate
        # velocity clocks with repeated zero-time supports.
        anchor_theta = torch.atan2(value[:, 0, 4], value[:, 0, 3])[:, None]
        anchor = torch.cat((value[:, 0, :2], anchor_theta, value[:, 0, 6:7]), dim=-1)
        native[:, 0] = anchor[:, : self.native_action_dim]
        native = native.reshape(*leading, self.action_horizon, self.native_action_dim)
        if squeeze:
            native = native.squeeze(0)
        return native if is_tensor else native.cpu().numpy()

    __call__ = decode
