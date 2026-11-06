"""Fixed-window planar decoding using the shared canonical ARC clocks."""

import numpy as np
import torch

from egomimic.rldb.zarr.arc_clock_bridge import ARRIVAL_CONTRACT
from egomimic.rldb.zarr.planar_arc_clock import (
    PlanarArcClockCodec,
    current_action_from_state,
)


class PlanarArcClockNativeDecoder:
    preserves_decoded_timing = True
    requires_initial_action = True

    def __init__(
        self, action_horizon=80, *, codec_contract, state_indices=None, **codec_kwargs
    ):
        if codec_contract != ARRIVAL_CONTRACT:
            raise ValueError(
                "Arrival ARC requires fresh targets, normalization and checkpoints"
            )
        self.codec = PlanarArcClockCodec(horizon=action_horizon, **codec_kwargs)
        self.action_horizon = self.codec.horizon
        self.native_action_dim = self.codec.native_action_dim
        self.state_indices = state_indices

    def context_from_observation(self, observations):
        state = observations["state_agent_obj"]
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        state = np.asarray(state)
        if state.ndim != 3 or state.shape[-2] < 1:
            raise ValueError("Expected raw (batch, history, state) observations")
        return {
            "initial_action": current_action_from_state(
                state[:, -1], self.native_action_dim, self.state_indices
            )
        }

    def decode(self, actions, context=None):
        if context is None or "initial_action" not in context:
            raise ValueError("Arrival decoding requires the measured current pose")
        tensor = torch.is_tensor(actions)
        value = (
            actions.detach().cpu().double().numpy() if tensor else np.asarray(actions)
        )
        if value.ndim < 2 or value.shape[-2:] != self.codec.shape:
            raise ValueError(f"Expected (...,{self.codec.shape}) ARC predictions")
        leading = value.shape[:-2]
        anchors = context["initial_action"]
        if torch.is_tensor(anchors):
            anchors = anchors.detach().cpu().numpy()
        anchors = np.broadcast_to(
            np.asarray(anchors), (*leading, self.native_action_dim)
        )
        native = np.stack(
            [
                self.codec.decode(row, initial_action=anchor)
                for row, anchor in zip(
                    value.reshape(-1, *self.codec.shape),
                    anchors.reshape(-1, self.native_action_dim),
                    strict=True,
                )
            ]
        ).reshape(*leading, self.action_horizon, self.native_action_dim)
        if tensor:
            return torch.as_tensor(native, device=actions.device, dtype=actions.dtype)
        return native.astype(value.dtype if value.dtype.kind == "f" else np.float32)

    def waypoint_clocks(self, actions, *, initial_action):
        value = (
            actions.detach().cpu().double().numpy()
            if torch.is_tensor(actions)
            else actions
        )
        return self.codec.waypoint_clocks(value, initial_action=initial_action)

    def representation_context(self):
        return self.codec.representation_context()

    __call__ = decode
