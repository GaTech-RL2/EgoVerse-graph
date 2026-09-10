"""Loader bindings for the grip-carrying ARC codec on the articulated corpus.

The u_socket-only codec family (``usocket_arc_velocity``) emits a width-6 token
with no engage channel. Six of the nine articulated embodiments latch, grasp or
suction, so a width-6 token throws away the one signal the corpus was
regenerated to contain. These bindings emit width 7 instead.

Two ordering constraints, both load-bearing:

* ``SliceActionTarget`` runs FIRST, so the tokenized window is the same window
  the Paper-DP baseline regresses on. Without it the ARC arms would tokenize
  ``raw_action_horizon + action_target_offset`` frames while DP regresses on a
  window shifted by one, and the comparison would not be apples to apples.
* ``PadPlanarAction`` runs SECOND, before tokenization. It maps every native
  layout -- (T,3) for u_socket/triangle/scoop and (T,4) for the six grasping
  tools -- onto the common-five layout, so ONE tokenizer serves all nine
  embodiments and grip always lives in the same column. Tokenizing native
  actions directly would silently mis-read a 4-DOF chunk: the parent
  ``_components`` treats width 4 as "no orientation" and returns theta = 0.

Tokenization still runs before dataset normalization, so D and R stay in native
pixels and radians.
"""

from __future__ import annotations

import numpy as np
import torch

from egomimic.pipeline.stages_arc import (
    ArcDetokenizeDurationGripStage,
    ArcDetokenizeStackedGripStage,
)
from egomimic.rldb.embodiment.pushshapes import SliceActionTarget
from egomimic.rldb.zarr.planar_arc import (
    PadPlanarAction,
    TokenizeArcDurationGrip,
    TokenizeArcVelocityStackedGrip,
)


def _prefix(keys, action_target_offset, raw_action_horizon):
    """Align the raw window, then widen it to common five."""
    steps = [PadPlanarAction(keys)]
    if action_target_offset:
        steps.insert(
            0,
            SliceActionTarget(
                keys, start=int(action_target_offset), horizon=int(raw_action_horizon)
            ),
        )
    return steps


def _check(keys):
    keys = keys or ["actions"]
    if len(keys) != 1:
        raise ValueError("ARC tokenization requires exactly one action key")
    return keys


def get_articulated_arc_stacked_grip_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 80.0,
    resampled_vector_length: int = 56,
    dt: float = 1.0 / 30.0,
    rotation_distance_unit: float | None = None,
    action_target_offset: int = 0,
    raw_action_horizon: int = 80,
    **_kwargs,
):
    """Loader transform emitting the width-7 stacked VELOCITY token."""
    keys = _check(keys)
    return _prefix(keys, action_target_offset, raw_action_horizon) + [
        TokenizeArcVelocityStackedGrip(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_distance_unit=rotation_distance_unit,
        )
    ]


def get_articulated_arc_duration_grip_transform_list(
    keys: list[str] | None = None,
    min_distance_unit: float = 80.0,
    resampled_vector_length: int = 56,
    dt: float = 1.0 / 30.0,
    rotation_distance_unit: float | None = None,
    action_target_offset: int = 0,
    raw_action_horizon: int = 80,
    **_kwargs,
):
    """Loader transform emitting the width-7 stacked DURATION token."""
    keys = _check(keys)
    return _prefix(keys, action_target_offset, raw_action_horizon) + [
        TokenizeArcDurationGrip(
            action_key=keys[0],
            output_action_key=keys[0],
            min_distance_unit=min_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            rotation_distance_unit=rotation_distance_unit,
        )
    ]


class _GripNativeDecoder:
    """Wrap a detokenize stage as an evaluator-side native decoder."""

    preserves_decoded_timing = True
    _stage_cls: type = ArcDetokenizeStackedGripStage

    def __init__(
        self,
        resampled_vector_length: int,
        action_horizon: int,
        native_action_dim: int = 4,
        dt: float = 1.0 / 30.0,
    ):
        self.action_horizon = int(action_horizon)
        self._decoder = self._stage_cls(
            resampled_vector_length=resampled_vector_length,
            action_horizon=action_horizon,
            native_action_dim=native_action_dim,
            dt=dt,
        )

    def decode(self, actions, context: dict | None = None):
        del context
        is_tensor = torch.is_tensor(actions)
        value = actions if is_tensor else torch.as_tensor(np.asarray(actions))
        squeeze = value.ndim == 2
        if squeeze:
            value = value.unsqueeze(0)
        native = self._decoder.forward({"pred_action": value})["pred_action_native"]
        if squeeze:
            native = native.squeeze(0)
        return native if is_tensor else native.cpu().numpy()

    __call__ = decode


class ArticulatedArcStackedGripNativeDecoder(_GripNativeDecoder):
    """Native decoder for the width-7 velocity token."""

    _stage_cls = ArcDetokenizeStackedGripStage


class ArticulatedArcDurationGripNativeDecoder(_GripNativeDecoder):
    """Native decoder for the width-7 duration token."""

    _stage_cls = ArcDetokenizeDurationGripStage
