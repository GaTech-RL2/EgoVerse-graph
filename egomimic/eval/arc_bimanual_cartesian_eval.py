"""Canonical ARC overlay and metric adapter."""

from dataclasses import replace

import numpy as np
import torch

from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian


class ArcBimanualCartesianEval(BimanualCartesianEval):
    def __init__(
        self,
        *args,
        min_distance_unit=0.4,
        rotation_distance_unit=1.7453292519943295,
        resampled_vector_length=100,
        action_horizon=100,
        dt=1 / 30,
        clock_mode="velocity",
        streams="full_separation",
        gripper_distance_unit=None,
        gripper_stream_mode="events",
        initial_action_key="arc.current_action",
        arc_metrics=True,
        **kwargs,
    ):
        if {"velocity_mode", "velocity_layout", "velocity_norm"}.intersection(kwargs):
            raise ValueError("Retired ARC options; use clock_mode and streams")
        super().__init__(*args, **kwargs)
        self.arc_metrics = bool(arc_metrics)
        self.action_horizon = int(action_horizon)
        self.min_distance_unit = float(min_distance_unit)
        self.resampled_vector_length = int(resampled_vector_length)
        if not isinstance(initial_action_key, str) or not initial_action_key:
            raise ValueError("ARC evaluator requires a measured initial_action_key")
        self.initial_action_key = initial_action_key
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if abs(self.arcmatch_distance - self.min_distance_unit) > 1e-9:
            raise ValueError("arcmatch_distance must equal min_distance_unit")
        self._tokenizer = TokenizeBimanualArcLengthCartesian(
            min_distance_unit=min_distance_unit,
            rotation_distance_unit=rotation_distance_unit,
            resampled_vector_length=resampled_vector_length,
            dt=dt,
            clock_mode=clock_mode,
            streams=streams,
            gripper_distance_unit=gripper_distance_unit,
            gripper_stream_mode=gripper_stream_mode,
        )

    def data_requirements(self):
        requirements = super().data_requirements()
        return replace(
            requirements,
            required_keys=tuple(
                dict.fromkeys((*requirements.required_keys, self.initial_action_key))
            ),
        )

    def _initial_actions(self, source_batch, batch_size):
        if source_batch is None or self.initial_action_key not in source_batch:
            raise ValueError(
                f"ARC evaluation requires native measured {self.initial_action_key!r}"
            )
        # This is physical metadata emitted by the frame transform, not a
        # normalized model input or the first ground-truth command target.
        anchors = source_batch[self.initial_action_key]
        if torch.is_tensor(anchors):
            anchors = anchors.detach().cpu().numpy()
        anchors = np.asarray(anchors, dtype=np.float64)
        if anchors.shape != (batch_size, 14) or not np.isfinite(anchors).all():
            raise ValueError("ARC measured anchors must be finite (B,14)")
        return anchors

    def _is_arc(self, actions):
        return (
            actions is not None
            and actions.ndim == 3
            and tuple(actions.shape[-2:]) == self._tokenizer.shape
        )

    def _viz_source(self, actions, embodiment_id, *, source_batch=None):
        if not self._is_arc(actions):
            if actions.shape[-1] != 14:
                raise ValueError("ARC evaluator and token shape disagree")
            return super()._viz_source(
                actions, embodiment_id, source_batch=source_batch
            )
        native = actions.detach().cpu().numpy()
        anchors = self._initial_actions(source_batch, len(native))
        decoded = np.stack(
            [
                self._tokenizer.detokenize(
                    row, self.action_horizon, initial_action=anchor
                )
                for row, anchor in zip(native, anchors)
            ]
        )
        return torch.as_tensor(decoded, dtype=actions.dtype)

    def _arc_pred_time_indexed(self, prediction, embodiment_id, *, source_batch=None):
        native = self._native(prediction, embodiment_id)
        if not self._is_arc(native):
            return native.detach().cpu().numpy().astype(np.float64, copy=False)
        return self._viz_source(
            native, embodiment_id, source_batch=source_batch
        ).numpy()

    def _arc_pred_for_arcmatch(
        self, prediction, ground_truth, embodiment_id, *, source_batch=None
    ):
        # Keep the existing metric switch; only the serialized token changed.
        from egomimic.eval.arc_metrics import combined_travel

        native = self._native(prediction, embodiment_id)
        if not self._is_arc(native):
            return native.detach().cpu().numpy().astype(np.float64, copy=False)
        if self.include_reconstruction_loss:
            return self._viz_source(
                native, embodiment_id, source_batch=source_batch
            ).numpy()
        anchors = self._initial_actions(source_batch, len(native))
        return [
            self._tokenizer.detokenize(
                token, self.action_horizon, initial_action=anchor
            )
            if combined_travel(gt) < self.min_distance_unit - 1e-12
            else token[:, :14].copy()
            for token, gt, anchor in zip(
                native.detach().cpu().numpy(), ground_truth, anchors
            )
        ]
