"""Canonical ARC overlay and metric adapter."""

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

    def _is_arc(self, actions):
        return (
            actions is not None
            and actions.ndim == 3
            and tuple(actions.shape[-2:]) == self._tokenizer.shape
        )

    def _viz_source(self, actions, embodiment_id):
        if not self._is_arc(actions):
            if actions.shape[-1] != 14:
                raise ValueError("ARC evaluator and token shape disagree")
            return super()._viz_source(actions, embodiment_id)
        native = actions.detach().cpu().numpy()
        decoded = np.stack(
            [self._tokenizer.detokenize(row, self.action_horizon) for row in native]
        )
        return torch.as_tensor(decoded, dtype=actions.dtype)

    def _arc_pred_time_indexed(self, prediction, embodiment_id):
        native = self._native(prediction, embodiment_id)
        if not self._is_arc(native):
            return super()._arc_pred_time_indexed(prediction, embodiment_id)
        return self._viz_source(native, embodiment_id).numpy()

    def _arc_pred_for_arcmatch(self, prediction, ground_truth, embodiment_id):
        # Keep the existing metric switch; only the serialized token changed.
        from egomimic.eval.arc_metrics import combined_travel

        native = self._native(prediction, embodiment_id)
        if not self._is_arc(native):
            return super()._arc_pred_for_arcmatch(
                prediction, ground_truth, embodiment_id
            )
        if self.include_reconstruction_loss:
            return self._arc_pred_time_indexed(prediction, embodiment_id)
        return [
            self._tokenizer.detokenize(token, self.action_horizon)
            if combined_travel(gt) < self.min_distance_unit - 1e-12
            else token[:, :14].copy()
            for token, gt in zip(native.detach().cpu().numpy(), ground_truth)
        ]
