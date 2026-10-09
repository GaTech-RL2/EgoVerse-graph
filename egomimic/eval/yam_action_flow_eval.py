"""YAM 14-D Cartesian Action Flow validation on complete action chunks.

The native layout is two seven-dimensional arms, each XYZ, YPR, gripper.
Euler differences are wrapped before native errors or stochastic distances.
"""

from __future__ import annotations

import torch

from egomimic.eval.energy_score import energy_score
from egomimic.eval.planar_action_eval import PlanarActionEval

YAM_ACTION_DIM = 14
YAM_ANGLE_INDICES = (3, 4, 5, 10, 11, 12)
YAM_TRANSLATION_INDICES = (0, 1, 2, 7, 8, 9)
YAM_GRIPPER_INDICES = (6, 13)
YAM_NATIVE_ERROR_CONTRACT = {
    "enabled": True,
    "type": "yam_cartesian_xyz_wrapped_ypr_gripper_mse_v1",
    "space": "unnormalized_cartesian_14d",
    "rotation_indices": list(YAM_ANGLE_INDICES),
    "wrap_period_radians": 6.283185307179586,
    "reduction": "mean_squared_error_over_horizon_and_native_coordinates",
    "normalizer": "bound_train_only_evaluator_normalizer",
}
YAM_ENERGY_DISTANCE_METADATA = {
    "type": "yam_cartesian_14d_chunk_v1",
    "space": "normalized_xyz_gripper_plus_unnormalized_wrapped_ypr",
    "formula": "(rms(normalized_xyz)+rms(wrap(native_ypr)/pi)+rms(normalized_gripper))/3",
    "complete_normalized_chunk_shape": [100, 14],
    "coverage": "all_100_steps_and_all_14_action_channels",
    "normalizer": "bound_train_only_evaluator_normalizer",
    "rotation_indices": list(YAM_ANGLE_INDICES),
    "translation_indices": list(YAM_TRANSLATION_INDICES),
    "gripper_indices": list(YAM_GRIPPER_INDICES),
}


def _require_yam_chunk(tensor: torch.Tensor) -> None:
    if tensor.ndim < 2 or tuple(tensor.shape[-2:]) != (100, YAM_ACTION_DIM):
        raise ValueError("YAM Action Flow requires complete (100,14) chunks")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("YAM Action Flow chunk is non-finite")


def yam_native_residual(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[-1] != YAM_ACTION_DIM or target.shape[-1] != YAM_ACTION_DIM:
        raise ValueError("YAM native residual requires two 14-D actions")
    try:
        torch.broadcast_shapes(prediction.shape, target.shape)
    except RuntimeError as exc:
        raise ValueError("YAM native residual shapes do not broadcast") from exc
    residual = prediction - target
    angles = residual[..., list(YAM_ANGLE_INDICES)]
    residual = residual.clone()
    residual[..., list(YAM_ANGLE_INDICES)] = torch.atan2(
        torch.sin(angles), torch.cos(angles)
    )
    return residual


class YamCartesianActionFlowEval(PlanarActionEval):
    """Reuse fixed-seed Action Flow diagnostics with YAM-native geometry."""

    def __init__(self, **kwargs):
        if kwargs.get("action_key", "actions_cartesian") != "actions_cartesian":
            raise ValueError("YAM evaluator requires actions_cartesian")
        if kwargs.get("native_decoder") is not None or kwargs.get("native_decoders"):
            raise ValueError("YAM Cartesian actions do not use a private decoder")
        kwargs["action_key"] = "actions_cartesian"
        super().__init__(**kwargs)
        if self.energy_score_distance is not None:
            raise ValueError("YAM evaluator has its own fixed Energy Score distance")
        self.energy_score_distance_metadata = dict(YAM_ENERGY_DISTANCE_METADATA)

    @staticmethod
    def _native_residual(prediction, target, decoder):
        if decoder is not None:
            raise ValueError("YAM native error must not use a decoder")
        return yam_native_residual(prediction, target)

    def _energy_values(self, samples, target, embodiment_id, label=None):
        if samples.ndim != 4 or samples.shape[0] != 32:
            raise ValueError("YAM EnergyScore@32 requires 32 sampled chunks")
        _require_yam_chunk(samples)
        _require_yam_chunk(target)
        if samples.shape[1:] != target.shape:
            raise ValueError("YAM sampled/target chunk shapes differ")

        def distance(left, right):
            _require_yam_chunk(left)
            _require_yam_chunk(right)
            native_left = self._native(left, embodiment_id, None)
            native_right = self._native(right, embodiment_id, None)
            native_residual = yam_native_residual(native_left, native_right)
            translation = (
                (
                    left[..., list(YAM_TRANSLATION_INDICES)]
                    - right[..., list(YAM_TRANSLATION_INDICES)]
                )
                .square()
                .mean(dim=(-2, -1))
                .sqrt()
            )
            rotation = (
                (native_residual[..., list(YAM_ANGLE_INDICES)] / torch.pi)
                .square()
                .mean(dim=(-2, -1))
                .sqrt()
            )
            gripper = (
                (
                    left[..., list(YAM_GRIPPER_INDICES)]
                    - right[..., list(YAM_GRIPPER_INDICES)]
                )
                .square()
                .mean(dim=(-2, -1))
                .sqrt()
            )
            return (translation + rotation + gripper) / 3.0

        values = energy_score(samples, target, distance_fn=distance)
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("YAM Energy Score produced non-finite values")
        return {name: value.detach() for name, value in values.items()}

    def _action_flow_native_error_fns(self, batch):
        runner = self._action_flow_diagnostics
        if runner is None or not runner.native_error_enabled:
            return None
        if runner.native_error != YAM_NATIVE_ERROR_CONTRACT:
            raise ValueError("YAM Action Flow native error contract changed")
        functions = {}
        for source_id, values in batch.items():
            embodiment_id, label = self._embodiment(values)
            if label.lower() != "yam_bimanual":
                raise ValueError("YAM evaluator received a different embodiment")

            def native_error(prediction, target, *, _id=embodiment_id):
                native_prediction = self._native(prediction, _id, None)
                native_target = self._native(target, _id, None)
                return self._native_mse_by_condition(
                    native_prediction, native_target, None
                )

            functions[source_id] = native_error
        return functions
