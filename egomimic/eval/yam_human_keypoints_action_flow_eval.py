"""Fixed-contract Action Flow validation for YAM Cartesian and human keypoints.

Human 138D layout is left wrist XYZ/YPR, left 21x3 keypoints, then the
corresponding right-hand blocks. Its camera-frame wrist angles are circular;
the keypoints and wrist translations are Euclidean. The two embodiments keep
separate normalization and native metrics while using one shared latent field.
"""

from __future__ import annotations

import torch

from egomimic.eval.energy_score import energy_score
from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.eval.yam_action_flow_eval import (
    YAM_ACTION_DIM,
    YAM_ENERGY_DISTANCE_METADATA,
    YamCartesianActionFlowEval,
    yam_native_residual,
)

HUMAN_ACTION_DIM = 138
HUMAN_WRIST_TRANSLATION_INDICES = (0, 1, 2, 69, 70, 71)
HUMAN_WRIST_ANGLE_INDICES = (3, 4, 5, 72, 73, 74)
HUMAN_KEYPOINT_INDICES = tuple(range(6, 69)) + tuple(range(75, 138))
ACTION_KEYS = {
    "yam_bimanual": "actions_cartesian",
    "human_bimanual": "actions_keypoints",
}
NATIVE_ERROR_CONTRACT = {
    "enabled": True,
    "type": "yam14_human138_headframe_wrapped_wrist_ypr_mse_v1",
    "space": "per_embodiment_unnormalized_native_actions",
    "human_wrist_angle_indices": list(HUMAN_WRIST_ANGLE_INDICES),
    "yam_wrist_angle_indices": [3, 4, 5, 10, 11, 12],
    "reduction": "mean_squared_error_over_horizon_and_native_coordinates",
    "normalizer": "bound_train_only_per_embodiment_evaluator_normalizer",
}
HUMAN_ENERGY_DISTANCE_METADATA = {
    "type": "human_headframe_keypoints_wrist_138d_chunk_v1",
    "space": "normalized_wrist_xyz_and_keypoints_plus_unnormalized_wrapped_wrist_ypr",
    "formula": "(rms(normalized_wrist_xyz)+rms(wrap(native_wrist_ypr)/pi)+rms(normalized_keypoints))/3",
    "complete_normalized_chunk_shape": [100, HUMAN_ACTION_DIM],
    "coverage": "all_100_steps_and_all_138_action_channels",
    "normalizer": "bound_train_only_per_embodiment_evaluator_normalizer",
    "wrist_translation_indices": list(HUMAN_WRIST_TRANSLATION_INDICES),
    "wrist_angle_indices": list(HUMAN_WRIST_ANGLE_INDICES),
    "keypoint_indices": list(HUMAN_KEYPOINT_INDICES),
}


def human_native_residual(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape[-1] != HUMAN_ACTION_DIM or target.shape[-1] != HUMAN_ACTION_DIM:
        raise ValueError("Human native residual requires 138-D keypoint/wrist actions")
    try:
        torch.broadcast_shapes(prediction.shape, target.shape)
    except RuntimeError as exc:
        raise ValueError("Human native residual shapes do not broadcast") from exc
    residual = prediction - target
    angles = residual[..., list(HUMAN_WRIST_ANGLE_INDICES)]
    residual = residual.clone()
    residual[..., list(HUMAN_WRIST_ANGLE_INDICES)] = torch.atan2(
        torch.sin(angles), torch.cos(angles)
    )
    return residual


class YamHumanKeypointsActionFlowEval(PlanarActionEval):
    """Use one evaluator with explicit action keys and native geometry."""

    def __init__(self, **kwargs):
        if kwargs.get("native_decoder") is not None or kwargs.get("native_decoders"):
            raise ValueError("YAM/human native actions do not use a private decoder")
        if kwargs.get("energy_score_distance") is not None:
            raise ValueError("YAM/human EnergyScore distance is fixed by embodiment")
        configured = kwargs.pop("action_keys_by_embodiment", ACTION_KEYS)
        if {str(k).lower(): str(v) for k, v in dict(configured).items()} != ACTION_KEYS:
            raise ValueError("YAM/human evaluator action-key contract changed")
        kwargs["action_keys_by_embodiment"] = ACTION_KEYS
        super().__init__(**kwargs)
        self._co_train_artifact_root = self.artifact_root
        self._co_train_diagnostic_root = (
            None
            if self._action_flow_diagnostics is None
            else self._action_flow_diagnostics.artifact_root
        )
        self.energy_score_distance_metadata = {
            "type": "yam_human_per_embodiment_fixed_v1",
            "yam_bimanual": dict(YAM_ENERGY_DISTANCE_METADATA),
            "human_bimanual": dict(HUMAN_ENERGY_DISTANCE_METADATA),
        }

    def set_validation_group(self, group_name) -> None:
        """Give each validation domain its own diagnostic budget and artifacts."""
        if group_name not in {"yam", "human"}:
            raise ValueError(f"Unexpected YAM/human validation group {group_name!r}")
        previous = self._validation_group
        super().set_validation_group(group_name)
        if self._co_train_artifact_root is not None:
            self.artifact_root = self._co_train_artifact_root / group_name
        if self._action_flow_diagnostics is not None:
            self._action_flow_diagnostics.artifact_root = (
                self._co_train_diagnostic_root / group_name
            )
            if previous != group_name:
                self._action_flow_diagnostics.reset()

    @staticmethod
    def _native_residual(prediction, target, decoder):
        if decoder is not None:
            raise ValueError("YAM/human native residual must not use a decoder")
        width = int(prediction.shape[-1])
        if width == YAM_ACTION_DIM:
            return yam_native_residual(prediction, target)
        if width == HUMAN_ACTION_DIM:
            return human_native_residual(prediction, target)
        raise ValueError(f"Unexpected YAM/human action width {width}")

    def _energy_values(self, samples, target, embodiment_id, label=None):
        if label == "yam_bimanual":
            return YamCartesianActionFlowEval._energy_values(
                self, samples, target, embodiment_id, label
            )
        if label != "human_bimanual":
            raise ValueError(f"Unexpected YAM/human evaluation embodiment {label!r}")
        if samples.ndim != 4 or samples.shape[0] != 32 or target.ndim != 3:
            raise ValueError("Human EnergyScore@32 requires 32 sampled chunks")
        if (
            tuple(target.shape[-2:]) != (100, HUMAN_ACTION_DIM)
            or samples.shape[1:] != target.shape
        ):
            raise ValueError(
                "Human EnergyScore requires complete matching (100,138) chunks"
            )
        if not bool(torch.isfinite(samples).all()) or not bool(
            torch.isfinite(target).all()
        ):
            raise ValueError("Human EnergyScore received non-finite actions")

        def distance(left, right):
            left_native = self._native(left, embodiment_id, None)
            right_native = self._native(right, embodiment_id, None)
            native_residual = human_native_residual(left_native, right_native)
            translation = (
                (
                    left[..., list(HUMAN_WRIST_TRANSLATION_INDICES)]
                    - right[..., list(HUMAN_WRIST_TRANSLATION_INDICES)]
                )
                .square()
                .mean(dim=(-2, -1))
                .sqrt()
            )
            rotation = (
                (native_residual[..., list(HUMAN_WRIST_ANGLE_INDICES)] / torch.pi)
                .square()
                .mean(dim=(-2, -1))
                .sqrt()
            )
            keypoints = (
                (
                    left[..., list(HUMAN_KEYPOINT_INDICES)]
                    - right[..., list(HUMAN_KEYPOINT_INDICES)]
                )
                .square()
                .mean(dim=(-2, -1))
                .sqrt()
            )
            return (translation + rotation + keypoints) / 3.0

        values = energy_score(samples, target, distance_fn=distance)
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("Human EnergyScore produced non-finite values")
        return {name: value.detach() for name, value in values.items()}

    def _action_flow_native_error_fns(self, batch):
        runner = self._action_flow_diagnostics
        if runner is None or not runner.native_error_enabled:
            return None
        if runner.native_error != NATIVE_ERROR_CONTRACT:
            raise ValueError("YAM/human diagnostic native-error contract changed")
        functions = {}
        for source_id, source_batch in batch.items():
            embodiment_id, label = self._embodiment(source_batch)
            if label not in ACTION_KEYS:
                raise ValueError(f"Unexpected diagnostic embodiment {label!r}")

            def native_error(prediction, target, *, _id=embodiment_id):
                prediction_native = self._native(prediction, _id, None)
                target_native = self._native(target, _id, None)
                return self._native_mse_by_condition(
                    prediction_native, target_native, None
                )

            functions[source_id] = native_error
        return functions
