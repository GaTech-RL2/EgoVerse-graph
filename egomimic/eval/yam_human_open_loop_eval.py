"""Add robot episode-segment scoring to the fixed YAM/human metrics.

Human keypoint chunks retain their existing metrics: their raw/resampled clocks
do not satisfy the robot control-frequency episode-walk contract. Complete-window
datasets also omit episode tails. Open-loop coverage therefore describes the
eligible validation-window-start range, not the entire raw recording.
"""

import torch

from egomimic.eval.open_loop_sim import OpenLoopSimEval
from egomimic.eval.yam_human_keypoints_action_flow_eval import (
    YamHumanKeypointsActionFlowEval,
)


def _select_batch_rows(value, mask):
    """Keep metadata and tensor rows aligned when scoring native-rate views."""
    batch_size = mask.numel()
    if isinstance(value, dict):
        return {key: _select_batch_rows(item, mask) for key, item in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
        return value[mask.to(value.device)]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        rows = [value[index] for index in mask.nonzero().flatten().tolist()]
        return tuple(rows) if isinstance(value, tuple) else rows
    return value


class RobotOpenLoopMixin:
    """Reuse ordered robot scoring with either native human action contract."""

    def __init__(
        self,
        *,
        requires_ordered_validation=True,
        robot_execute_fraction=0.25,
        robot_control_dt=1.0 / 30.0,
        robot_prediction_horizon=100,
        **kwargs,
    ):
        if not requires_ordered_validation:
            raise ValueError("Robot open-loop scoring requires ordered validation")
        limit = kwargs.get("limit_val_batches")
        if not isinstance(limit, float) or limit != 1.0:
            raise ValueError("Robot open-loop scoring requires limit_val_batches: 1.0")
        if type(robot_prediction_horizon) is not int or robot_prediction_horizon < 1:
            raise ValueError("Robot prediction horizon must be positive integer")
        self.robot_prediction_horizon = robot_prediction_horizon
        super().__init__(**kwargs)
        self.yam_action_horizon = robot_prediction_horizon
        if "yam_bimanual" in self.energy_score_distance_metadata:
            self.energy_score_distance_metadata["yam_bimanual"].update(
                complete_normalized_chunk_shape=[robot_prediction_horizon, 14],
                coverage=f"all_{robot_prediction_horizon}_steps_and_all_14_action_channels",
            )
        self.requires_ordered_validation = True
        self.robot_open_loop = OpenLoopSimEval(
            action_key="actions_cartesian",
            ground_truth_action_key="actions_cartesian",
            action_mode="baseline",
            control_horizon=robot_prediction_horizon,
            execute_fraction=robot_execute_fraction,
            control_dt=robot_control_dt,
        )

    def bind_data_context(self, *, normalizer):
        super().bind_data_context(normalizer=normalizer)
        self.robot_open_loop.bind_data_context(normalizer=normalizer)

    def on_validation_start(self):
        if self.trainer is not None:
            limit = self.trainer.limit_val_batches
            if not isinstance(limit, float) or limit != 1.0:
                raise ValueError(
                    "trainer.limit_val_batches must be 1.0 for robot open-loop scoring"
                )
            if getattr(self.trainer, "num_sanity_val_steps", 0) != 0:
                raise ValueError(
                    "Robot open-loop scoring requires num_sanity_val_steps: 0"
                )
        super().on_validation_start()
        self.robot_open_loop.model = self.model
        self.robot_open_loop.trainer = self.trainer
        self.robot_open_loop.on_validation_start()

    def _collect_validation_predictions(self, batch, result):
        self.robot_open_loop.set_validation_group(self._validation_group)
        for source_id, source_batch in batch.items():
            _, label = self._embodiment(source_batch)
            if label != "yam_bimanual":
                continue
            prediction = result[source_id]["pred_action"]
            target = source_batch["actions_cartesian"]
            if prediction.shape != target.shape or tuple(target.shape[-2:]) != (
                self.robot_prediction_horizon,
                14,
            ):
                raise ValueError(
                    f"Robot open-loop scoring requires matching [B,{self.robot_prediction_horizon},14] chunks"
                )
            if not bool(
                torch.isfinite(prediction).all() and torch.isfinite(target).all()
            ):
                raise ValueError("Robot open-loop scoring received non-finite actions")
            rate = source_batch.get("retiming_rate")
            if rate is None:
                raise ValueError(
                    "Robot open-loop scoring requires retiming_rate metadata"
                )
            rate = torch.as_tensor(rate, device=prediction.device).reshape(-1)
            if rate.numel() != prediction.shape[0] or not bool(
                torch.isfinite(rate).all() and rate.gt(0).all()
            ):
                raise ValueError(
                    "Robot open-loop scoring requires one finite positive rate per sample"
                )
            identity = rate.eq(1.0)
            if not bool(identity.any()):
                continue
            if not bool(identity.all()):
                source_batch = _select_batch_rows(source_batch, identity)
                prediction = prediction[identity]
            self.robot_open_loop._append_source_records(
                source_id, source_batch, prediction
            )

    def on_validation_end(self):
        super().on_validation_end()
        return self.robot_open_loop.on_validation_end()


class YamHumanOpenLoopEval(RobotOpenLoopMixin, YamHumanKeypointsActionFlowEval):
    """Existing 14D robot / 138D human checkpoint contract."""
