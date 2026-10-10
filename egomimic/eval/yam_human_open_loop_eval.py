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


class RobotOpenLoopMixin:
    """Reuse ordered robot scoring with either native human action contract."""

    def __init__(
        self,
        *,
        requires_ordered_validation=True,
        robot_execute_fraction=0.25,
        robot_control_dt=1.0 / 30.0,
        **kwargs,
    ):
        if not requires_ordered_validation:
            raise ValueError("Robot open-loop scoring requires ordered validation")
        limit = kwargs.get("limit_val_batches")
        if not isinstance(limit, float) or limit != 1.0:
            raise ValueError("Robot open-loop scoring requires limit_val_batches: 1.0")
        super().__init__(**kwargs)
        self.requires_ordered_validation = True
        self.robot_open_loop = OpenLoopSimEval(
            action_key="actions_cartesian",
            ground_truth_action_key="actions_cartesian",
            action_mode="baseline",
            control_horizon=100,
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
                100,
                14,
            ):
                raise ValueError(
                    "Robot open-loop scoring requires matching [B,100,14] chunks"
                )
            if not bool(
                torch.isfinite(prediction).all() and torch.isfinite(target).all()
            ):
                raise ValueError("Robot open-loop scoring received non-finite actions")
            rate = source_batch.get("retiming_rate")
            if rate is None or not bool(torch.as_tensor(rate).eq(1.0).all()):
                raise ValueError(
                    "Robot open-loop scoring requires the identity retiming_rate 1.0"
                )
            self.robot_open_loop._append_source_records(
                source_id, source_batch, prediction
            )

    def on_validation_end(self):
        super().on_validation_end()
        return self.robot_open_loop.on_validation_end()


class YamHumanOpenLoopEval(RobotOpenLoopMixin, YamHumanKeypointsActionFlowEval):
    """Existing 14D robot / 138D human checkpoint contract."""
