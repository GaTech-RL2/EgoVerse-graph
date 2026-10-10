"""Common Cartesian14 metrics with the maintained robot open-loop collector."""

from egomimic.eval.yam_action_flow_eval import YamCartesianActionFlowEval
from egomimic.eval.yam_human_open_loop_eval import RobotOpenLoopMixin


class YamHumanCartesianDPEval(RobotOpenLoopMixin, YamCartesianActionFlowEval):
    def __init__(self, **kwargs):
        keys = {
            "yam_bimanual": "actions_cartesian",
            "human_bimanual": "actions_cartesian",
        }
        if dict(kwargs.get("action_keys_by_embodiment", keys)) != keys:
            raise ValueError(
                "Shared Cartesian14 evaluator requires Cartesian actions for both domains"
            )
        kwargs["action_keys_by_embodiment"] = keys
        super().__init__(**kwargs)
        self._shared_artifact_root = self.artifact_root

    def set_validation_group(self, group_name):
        if group_name not in {"yam", "human"}:
            raise ValueError("Unexpected validation group")
        super().set_validation_group(group_name)
        if self._shared_artifact_root is not None:
            self.artifact_root = self._shared_artifact_root / group_name
