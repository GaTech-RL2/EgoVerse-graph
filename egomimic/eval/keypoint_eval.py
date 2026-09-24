"""MANO sequence validation using shared graph, metrics and video lifecycle."""

from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval


class KeypointEval(BimanualCartesianEval):
    def __init__(self, **kwargs):
        kwargs.setdefault("action_key", "actions_keypoints")
        kwargs.setdefault("obs_pose_key", "observations.state.keypoints")
        kwargs.setdefault("pose_metrics", True)
        super().__init__(**kwargs)

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        for source in batch.values():
            if source[self.action_key].shape[-1] != 138:
                raise ValueError(
                    "MANO bimanual actions require two [wrist pose(6), keypoints(63)] blocks"
                )
        return super().on_validation_step(batch, batch_idx, dataloader_idx)
