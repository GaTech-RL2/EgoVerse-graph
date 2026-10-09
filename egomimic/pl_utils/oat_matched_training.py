"""Released-observation checkpoint metadata for original Action Flow training."""

from egomimic.pl_utils.oat_training import OATEMACallback, OATTrainingBehavior
from egomimic.pl_utils.training_behavior_action_flow import ActionFlowTrainingBehavior
from egomimic.utils.ema_callback import EMACallback


class OATObservationActionFlowTrainingBehavior(ActionFlowTrainingBehavior):
    """Inherit the original optimizer/loss path; share OAT checkpoint context."""

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        OATTrainingBehavior.on_save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        OATTrainingBehavior.on_load_checkpoint(self, checkpoint)


class ActionFlowFixedEMACallback(EMACallback):
    """Original AF EMA counter, with the shared terminal-checkpoint hook."""

    def __init__(self, final_checkpoint_path=None, **kwargs):
        super().__init__(**kwargs)
        self.final_checkpoint_path = final_checkpoint_path

    on_train_end = OATEMACallback.on_train_end
