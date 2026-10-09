"""Released-observation checkpoint metadata for original Action Flow training."""
from egomimic.pl_utils.training_behavior_action_flow import ActionFlowTrainingBehavior
from egomimic.pl_utils.oat_training import OATTrainingBehavior


class OATObservationActionFlowTrainingBehavior(ActionFlowTrainingBehavior):
    """Inherit the original optimizer/loss path; share OAT checkpoint context."""
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        OATTrainingBehavior.on_save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        OATTrainingBehavior.on_load_checkpoint(self, checkpoint)
