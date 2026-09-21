"""Optimizer lifecycle for graph stages with Polyak target networks."""
from egomimic.pl_utils.training_behavior import TrainingBehavior


class TargetNetworkBehavior(TrainingBehavior):
    def on_before_optimizer_step(self, optimizer):
        for module in self.context.nets.modules():
            update = getattr(module, "update_target_before_optimizer", None)
            if update is not None:
                update()
        return super().on_before_optimizer_step(optimizer)
