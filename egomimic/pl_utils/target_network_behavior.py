"""Optimizer lifecycle for graph stages with Polyak target networks."""
from egomimic.pl_utils.training_behavior import TrainingBehavior


class TargetNetworkBehavior(TrainingBehavior):
    def __init__(self, prediction_log_interval=1):
        super().__init__()
        if int(prediction_log_interval) != prediction_log_interval or prediction_log_interval < 1:
            raise ValueError("prediction_log_interval must be a positive integer")
        self.prediction_log_interval = int(prediction_log_interval)

    def training_step(self, batch, batch_idx):
        # global_step is restored by Lightning, so diagnostics remain aligned
        # with logger flushes even when resuming partway through an interval.
        due = (self.context.global_step + 1) % self.prediction_log_interval == 0
        batch = {source: {**values, "log_predictions": due} for source, values in batch.items()}
        return super().training_step(batch, batch_idx)

    def on_before_optimizer_step(self, optimizer):
        for module in self.context.nets.modules():
            update = getattr(module, "update_target_before_optimizer", None)
            if update is not None:
                update()
        return super().on_before_optimizer_step(optimizer)
