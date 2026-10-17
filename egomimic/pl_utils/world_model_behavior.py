"""Use the shared Lightning wrapper; schedule follows actual optimizer steps."""

from egomimic.pl_utils.training_behavior import TrainingBehavior


class WorldModelTrainingBehavior(TrainingBehavior):
    def on_validation_epoch_end(self):
        if self.context.evaluator is not None:
            self.context.evaluator.log_epoch_metrics()

    def training_step(self, batch, batch_idx):
        batch = {
            name: {**values, "optimizer_step": int(self.context.global_step)}
            for name, values in batch.items()
        }
        return self.context._default_training_step(batch, batch_idx)
