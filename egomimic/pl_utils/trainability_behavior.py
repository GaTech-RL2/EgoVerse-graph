"""Configured freeze/unfreeze schedules without owning model parameters."""

from egomimic.pipeline.initialization import configure_trainability
from egomimic.pl_utils.training_behavior import TrainingBehavior


class TrainabilitySchedule(TrainingBehavior):
    def __init__(self, schedule):
        super().__init__()
        self.schedule = tuple(schedule)
        steps = [entry["at_step"] for entry in self.schedule]
        if any(type(step) is not int or step < 0 for step in steps) or steps != sorted(
            set(steps)
        ):
            raise ValueError(
                "Trainability schedule steps must be unique, sorted nonnegative integers"
            )
        self._applied = -1

    def _apply(self, step):
        for entry in self.schedule:
            if self._applied < entry["at_step"] <= step:
                configure_trainability(self.context.model.pipeline, entry["targets"])
                self._applied = entry["at_step"]

    def on_bind(self):
        self._apply(0)

    def training_step(self, batch, batch_idx):
        self._apply(self.context.global_step)
        return super().training_step(batch, batch_idx)

    def on_load_checkpoint(self, checkpoint):
        self._applied = -1
        self._apply(int(checkpoint["global_step"]))
