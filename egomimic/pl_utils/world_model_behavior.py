"""Use the shared Lightning wrapper; schedule follows actual optimizer steps."""

import random

import numpy as np
import torch

from egomimic.pl_utils.training_behavior import TrainingBehavior


class WorldModelTrainingBehavior(TrainingBehavior):
    def on_validation_epoch_end(self):
        if self.context.evaluator is not None:
            self.context.evaluator.log_epoch_metrics()

    def training_step(self, batch, batch_idx):
        pending = getattr(self, "_pending_rng", None)
        if pending is not None:
            torch.set_rng_state(pending["torch_cpu"].cpu())
            if pending["torch_cuda"]:
                if not torch.cuda.is_available():
                    raise ValueError("Cannot restore CUDA RNG on a CPU training runtime")
                torch.cuda.set_rng_state_all([state.cpu() for state in pending["torch_cuda"]])
            np.random.set_state(pending["numpy"])
            random.setstate(pending["python"])
            self._pending_rng = None
        data = self.context.trainer.datamodule
        if getattr(data, "checkpointable_sampling", False):
            cursors = {values["sampling_cursor"] for values in batch.values()}
            if len(cursors) != 1:
                raise ValueError("Expected one consumed MoT batch cursor")
            data.acknowledge_batch(cursors.pop())
        batch = {
            name: {**values, "optimizer_step": int(self.context.global_step)}
            for name, values in batch.items()
        }
        return self.context._default_training_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["mot_rng_state"] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate(),
        }

    def on_load_checkpoint(self, checkpoint):
        self._pending_rng = checkpoint.get("mot_rng_state")
        if (getattr(self.context.trainer.datamodule, "checkpointable_sampling", False)
                and self._pending_rng is None):
            raise ValueError("Exact MoT resume requires saved model RNG state")
