"""Atomic progress receipts for remotely supervised training runs."""
import json
from pathlib import Path
import time

from lightning.pytorch.callbacks import Callback
import torch


class RunProgressCallback(Callback):
    def __init__(self, path, every_n_steps=100, require_homogeneous=False):
        self.path = Path(path)
        self.every_n_steps = int(every_n_steps)
        self.require_homogeneous = bool(require_homogeneous)

    def on_train_start(self, trainer, pl_module):
        self.started = time.monotonic()
        self.start_step = int(trainer.global_step)
        if trainer.is_global_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del batch_idx
        step = int(trainer.global_step)
        if not trainer.is_global_zero or (step % self.every_n_steps and step != self.start_step + 1 and step != trainer.max_steps):
            return
        layout = getattr(pl_module.model, "last_training_batch_layout", {})
        if self.require_homogeneous and not layout.get("fused"):
            raise RuntimeError("Expected one homogeneous forward pass across embodiments")
        elapsed = time.monotonic() - self.started
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        value = {"global_step": step, "start_step": self.start_step,
            "elapsed_training_seconds": elapsed,
            "seconds_per_update": elapsed / max(1, step - self.start_step),
            "trainable_parameters": sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            "batch_layout": layout, "time": time.time(),
            "loss": float(loss.detach().cpu()) if torch.is_tensor(loss) else None}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
        temporary.replace(self.path)
