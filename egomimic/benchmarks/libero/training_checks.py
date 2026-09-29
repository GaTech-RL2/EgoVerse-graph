"""Training receipts and checkpoint identity checks, never a training loop."""

from pathlib import Path
import time

from lightning import Callback
import numpy as np
from omegaconf import OmegaConf
import torch

from egomimic.benchmarks.libero.cluster import write_json


class StepTiming(Callback):
    """Use an explicit output path: saved model configs omit launch paths."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.steps, self.validations = [], []
        self.previous = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.started = self.previous or time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        self.steps.append({"step": trainer.global_step, "seconds": now - self.started})
        self.previous = now

    def on_validation_start(self, trainer, pl_module):
        assert not trainer.sanity_checking and trainer.global_step > 0
        self.valid_started = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        self.validations.append(
            {
                "step": trainer.global_step,
                "seconds": time.perf_counter() - self.valid_started,
            }
        )
        self.previous = None

    def on_train_end(self, trainer, pl_module):
        write_json(
            self.output_dir / f"timing-rank-{trainer.global_rank}.json",
            {
                "rank": trainer.global_rank,
                "world": trainer.world_size,
                "global_step": trainer.global_step,
                "steps": self.steps,
                "validations": self.validations,
            },
        )


def check_weights_only_payload(payload, cfg, expected_step, normalizer):
    """Validate against the entry point's real minimal checkpoint schema."""
    from egomimic.trainHydra import _build_model_config_tree

    assert (
        payload["global_step"] == expected_step
        and payload["ema_num_updates"] == expected_step
    )
    assert "optimizer_states" not in payload and "ema_state_dict" in payload
    budget = payload["training_budget"]
    assert budget["world_size"] == cfg.trainer.devices
    assert budget["global_batch_size"] == cfg.callbacks.batch_budget.global_batch_size
    assert budget["total_optimizer_steps"] == cfg.trainer.max_steps
    saved = OmegaConf.create(payload["hyper_parameters"]["config_tree"])
    expected = _build_model_config_tree(cfg)
    assert OmegaConf.to_container(saved, resolve=True) == OmegaConf.to_container(
        expected, resolve=True
    )

    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (np.ndarray, torch.Tensor)):
            assert np.array_equal(np.asarray(a), np.asarray(b))
        else:
            assert a == b

    compare(payload["normalizer_state"], normalizer)
