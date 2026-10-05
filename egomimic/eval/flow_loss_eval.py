"""Validation flow-matching loss: the model's own training objective on validation batches.

The stationery E1 runs never log a validation loss -- their validation step hands every batch to the
evaluator, which reports task metrics. This evaluator computes ``compute_losses(forward_training(batch))``
on each (already normalized) validation batch under ``torch.no_grad`` with the noise / flow-time draws
pinned per batch (``seed + batch_idx``), so checkpoints of one run, and runs of one token layout, are scored
on identical draws. The loss lives in each run's own normalized action space: compare checkpoints within a
run, not values across token layouts.

Eval-only use: unknown keyword arguments are accepted and ignored, because an experiment's inline evaluator
block (e.g. the open-loop settings) merges into whichever evaluator group is selected.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from egomimic.eval.eval import Eval


class FlowLossEval(Eval):
    def __init__(self, seed: int = 0, results_path: str | None = None, **ignored):
        self.seed = int(seed)
        self.results_path = results_path
        self.ignored = sorted(ignored)
        # trainHydra (mode=eval) copies this onto the trainer config: score the whole validation set.
        self.override_dict = {"limit_val_batches": 1.0}
        self.model = None
        self._sum, self._n, self._per_source = 0.0, 0, {}

    def bind_data_context(self, *, normalizer):
        del normalizer

    def on_validation_start(self):
        self._sum, self._n, self._per_source = 0.0, 0, {}

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        algo = self.model.model  # ModelWrapper -> PipelineAlgo
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed + int(batch_idx))
            losses = algo.compute_losses(algo.forward_training(batch), batch)
        n = sum(int(next(v for v in src.values() if torch.is_tensor(v)).shape[0]) for src in batch.values())
        self._sum += float(losses["loss"]) * n
        self._n += n
        for key, value in losses.items():
            if key != "loss" and torch.is_tensor(value) and value.numel() == 1:
                s = self._per_source.setdefault(key, [0.0, 0])
                s[0] += float(value) * n
                s[1] += n
        return {}

    def on_validation_end(self):
        result = {
            "flow_loss": self._sum / max(self._n, 1),
            "samples": self._n,
            "seed": self.seed,
            "components": {k: v[0] / max(v[1], 1) for k, v in self._per_source.items()},
        }
        print("FLOW_LOSS", json.dumps(result), flush=True)
        if self.results_path:
            Path(self.results_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.results_path).write_text(json.dumps(result, indent=1))
        return result
