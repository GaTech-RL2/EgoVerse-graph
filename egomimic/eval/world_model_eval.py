"""Normalized prediction metrics for the offline world-model family."""

import torch

from egomimic.eval.eval import Eval, EvaluationDataRequirements
from egomimic.models.world_model import MODALITIES


class WorldModelEval(Eval):
    def __init__(self, sample_seed=42):
        self.sample_seed = int(sample_seed)
        self.totals = {}

    def bind_data_context(self, *, normalizer):
        if normalizer is not None:
            raise ValueError("World-model evaluator expects prepared normalized caches")

    def data_requirements(self):
        return EvaluationDataRequirements(required_keys=("world", "embodiment"))

    def trainer_overrides(self):
        return {}  # Respect configured step-based validation; no hidden override.

    def on_validation_start(self):
        self.totals = {}

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.model.device.type == "cuda":
            devices = [self.model.device.index or 0]
        else:
            devices = []
        # Repeatable validation without advancing the training sampler RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(
                self.sample_seed + int(batch_idx) + 100000 * dataloader_idx
            )
            output = self.model.forward_eval(batch)
        for source, item in batch.items():
            for name in MODALITIES:
                s = item["world"][name]
                weight = (
                    s.get("target_valid", s["valid"]) & s["valid"] & ~s["observed"]
                ).float()
                if "confidence" in s:
                    weight *= s["confidence"]
                active = weight > 0
                target = torch.where(active[..., None], s["value"], 0)
                prediction = output[source]["world_prediction"][name]
                error = (prediction.float() - target.float()).square().mean(-1)
                if not torch.isfinite(error[active]).all():
                    raise ValueError(
                        "Non-finite world-model validation prediction/target"
                    )
                key = f"val/{source}/{name}_normalized_mse"
                numerator = (error * weight).sum().double()
                denominator = weight.sum().double()
                old = self.totals.get(
                    key, (numerator.new_zeros(()), denominator.new_zeros(()))
                )
                self.totals[key] = (old[0] + numerator, old[1] + denominator)
        return output

    def log_epoch_metrics(self):
        for key, (numerator, denominator) in self.totals.items():
            if denominator > 0:
                self.model.log(key, (numerator / denominator).float(), sync_dist=False)

    def on_validation_end(self):
        return None
