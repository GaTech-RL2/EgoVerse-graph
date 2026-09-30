"""Native LIBERO validation and fixed-bank Action Flow EnergyScore@32."""

import hashlib
import json
from pathlib import Path

import torch

from egomimic.eval.energy_score import energy_score
from egomimic.eval.libero_eval import LiberoActionEvaluator, action_metrics
from egomimic.rldb.zarr.libero_dataset import EMBODIMENT


class LiberoActionFlowEvaluator(LiberoActionEvaluator):
    def __init__(
        self,
        energy_sample_count=32,
        energy_seed_bank_path=None,
        energy_seed_bank_sha256=None,
        diagnostic_raw_noise_levels=(0.0, 0.25, 0.5, 0.75, 1.0),
    ):
        super().__init__()
        self.energy_sample_count = int(energy_sample_count)
        if self.energy_sample_count != 32:
            raise ValueError("LIBERO Action Flow requires EnergyScore@32")
        path = Path(str(energy_seed_bank_path))
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != energy_seed_bank_sha256:
            raise ValueError("LIBERO EnergyScore seed-bank SHA mismatch")
        seeds = json.loads(raw)["seeds"]
        if len(seeds) != 32 or len(set(seeds)) != 32:
            raise ValueError("LIBERO EnergyScore needs 32 distinct seeds")
        self.seeds = tuple(int(seed) for seed in seeds)
        self.diagnostic_raw_noise_levels = tuple(
            float(level) for level in diagnostic_raw_noise_levels
        )

    def _predict(self, batch, seed):
        devices = sorted({
            int(value.device.index)
            for source_batch in batch.values()
            for value in source_batch.values()
            if torch.is_tensor(value)
            and value.device.type == "cuda"
            and value.device.index is not None
        })
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return self.model.forward_eval(batch)

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        first = self._predict(batch, self.seeds[0])
        prefix = "Valid" if self.group == "valid" else f"Valid_{self.group}"
        for source, values in batch.items():
            target_norm = values["actions"]
            prediction_norm = first[source]["pred_action"]
            target = self.normalizer.unnormalize(
                {"actions": target_norm}, EMBODIMENT
            )["actions"]
            prediction = self.normalizer.unnormalize(
                {"actions": prediction_norm}, EMBODIMENT
            )["actions"]
            metrics = action_metrics(prediction, target)
            metrics["normalized_reconst_mse"] = (
                prediction_norm - target_norm
            ).square().mean()
            if batch_idx == 0:
                sample_norm = torch.stack([
                    self._predict(batch, seed)[source]["pred_action"].detach()
                    for seed in self.seeds
                ])
                sample_native = self.normalizer.unnormalize(
                    {"actions": sample_norm}, EMBODIMENT
                )["actions"]

                def native_chunk_distance(left, right):
                    # The OSC action components have distinct units. Report
                    # native per-component distances, not a pooled pseudo-unit.
                    delta = left - right
                    translation = delta[..., :3].square().mean(dim=(-2, -1)).sqrt()
                    rotation = delta[..., 3:6].square().mean(dim=(-2, -1)).sqrt()
                    gripper = delta[..., 6:].square().mean(dim=(-2, -1)).sqrt()
                    return (translation + rotation + gripper) / 3

                scores = energy_score(
                    sample_native, target, distance_fn=native_chunk_distance
                )
                metrics.update({
                    "energy_score32_native_equal_components": scores["score"],
                    "energy_accuracy32_native_equal_components": scores["accuracy"],
                    "energy_diversity32_native_equal_components": scores["diversity"],
                })
                diagnostics = self.model.forward_action_flow_diagnostics(
                    batch,
                    raw_noise_levels=self.diagnostic_raw_noise_levels,
                    noise_seed=self.seeds[0],
                    max_samples=min(8, len(target)),
                    jacobian_samples=2,
                    capture_activations=True,
                )[source]
                for key in ("clean_latent", "clean_decoded_action_normalized"):
                    tensor = diagnostics[key]
                    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"non-finite LIBERO Action Flow diagnostic: {key}")
                    metrics[f"diagnostic_{key}_rms"] = tensor.square().mean().sqrt()
            for key, metric in metrics.items():
                if not bool(torch.isfinite(metric)):
                    raise ValueError(f"non-finite LIBERO validation metric: {key}")
                self.model.log(
                    f"{prefix}/{key}", metric, batch_size=len(target),
                    on_step=False, on_epoch=True, sync_dist=True,
                )
