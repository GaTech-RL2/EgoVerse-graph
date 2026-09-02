"""Teacher-forced Planar MSE and Energy Score without video/report machinery."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from egomimic.eval.energy_score import energy_score
from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment


class PlanarActionEval(Eval):
    """Log normalized/native MSE and seeded EnergyScore@32 per embodiment."""

    def __init__(
        self,
        limit_val_batches: int = 400,
        seed_bank_path: str | None = None,
        seed_bank_sha256: str | None = None,
        artifact_root: str | None = None,
        semantic_blocks=((0, 2), (2, 4), (4, 5)),
        energy_score_enabled: bool = True,
    ):
        self.trainer = None
        self.model = None
        self.blocks = tuple(tuple(map(int, block)) for block in semantic_blocks)
        self.energy_score_enabled = bool(energy_score_enabled)
        self.artifact_root = None if artifact_root is None else Path(artifact_root)
        self.seeds = []
        self.seed_bank_sha256 = None
        if self.energy_score_enabled:
            path = Path(str(seed_bank_path)).resolve()
            digest = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None
            )
            if digest != seed_bank_sha256:
                raise ValueError(f"Energy Score seed-bank identity mismatch: {path}")
            payload = json.loads(path.read_text())
            self.seeds = [int(seed) for seed in payload.get("seeds", [])]
            if len(self.seeds) != 32 or len(set(self.seeds)) != 32:
                raise ValueError("EnergyScore@32 requires 32 unique seeds")
            if self.artifact_root is None:
                raise ValueError("Energy Score needs a non-empty artifact_root")
            self.seed_bank_sha256 = digest
        self.override_dict = {
            "limit_train_batches": 0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": 1,
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def on_validation_start(self):
        return None

    def on_validation_end(self):
        return None

    def _seeded_predictions(self, batch):
        predictions = {embodiment_id: [] for embodiment_id in batch}
        cuda_devices = sorted(
            {
                int(value.device.index)
                for domain_batch in batch.values()
                for value in domain_batch.values()
                if torch.is_tensor(value)
                and value.device.type == "cuda"
                and value.device.index is not None
            }
        )
        for seed in self.seeds:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                result = self.model.forward_eval(batch)
            for embodiment_id in batch:
                action_key = self.model.resolved_ac_keys[embodiment_id]
                predictions[embodiment_id].append(
                    result[f"emb{embodiment_id}_{action_key}"].detach()
                )
        return {
            embodiment_id: torch.stack(values)
            for embodiment_id, values in predictions.items()
        }

    def _native(self, normalized, embodiment_id):
        action_key = self.model.resolved_ac_keys[embodiment_id]
        unnormalized = self.model.norm_stats.unnormalize(
            {action_key: normalized}, embodiment_id
        )[action_key]
        adapter = self.model.rollout_adapter_for(embodiment_id)
        return adapter.decode(unnormalized) if adapter is not None else unnormalized

    def _save_artifact(self, batch_idx, samples, batch):
        destination = (
            self.artifact_root
            / f"epoch-{int(self.trainer.current_epoch)}-step-{int(self.trainer.global_step)}"
            / f"rank-{int(self.trainer.global_rank)}-batch-{int(batch_idx)}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        domains = {}
        for embodiment_id, values in samples.items():
            action_key = self.model.resolved_ac_keys[embodiment_id]
            domains[get_embodiment(embodiment_id)] = {
                "samples": values.float().cpu(),
                "target": batch[embodiment_id][action_key].detach().float().cpu(),
            }
        torch.save(
            {
                "schema_version": 1,
                "metric": "EnergyScore@32",
                "seeds": self.seeds,
                "seed_bank_sha256": self.seed_bank_sha256,
                "semantic_blocks": self.blocks,
                "domains": domains,
            },
            temporary,
        )
        os.replace(temporary, destination)

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        result = self.model.forward_eval(batch)
        metrics = {}
        normalized_values = []
        native_values = []
        for embodiment_id, domain_batch in batch.items():
            domain = get_embodiment(embodiment_id).lower()
            action_key = self.model.resolved_ac_keys[embodiment_id]
            prediction = result[f"emb{embodiment_id}_{action_key}"]
            target = domain_batch[action_key]
            normalized_mse = (prediction - target).square().mean()
            native_mse = (
                (
                    self._native(prediction, embodiment_id)
                    - self._native(target, embodiment_id)
                )
                .square()
                .mean()
            )
            metrics[f"Valid/MSE/{domain}"] = normalized_mse
            metrics[f"Valid/Native_MSE/{domain}"] = native_mse
            normalized_values.append(normalized_mse)
            native_values.append(native_mse)
        metrics["Valid/MSE"] = torch.stack(normalized_values).mean()
        metrics["Valid/Native_MSE"] = torch.stack(native_values).mean()

        if self.energy_score_enabled:
            sampled = self._seeded_predictions(batch)
            scores = []
            for embodiment_id, samples in sampled.items():
                domain = get_embodiment(embodiment_id).lower()
                action_key = self.model.resolved_ac_keys[embodiment_id]
                values = energy_score(
                    samples, batch[embodiment_id][action_key], self.blocks
                )
                metrics[f"Valid/EnergyScore@32/{domain}"] = values["score"]
                metrics[f"Valid/EnergyScoreAccuracy@32/{domain}"] = values["accuracy"]
                metrics[f"Valid/EnergyScoreDiversity@32/{domain}"] = values["diversity"]
                scores.append(values)
            for key, metric_name in (
                ("score", "Valid/EnergyScore@32"),
                ("accuracy", "Valid/EnergyScoreAccuracy@32"),
                ("diversity", "Valid/EnergyScoreDiversity@32"),
            ):
                metrics[metric_name] = torch.stack(
                    [value[key] for value in scores]
                ).mean()
            self._save_artifact(batch_idx, sampled, batch)
        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
