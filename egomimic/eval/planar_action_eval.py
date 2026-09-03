"""Teacher-forced Planar MSE and Energy Score without video/report machinery."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from egomimic.eval.energy_score import energy_score
from egomimic.eval.eval import Eval
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.rldb.embodiment.embodiment import get_embodiment

_DEFAULT_DETERMINISTIC_SEED = 420042


class PlanarActionEval(Eval):
    """Log deterministic normalized/native MSE and seeded EnergyScore@32."""

    def __init__(
        self,
        limit_val_batches: int = 400,
        seed_bank_path: str | None = None,
        seed_bank_sha256: str | None = None,
        artifact_root: str | None = None,
        semantic_blocks=((0, 2), (2, 4), (4, 5)),
        energy_score_enabled: bool = True,
        action_key: str = "actions",
        native_decoder=None,
        deterministic_seed: int = _DEFAULT_DETERMINISTIC_SEED,
        energy_score_max_batches_per_rank: int | None = None,
        energy_score_validation_view: Mapping | None = None,
        energy_score_provenance: Mapping | None = None,
    ):
        self.trainer = None
        self.model = None
        self.normalizer = None
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.native_decoder = native_decoder
        self.blocks = tuple(tuple(map(int, block)) for block in semantic_blocks)
        self.energy_score_enabled = bool(energy_score_enabled)
        self.deterministic_seed = int(deterministic_seed)
        self.energy_score_max_batches_per_rank = energy_score_max_batches_per_rank
        self.energy_score_validation_view = self._metadata_copy(
            energy_score_validation_view,
            label="energy_score_validation_view",
        )
        self.energy_score_provenance = self._metadata_copy(
            energy_score_provenance,
            label="energy_score_provenance",
        )
        if (
            energy_score_max_batches_per_rank is not None
            and int(energy_score_max_batches_per_rank) <= 0
        ):
            raise ValueError("energy_score_max_batches_per_rank must be positive")
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
            if self.seeds[0] != self.deterministic_seed:
                raise ValueError(
                    "deterministic_seed must equal the first Energy Score seed "
                    f"for prediction reuse: {self.deterministic_seed} != "
                    f"{self.seeds[0]}"
                )
            self.seed_bank_sha256 = digest
        self.override_dict = {
            "limit_train_batches": 0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": 1,
            "max_epochs": 1,
            "min_epochs": 1,
        }

    @staticmethod
    def _metadata_copy(value, *, label):
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping or None")

        def to_plain(item):
            if isinstance(item, Mapping):
                return {str(key): to_plain(child) for key, child in item.items()}
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                return [to_plain(child) for child in item]
            return copy.deepcopy(item)

        return to_plain(value)

    def on_validation_start(self):
        return None

    def on_validation_end(self):
        return None

    def bind_data_context(self, *, normalizer):
        """Attach data-owned normalization without putting it on PipelineAlgo."""
        self.normalizer = normalizer

    @staticmethod
    def _embodiment(source_batch):
        if "embodiment" not in source_batch:
            raise KeyError("Planar evaluation batch is missing embodiment metadata")
        embodiment_id = int(
            resolve_homogeneous_scalar(source_batch["embodiment"], label="embodiment")
        )
        name = get_embodiment(embodiment_id)
        if name is None:
            raise KeyError(f"Unknown Planar embodiment id {embodiment_id}")
        return embodiment_id, name.lower()

    @staticmethod
    def _cuda_devices(batch):
        return sorted(
            {
                int(value.device.index)
                for source_batch in batch.values()
                for value in source_batch.values()
                if torch.is_tensor(value)
                and value.device.type == "cuda"
                and value.device.index is not None
            }
        )

    def _forward_with_seed(self, batch, seed):
        """Run one stochastic prediction without changing caller RNG state."""
        with torch.random.fork_rng(devices=self._cuda_devices(batch)):
            torch.manual_seed(int(seed))
            return self.model.forward_eval(batch)

    def _seeded_predictions(self, batch):
        predictions = {source_id: [] for source_id in batch}
        first_result = None
        for index, seed in enumerate(self.seeds):
            result = self._forward_with_seed(batch, seed)
            if index == 0:
                first_result = result
            for source_id in batch:
                predictions[source_id].append(
                    result[source_id]["pred_action"].detach()
                )
        if first_result is None:  # guarded by the seed-bank constructor check
            raise RuntimeError("Energy Score seed bank is empty")
        return (
            {
                source_id: torch.stack(values)
                for source_id, values in predictions.items()
            },
            first_result,
        )

    def _native(self, normalized, embodiment_id):
        if self.normalizer is None:
            raise RuntimeError("Planar evaluator data context was not bound")
        unnormalized = self.normalizer.unnormalize(
            {self.action_key: normalized}, embodiment_id
        )[self.action_key]
        if self.native_decoder is None:
            return unnormalized
        return self.native_decoder.decode(unnormalized)

    def _native_mse(self, prediction, target):
        """Measure native Planar actions with a circular theta residual."""
        if prediction.shape != target.shape:
            raise ValueError(
                "native prediction and target shapes differ: "
                f"{prediction.shape} != {target.shape}"
            )
        # Without a decoder, rotation remains encoded as cos/sin in common-five
        # space, where ordinary subtraction is already wrap-safe.
        if self.native_decoder is None or prediction.shape[-1] < 3:
            return (prediction - target).square().mean()

        residual = prediction - target
        theta_residual = torch.atan2(
            torch.sin(residual[..., 2]), torch.cos(residual[..., 2])
        )
        residual = torch.cat(
            (residual[..., :2], theta_residual.unsqueeze(-1), residual[..., 3:]),
            dim=-1,
        )
        return residual.square().mean()

    def _energy_values(self, samples, target):
        if samples.ndim != 4 or samples.shape[0] != 32:
            raise ValueError("EnergyScore@32 requires exactly 32 samples")
        values = {
            name: value.detach()
            for name, value in energy_score(samples, target, self.blocks).items()
        }
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("Energy Score produced non-finite values")
        return values

    def _save_artifact(self, batch_idx, samples, scores, batch):
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
        if set(samples) != set(scores) or set(samples) != set(batch):
            raise ValueError("Energy Score source identities do not match")
        for source_id, predictions in samples.items():
            embodiment_id, name = self._embodiment(batch[source_id])
            if name in domains:
                raise ValueError(f"Duplicate Planar evaluation embodiment {name!r}")
            values = scores[source_id]
            domains[name] = {
                "source_id": source_id,
                "embodiment_id": embodiment_id,
                "action_key": self.action_key,
                "predictions": predictions.float().cpu(),
                "targets": batch[source_id][self.action_key].detach().float().cpu(),
                "accuracy_by_condition": values["accuracy_by_condition"].float().cpu(),
                "diversity_by_condition": values["diversity_by_condition"]
                .float()
                .cpu(),
                "score_by_condition": values["score_by_condition"].float().cpu(),
            }
        precision = getattr(self.trainer, "precision", None)
        torch.save(
            {
                "schema_version": 1,
                "metric": "EnergyScore@32",
                "sample_count": 32,
                "seed_bank": self.seeds,
                "seed_bank_sha256": self.seed_bank_sha256,
                "deterministic_seed": self.deterministic_seed,
                "distance": {
                    "space": "normalized_action_chunk",
                    "formula": "mean_equal_weight_semantic_block_rms",
                    "semantic_blocks": self.blocks,
                },
                "aggregation": "condition_mean_then_equal_domain_macro_mean",
                "global_step": int(self.trainer.global_step),
                "epoch": int(self.trainer.current_epoch),
                "rank": int(self.trainer.global_rank),
                "batch_idx": int(batch_idx),
                "precision": None if precision is None else str(precision),
                "validation_view": self.energy_score_validation_view,
                "provenance": self.energy_score_provenance,
                "domains": domains,
            },
            temporary,
        )
        os.replace(temporary, destination)

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        sampled = None
        score_this_batch = self.energy_score_enabled and (
            self.energy_score_max_batches_per_rank is None
            or batch_idx < int(self.energy_score_max_batches_per_rank)
        )
        if score_this_batch:
            sampled, result = self._seeded_predictions(batch)
        else:
            result = self._forward_with_seed(batch, self.deterministic_seed)
        metrics = {}
        normalized_values = []
        native_values = []
        labels = set()
        for source_id, source_batch in batch.items():
            embodiment_id, label = self._embodiment(source_batch)
            if label in labels:
                raise ValueError(f"Duplicate Planar evaluation embodiment {label!r}")
            labels.add(label)
            prediction = result[source_id]["pred_action"]
            target = source_batch[self.action_key]
            normalized_mse = (prediction - target).square().mean()
            native_mse = self._native_mse(
                self._native(prediction, embodiment_id),
                self._native(target, embodiment_id),
            )
            metrics[f"Valid/MSE/{label}"] = normalized_mse
            metrics[f"Valid/Native_MSE/{label}"] = native_mse
            normalized_values.append(normalized_mse)
            native_values.append(native_mse)
        metrics["Valid/MSE"] = torch.stack(normalized_values).mean()
        metrics["Valid/Native_MSE"] = torch.stack(native_values).mean()

        if sampled is not None:
            scores = []
            scores_by_source = {}
            for source_id, samples in sampled.items():
                _, label = self._embodiment(batch[source_id])
                values = self._energy_values(samples, batch[source_id][self.action_key])
                metrics[f"Valid/EnergyScore@32/{label}"] = values["score"]
                metrics[f"Valid/EnergyScoreAccuracy@32/{label}"] = values["accuracy"]
                metrics[f"Valid/EnergyScoreDiversity@32/{label}"] = values["diversity"]
                scores.append(values)
                scores_by_source[source_id] = values
            for key, metric_name in (
                ("score", "Valid/EnergyScore@32"),
                ("accuracy", "Valid/EnergyScoreAccuracy@32"),
                ("diversity", "Valid/EnergyScoreDiversity@32"),
            ):
                metrics[metric_name] = torch.stack(
                    [value[key] for value in scores]
                ).mean()
            self._save_artifact(batch_idx, sampled, scores_by_source, batch)
        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
