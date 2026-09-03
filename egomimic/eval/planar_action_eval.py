"""Teacher-forced Planar MSE and Energy Score without video/report machinery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
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
        unite_diagnostics: Mapping | None = None,
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
        self.unite_diagnostics = (
            self._metadata_copy(
                unite_diagnostics,
                label="unite_diagnostics",
            )
            or {}
        )
        self.unite_diagnostics_enabled = bool(
            self.unite_diagnostics.get("enabled", False)
        )
        self._unite_diagnostics_batches_done = 0
        self._unite_noise_seed_bank = []
        self._unite_noise_seed_bank_sha256 = None
        self._unite_raw_noise_levels = []
        self._unite_noise_labels = []
        self._unite_knn_k = 0
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
        if self.unite_diagnostics_enabled:
            self._configure_unite_diagnostics()
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
        self._unite_diagnostics_batches_done = 0
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
                predictions[source_id].append(result[source_id]["pred_action"].detach())
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

    @staticmethod
    def _noise_label(raw_level):
        return f"t{int(round(float(raw_level) * 1000.0)):03d}"

    def _configure_unite_diagnostics(self):
        path = Path(
            str(self.unite_diagnostics.get("noise_seed_bank_path", ""))
        ).resolve()
        expected_sha = str(self.unite_diagnostics.get("noise_seed_bank_sha256", ""))
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if digest != expected_sha:
            raise ValueError(f"UNITE diagnostic noise seed-bank mismatch: {path}")
        payload = json.loads(path.read_text())
        seeds = payload.get("seeds") if isinstance(payload, Mapping) else None
        if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(
                "UNITE diagnostic noise seed bank must be non-empty and unique"
            )
        self._unite_noise_seed_bank = [int(seed) for seed in seeds]
        self._unite_noise_seed_bank_sha256 = digest

        levels = [
            float(value) for value in self.unite_diagnostics.get("raw_noise_levels", [])
        ]
        if (
            len(levels) < 2
            or any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in levels
            )
            or any(right <= left for left, right in zip(levels, levels[1:]))
        ):
            raise ValueError(
                "UNITE raw_noise_levels must be finite, strictly increasing values in [0, 1]"
            )
        labels = [self._noise_label(value) for value in levels]
        if len(set(labels)) != len(labels):
            raise ValueError(
                "UNITE raw_noise_levels collide after metric label rounding"
            )
        self._unite_raw_noise_levels = levels
        self._unite_noise_labels = labels

        self._unite_knn_k = int(self.unite_diagnostics.get("cknna_k", 0))
        validation_view = dict(self.unite_diagnostics.get("validation_view", {}))
        per_rank_batch_size = int(validation_view.get("per_rank_batch_size", 0))
        world_size = int(validation_view.get("world_size", 0))
        if self._unite_knn_k < 2 or per_rank_batch_size <= self._unite_knn_k:
            raise ValueError(
                "UNITE CKNNA requires 2 <= k < per-rank validation batch size"
            )
        if world_size <= 0 or len(self._unite_noise_seed_bank) < world_size:
            raise ValueError(
                "UNITE diagnostic seed bank must provide at least one seed per rank"
            )
        if int(self.unite_diagnostics.get("max_batches_per_rank", 0)) <= 0:
            raise ValueError("UNITE diagnostic max_batches_per_rank must be positive")
        if not str(self.unite_diagnostics.get("artifact_root", "")):
            raise ValueError("UNITE diagnostics require an artifact_root")

    @staticmethod
    def _centered_linear_cka(left, right):
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError(
                f"CKA expects paired 2D features, got {left.shape} and {right.shape}"
            )
        left = left.float() - left.float().mean(dim=0, keepdim=True)
        right = right.float() - right.float().mean(dim=0, keepdim=True)
        numerator = (left.T @ right).square().sum()
        denominator = (
            (left.T @ left).square().sum() * (right.T @ right).square().sum()
        ).sqrt()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            raise ValueError("CKA encountered a zero or non-finite centered norm")
        value = numerator / denominator
        if not bool(torch.isfinite(value)):
            raise ValueError("CKA produced a non-finite value")
        return value

    @staticmethod
    def _unbiased_hsic(left, right):
        if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 4:
            raise ValueError("Unbiased HSIC requires paired square kernels with N >= 4")
        sample_count = int(left.shape[0])
        left = left.clone().fill_diagonal_(0.0)
        right = right.clone().fill_diagonal_(0.0)
        return (
            (left * right.T).sum()
            + left.sum() * right.sum() / ((sample_count - 1) * (sample_count - 2))
            - 2.0 * (left @ right).sum() / (sample_count - 2)
        ) / (sample_count * (sample_count - 3))

    @classmethod
    def _cknna(cls, left, right, *, topk):
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError(
                f"CKNNA expects paired 2D features, got {left.shape} and {right.shape}"
            )
        sample_count = int(left.shape[0])
        if not 2 <= int(topk) < sample_count:
            raise ValueError(f"CKNNA requires 2 <= topk < {sample_count}")

        def normalize(features):
            features = features.float()
            norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
            if not bool(torch.isfinite(norms).all()) or bool((norms <= 0.0).any()):
                raise ValueError("CKNNA encountered a zero or non-finite feature norm")
            return features / norms

        def local_hsic(first, second):
            first_search = first.clone().fill_diagonal_(float("-inf"))
            second_search = second.clone().fill_diagonal_(float("-inf"))
            first_indices = torch.topk(first_search, topk, dim=1).indices
            second_indices = torch.topk(second_search, topk, dim=1).indices
            first_mask = torch.zeros_like(first).scatter_(1, first_indices, 1.0)
            second_mask = torch.zeros_like(second).scatter_(1, second_indices, 1.0)
            intersection = first_mask * second_mask
            return cls._unbiased_hsic(intersection * first, intersection * second)

        left_kernel = normalize(left) @ normalize(left).T
        right_kernel = normalize(right) @ normalize(right).T
        cross = local_hsic(left_kernel, right_kernel)
        product = local_hsic(left_kernel, left_kernel) * local_hsic(
            right_kernel, right_kernel
        )
        if not bool(torch.isfinite(product)) or float(product) <= 0.0:
            raise ValueError("CKNNA encountered a zero or non-finite self-alignment")
        value = cross / product.sqrt()
        if not bool(torch.isfinite(value)):
            raise ValueError("CKNNA produced a non-finite value")
        return value

    @staticmethod
    def _cpu_tree(value):
        if torch.is_tensor(value):
            return value.detach().float().cpu()
        if isinstance(value, Mapping):
            return {
                str(key): PlanarActionEval._cpu_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PlanarActionEval._cpu_tree(item) for item in value]
        return value

    def _unite_metrics_and_artifact(self, batch, batch_idx):
        rank = int(self.trainer.global_rank)
        if rank >= len(self._unite_noise_seed_bank):
            raise ValueError(f"No UNITE diagnostic seed configured for rank {rank}")
        noise_seed = self._unite_noise_seed_bank[rank]
        with torch.random.fork_rng(devices=self._cuda_devices(batch)):
            torch.manual_seed(noise_seed)
            diagnostics = self.model.forward_unite_diagnostics(
                batch, raw_noise_levels=self._unite_raw_noise_levels
            )

        metrics = {}
        macro_values = {}
        artifact_domains = {}

        def add_metric(base, domain, value):
            value = value.detach().float()
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError(f"Invalid UNITE diagnostic metric {base}/{domain}")
            metrics[f"{base}/{domain}"] = value
            macro_values.setdefault(base, []).append(value)

        for source_id, diagnostic in diagnostics.items():
            embodiment_id, label = self._embodiment(batch[source_id])
            normalized_target = batch[source_id][self.action_key]
            native_target = self._native(normalized_target, embodiment_id)
            clean_latent = diagnostic["clean_latent"]
            sampler_latents = diagnostic["sampler_latents"]
            decoded_normalized = diagnostic["decoded_actions_normalized"]
            if sampler_latents.shape[1:] != clean_latent.shape:
                raise ValueError(
                    "UNITE sampler trajectory and clean latent shapes differ"
                )
            if decoded_normalized.shape[1:] != normalized_target.shape:
                raise ValueError(
                    "UNITE decoded trajectory and action target shapes differ"
                )
            latent_mse = (
                (sampler_latents.float() - clean_latent.float().unsqueeze(0))
                .square()
                .mean(dim=(-2, -1))
            )
            decoded_mse = (
                (decoded_normalized.float() - normalized_target.float().unsqueeze(0))
                .square()
                .mean(dim=(-2, -1))
            )
            decoded_native = torch.stack(
                [self._native(state, embodiment_id) for state in decoded_normalized],
                dim=0,
            )
            native_mse = (
                (decoded_native.float() - native_target.float().unsqueeze(0))
                .square()
                .mean(dim=(-2, -1))
            )
            for step_index in range(int(sampler_latents.shape[0])):
                add_metric(
                    f"Valid/DenoisingTrajectory/LatentMSE/step_{step_index}",
                    label,
                    latent_mse[step_index].mean(),
                )
                add_metric(
                    f"Valid/DenoisingTrajectory/DecodedMSE/step_{step_index}",
                    label,
                    decoded_mse[step_index].mean(),
                )
                add_metric(
                    f"Valid/DenoisingTrajectory/DecodedNativeMSE/step_{step_index}",
                    label,
                    native_mse[step_index].mean(),
                )

            final_predictions = diagnostic["noise_level_final_predictions"].float()
            clean_flat = clean_latent.float().reshape(clean_latent.shape[0], -1)
            predicted_flat = final_predictions.reshape(
                final_predictions.shape[0], final_predictions.shape[1], -1
            )
            clean_norm = torch.linalg.vector_norm(clean_flat, dim=1)
            predicted_norm = torch.linalg.vector_norm(predicted_flat, dim=2)
            if bool((clean_norm <= 0).any()) or bool((predicted_norm <= 0).any()):
                raise ValueError("Final-latent cosine encountered an invalid norm")
            cosine = (predicted_flat * clean_flat.unsqueeze(0)).sum(dim=2) / (
                predicted_norm * clean_norm.unsqueeze(0)
            )
            alignment_values = {}
            for noise_index, noise_label in enumerate(self._unite_noise_labels):
                add_metric(
                    f"Valid/Alignment/FinalLatentCosine/{noise_label}",
                    label,
                    cosine[noise_index].mean(),
                )
                add_metric(
                    f"Valid/Alignment/FinalLatentCosineStd/{noise_label}",
                    label,
                    cosine[noise_index].std(unbiased=False),
                )
                for layer_name, tokenization_activation in diagnostic[
                    "tokenization_activations"
                ].items():
                    denoising_activation = diagnostic["denoising_activations"][
                        layer_name
                    ][noise_index]
                    left = tokenization_activation.float().mean(dim=1)
                    right = denoising_activation.float().mean(dim=1)
                    cka = self._centered_linear_cka(left, right)
                    cknna = self._cknna(left, right, topk=self._unite_knn_k)
                    add_metric(
                        f"Valid/Alignment/CKA/{layer_name}/{noise_label}", label, cka
                    )
                    add_metric(
                        f"Valid/Alignment/CKNNA/{layer_name}/{noise_label}",
                        label,
                        cknna,
                    )
                    alignment_values.setdefault(layer_name, {})[noise_label] = {
                        "cka": cka,
                        "cknna": cknna,
                    }

            artifact_domains[label] = self._cpu_tree(
                {
                    **diagnostic,
                    "source_id": source_id,
                    "embodiment_id": embodiment_id,
                    "target_normalized": normalized_target,
                    "target_native": native_target,
                    "decoded_actions_native": decoded_native,
                    "trajectory_latent_mse_by_condition": latent_mse,
                    "trajectory_decoded_mse_by_condition": decoded_mse,
                    "trajectory_decoded_native_mse_by_condition": native_mse,
                    "final_latent_cosine_by_condition": cosine,
                    "alignment": alignment_values,
                }
            )

        for base, values in macro_values.items():
            metrics[base] = torch.stack(values).mean()
        root = Path(str(self.unite_diagnostics["artifact_root"])).resolve()
        destination = (
            root
            / f"epoch-{int(self.trainer.current_epoch)}-step-{int(self.trainer.global_step)}"
            / f"rank-{rank}-batch-{int(batch_idx)}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        torch.save(
            {
                "schema_version": 1,
                "metric": "UNITETrainingDiagnostics",
                "global_step": int(self.trainer.global_step),
                "epoch": int(self.trainer.current_epoch),
                "rank": rank,
                "batch_idx": int(batch_idx),
                "noise_seed": noise_seed,
                "noise_seed_bank_sha256": self._unite_noise_seed_bank_sha256,
                "raw_noise_levels": self._unite_raw_noise_levels,
                "noise_labels": self._unite_noise_labels,
                "sampler_states": int(
                    next(iter(diagnostics.values()))["sampler_latents"].shape[0]
                ),
                "validation_view": self.unite_diagnostics.get("validation_view"),
                "provenance": self.unite_diagnostics.get("provenance"),
                "domains": artifact_domains,
            },
            temporary,
        )
        os.replace(temporary, destination)
        return metrics

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

        if (
            self.unite_diagnostics_enabled
            and self._unite_diagnostics_batches_done
            < int(self.unite_diagnostics["max_batches_per_rank"])
        ):
            metrics.update(
                self._unite_metrics_and_artifact(
                    batch, self._unite_diagnostics_batches_done
                )
            )
            self._unite_diagnostics_batches_done += 1

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
