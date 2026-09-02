"""Teacher-forced Fold action metrics and GT/prediction overlay videos."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from egomimic.eval.eval_video import EvalVideo
from egomimic.pipeline import packed
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment


class HumanRobotOverlayEval(EvalVideo):
    """Evaluate denoised actions in one shared, unnormalized action space.

    Predictions and targets are both converted from the training normalizer
    before metrics or rendering. This makes the reported MSE directly
    comparable across Pipeline and diffusion-policy baselines.
    """

    def __init__(
        self,
        chunk_len: int | None = None,
        image_key: str = "front_img_1",
        frame_stride: int = 10,
        max_frames: int | None = 120,
        max_frames_by_embodiment: dict[str, int] | None = None,
        log_normalized_mse: bool | None = None,
        log_native_mse: bool | None = None,
        deterministic_seed: int | None = None,
        exact_epoch_metrics: bool = False,
        energy_score: dict[str, Any] | None = None,
        unite_diagnostics: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_len = None if chunk_len is None else int(chunk_len)
        self.image_key = str(image_key)
        self.frame_stride = int(frame_stride)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.max_frames_by_embodiment = {
            str(name).lower(): int(limit)
            for name, limit in (max_frames_by_embodiment or {}).items()
        }
        # ``None`` preserves the UNITE evaluator contract: log normalized and
        # unnormalized model-action aliases and macro-average domains. Paper
        # DP opts in explicitly and requires adapter-decoded native actions and
        # element-weighted batch aggregates.
        self.log_normalized_mse = (
            True if log_normalized_mse is None else bool(log_normalized_mse)
        )
        self.log_native_mse = True if log_native_mse is None else bool(log_native_mse)
        self._normalized_mse_is_explicit = log_normalized_mse is not None
        self._native_mse_is_explicit = log_native_mse is not None
        self._decode_native_mse = log_native_mse is True
        self.deterministic_seed = (
            None if deterministic_seed is None else int(deterministic_seed)
        )
        self.exact_epoch_metrics = bool(exact_epoch_metrics)
        self.energy_score = dict(energy_score or {})
        self.energy_score_enabled = bool(self.energy_score.get("enabled", False))
        self._energy_seed_bank: list[int] = []
        self._energy_seed_bank_sha256: str | None = None
        self._energy_batches_done = 0
        self._energy_schema: str | None = None
        self._energy_action_dims: dict[str, int] = {}
        self._energy_distance_blocks: dict[str, dict[str, dict[str, Any]]] = {}
        self._energy_shared_action_dim: int | None = None
        self._energy_shared_distance_blocks: dict[str, dict[str, Any]] | None = None
        if self.energy_score_enabled:
            self._configure_energy_score()
        self.unite_diagnostics = dict(unite_diagnostics or {})
        self.unite_diagnostics_enabled = bool(
            self.unite_diagnostics.get("enabled", False)
        )
        self._unite_noise_seed_bank: list[int] = []
        self._unite_noise_seed_bank_sha256: str | None = None
        self._unite_raw_noise_levels: list[float] = []
        self._unite_noise_labels: list[str] = []
        self._unite_knn_k = 0
        self._unite_diagnostics_batches_done = 0
        if self.unite_diagnostics_enabled:
            self._configure_unite_diagnostics()
        self._rendered_frames: dict[int, int] = {}
        self._exact_sums: dict[int, dict[str, torch.Tensor | int]] = {}

    def on_validation_start(self):
        self._rendered_frames.clear()
        self._exact_sums.clear()
        self._energy_batches_done = 0
        self._unite_diagnostics_batches_done = 0
        super().on_validation_start()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_distance_blocks(
        embodiment_name: str,
        action_dim: int,
        blocks: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        if action_dim <= 0:
            raise ValueError(
                f"Energy Score action dimension for {embodiment_name} must be positive"
            )
        if not blocks:
            raise ValueError(
                f"Energy Score requires semantic blocks for {embodiment_name}"
            )
        validated = {}
        covered: set[int] = set()
        for name, raw in blocks.items():
            indices = [int(index) for index in raw.get("indices", [])]
            weight = float(raw.get("weight", 0.0))
            if (
                not indices
                or len(indices) != len(set(indices))
                or weight <= 0.0
                or covered.intersection(indices)
            ):
                raise ValueError(
                    f"Invalid Energy Score block {embodiment_name}/{name}: {raw}"
                )
            covered.update(indices)
            validated[str(name)] = {"indices": indices, "weight": weight}
        if covered != set(range(action_dim)):
            raise ValueError(
                f"Energy Score blocks for {embodiment_name} must partition "
                f"all {action_dim} action channels"
            )
        return validated

    def _configure_energy_score(self) -> None:
        sample_count = int(self.energy_score.get("sample_count", 0))
        if sample_count != 32:
            raise ValueError("Energy Score requires sample_count=32")
        path = Path(str(self.energy_score.get("seed_bank_path", ""))).resolve()
        expected_sha = str(self.energy_score.get("seed_bank_sha256", ""))
        if not path.is_file() or self._sha256(path) != expected_sha:
            raise ValueError(f"Energy Score seed bank identity mismatch: {path}")
        payload = json.loads(path.read_text())
        seeds = payload.get("seeds") if isinstance(payload, dict) else None
        if not isinstance(seeds, list) or len(seeds) != sample_count:
            raise ValueError("Energy Score seed bank must contain exactly 32 seeds")
        self._energy_seed_bank = [int(seed) for seed in seeds]
        if len(set(self._energy_seed_bank)) != sample_count:
            raise ValueError("Energy Score seeds must be unique")
        self._energy_seed_bank_sha256 = expected_sha

        raw_action_dims = self.energy_score.get("action_dims")
        raw_action_dim = self.energy_score.get("action_dim")
        configured_blocks = dict(self.energy_score.get("distance_blocks", {}))
        if raw_action_dims is not None and raw_action_dim is not None:
            raise ValueError(
                "Energy Score must use either action_dim or action_dims, not both"
            )
        if raw_action_dims is not None:
            action_dims = {
                str(name).lower(): int(width)
                for name, width in dict(raw_action_dims).items()
            }
            blocks_by_embodiment = {
                str(name).lower(): dict(blocks)
                for name, blocks in configured_blocks.items()
            }
            if not action_dims or set(action_dims) != set(blocks_by_embodiment):
                raise ValueError(
                    "Energy Score action_dims and distance_blocks must name the "
                    "same embodiments"
                )
            self._energy_schema = "per_domain"
            self._energy_action_dims = action_dims
            self._energy_distance_blocks = {
                name: self._validate_distance_blocks(
                    name, action_dims[name], blocks_by_embodiment[name]
                )
                for name in action_dims
            }
        elif raw_action_dim is not None:
            action_dim = int(raw_action_dim)
            self._energy_schema = "shared"
            self._energy_shared_action_dim = action_dim
            self._energy_shared_distance_blocks = self._validate_distance_blocks(
                "shared", action_dim, configured_blocks
            )
        else:
            raise ValueError("Energy Score requires action_dim or action_dims")
        if int(self.energy_score.get("max_batches_per_rank", 0)) <= 0:
            raise ValueError("Energy Score max_batches_per_rank must be positive")
        if not str(self.energy_score.get("artifact_root", "")):
            raise ValueError("Energy Score requires an artifact_root")

    @staticmethod
    def _noise_label(raw_level: float) -> str:
        return f"t{int(round(float(raw_level) * 1000.0)):03d}"

    def _configure_unite_diagnostics(self) -> None:
        path = Path(
            str(self.unite_diagnostics.get("noise_seed_bank_path", ""))
        ).resolve()
        expected_sha = str(self.unite_diagnostics.get("noise_seed_bank_sha256", ""))
        if not path.is_file() or self._sha256(path) != expected_sha:
            raise ValueError(f"UNITE diagnostic noise seed bank mismatch: {path}")
        payload = json.loads(path.read_text())
        seeds = payload.get("seeds") if isinstance(payload, dict) else None
        if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(
                "UNITE diagnostic noise seed bank must be non-empty and unique"
            )
        self._unite_noise_seed_bank = [int(seed) for seed in seeds]
        self._unite_noise_seed_bank_sha256 = expected_sha

        levels = [
            float(value) for value in self.unite_diagnostics.get("raw_noise_levels", [])
        ]
        if (
            len(levels) < 2
            or any(
                not np.isfinite(value) or value < 0.0 or value > 1.0 for value in levels
            )
            or any(
                right <= left for left, right in zip(levels, levels[1:], strict=False)
            )
        ):
            raise ValueError(
                "UNITE raw_noise_levels must be finite, strictly increasing values in [0, 1]"
            )
        labels = [self._noise_label(value) for value in levels]
        if len(labels) != len(set(labels)):
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
    def _centered_linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError(
                f"CKA expects paired 2D features, got {left.shape} and {right.shape}"
            )
        left = left.float() - left.float().mean(dim=0, keepdim=True)
        right = right.float() - right.float().mean(dim=0, keepdim=True)
        cross = left.T @ right
        left_gram = left.T @ left
        right_gram = right.T @ right
        numerator = cross.square().sum()
        denominator = (left_gram.square().sum() * right_gram.square().sum()).sqrt()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            raise ValueError("CKA encountered a zero or non-finite centered norm")
        value = numerator / denominator
        if not bool(torch.isfinite(value)):
            raise ValueError("CKA produced a non-finite value")
        return value

    @staticmethod
    def _unbiased_hsic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 4:
            raise ValueError("Unbiased HSIC requires paired square kernels with N >= 4")
        sample_count = int(left.shape[0])
        left = left.clone().fill_diagonal_(0.0)
        right = right.clone().fill_diagonal_(0.0)
        value = (
            (left * right.T).sum()
            + left.sum() * right.sum() / ((sample_count - 1) * (sample_count - 2))
            - 2.0 * (left @ right).sum() / (sample_count - 2)
        ) / (sample_count * (sample_count - 3))
        return value

    @classmethod
    def _cknna(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        topk: int,
    ) -> torch.Tensor:
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError(
                f"CKNNA expects paired 2D features, got {left.shape} and {right.shape}"
            )
        sample_count = int(left.shape[0])
        if not 2 <= int(topk) < sample_count:
            raise ValueError(f"CKNNA requires 2 <= topk < {sample_count}")

        def l2_normalize(features: torch.Tensor) -> torch.Tensor:
            features = features.float()
            norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
            if not bool(torch.isfinite(norms).all()) or bool((norms <= 0.0).any()):
                raise ValueError("CKNNA encountered a zero or non-finite feature norm")
            return features / norms

        left = l2_normalize(left)
        right = l2_normalize(right)
        left_kernel = left @ left.T
        right_kernel = right @ right.T

        def local_hsic(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            first_search = first.clone().fill_diagonal_(float("-inf"))
            second_search = second.clone().fill_diagonal_(float("-inf"))
            first_indices = torch.topk(first_search, topk, dim=1).indices
            second_indices = torch.topk(second_search, topk, dim=1).indices
            first_mask = torch.zeros_like(first).scatter_(1, first_indices, 1.0)
            second_mask = torch.zeros_like(second).scatter_(1, second_indices, 1.0)
            intersection = first_mask * second_mask
            return cls._unbiased_hsic(intersection * first, intersection * second)

        cross = local_hsic(left_kernel, right_kernel)
        left_self = local_hsic(left_kernel, left_kernel)
        right_self = local_hsic(right_kernel, right_kernel)
        product = left_self * right_self
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
        if isinstance(value, dict):
            return {
                str(key): HumanRobotOverlayEval._cpu_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [HumanRobotOverlayEval._cpu_tree(item) for item in value]
        return value

    def _unite_metrics_and_artifact(
        self,
        batch: dict[int, dict[str, Any]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        rank = int(self.trainer.global_rank)
        if rank >= len(self._unite_noise_seed_bank):
            raise ValueError(f"No UNITE diagnostic seed configured for rank {rank}")
        noise_seed = self._unite_noise_seed_bank[rank]
        cuda_devices = sorted(
            {
                int(value.device.index)
                for embodiment_batch in batch.values()
                for value in embodiment_batch.values()
                if torch.is_tensor(value)
                and value.device.type == "cuda"
                and value.device.index is not None
            }
        )
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(noise_seed)
            diagnostics = self.model.forward_unite_diagnostics(
                batch,
                raw_noise_levels=self._unite_raw_noise_levels,
            )

        metrics: dict[str, torch.Tensor] = {}
        macro_values: dict[str, list[torch.Tensor]] = {}
        artifact_domains: dict[str, Any] = {}

        def add_metric(base: str, domain: str, value: torch.Tensor) -> None:
            value = value.detach().float()
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError(f"Invalid UNITE diagnostic metric {base}/{domain}")
            metrics[f"{base}/{domain}"] = value
            macro_values.setdefault(base, []).append(value)

        for emb_id, diagnostic in diagnostics.items():
            embodiment_name = get_embodiment(emb_id).lower()
            action_key = self.model.resolved_ac_keys[emb_id]
            normalized_target = self._normalized_target(batch[emb_id], action_key)
            _, native_target = self._unnormalized_target(
                batch[emb_id], emb_id, action_key
            )
            clean_latent = diagnostic["clean_latent"]
            sampler_latents = diagnostic["sampler_latents"]
            decoded_normalized = diagnostic["decoded_actions_normalized"]
            if sampler_latents.ndim != 4 or decoded_normalized.ndim != 4:
                raise ValueError(
                    "UNITE trajectory tensors must be (step, batch, token, dim)"
                )
            if sampler_latents.shape[0] != decoded_normalized.shape[0]:
                raise ValueError(
                    "UNITE latent and decoded trajectories disagree on steps"
                )
            if sampler_latents.shape[1:] != clean_latent.shape:
                raise ValueError(
                    "UNITE sampler trajectory and clean latent shapes differ"
                )
            if decoded_normalized.shape[1:] != normalized_target.shape:
                raise ValueError(
                    "UNITE decoded trajectory and action target shapes differ"
                )

            latent_mse_by_condition = (
                (sampler_latents.float() - clean_latent.float().unsqueeze(0))
                .square()
                .mean(dim=(-2, -1))
            )
            decoded_mse_by_condition = (
                (decoded_normalized.float() - normalized_target.float().unsqueeze(0))
                .square()
                .mean(dim=(-2, -1))
            )
            decoded_native = torch.stack(
                [
                    self._unnormalize_prediction(state, emb_id, action_key)
                    for state in decoded_normalized
                ],
                dim=0,
            )
            native_mse_by_condition = (
                (decoded_native.float() - native_target.float().unsqueeze(0))
                .square()
                .mean(dim=(-2, -1))
            )
            for step_index in range(int(sampler_latents.shape[0])):
                add_metric(
                    f"Valid/DenoisingTrajectory/LatentMSE/step_{step_index}",
                    embodiment_name,
                    latent_mse_by_condition[step_index].mean(),
                )
                add_metric(
                    f"Valid/DenoisingTrajectory/DecodedMSE/step_{step_index}",
                    embodiment_name,
                    decoded_mse_by_condition[step_index].mean(),
                )
                add_metric(
                    f"Valid/DenoisingTrajectory/DecodedNativeMSE/step_{step_index}",
                    embodiment_name,
                    native_mse_by_condition[step_index].mean(),
                )

            final_predictions = diagnostic["noise_level_final_predictions"].float()
            clean_flat = clean_latent.float().reshape(clean_latent.shape[0], -1)
            predicted_flat = final_predictions.reshape(
                final_predictions.shape[0], final_predictions.shape[1], -1
            )
            clean_norm = torch.linalg.vector_norm(clean_flat, dim=1)
            predicted_norm = torch.linalg.vector_norm(predicted_flat, dim=2)
            if (
                not bool(torch.isfinite(clean_norm).all())
                or not bool(torch.isfinite(predicted_norm).all())
                or bool((clean_norm <= 0.0).any())
                or bool((predicted_norm <= 0.0).any())
            ):
                raise ValueError("Final-latent cosine encountered an invalid norm")
            cosine_by_condition = (predicted_flat * clean_flat.unsqueeze(0)).sum(
                dim=2
            ) / (predicted_norm * clean_norm.unsqueeze(0))

            tokenization = diagnostic["tokenization_activations"]
            denoising = diagnostic["denoising_activations"]
            alignment_values: dict[str, Any] = {}
            for noise_index, noise_label in enumerate(self._unite_noise_labels):
                add_metric(
                    f"Valid/Alignment/FinalLatentCosine/{noise_label}",
                    embodiment_name,
                    cosine_by_condition[noise_index].mean(),
                )
                add_metric(
                    f"Valid/Alignment/FinalLatentCosineStd/{noise_label}",
                    embodiment_name,
                    cosine_by_condition[noise_index].std(unbiased=False),
                )
                for layer_name, tokenization_activation in tokenization.items():
                    if layer_name not in denoising:
                        raise ValueError(
                            f"Missing denoising activation for {layer_name}"
                        )
                    left = tokenization_activation.float().mean(dim=1)
                    right = denoising[layer_name][noise_index].float().mean(dim=1)
                    cka = self._centered_linear_cka(left, right)
                    cknna = self._cknna(left, right, topk=self._unite_knn_k)
                    add_metric(
                        f"Valid/Alignment/CKA/{layer_name}/{noise_label}",
                        embodiment_name,
                        cka,
                    )
                    add_metric(
                        f"Valid/Alignment/CKNNA/{layer_name}/{noise_label}",
                        embodiment_name,
                        cknna,
                    )
                    alignment_values.setdefault(layer_name, {})[noise_label] = {
                        "cka": cka,
                        "cknna": cknna,
                    }

            artifact_domains[embodiment_name] = self._cpu_tree(
                {
                    **diagnostic,
                    "embodiment_id": int(emb_id),
                    "action_key": action_key,
                    "target_normalized": normalized_target,
                    "target_native": native_target,
                    "decoded_actions_native": decoded_native,
                    "trajectory_latent_mse_by_condition": latent_mse_by_condition,
                    "trajectory_decoded_mse_by_condition": decoded_mse_by_condition,
                    "trajectory_decoded_native_mse_by_condition": native_mse_by_condition,
                    "final_latent_cosine_by_condition": cosine_by_condition,
                    "alignment": alignment_values,
                }
            )

        for base, values in macro_values.items():
            metrics[base] = torch.stack(values).mean()

        root = Path(str(self.unite_diagnostics["artifact_root"])).resolve()
        step = int(self.trainer.global_step)
        epoch = int(self.trainer.current_epoch)
        destination = (
            root
            / f"epoch-{epoch}-step-{step}"
            / f"rank-{rank}-batch-{int(batch_idx)}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        payload = {
            "schema_version": 1,
            "metric": "UNITETrainingDiagnostics",
            "global_step": step,
            "epoch": epoch,
            "rank": rank,
            "batch_idx": int(batch_idx),
            "noise_seed": noise_seed,
            "noise_seed_bank_sha256": self._unite_noise_seed_bank_sha256,
            "raw_noise_levels": self._unite_raw_noise_levels,
            "noise_labels": self._unite_noise_labels,
            "sampler_steps": int(
                next(iter(diagnostics.values()))["sampler_latents"].shape[0] - 1
            ),
            "alignment": {
                "pooling": "mean_over_tokens_then_one_row_per_validation_condition",
                "cka": "biased_centered_linear_cka",
                "cknna": "platonic_rep_unbiased_hsic_intersection_knn",
                "cknna_k": self._unite_knn_k,
                "cknna_feature_normalization": "per_sample_l2",
                "cosine": "flatten_all_tokens_and_latent_channels_per_condition",
            },
            "aggregation": "condition_mean_then_equal_domain_macro_mean",
            "precision": str(self.trainer.precision),
            "validation_view": self.unite_diagnostics.get("validation_view"),
            "provenance": self.unite_diagnostics.get("provenance"),
            "domains": artifact_domains,
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return metrics

    def _energy_distance(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        embodiment_name: str,
    ) -> torch.Tensor:
        """Return a block-balanced normalized chunk distance.

        The tensors may carry any shared leading axes. Their last two axes are
        horizon and action channel; every semantic block contributes one RMS
        term regardless of its coordinate count.
        """

        if left.shape != right.shape:
            left, right = torch.broadcast_tensors(left, right)
        action_width = int(left.shape[-1])
        if self._energy_schema == "per_domain":
            if embodiment_name not in self._energy_distance_blocks:
                raise ValueError(
                    f"No Energy Score distance contract for {embodiment_name}"
                )
            expected_width = self._energy_action_dims[embodiment_name]
            blocks = self._energy_distance_blocks[embodiment_name]
        elif self._energy_schema == "shared":
            expected_width = int(self._energy_shared_action_dim)
            blocks = self._energy_shared_distance_blocks
        else:
            raise RuntimeError("Energy Score distance contract is not configured")
        if action_width != expected_width:
            raise ValueError(
                f"Energy Score action width for {embodiment_name} is {action_width}, "
                f"expected {expected_width}"
            )
        terms = []
        weights = []
        for raw in blocks.values():
            indices = raw["indices"]
            weight = raw["weight"]
            delta = left[..., indices] - right[..., indices]
            terms.append(delta.float().square().mean(dim=(-2, -1)).sqrt() * weight)
            weights.append(weight)
        return torch.stack(terms).sum(dim=0) / sum(weights)

    def _sample_energy_predictions(
        self, batch: dict[int, dict[str, Any]]
    ) -> dict[int, torch.Tensor]:
        samples: dict[int, list[torch.Tensor]] = {emb_id: [] for emb_id in batch}
        cuda_devices = sorted(
            {
                int(value.device.index)
                for embodiment_batch in batch.values()
                for value in embodiment_batch.values()
                if torch.is_tensor(value)
                and value.device.type == "cuda"
                and value.device.index is not None
            }
        )
        for seed in self._energy_seed_bank:
            # Fixed validation sampling must not perturb subsequent training RNG.
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                torch.manual_seed(seed)
                prediction = self.model.forward_eval(batch)
            for emb_id in batch:
                action_key = self.model.resolved_ac_keys[emb_id]
                samples[emb_id].append(prediction[f"emb{emb_id}_{action_key}"].detach())
        return {
            emb_id: torch.stack(per_seed, dim=0) for emb_id, per_seed in samples.items()
        }

    def _energy_metrics_and_artifact(
        self,
        batch: dict[int, dict[str, Any]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        sampled = self._sample_energy_predictions(batch)
        metrics: dict[str, torch.Tensor] = {}
        artifact_domains: dict[str, Any] = {}
        domain_scores = []
        domain_accuracies = []
        domain_diversities = []
        for emb_id, predictions in sampled.items():
            embodiment_name = get_embodiment(emb_id).lower()
            action_key = self.model.resolved_ac_keys[emb_id]
            target = self._normalized_target(batch[emb_id], action_key)
            if predictions.shape[1:] != target.shape:
                raise ValueError(
                    f"Energy Score prediction/target mismatch for "
                    f"{embodiment_name}: {tuple(predictions.shape)} vs "
                    f"{tuple(target.shape)}"
                )
            accuracy_by_condition = self._energy_distance(
                predictions,
                target.unsqueeze(0).expand_as(predictions),
                embodiment_name,
            ).mean(dim=0)
            pairwise = self._energy_distance(
                predictions[:, None], predictions[None, :], embodiment_name
            )
            sample_count = int(predictions.shape[0])
            off_diagonal = ~torch.eye(
                sample_count, device=pairwise.device, dtype=torch.bool
            )
            diversity_by_condition = (
                pairwise[off_diagonal]
                .reshape(sample_count * (sample_count - 1), predictions.shape[1])
                .mean(dim=0)
            )
            score_by_condition = accuracy_by_condition - 0.5 * diversity_by_condition
            accuracy = accuracy_by_condition.mean().detach()
            diversity = diversity_by_condition.mean().detach()
            score = score_by_condition.mean().detach()
            if not bool(
                torch.isfinite(torch.stack((score, accuracy, diversity))).all()
            ):
                raise ValueError(f"Non-finite Energy Score for {embodiment_name}")
            metrics[f"Valid/EnergyScore@32/{embodiment_name}"] = score
            metrics[f"Valid/EnergyScoreAccuracy@32/{embodiment_name}"] = accuracy
            metrics[f"Valid/EnergyScoreDiversity@32/{embodiment_name}"] = diversity
            domain_scores.append(score)
            domain_accuracies.append(accuracy)
            domain_diversities.append(diversity)
            artifact_domains[embodiment_name] = {
                "embodiment_id": int(emb_id),
                "action_key": action_key,
                "predictions": predictions.float().cpu(),
                "targets": target.detach().float().cpu(),
                "accuracy_by_condition": accuracy_by_condition.float().cpu(),
                "diversity_by_condition": diversity_by_condition.float().cpu(),
                "score_by_condition": score_by_condition.float().cpu(),
            }

        metrics["Valid/EnergyScore@32"] = torch.stack(domain_scores).mean()
        metrics["Valid/EnergyScoreAccuracy@32"] = torch.stack(domain_accuracies).mean()
        metrics["Valid/EnergyScoreDiversity@32"] = torch.stack(
            domain_diversities
        ).mean()

        root = Path(str(self.energy_score["artifact_root"])).resolve()
        step = int(self.trainer.global_step)
        epoch = int(self.trainer.current_epoch)
        rank = int(self.trainer.global_rank)
        destination = (
            root
            / f"epoch-{epoch}-step-{step}"
            / f"rank-{rank}-batch-{int(batch_idx)}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        distance_contract = {
            "space": "normalized_action_chunk",
            "formula": "mean_weighted_semantic_block_rms",
        }
        if self._energy_schema == "per_domain":
            distance_contract["blocks_by_embodiment"] = self._energy_distance_blocks
        else:
            distance_contract["blocks"] = self._energy_shared_distance_blocks
        payload = {
            "schema_version": 1,
            "metric": "EnergyScore@32",
            "sample_count": 32,
            "seed_bank": self._energy_seed_bank,
            "seed_bank_sha256": self._energy_seed_bank_sha256,
            "distance": distance_contract,
            "aggregation": "condition_mean_then_equal_domain_macro_mean",
            "global_step": step,
            "epoch": epoch,
            "rank": rank,
            "batch_idx": int(batch_idx),
            "precision": str(self.trainer.precision),
            "validation_view": self.energy_score.get("validation_view"),
            "provenance": self.energy_score.get("provenance"),
            "domains": artifact_domains,
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return metrics

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.deterministic_seed is None:
            return super().on_validation_step(batch, batch_idx, dataloader_idx)

        device = self.trainer.lightning_module.device
        devices = []
        if device.type == "cuda":
            devices = [
                (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        rank = int(getattr(self.trainer, "global_rank", 0))
        seed = self.deterministic_seed + int(batch_idx) + rank * 1_000_003
        # Validation must neither inherit moving training RNG state nor change
        # the next training sample/noise. The same batch on the same rank gets
        # the same latent noise at every checkpoint and every validation pass.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            return super().on_validation_step(batch, batch_idx, dataloader_idx)

    def _accumulate_exact(
        self,
        emb_id: int,
        normalized_squared_error: torch.Tensor | None,
        native_squared_error: torch.Tensor | None,
    ) -> None:
        if normalized_squared_error is None and native_squared_error is None:
            return
        reference = (
            normalized_squared_error
            if normalized_squared_error is not None
            else native_squared_error
        )
        item = self._exact_sums.setdefault(
            int(emb_id),
            {
                "normalized_sum": torch.zeros(
                    (), device=reference.device, dtype=torch.float64
                ),
                "normalized_count": 0,
                "native_sum": torch.zeros(
                    (), device=reference.device, dtype=torch.float64
                ),
                "native_count": 0,
            },
        )
        if normalized_squared_error is not None:
            item["normalized_sum"] += normalized_squared_error.double().sum().detach()
            item["normalized_count"] += normalized_squared_error.numel()
        if native_squared_error is not None:
            item["native_sum"] += native_squared_error.double().sum().detach()
            item["native_count"] += native_squared_error.numel()

    def on_validation_end(self):
        if self.exact_epoch_metrics:
            metrics = {}
            normalized_mses = []
            native_mses = []
            device = self.trainer.lightning_module.device
            distributed = dist.is_available() and dist.is_initialized()
            embodiment_ids = sorted(self._exact_sums)
            if distributed:
                gathered_ids: list[list[int] | None] = [
                    None for _ in range(dist.get_world_size())
                ]
                dist.all_gather_object(gathered_ids, embodiment_ids)
                embodiment_ids = sorted(
                    {
                        int(emb_id)
                        for rank_ids in gathered_ids
                        for emb_id in (rank_ids or [])
                    }
                )

            for emb_id in embodiment_ids:
                item = self._exact_sums.get(emb_id)
                totals = torch.zeros(4, dtype=torch.float64, device=device)
                if item is not None:
                    totals[0] = item["normalized_sum"].item()
                    totals[1] = item["normalized_count"]
                    totals[2] = item["native_sum"].item()
                    totals[3] = item["native_count"]
                if distributed:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                embodiment_name = get_embodiment(emb_id).lower()
                if totals[1] > 0:
                    normalized_mse = totals[0] / totals[1]
                    metrics[f"Valid/MSE/{embodiment_name}"] = normalized_mse.float()
                    metrics[f"Valid/Element_Count/{embodiment_name}"] = totals[1]
                    normalized_mses.append(normalized_mse)
                if totals[3] > 0:
                    native_mse = totals[2] / totals[3]
                    metrics[f"Valid/Native_MSE/{embodiment_name}"] = native_mse.float()
                    native_mses.append(native_mse)
            if normalized_mses:
                metrics["Valid/MSE"] = torch.stack(normalized_mses).mean().float()
            if native_mses:
                metrics["Valid/Native_MSE"] = torch.stack(native_mses).mean().float()
            if metrics:
                self.trainer.lightning_module.log_dict(
                    metrics, on_step=False, on_epoch=True, sync_dist=False
                )
        super().on_validation_end()

    @staticmethod
    def _error_metrics(prefix: str, prediction, target) -> dict:
        squared_error = (prediction - target).square().reshape(-1)
        return {
            f"{prefix}_mse": squared_error.mean().detach(),
            f"{prefix}_squared_error_median": squared_error.median().detach(),
            f"{prefix}_squared_error_p95": torch.quantile(
                squared_error.float(), 0.95
            ).detach(),
            f"{prefix}_squared_error_max": squared_error.max().detach(),
        }

    def _unnormalized_target(self, batch, emb_id, action_key):
        unnormalized = self.model.norm_stats.unnormalize(batch, emb_id)
        target = unnormalized[action_key]
        if "cu_seqlens" not in batch:
            if target.ndim != 3:
                raise ValueError(
                    f"Expected standard actions (batch, horizon, dim), got "
                    f"{tuple(target.shape)}"
                )
            return unnormalized, target
        if self.chunk_len is None:
            raise ValueError("chunk_len is required for a packed validation loader")
        cu_seqlens = batch["cu_seqlens"].to(target.device)
        return unnormalized, packed.chunk_targets(target, cu_seqlens, self.chunk_len)

    def _normalized_target(self, batch, action_key):
        """Return the normalized target in the same chunk shape as prediction."""
        target = batch[action_key]
        if "cu_seqlens" not in batch:
            if target.ndim != 3:
                raise ValueError(
                    f"Expected standard actions (batch, horizon, dim), got "
                    f"{tuple(target.shape)}"
                )
            return target
        if self.chunk_len is None:
            raise ValueError("chunk_len is required for a packed validation loader")
        cu_seqlens = batch["cu_seqlens"].to(target.device)
        return packed.chunk_targets(target, cu_seqlens, self.chunk_len)

    def _unnormalize_prediction(self, prediction, emb_id, action_key):
        if prediction.ndim != 3:
            raise ValueError(
                "Expected prediction shaped (batch, horizon, action_dim), got "
                f"{tuple(prediction.shape)}"
            )
        # Keep the horizon axis intact. MultiDataset statistics may be either
        # per-dimension (D,) or slotwise (H, D); both broadcast correctly into
        # (B, H, D), while flattening B and H breaks slotwise arc-token stats.
        return self.model.norm_stats.unnormalize({action_key: prediction}, emb_id)[
            action_key
        ]

    @staticmethod
    def _finite_mse(prediction, target, label: str):
        if prediction.shape != target.shape:
            raise ValueError(
                f"{label} prediction/target mismatch: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if prediction.numel() == 0:
            raise ValueError(f"{label} received an empty prediction/target")
        squared_error = (prediction - target).float().square().reshape(-1)
        if not bool(torch.isfinite(squared_error).all()):
            raise ValueError(f"{label} contains non-finite squared error")
        return (
            squared_error.mean().detach(),
            squared_error.sum().detach(),
            squared_error.numel(),
        )

    def compute_metrics_and_viz(
        self, batch: dict[int, dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], dict[int, np.ndarray]]:
        predictions = self.model.forward_eval(batch)
        metrics = {}
        images = {}
        normalized_domain_mses = []
        native_domain_mses = []
        normalized_squared_error_sum = None
        normalized_element_count = 0
        native_squared_error_sum = None
        native_element_count = 0

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

        if self.energy_score_enabled and self._energy_batches_done < int(
            self.energy_score["max_batches_per_rank"]
        ):
            metrics.update(
                self._energy_metrics_and_artifact(batch, self._energy_batches_done)
            )
            self._energy_batches_done += 1

        for emb_id, embodiment_batch in batch.items():
            embodiment_name = get_embodiment(emb_id).lower()
            action_key = self.model.resolved_ac_keys[emb_id]
            prediction_key = f"emb{emb_id}_{action_key}"
            if prediction_key not in predictions:
                raise KeyError(
                    f"Missing {prediction_key!r}; available={sorted(predictions)}"
                )

            normalized_prediction = predictions[prediction_key]
            normalized_target = self._normalized_target(embodiment_batch, action_key)
            normalized_count = min(
                normalized_target.shape[0], normalized_prediction.shape[0]
            )
            normalized_target = normalized_target[:normalized_count]
            normalized_prediction = normalized_prediction[:normalized_count]
            unnormalized, target = self._unnormalized_target(
                embodiment_batch, emb_id, action_key
            )
            prediction = self._unnormalize_prediction(
                normalized_prediction, emb_id, action_key
            )
            count = min(target.shape[0], prediction.shape[0])
            normalized_target = normalized_target[:count]
            normalized_prediction = normalized_prediction[:count]
            if normalized_target.shape != normalized_prediction.shape:
                raise ValueError(
                    f"Normalized prediction/target mismatch: "
                    f"{tuple(normalized_prediction.shape)} vs "
                    f"{tuple(normalized_target.shape)}"
                )
            target = target[:count]
            prediction = prediction[:count]
            if target.shape != prediction.shape:
                raise ValueError(
                    f"Prediction/target mismatch after unnormalizing: "
                    f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
                )

            metric_prefix = f"Valid/emb{emb_id}_{action_key}_action"
            metrics.update(self._error_metrics(metric_prefix, prediction, target))
            metrics[f"Valid/emb{emb_id}_{action_key}_copybaseline_mse"] = (
                (target[:, :1] - target).square().mean().detach()
            )

            normalized_squared_error = None
            if self.log_normalized_mse:
                normalized_mse, squared_error_sum, element_count = self._finite_mse(
                    normalized_prediction,
                    normalized_target,
                    f"normalized {embodiment_name}",
                )
                normalized_squared_error = (
                    (normalized_prediction - normalized_target).float().square()
                )
                if not self.exact_epoch_metrics:
                    metrics[f"Valid/MSE/{embodiment_name}"] = normalized_mse
                    if self._normalized_mse_is_explicit:
                        normalized_squared_error_sum = (
                            squared_error_sum
                            if normalized_squared_error_sum is None
                            else normalized_squared_error_sum + squared_error_sum
                        )
                        normalized_element_count += element_count
                    else:
                        normalized_domain_mses.append(normalized_mse)

            native_squared_error = None
            if self.log_native_mse:
                if self._decode_native_mse:
                    adapter = self.model.rollout_adapter_for(emb_id)
                    if adapter is None:
                        raise ValueError(
                            f"Native MSE requested but {embodiment_name} has no "
                            "rollout adapter"
                        )
                    native_prediction = adapter.decode(prediction, unnormalized)
                    native_target = adapter.decode(target, unnormalized)
                    if not torch.is_tensor(native_prediction):
                        native_prediction = torch.as_tensor(
                            native_prediction, device=prediction.device
                        )
                    if not torch.is_tensor(native_target):
                        native_target = torch.as_tensor(
                            native_target, device=target.device
                        )
                else:
                    # UNITE action heads already emit the action representation
                    # used by its historical Native_MSE validation contract.
                    native_prediction = prediction
                    native_target = target
                native_mse, squared_error_sum, element_count = self._finite_mse(
                    native_prediction,
                    native_target,
                    f"native {embodiment_name}",
                )
                native_squared_error = (
                    (native_prediction - native_target).float().square()
                )
                if not self.exact_epoch_metrics:
                    metrics[f"Valid/Native_MSE/{embodiment_name}"] = native_mse
                    if self._native_mse_is_explicit:
                        native_squared_error_sum = (
                            squared_error_sum
                            if native_squared_error_sum is None
                            else native_squared_error_sum + squared_error_sum
                        )
                        native_element_count += element_count
                    else:
                        native_domain_mses.append(native_mse)

            if self.exact_epoch_metrics:
                self._accumulate_exact(
                    emb_id, normalized_squared_error, native_squared_error
                )

            raw_images = embodiment_batch.get(self.image_key)
            viz = None if self.viz_func is None else self.viz_func.get(embodiment_name)
            if raw_images is None or viz is None:
                continue
            if raw_images.ndim == 5:
                raw_images = raw_images[:, -1]

            selected = torch.arange(0, count, self.frame_stride)
            limit = self.max_frames_by_embodiment.get(embodiment_name, self.max_frames)
            rendered = self._rendered_frames.get(emb_id, 0)
            if limit is not None and limit > 0:
                selected = selected[: max(0, limit - rendered)]
            if len(selected) == 0:
                continue
            self._rendered_frames[emb_id] = rendered + len(selected)

            def take(value):
                return value.index_select(0, selected.to(value.device))

            viz_batch = {
                "embodiment": torch.full((len(selected),), emb_id, dtype=torch.long),
                self.image_key: take(raw_images),
                action_key: take(target),
            }
            for key in (
                "state_ee_pose",
                "viz_current_wrist_poses",
                "front_intrinsics",
                "left_camera_extrinsics",
                "right_camera_extrinsics",
            ):
                source = unnormalized if key in unnormalized else embodiment_batch
                if key in source and torch.is_tensor(source[key]):
                    viz_batch[key] = take(source[key])
            viz_predictions = {f"{embodiment_name}_{action_key}": take(prediction)}

            transforms = self.transform_lists.get(embodiment_name)
            if transforms:
                gt_transformed = Embodiment.apply_transform(viz_batch, transforms)
                pred_input = dict(viz_batch)
                pred_input[action_key] = viz_predictions[
                    f"{embodiment_name}_{action_key}"
                ]
                pred_transformed = Embodiment.apply_transform(pred_input, transforms)
                viz_batch.update(gt_transformed)
                viz_predictions[f"{embodiment_name}_{action_key}"] = pred_transformed[
                    action_key
                ]
                metrics.update(
                    self._error_metrics(
                        f"Valid/{embodiment_name}_{action_key}_camera_action",
                        pred_transformed[action_key],
                        gt_transformed[action_key],
                    )
                )

            try:
                images[emb_id] = np.asarray(
                    viz(predictions=viz_predictions, batch=viz_batch)
                )
            except Exception as error:
                print(
                    f"[HumanRobotOverlayEval] visualization skipped for "
                    f"{embodiment_name}: {type(error).__name__}: {error}",
                    flush=True,
                )

        if not self.exact_epoch_metrics and self.log_normalized_mse:
            if self._normalized_mse_is_explicit:
                if (
                    normalized_squared_error_sum is None
                    or normalized_element_count == 0
                ):
                    raise ValueError(
                        "Normalized MSE requested for an empty validation batch"
                    )
                metrics["Valid/MSE"] = (
                    normalized_squared_error_sum / normalized_element_count
                ).detach()
            elif normalized_domain_mses:
                metrics["Valid/MSE"] = torch.stack(normalized_domain_mses).mean()

        if not self.exact_epoch_metrics and self.log_native_mse:
            if self._native_mse_is_explicit:
                if native_squared_error_sum is None or native_element_count == 0:
                    raise ValueError(
                        "Native MSE requested for an empty validation batch"
                    )
                metrics["Valid/Native_MSE"] = (
                    native_squared_error_sum / native_element_count
                ).detach()
            elif native_domain_mses:
                metrics["Valid/Native_MSE"] = torch.stack(native_domain_mses).mean()

        return metrics, images


FoldOverlayEval = HumanRobotOverlayEval
