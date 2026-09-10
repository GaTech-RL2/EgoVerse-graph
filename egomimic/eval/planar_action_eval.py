"""Teacher-forced Planar MSE and Energy Score without video/report machinery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from egomimic.eval.action_flow_diagnostics import ActionFlowDiagnostics
from egomimic.eval.artifact_paths import (
    artifact_destination,
    artifact_execution_identity,
)
from egomimic.eval.energy_score import (
    USOCKET_NATIVE_DECODER,
    energy_score,
    normalize_usocket_energy_distance_config,
    usocket_energy_distance_metadata,
    usocket_xy_theta_chunk_distance,
)
from egomimic.eval.eval import Eval
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.rldb.embodiment.embodiment import get_embodiment

_DEFAULT_DETERMINISTIC_SEED = 420042
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
USOCKET_NATIVE_ERROR_CONFIG = {
    "enabled": True,
    "type": "usocket_native_xy_wrapped_theta_mse_v1",
    "space": "decoded_native_x_y_theta_radians",
    "native_theta_index": 2,
    "wrap_period_radians": 2.0 * math.pi,
    "reduction": "mean_squared_error_over_horizon_and_native_coordinates",
    "normalizer": "bound_train_only_evaluator_normalizer",
    "native_decoder": USOCKET_NATIVE_DECODER,
}


def normalize_usocket_native_error_config(value: Mapping) -> dict:
    """Validate the typed native/circular Action Flow diagnostic contract."""
    if not isinstance(value, Mapping):
        raise TypeError("USocket native error contract must be a mapping")
    expected_keys = set(USOCKET_NATIVE_ERROR_CONFIG)
    actual_keys = {str(key) for key in value}
    if actual_keys != expected_keys:
        raise ValueError(
            "USocket native error keys differ: "
            f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
    normalized = {
        "enabled": bool(value["enabled"]),
        "type": str(value["type"]),
        "space": str(value["space"]),
        "native_theta_index": int(value["native_theta_index"]),
        "wrap_period_radians": float(value["wrap_period_radians"]),
        "reduction": str(value["reduction"]),
        "normalizer": str(value["normalizer"]),
        "native_decoder": str(value["native_decoder"]),
    }
    if normalized != USOCKET_NATIVE_ERROR_CONFIG:
        raise ValueError(
            "unsupported USocket native error contract: "
            f"expected {USOCKET_NATIVE_ERROR_CONFIG!r}, got {normalized!r}"
        )
    return copy.deepcopy(normalized)


class PlanarActionEval(Eval):
    """Log deterministic normalized/native MSE and seeded EnergyScore@32."""

    def __init__(
        self,
        limit_val_batches: int = 400,
        seed_bank_path: str | None = None,
        seed_bank_sha256: str | None = None,
        artifact_root: str | None = None,
        semantic_blocks=((0, 2), (2, 4), (4, 5)),
        semantic_blocks_by_source: Mapping | None = None,
        energy_score_enabled: bool = True,
        action_key: str = "actions",
        native_decoder=None,
        native_decoders=None,
        deterministic_seed: int = _DEFAULT_DETERMINISTIC_SEED,
        energy_score_max_batches_per_rank: int | None = None,
        energy_score_validation_view: Mapping | None = None,
        energy_score_provenance: Mapping | None = None,
        energy_score_distance: Mapping | None = None,
        unite_diagnostics: Mapping | None = None,
        action_flow_diagnostics: Mapping | None = None,
    ):
        self.trainer = None
        self.model = None
        self.normalizer = None
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        if native_decoder is not None and native_decoders is not None:
            raise ValueError("configure native_decoder or native_decoders, not both")
        self.native_decoder = native_decoder
        self.native_decoders = dict(native_decoders or {})
        self.blocks = tuple(tuple(map(int, block)) for block in semantic_blocks)
        self.blocks_by_source = {
            str(source).lower(): tuple(tuple(map(int, block)) for block in blocks)
            for source, blocks in dict(semantic_blocks_by_source or {}).items()
        }
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
        self.energy_score_distance = (
            None
            if energy_score_distance is None
            else normalize_usocket_energy_distance_config(energy_score_distance)
        )
        self.energy_score_distance_metadata = (
            {
                "space": "normalized_action_chunk",
                "formula": "mean_equal_weight_semantic_block_rms",
                "semantic_blocks": (
                    [list(block) for block in self.blocks]
                    if self.blocks_by_source
                    else self.blocks
                ),
                **(
                    {
                        "semantic_blocks_by_source": {
                            source: [list(block) for block in blocks]
                            for source, blocks in self.blocks_by_source.items()
                        }
                    }
                    if self.blocks_by_source
                    else {}
                ),
            }
            if self.energy_score_distance is None
            else usocket_energy_distance_metadata(self.energy_score_distance)
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
        self._unite_diagnostic_batches_done = 0
        self._unite_noise_seeds = []
        self._unite_noise_seed_sha256 = None
        self._unite_noise_levels = []
        self._unite_noise_labels = []
        self._unite_cknna_k = 0
        self.action_flow_diagnostics = (
            self._metadata_copy(
                action_flow_diagnostics,
                label="action_flow_diagnostics",
            )
            or {}
        )
        self.action_flow_diagnostics_enabled = bool(
            self.action_flow_diagnostics.get("enabled", False)
        )
        self._action_flow_diagnostics = None
        if (
            energy_score_max_batches_per_rank is not None
            and int(energy_score_max_batches_per_rank) <= 0
        ):
            raise ValueError("energy_score_max_batches_per_rank must be positive")
        self.artifact_root = None if artifact_root is None else Path(artifact_root)
        self.artifact_execution = artifact_execution_identity()
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
        if self.action_flow_diagnostics_enabled:
            self._action_flow_diagnostics = ActionFlowDiagnostics(
                {
                    key: value
                    for key, value in self.action_flow_diagnostics.items()
                    if key != "enabled"
                }
            )
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
        self._unite_diagnostic_batches_done = 0
        if self._action_flow_diagnostics is not None:
            self._action_flow_diagnostics.reset()

    def on_validation_end(self):
        return None

    def _artifact_destination(self, root, batch_idx):
        return artifact_destination(
            root, self.artifact_execution,
            epoch=self.trainer.current_epoch, global_step=self.trainer.global_step,
            rank=self.trainer.global_rank, batch_idx=batch_idx,
        )

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

    @staticmethod
    def _noise_label(raw_level):
        return f"t{int(round(float(raw_level) * 1000.0)):03d}"

    def _configure_unite_diagnostics(self):
        path = Path(str(self.unite_diagnostics.get("noise_seed_bank_path", "")))
        expected = str(self.unite_diagnostics.get("noise_seed_bank_sha256", ""))
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if digest != expected:
            raise ValueError(f"UNITE diagnostic seed-bank identity mismatch: {path}")
        payload = json.loads(path.read_text())
        seeds = [int(seed) for seed in payload.get("seeds", [])]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("UNITE diagnostic seeds must be non-empty and unique")
        self._unite_noise_seeds = seeds
        self._unite_noise_seed_sha256 = digest
        self._unite_noise_levels = [
            float(value) for value in self.unite_diagnostics.get("raw_noise_levels", [])
        ]
        if (
            len(self._unite_noise_levels) < 2
            or any(not 0.0 <= value <= 1.0 for value in self._unite_noise_levels)
            or any(
                right <= left
                for left, right in zip(
                    self._unite_noise_levels,
                    self._unite_noise_levels[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("UNITE diagnostic noise levels must increase in [0, 1]")
        self._unite_noise_labels = [
            self._noise_label(value) for value in self._unite_noise_levels
        ]
        if len(self._unite_noise_labels) != len(set(self._unite_noise_labels)):
            raise ValueError("UNITE diagnostic noise labels collide")
        self._unite_cknna_k = int(self.unite_diagnostics.get("cknna_k", 0))
        view = dict(self.unite_diagnostics.get("validation_view", {}))
        batch_size = int(view.get("per_rank_batch_size", 0))
        world_size = int(view.get("world_size", 0))
        if not 2 <= self._unite_cknna_k < batch_size:
            raise ValueError("UNITE CKNNA requires 2 <= k < per-rank batch size")
        if world_size <= 0 or len(seeds) < world_size:
            raise ValueError("UNITE diagnostics need one seed per configured rank")
        if int(self.unite_diagnostics.get("max_batches_per_rank", 0)) <= 0:
            raise ValueError("UNITE diagnostic batch limit must be positive")
        if not str(self.unite_diagnostics.get("artifact_root", "")):
            raise ValueError("UNITE diagnostics need an artifact root")

    @staticmethod
    def _centered_linear_cka(left, right):
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError("CKA requires paired two-dimensional features")
        left = left.float() - left.float().mean(dim=0, keepdim=True)
        right = right.float() - right.float().mean(dim=0, keepdim=True)
        numerator = (left.T @ right).square().sum()
        denominator = (
            (left.T @ left).square().sum() * (right.T @ right).square().sum()
        ).sqrt()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            raise ValueError("CKA has a zero or non-finite centered norm")
        value = numerator / denominator
        if not bool(torch.isfinite(value)):
            raise ValueError("CKA is non-finite")
        return value

    @staticmethod
    def _unbiased_hsic(left, right):
        if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 4:
            raise ValueError("Unbiased HSIC requires paired square kernels and N >= 4")
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
            raise ValueError("CKNNA requires paired two-dimensional features")
        sample_count = int(left.shape[0])
        if not 2 <= int(topk) < sample_count:
            raise ValueError(f"CKNNA requires 2 <= k < {sample_count}")

        def normalize(value):
            value = value.float()
            norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
            if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
                raise ValueError("CKNNA feature norm is zero or non-finite")
            return value / norms

        left_kernel = normalize(left) @ normalize(left).T
        right_kernel = normalize(right) @ normalize(right).T

        def local_hsic(first, second):
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
            raise ValueError("CKNNA self-alignment is zero or non-finite")
        value = cross / product.sqrt()
        if not bool(torch.isfinite(value)):
            raise ValueError("CKNNA is non-finite")
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
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [PlanarActionEval._cpu_tree(item) for item in value]
        return copy.deepcopy(value)

    def _unite_metrics_and_artifact(self, batch, batch_idx):
        rank = int(self.trainer.global_rank)
        if rank >= len(self._unite_noise_seeds):
            raise ValueError(f"No UNITE diagnostic seed for rank {rank}")
        noise_seed = self._unite_noise_seeds[rank]
        with torch.random.fork_rng(devices=self._cuda_devices(batch)):
            torch.manual_seed(noise_seed)
            diagnostics = self.model.forward_unite_diagnostics(
                batch, raw_noise_levels=self._unite_noise_levels
            )

        metrics = {}
        macro = {}
        artifact_domains = {}

        def add_metric(base, domain, value):
            value = value.detach().float()
            if value.ndim or not bool(torch.isfinite(value)):
                raise ValueError(f"Invalid UNITE diagnostic metric {base}/{domain}")
            metrics[f"{base}/{domain}"] = value
            macro.setdefault(base, []).append(value)

        for source_id, diagnostic in diagnostics.items():
            embodiment_id, domain = self._embodiment(batch[source_id])
            decoder = self._native_decoder(embodiment_id)
            target = batch[source_id][self.action_key]
            clean = diagnostic["clean_latent"].float()
            states = diagnostic["sampler_latents"].float()
            decoded = diagnostic["decoded_actions_normalized"].float()
            if states.shape[1:] != clean.shape or decoded.shape[1:] != target.shape:
                raise ValueError("UNITE diagnostic trajectory shapes do not align")
            latent_mse = (states - clean.unsqueeze(0)).square().mean(dim=(-2, -1))
            decoded_mse = (
                (decoded - target.float().unsqueeze(0)).square().mean(dim=(-2, -1))
            )
            native_target = self._native(target, embodiment_id, decoder)
            decoded_native = torch.stack(
                [self._native(state, embodiment_id, decoder) for state in decoded],
                dim=0,
            )
            native_mse = (
                (decoded_native - native_target.unsqueeze(0))
                .square()
                .mean(dim=(-2, -1))
            )
            for step in range(int(states.shape[0])):
                add_metric(
                    f"Valid/DenoisingTrajectory/LatentMSE/step_{step}",
                    domain,
                    latent_mse[step].mean(),
                )
                add_metric(
                    f"Valid/DenoisingTrajectory/DecodedMSE/step_{step}",
                    domain,
                    decoded_mse[step].mean(),
                )
                add_metric(
                    f"Valid/DenoisingTrajectory/DecodedNativeMSE/step_{step}",
                    domain,
                    native_mse[step].mean(),
                )

            predictions = diagnostic["noise_level_final_predictions"].float()
            clean_flat = clean.reshape(clean.shape[0], -1)
            prediction_flat = predictions.reshape(
                predictions.shape[0], predictions.shape[1], -1
            )
            clean_norm = torch.linalg.vector_norm(clean_flat, dim=1)
            prediction_norm = torch.linalg.vector_norm(prediction_flat, dim=2)
            if bool((clean_norm <= 0).any()) or bool((prediction_norm <= 0).any()):
                raise ValueError("UNITE final-latent cosine has a zero norm")
            cosine = (prediction_flat * clean_flat.unsqueeze(0)).sum(dim=2) / (
                prediction_norm * clean_norm.unsqueeze(0)
            )
            alignment = {}
            tokenization = diagnostic["tokenization_activations"]
            denoising = diagnostic["denoising_activations"]
            for noise_index, label in enumerate(self._unite_noise_labels):
                add_metric(
                    f"Valid/Alignment/FinalLatentCosine/{label}",
                    domain,
                    cosine[noise_index].mean(),
                )
                add_metric(
                    f"Valid/Alignment/FinalLatentCosineStd/{label}",
                    domain,
                    cosine[noise_index].std(unbiased=False),
                )
                for layer, tokenized in tokenization.items():
                    left = tokenized.float().mean(dim=1)
                    right = denoising[layer][noise_index].float().mean(dim=1)
                    cka = self._centered_linear_cka(left, right)
                    cknna = self._cknna(left, right, topk=self._unite_cknna_k)
                    add_metric(f"Valid/Alignment/CKA/{layer}/{label}", domain, cka)
                    add_metric(f"Valid/Alignment/CKNNA/{layer}/{label}", domain, cknna)
                    alignment.setdefault(layer, {})[label] = {
                        "cka": cka,
                        "cknna": cknna,
                    }
            artifact_domains[domain] = self._cpu_tree(
                {
                    **diagnostic,
                    "source_id": source_id,
                    "embodiment_id": embodiment_id,
                    "target_normalized": target,
                    "target_native": native_target,
                    "decoded_actions_native": decoded_native,
                    "trajectory_latent_mse_by_condition": latent_mse,
                    "trajectory_decoded_mse_by_condition": decoded_mse,
                    "trajectory_decoded_native_mse_by_condition": native_mse,
                    "final_latent_cosine_by_condition": cosine,
                    "alignment": alignment,
                }
            )
        for base, values in macro.items():
            metrics[base] = torch.stack(values).mean()

        root = Path(str(self.unite_diagnostics["artifact_root"])).resolve()
        destination = self._artifact_destination(root, batch_idx)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        torch.save(
            {
                "schema_version": 1,
                "metric": "ReleasedUNITETrainingDiagnostics",
                "execution": self.artifact_execution,
                "global_step": int(self.trainer.global_step),
                "epoch": int(self.trainer.current_epoch),
                "rank": rank,
                "batch_idx": int(batch_idx),
                "noise_seed": noise_seed,
                "noise_seed_bank_sha256": self._unite_noise_seed_sha256,
                "raw_noise_levels": self._unite_noise_levels,
                "noise_labels": self._unite_noise_labels,
                "sampler": "dopri5",
                "sampler_output_points": int(
                    next(iter(diagnostics.values()))["sampler_latents"].shape[0]
                ),
                "alignment": {
                    "pooling": "mean_tokens_then_validation_condition_rows",
                    "cka": "biased_centered_linear_cka",
                    "cknna": "unbiased_hsic_intersection_knn",
                    "cknna_k": self._unite_cknna_k,
                    "cosine": "flatten_tokens_and_latent_channels_per_condition",
                },
                "validation_view": self.unite_diagnostics.get("validation_view"),
                "provenance": self.unite_diagnostics.get("provenance"),
                "domains": artifact_domains,
            },
            temporary,
        )
        os.replace(temporary, destination)
        return metrics

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

    def _native_decoder(self, embodiment_id):
        if not self.native_decoders:
            return self.native_decoder
        name = get_embodiment(embodiment_id)
        if name is None:
            raise KeyError(f"Unknown Planar embodiment id {embodiment_id}")
        label = name.lower()
        if label not in self.native_decoders:
            raise KeyError(f"No native decoder configured for {label!r}")
        return self.native_decoders[label]

    @staticmethod
    def _decoder_identity(decoder) -> str | None:
        if decoder is None:
            return None
        decoder_type = type(decoder)
        return f"{decoder_type.__module__}.{decoder_type.__qualname__}"

    @classmethod
    def _require_usocket_decoder(cls, decoder) -> None:
        actual = cls._decoder_identity(decoder)
        if actual != USOCKET_NATIVE_DECODER:
            raise TypeError(
                "typed USocket metric requires the exact native decoder "
                f"{USOCKET_NATIVE_DECODER!r}, got {actual!r}"
            )

    def _native(self, normalized, embodiment_id, decoder):
        if self.normalizer is None:
            raise RuntimeError("Planar evaluator data context was not bound")
        unnormalized = self.normalizer.unnormalize(
            {self.action_key: normalized}, embodiment_id
        )[self.action_key]
        if decoder is None:
            return unnormalized
        if unnormalized.ndim < 3:
            raise ValueError(
                "native action decoding requires (..., horizon, action_dim), "
                f"got {tuple(unnormalized.shape)}"
            )
        leading_shape = tuple(unnormalized.shape[:-2])
        horizon, action_dim = tuple(unnormalized.shape[-2:])
        decoded = decoder.decode(
            unnormalized.reshape(-1, horizon, action_dim)
        )
        return decoded.reshape(*leading_shape, *decoded.shape[-2:])

    @staticmethod
    def _native_mse_by_condition(prediction, target, decoder):
        """Measure native Planar chunks with a circular theta residual."""
        if prediction.shape != target.shape:
            raise ValueError(
                "native prediction and target shapes differ: "
                f"{prediction.shape} != {target.shape}"
            )
        # Without a decoder, rotation remains encoded as cos/sin in common-five
        # space, where ordinary subtraction is already wrap-safe.
        if decoder is None or prediction.shape[-1] < 3:
            return (prediction - target).square().mean(dim=(-2, -1))

        residual = prediction - target
        theta_residual = torch.atan2(
            torch.sin(residual[..., 2]), torch.cos(residual[..., 2])
        )
        residual = torch.cat(
            (residual[..., :2], theta_residual.unsqueeze(-1), residual[..., 3:]),
            dim=-1,
        )
        return residual.square().mean(dim=(-2, -1))

    @classmethod
    def _native_mse(cls, prediction, target, decoder):
        return cls._native_mse_by_condition(prediction, target, decoder).mean()

    def _energy_values(self, samples, target, embodiment_id):
        if samples.ndim != 4 or samples.shape[0] != 32:
            raise ValueError("EnergyScore@32 requires exactly 32 samples")
        distance_fn = None
        if self.energy_score_distance is not None:
            decoder = self._native_decoder(embodiment_id)
            self._require_usocket_decoder(decoder)

            def distance_fn(left, right):
                return usocket_xy_theta_chunk_distance(
                    left,
                    right,
                    self._native(left, embodiment_id, decoder),
                    self._native(right, embodiment_id, decoder),
                    config=self.energy_score_distance,
                )

        embodiment_name = get_embodiment(embodiment_id)
        if embodiment_name is None:
            raise KeyError(f"Unknown Planar embodiment id {embodiment_id}")
        blocks = self.blocks_by_source.get(embodiment_name.lower(), self.blocks)
        values = {
            name: value.detach()
            for name, value in energy_score(
                samples,
                target,
                blocks,
                distance_fn=distance_fn,
            ).items()
        }
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("Energy Score produced non-finite values")
        return values

    @staticmethod
    def _json_sha256(value) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _tensor_sha256(value: torch.Tensor) -> str:
        value = value.detach().float().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _batch_identity_values(value, batch_size: int):
        if torch.is_tensor(value):
            if value.ndim != 1 or int(value.shape[0]) != batch_size:
                return None
            return [item.item() for item in value.detach().cpu()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != batch_size:
                return None
            result = []
            for item in value:
                if torch.is_tensor(item):
                    if item.ndim != 0:
                        return None
                    item = item.item()
                if not isinstance(item, (str, int, float, bool)) and item is not None:
                    return None
                result.append(item)
            return result
        return None

    @classmethod
    def _condition_ids(cls, source_batch, target: torch.Tensor) -> list[dict]:
        target = target.detach().float().cpu()
        batch_size = int(target.shape[0])
        fields = {}
        for key in (
            "episode_hash",
            "episode_id",
            "frame_index",
            "frame_idx",
            "sample_index",
        ):
            if key in source_batch:
                values = cls._batch_identity_values(source_batch[key], batch_size)
                if values is not None:
                    fields[key] = values
        identities = []
        for index in range(batch_size):
            identity = {
                "batch_position": index,
                "normalized_target_sha256": cls._tensor_sha256(target[index]),
            }
            for key, values in fields.items():
                identity[key] = values[index]
            identities.append(identity)
        return identities

    def _typed_artifact_identity(self, *, domains, global_step: int) -> dict:
        provenance = self.energy_score_provenance
        if not isinstance(provenance, Mapping):
            raise ValueError(
                "typed USocket EnergyScore requires energy_score_provenance"
            )
        source_commit = str(provenance.get("source_commit", "")).lower()
        normalization_sha256 = str(provenance.get("normalization_sha256", "")).lower()
        split_sha256 = str(provenance.get("split_manifest_sha256", "")).lower()
        if not _COMMIT_RE.fullmatch(source_commit):
            raise ValueError("EnergyScore provenance source_commit is not exact")
        for label, value in (
            ("normalization", normalization_sha256),
            ("split manifest", split_sha256),
        ):
            if not _SHA256_RE.fullmatch(value):
                raise ValueError(f"EnergyScore provenance {label} SHA-256 is not exact")
        validation_split = str(
            (self.energy_score_validation_view or {}).get("split_manifest_sha256", "")
        ).lower()
        if validation_split != split_sha256:
            raise ValueError(
                "EnergyScore validation/provenance split identities differ"
            )

        distance_contract = provenance.get("distance_contract")
        if self.energy_score_distance is None:
            expected_distance_contract = (
                self.energy_score_distance_metadata
                if self.blocks_by_source
                else None
            )
            if distance_contract != expected_distance_contract:
                raise ValueError("generic EnergyScore distance contract differs")
        else:
            normalized_distance = normalize_usocket_energy_distance_config(
                distance_contract
            )
            if normalized_distance != self.energy_score_distance:
                raise ValueError("EnergyScore provenance distance contract differs")

        config_path = Path(str(provenance.get("resolved_config_path", ""))).expanduser()
        try:
            config_path = config_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"EnergyScore resolved config does not exist: {config_path}"
            ) from error
        if (
            not config_path.is_file()
            or config_path.name != "config.yaml"
            or config_path.parent.name != ".hydra"
        ):
            raise ValueError(
                f"EnergyScore resolved config path is not .hydra/config.yaml: {config_path}"
            )
        resolved_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

        wandb = provenance.get("wandb")
        if not isinstance(wandb, Mapping):
            raise ValueError("EnergyScore provenance W&B identity is missing")
        wandb_identity = {}
        for key in ("entity", "project", "run_id"):
            value = wandb.get(key)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value.strip().lower() in {"none", "null"}
                or "${" in value
            ):
                raise ValueError("EnergyScore provenance W&B identity is incomplete")
            wandb_identity[key] = value.strip()

        dataset_content = provenance.get("dataset_content")
        if not isinstance(dataset_content, Mapping):
            raise ValueError("EnergyScore dataset-content identity is missing")
        manifest_path = Path(str(dataset_content.get("manifest_path", ""))).expanduser()
        try:
            manifest_path = manifest_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"EnergyScore dataset-content manifest does not exist: {manifest_path}"
            ) from error
        if not manifest_path.is_file():
            raise ValueError(
                f"EnergyScore dataset-content manifest is not a file: {manifest_path}"
            )
        manifest_sha256 = str(dataset_content.get("manifest_sha256", "")).lower()
        aggregate_sha256 = str(dataset_content.get("aggregate_sha256", "")).lower()
        for label, value in (
            ("dataset-content manifest", manifest_sha256),
            ("dataset aggregate content", aggregate_sha256),
        ):
            if not _SHA256_RE.fullmatch(value):
                raise ValueError(f"EnergyScore {label} SHA-256 is not exact")
        actual_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if actual_manifest_sha256 != manifest_sha256:
            raise ValueError("EnergyScore dataset-content manifest hash differs")
        manifest_payload = json.loads(manifest_path.read_text())
        if (
            str(manifest_payload.get("aggregate_sha256", "")).lower()
            != aggregate_sha256
        ):
            raise ValueError("EnergyScore dataset aggregate content hash differs")

        validation_conditions = {
            label: domain["condition_ids"] for label, domain in domains.items()
        }
        for label, conditions in validation_conditions.items():
            if not isinstance(conditions, list) or not conditions:
                raise ValueError(
                    f"EnergyScore validation conditions are empty for {label!r}"
                )
            for condition in conditions:
                if not isinstance(condition, Mapping) or not {
                    "batch_position",
                    "episode_hash",
                    "frame_index",
                    "normalized_target_sha256",
                }.issubset(condition):
                    raise ValueError(
                        "typed USocket EnergyScore condition identity requires "
                        "batch position, episode hash, frame index, and target hash"
                    )
        return {
            "schema": "energy-score-validation-artifact/v2",
            "metric": {
                "name": "EnergyScore@32",
                "sample_count": 32,
                "deterministic_seed": self.deterministic_seed,
            },
            "source_commit": source_commit,
            "normalization_sha256": normalization_sha256,
            "split_manifest_sha256": split_sha256,
            "resolved_config_path": str(config_path),
            "resolved_config_sha256": resolved_config_sha256,
            "wandb": wandb_identity,
            "dataset_content": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "aggregate_sha256": aggregate_sha256,
            },
            "distance": self.energy_score_distance_metadata,
            "seed_bank_sha256": self.seed_bank_sha256,
            "validation_view": self.energy_score_validation_view,
            "validation_conditions": validation_conditions,
            "checkpoint_binding": {
                "global_step": int(global_step),
                "checkpoint_sha256": None,
                "sha256_status": (
                    "unavailable_during_validation_artifact_write; post-run_smoke_"
                    "verifier_binds_checkpoint_and_artifact_by_global_step_and_records_"
                    "both_file_hashes"
                ),
            },
        }

    def _action_flow_native_error_fns(self, batch):
        runner = self._action_flow_diagnostics
        if runner is None or not runner.native_error_enabled:
            return None
        contract = normalize_usocket_native_error_config(runner.native_error)
        functions = {}
        for source_id, source_batch in batch.items():
            embodiment_id, _ = self._embodiment(source_batch)
            decoder = self._native_decoder(embodiment_id)
            self._require_usocket_decoder(decoder)

            def native_error(
                prediction, target, *, _id=embodiment_id, _decoder=decoder
            ):
                prediction_native = self._native(prediction, _id, _decoder)
                target_native = self._native(target, _id, _decoder)
                return self._native_mse_by_condition(
                    prediction_native,
                    target_native,
                    _decoder,
                )

            functions[source_id] = native_error
        if contract != USOCKET_NATIVE_ERROR_CONFIG:
            raise ValueError("Action Flow native error contract changed")
        return functions

    def _save_artifact(self, batch_idx, samples, scores, batch):
        destination = self._artifact_destination(self.artifact_root, batch_idx)
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
            target = batch[source_id][self.action_key].detach().float().cpu()
            domain = {
                "source_id": source_id,
                "embodiment_id": embodiment_id,
                "action_key": self.action_key,
                "predictions": predictions.float().cpu(),
                "targets": target,
                "accuracy_by_condition": values["accuracy_by_condition"].float().cpu(),
                "diversity_by_condition": values["diversity_by_condition"]
                .float()
                .cpu(),
                "score_by_condition": values["score_by_condition"].float().cpu(),
            }
            if (
                self.energy_score_distance is not None
                or self.native_decoder is not None
                or self.native_decoders
            ):
                decoder = self._native_decoder(embodiment_id)
                if self.energy_score_distance is not None:
                    self._require_usocket_decoder(decoder)
                domain["condition_ids"] = self._condition_ids(batch[source_id], target)
                domain["native_predictions"] = (
                    self._native(predictions, embodiment_id, decoder)
                    .detach()
                    .float()
                    .cpu()
                )
                domain["native_targets"] = (
                    self._native(
                        batch[source_id][self.action_key], embodiment_id, decoder
                    )
                    .detach()
                    .float()
                    .cpu()
                )
            domains[name] = domain
        precision = getattr(self.trainer, "precision", None)
        payload = {
            "schema_version": 1,
            "metric": "EnergyScore@32",
            "execution": self.artifact_execution,
            "sample_count": 32,
            "seed_bank": self.seeds,
            "seed_bank_sha256": self.seed_bank_sha256,
            "deterministic_seed": self.deterministic_seed,
            "distance": self.energy_score_distance_metadata,
            "aggregation": "condition_mean_then_equal_domain_macro_mean",
            "global_step": int(self.trainer.global_step),
            "epoch": int(self.trainer.current_epoch),
            "rank": int(self.trainer.global_rank),
            "batch_idx": int(batch_idx),
            "precision": None if precision is None else str(precision),
            "validation_view": self.energy_score_validation_view,
            "provenance": self.energy_score_provenance,
            "domains": domains,
        }
        if (
            self.energy_score_distance is not None
            or self.native_decoder is not None
            or self.native_decoders
        ):
            payload["schema_version"] = 2
            identity = self._typed_artifact_identity(
                domains=domains,
                global_step=int(self.trainer.global_step),
            )
            payload["identity"] = identity
            payload["identity_sha256"] = self._json_sha256(identity)
        torch.save(
            payload,
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
        if self.unite_diagnostics_enabled and self._unite_diagnostic_batches_done < int(
            self.unite_diagnostics["max_batches_per_rank"]
        ):
            metrics.update(
                self._unite_metrics_and_artifact(
                    batch, self._unite_diagnostic_batches_done
                )
            )
            self._unite_diagnostic_batches_done += 1
        if (
            self._action_flow_diagnostics is not None
            and self._action_flow_diagnostics.should_run()
        ):
            source_labels = {
                source_id: self._embodiment(source_batch)[1]
                for source_id, source_batch in batch.items()
            }
            metrics.update(
                self._action_flow_diagnostics.run(
                    model=self.model,
                    batch=batch,
                    batch_idx=batch_idx,
                    rank=int(self.trainer.global_rank),
                    epoch=int(self.trainer.current_epoch),
                    global_step=int(self.trainer.global_step),
                    precision=getattr(self.trainer, "precision", None),
                    source_labels=source_labels,
                    cuda_devices=self._cuda_devices(batch),
                    native_error_fns=self._action_flow_native_error_fns(batch),
                )
            )
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
            decoder = self._native_decoder(embodiment_id)
            native_mse = self._native_mse(
                self._native(prediction, embodiment_id, decoder),
                self._native(target, embodiment_id, decoder),
                decoder,
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
                embodiment_id, label = self._embodiment(batch[source_id])
                values = self._energy_values(
                    samples,
                    batch[source_id][self.action_key],
                    embodiment_id,
                )
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
