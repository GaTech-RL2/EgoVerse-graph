"""Strict validation diagnostics for context-conditioned latent action flow."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _plain(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")

    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return [convert(child) for child in item]
        return copy.deepcopy(item)

    result = convert(value)
    try:
        json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{label} must contain JSON-serializable finite values"
        ) from error
    return result


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _noise_label(level: float) -> str:
    return f"t{int(round(float(level) * 1000.0)):04d}"


def _tensor(
    diagnostic: Mapping[str, Any],
    key: str,
    *,
    ndim: int | None = None,
) -> torch.Tensor:
    value = diagnostic.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"Action Flow diagnostic {key!r} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"Action Flow diagnostic {key!r} must have rank {ndim}, "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Action Flow diagnostic {key!r} is non-finite")
    return value


def _integer(value: Any, *, label: str) -> int:
    if torch.is_tensor(value):
        if value.ndim != 0:
            raise TypeError(f"{label} must be an integer scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _assert_close(left: torch.Tensor, right: torch.Tensor, *, label: str) -> None:
    if left.shape != right.shape or not bool(
        torch.allclose(left.float(), right.float(), rtol=1.0e-4, atol=1.0e-5)
    ):
        raise ValueError(f"Action Flow diagnostic endpoint mismatch: {label}")


class ActionFlowDiagnostics:
    """Validate, summarize, and immutably persist Action Flow diagnostics.

    The model owns diagnostic forward computation. This class deliberately
    consumes a strict tensor schema instead of reaching into model modules, so
    validation cannot silently reconstruct a different sampler or activation
    pathway.
    """

    schema = "action-flow-validation-diagnostics/v1"

    def __init__(self, config: Mapping[str, Any]):
        config = _plain(config, label="action_flow_diagnostics")
        self.max_batches_per_rank = int(config.get("max_batches_per_rank", 0))
        if self.max_batches_per_rank <= 0:
            raise ValueError("Action Flow diagnostic batch limit must be positive")

        artifact_root = str(config.get("artifact_root", ""))
        if not artifact_root:
            raise ValueError("Action Flow diagnostics need an artifact root")
        self.artifact_root = Path(artifact_root).expanduser().resolve()

        seed_path = Path(str(config.get("noise_seed_bank_path", ""))).expanduser()
        expected_seed_hash = str(config.get("noise_seed_bank_sha256", ""))
        actual_seed_hash = (
            hashlib.sha256(seed_path.read_bytes()).hexdigest()
            if seed_path.is_file()
            else None
        )
        if actual_seed_hash != expected_seed_hash:
            raise ValueError(
                f"Action Flow diagnostic seed-bank identity mismatch: {seed_path}"
            )
        seed_payload = json.loads(seed_path.read_text())
        self.noise_seeds = [int(seed) for seed in seed_payload.get("seeds", [])]
        if not self.noise_seeds or len(self.noise_seeds) != len(set(self.noise_seeds)):
            raise ValueError(
                "Action Flow diagnostic seeds must be non-empty and unique"
            )
        self.noise_seed_bank_path = str(seed_path.resolve())
        self.noise_seed_bank_sha256 = str(actual_seed_hash)

        self.noise_levels = [
            float(value) for value in config.get("raw_noise_levels", [])
        ]
        if (
            len(self.noise_levels) < 2
            or any(
                not math.isfinite(level) or not 0.0 <= level <= 1.0
                for level in self.noise_levels
            )
            or any(
                right <= left
                for left, right in zip(
                    self.noise_levels, self.noise_levels[1:], strict=False
                )
            )
        ):
            raise ValueError(
                "Action Flow diagnostic noise levels must strictly increase in [0, 1]"
            )
        self.noise_labels = [_noise_label(level) for level in self.noise_levels]
        if len(self.noise_labels) != len(set(self.noise_labels)):
            raise ValueError("Action Flow diagnostic noise labels collide")

        raw_max_samples = config.get("max_samples")
        self.max_samples = None if raw_max_samples is None else int(raw_max_samples)
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("Action Flow diagnostic max_samples must be positive")
        self.jacobian_samples = int(config.get("jacobian_samples", 0))
        if self.jacobian_samples <= 0:
            raise ValueError("Action Flow diagnostic jacobian_samples must be positive")
        if self.max_samples is not None and self.jacobian_samples > self.max_samples:
            raise ValueError("jacobian_samples cannot exceed max_samples")

        self.capture_activations = bool(config.get("capture_activations", False))
        raw_layer_map = config.get("activation_layer_map", {})
        if not isinstance(raw_layer_map, Mapping):
            raise TypeError("activation_layer_map must map encoder to field indices")
        self.activation_layer_map = tuple(
            sorted(
                (int(encoder), int(field)) for encoder, field in raw_layer_map.items()
            )
        )
        if len({pair[0] for pair in self.activation_layer_map}) != len(
            self.activation_layer_map
        ):
            raise ValueError("activation_layer_map repeats an encoder block")
        if any(
            encoder < 0 or field < 0 for encoder, field in self.activation_layer_map
        ):
            raise ValueError("activation layer indices must be non-negative")
        self.cknna_k = int(config.get("cknna_k", 0))
        if self.capture_activations:
            if not self.activation_layer_map:
                raise ValueError(
                    "activation capture requires a non-empty activation_layer_map"
                )
            if self.cknna_k < 2:
                raise ValueError("Action Flow CKNNA requires k >= 2")
        elif self.activation_layer_map or self.cknna_k:
            raise ValueError(
                "activation_layer_map and cknna_k require capture_activations=true"
            )

        self.validation_view = _plain(
            config.get("validation_view", {}), label="validation_view"
        )
        self.provenance = _plain(config.get("provenance", {}), label="provenance")
        if not self.validation_view or not self.provenance:
            raise ValueError(
                "Action Flow diagnostics require validation_view and provenance"
            )
        native_error = config.get("native_error")
        self.native_error = (
            None if native_error is None else _plain(native_error, label="native_error")
        )
        self.native_error_enabled = bool(
            self.native_error is not None and self.native_error.get("enabled", False)
        )
        if self.native_error is not None and not self.native_error_enabled:
            raise ValueError(
                "configured Action Flow native_error must have enabled=true"
            )
        world_size = int(self.validation_view.get("world_size", 0))
        if world_size <= 0 or len(self.noise_seeds) < world_size:
            raise ValueError(
                "Action Flow diagnostics need one noise seed per configured rank"
            )
        configured_batch_size = int(self.validation_view.get("per_rank_batch_size", 0))
        if configured_batch_size <= 0:
            raise ValueError("validation_view.per_rank_batch_size must be positive")
        if self.max_samples is not None:
            configured_batch_size = min(configured_batch_size, self.max_samples)
        self.expected_sample_count = configured_batch_size
        if self.capture_activations and not 2 <= self.cknna_k < configured_batch_size:
            raise ValueError(
                "Action Flow CKNNA requires 2 <= k < diagnostic batch size"
            )

        identity = {
            "schema": self.schema,
            "noise_seed_bank_path": self.noise_seed_bank_path,
            "noise_seed_bank_sha256": self.noise_seed_bank_sha256,
            "raw_noise_levels": self.noise_levels,
            "max_samples": self.max_samples,
            "jacobian_samples": self.jacobian_samples,
            "capture_activations": self.capture_activations,
            "activation_layer_map": [list(pair) for pair in self.activation_layer_map],
            "cknna_k": self.cknna_k,
            "validation_view": self.validation_view,
            "provenance": self.provenance,
        }
        if self.native_error is not None:
            identity["native_error"] = self.native_error
        self.identity = identity
        self.identity_sha256 = _sha256_json(identity)
        self.batches_done = 0

    def reset(self) -> None:
        self.batches_done = 0

    def should_run(self) -> bool:
        return self.batches_done < self.max_batches_per_rank

    @staticmethod
    def _centered_linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError("Action Flow CKA requires paired matrix features")
        left = left.float() - left.float().mean(dim=0, keepdim=True)
        right = right.float() - right.float().mean(dim=0, keepdim=True)
        numerator = (left.T @ right).square().sum()
        denominator = (
            (left.T @ left).square().sum() * (right.T @ right).square().sum()
        ).sqrt()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            raise ValueError("Action Flow CKA has zero or non-finite centered norm")
        value = numerator / denominator
        if not bool(torch.isfinite(value)):
            raise ValueError("Action Flow CKA is non-finite")
        return value

    @staticmethod
    def _unbiased_hsic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 4:
            raise ValueError(
                "Action Flow unbiased HSIC needs paired square kernels and N >= 4"
            )
        count = int(left.shape[0])
        left = left.clone().fill_diagonal_(0.0)
        right = right.clone().fill_diagonal_(0.0)
        return (
            (left * right.T).sum()
            + left.sum() * right.sum() / ((count - 1) * (count - 2))
            - 2.0 * (left @ right).sum() / (count - 2)
        ) / (count * (count - 3))

    @classmethod
    def _cknna(
        cls, left: torch.Tensor, right: torch.Tensor, *, topk: int
    ) -> torch.Tensor:
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError("Action Flow CKNNA requires paired matrix features")
        count = int(left.shape[0])
        if not 2 <= int(topk) < count:
            raise ValueError(f"Action Flow CKNNA requires 2 <= k < {count}")

        def normalized_kernel(value: torch.Tensor) -> torch.Tensor:
            value = value.float()
            norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
            if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
                raise ValueError("Action Flow CKNNA feature norm is invalid")
            value = value / norms
            return value @ value.T

        left_kernel = normalized_kernel(left)
        right_kernel = normalized_kernel(right)

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
            raise ValueError("Action Flow CKNNA self-alignment is invalid")
        value = cross / product.sqrt()
        if not bool(torch.isfinite(value)):
            raise ValueError("Action Flow CKNNA is non-finite")
        return value

    @staticmethod
    def _latent_summary(value: torch.Tensor) -> dict[str, torch.Tensor]:
        value = value.float()
        if value.ndim != 3 or int(value.shape[0] * value.shape[1]) < 2:
            raise ValueError("latent statistics need shape (B, H, D) and B*H >= 2")
        flat = value.reshape(-1, value.shape[-1]).double()
        centered = flat - flat.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(flat.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance)
        tolerance = 1.0e-8 * max(float(eigenvalues.abs().max()), 1.0)
        if float(eigenvalues.min()) < -tolerance:
            raise ValueError("latent covariance has a materially negative eigenvalue")
        eigenvalues = eigenvalues.clamp_min(0.0).flip(0).float()
        trace = eigenvalues.sum()
        if float(trace) > 0.0:
            probabilities = eigenvalues / trace
            positive = probabilities > 0.0
            effective_rank = torch.exp(
                -(probabilities[positive] * probabilities[positive].log()).sum()
            )
        else:
            effective_rank = trace.new_zeros(())
        by_condition = value.square().mean(dim=(-2, -1)).sqrt()
        result = {
            "rms": value.square().mean().sqrt(),
            "rms_by_condition": by_condition,
            "rms_std": by_condition.std(unbiased=False),
            "covariance_eigenvalues": eigenvalues,
            "covariance_trace": trace,
            "effective_rank": effective_rank,
        }
        if not all(bool(torch.isfinite(item).all()) for item in result.values()):
            raise ValueError("latent statistics are non-finite")
        return result

    @staticmethod
    def _jacobian_summary(singular_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if singular_values.ndim != 2 or not singular_values.numel():
            raise ValueError("decoder Jacobian singular values need shape (J, R)")
        values = singular_values.float()
        if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
            raise ValueError("decoder Jacobian singular values are invalid")
        mean_spectrum = values.mean(dim=0)
        largest = values[:, 0]
        smallest = values[:, -1]
        epsilon = torch.finfo(values.dtype).eps
        condition = largest / smallest.clamp_min(epsilon)
        threshold = largest.unsqueeze(1) * 1.0e-5
        numerical_rank = (values > threshold).sum(dim=1).float()
        result = {
            "mean_spectrum": mean_spectrum,
            "spectral_norm_mean": largest.mean(),
            "smallest_singular_mean": smallest.mean(),
            "frobenius_norm_mean": values.square().sum(dim=1).sqrt().mean(),
            "condition_number_mean": condition.mean(),
            "numerical_rank_mean": numerical_rank.mean(),
        }
        if not all(bool(torch.isfinite(item).all()) for item in result.values()):
            raise ValueError("decoder Jacobian summaries are non-finite")
        return result

    @staticmethod
    def _decoded_noise_summary(value: torch.Tensor) -> dict[str, torch.Tensor]:
        if value.ndim != 3:
            raise ValueError("decoded full noise must have shape (B, H, A)")
        value = value.float()
        flat = value.reshape(-1, value.shape[-1])
        token_radius = torch.linalg.vector_norm(value, dim=-1)
        chunk_radius = value.square().mean(dim=(-2, -1)).sqrt()
        result = {
            "coordinate_mean": flat.mean(dim=0),
            "coordinate_std": flat.std(dim=0, unbiased=False),
            "coordinate_second_moment": flat.square().mean(dim=0),
            "token_radius_mean": token_radius.mean(),
            "token_radius_std": token_radius.std(unbiased=False),
            "token_radius_max": token_radius.max(),
            "chunk_rms_radius_by_condition": chunk_radius,
            "chunk_rms_radius_mean": chunk_radius.mean(),
            "chunk_rms_radius_std": chunk_radius.std(unbiased=False),
        }
        if not all(bool(torch.isfinite(item).all()) for item in result.values()):
            raise ValueError("decoded full-noise summaries are non-finite")
        return result

    @staticmethod
    def _native_error_values(
        function,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        expected_shape: tuple[int, ...],
        label: str,
    ) -> torch.Tensor:
        values = function(prediction, target)
        if not torch.is_tensor(values) or tuple(values.shape) != expected_shape:
            actual = None if not torch.is_tensor(values) else tuple(values.shape)
            raise ValueError(
                f"Action Flow native error {label} must have shape "
                f"{expected_shape}, got {actual}"
            )
        values = values.detach().float()
        if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
            raise ValueError(f"Action Flow native error {label} is invalid")
        return values

    @staticmethod
    def _cpu_tree(value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            return value.float() if value.is_floating_point() else value
        if isinstance(value, Mapping):
            return {
                str(key): ActionFlowDiagnostics._cpu_tree(child)
                for key, child in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [ActionFlowDiagnostics._cpu_tree(child) for child in value]
        return copy.deepcopy(value)

    @staticmethod
    def _validate_tensor_tree(value: Any, path: tuple[str, ...] = ()) -> None:
        if torch.is_tensor(value):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    "non-finite Action Flow diagnostic output " + "/".join(path)
                )
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                ActionFlowDiagnostics._validate_tensor_tree(child, (*path, str(key)))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, child in enumerate(value):
                ActionFlowDiagnostics._validate_tensor_tree(child, (*path, str(index)))

    def _analyze_source(
        self,
        diagnostic: Mapping[str, Any],
        *,
        source_id: Any,
        source_label: str,
        add_metric,
        expected_noise_seed: int,
        native_error_fn=None,
    ) -> dict[str, Any]:
        self._validate_tensor_tree(diagnostic)
        if diagnostic.get("schema") != self.schema:
            raise ValueError(
                f"unsupported Action Flow diagnostic schema for {source_id!r}"
            )
        if _integer(diagnostic.get("noise_seed"), label="noise_seed") != int(
            expected_noise_seed
        ):
            raise ValueError("Action Flow diagnostic noise seed does not match request")
        levels = _tensor(diagnostic, "noise_levels", ndim=1).float()
        expected_levels = levels.new_tensor(self.noise_levels)
        if levels.shape != expected_levels.shape or not bool(
            torch.allclose(levels, expected_levels, rtol=0.0, atol=1.0e-7)
        ):
            raise ValueError("Action Flow diagnostic noise levels do not match config")

        target = _tensor(diagnostic, "target", ndim=3).float()
        condition = _tensor(diagnostic, "condition", ndim=2)
        batch_size, horizon, action_dim = map(int, target.shape)
        if batch_size <= 0 or condition.shape[0] != batch_size:
            raise ValueError("Action Flow diagnostic target/condition batch mismatch")
        if batch_size != self.expected_sample_count:
            raise ValueError(
                "Action Flow diagnostic returned sample count differs from its "
                "fixed validation view"
            )
        if (
            _integer(diagnostic.get("returned_samples"), label="returned_samples")
            != batch_size
        ):
            raise ValueError("Action Flow diagnostic returned_samples is inconsistent")
        if str(diagnostic.get("source")) != str(source_id):
            raise ValueError("Action Flow diagnostic source identity is inconsistent")
        expected_integration = (
            "reverse_euler_partial_to_lower_canonical_grid_then_uniform_to_zero"
        )
        if diagnostic.get("integration_semantics") != expected_integration:
            raise ValueError("Action Flow diagnostic integration semantics changed")

        def latent(key: str, *, leading: tuple[int, ...] = ()) -> torch.Tensor:
            value = _tensor(diagnostic, key, ndim=3 + len(leading)).float()
            expected_prefix = (*leading, batch_size, horizon)
            if tuple(value.shape[: len(expected_prefix)]) != expected_prefix:
                raise ValueError(f"Action Flow diagnostic {key!r} shape mismatch")
            return value

        clean = latent("latent/clean")
        latent_dim = int(clean.shape[-1])
        if latent_dim <= 0:
            raise ValueError("Action Flow latent width must be positive")
        noise = latent("latent/noise")
        generated = latent("latent/generated")
        if noise.shape != clean.shape or generated.shape != clean.shape:
            raise ValueError("Action Flow clean/noise/generated latent shapes differ")

        level_count = len(self.noise_levels)
        fixed_latent_keys = (
            "latent/fixed_states",
            "latent/predicted_clean",
            "latent/fixed_final",
            "field/predicted_velocity",
            "field/velocity_residual",
        )
        fixed_latents = {
            key: latent(key, leading=(level_count,)) for key in fixed_latent_keys
        }
        if any(value.shape[-1] != latent_dim for value in fixed_latents.values()):
            raise ValueError("Action Flow fixed-level latent widths differ")

        inference_steps = _integer(
            diagnostic.get("num_inference_steps"), label="num_inference_steps"
        )
        if inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if int(self.provenance.get("sampler_steps", -1)) != inference_steps:
            raise ValueError(
                "Action Flow diagnostic inference steps do not match provenance"
            )
        trajectory = latent("latent/trajectory", leading=(inference_steps + 1,))
        if trajectory.shape[-1] != latent_dim:
            raise ValueError("Action Flow trajectory latent width differs")

        def decoded(key: str, *, leading: tuple[int, ...] = ()) -> torch.Tensor:
            value = _tensor(diagnostic, key, ndim=3 + len(leading)).float()
            expected = (*leading, batch_size, horizon, action_dim)
            if tuple(value.shape) != expected:
                raise ValueError(f"Action Flow diagnostic {key!r} shape mismatch")
            return value

        reconstruction = decoded("decoded/reconstruction")
        decoded_noise = decoded("decoded/noise")
        decoded_generated = decoded("decoded/generated")
        fixed_decoded = {
            key: decoded(key, leading=(level_count,))
            for key in (
                "decoded/fixed_states",
                "decoded/predicted_clean",
                "decoded/fixed_final",
            )
        }
        decoded_trajectory = decoded(
            "decoded/trajectory", leading=(inference_steps + 1,)
        )
        field_evaluations = _tensor(diagnostic, "fixed_level_field_evaluations", ndim=1)
        if field_evaluations.shape != levels.shape or bool(
            (field_evaluations < 0).any()
        ):
            raise ValueError("invalid fixed-level field-evaluation counts")
        expected_evaluations = []
        for level in self.noise_levels:
            scaled = level * inference_steps
            rounded = round(scaled)
            expected_evaluations.append(
                int(rounded)
                if math.isclose(scaled, rounded, rel_tol=0.0, abs_tol=1.0e-7)
                else int(math.ceil(scaled))
            )
        if not torch.equal(
            field_evaluations.long(),
            field_evaluations.new_tensor(expected_evaluations).long(),
        ):
            raise ValueError("fixed-level field-evaluation counts changed")
        add_metric(
            "Valid/ActionFlow/Compute/InferenceFieldEvaluations",
            source_label,
            trajectory.new_tensor(float(inference_steps)),
        )
        for index, label in enumerate(self.noise_labels):
            add_metric(
                f"Valid/ActionFlow/Compute/FixedLevelFieldEvaluations/{label}",
                source_label,
                field_evaluations[index].float(),
            )

        _assert_close(trajectory[0], noise, label="trajectory start versus noise")
        _assert_close(
            trajectory[-1], generated, label="trajectory end versus generated"
        )
        _assert_close(
            decoded_trajectory[0],
            decoded_noise,
            label="decoded trajectory start versus decoded noise",
        )
        _assert_close(
            decoded_trajectory[-1],
            decoded_generated,
            label="decoded trajectory end versus decoded generated",
        )
        for index, level in enumerate(self.noise_levels):
            if level == 0.0:
                _assert_close(
                    fixed_latents["latent/fixed_states"][index],
                    clean,
                    label="t=0 fixed state versus clean latent",
                )
                _assert_close(
                    fixed_latents["latent/fixed_final"][index],
                    clean,
                    label="t=0 final latent versus clean latent",
                )
                _assert_close(
                    fixed_decoded["decoded/fixed_states"][index],
                    reconstruction,
                    label="t=0 decoded state versus reconstruction",
                )
            if level == 1.0:
                _assert_close(
                    fixed_latents["latent/fixed_states"][index],
                    noise,
                    label="t=1 fixed state versus noise",
                )

        computed: dict[str, Any] = {}
        latent_statistics = {}
        for name, value in (
            ("clean", clean),
            ("noise", noise),
            ("generated", generated),
        ):
            summary = self._latent_summary(value)
            latent_statistics[name] = summary
            prefix = f"Valid/ActionFlow/Latent/{name}"
            add_metric(f"{prefix}/RMS", source_label, summary["rms"])
            add_metric(f"{prefix}/RMSStd", source_label, summary["rms_std"])
            add_metric(
                f"{prefix}/CovarianceTrace",
                source_label,
                summary["covariance_trace"],
            )
            add_metric(
                f"{prefix}/EffectiveRank", source_label, summary["effective_rank"]
            )
            for eigen_index, eigenvalue in enumerate(summary["covariance_eigenvalues"]):
                add_metric(
                    f"{prefix}/CovarianceEigenvalue/eig_{eigen_index:02d}",
                    source_label,
                    eigenvalue,
                )
        computed["latent_statistics"] = latent_statistics

        decoder_statistics = {}
        jacobian_specs = (
            ("clean", "decoder_jacobian/clean_singular_values"),
            ("noise", "decoder_jacobian/noise_singular_values"),
        )
        for name, key in jacobian_specs:
            singular = _tensor(diagnostic, key, ndim=2)
            if int(singular.shape[0]) > self.jacobian_samples:
                raise ValueError("decoder Jacobian sample count exceeds configured cap")
            summary = self._jacobian_summary(singular)
            decoder_statistics[name] = summary
            prefix = f"Valid/ActionFlow/DecoderJacobian/{name}"
            for metric_name in (
                "spectral_norm_mean",
                "smallest_singular_mean",
                "frobenius_norm_mean",
                "condition_number_mean",
                "numerical_rank_mean",
            ):
                add_metric(
                    f"{prefix}/{metric_name}", source_label, summary[metric_name]
                )
            for singular_index, singular_value in enumerate(summary["mean_spectrum"]):
                add_metric(
                    f"{prefix}/Spectrum/sv_{singular_index:03d}",
                    source_label,
                    singular_value,
                )

        fixed_singular = _tensor(
            diagnostic, "decoder_jacobian/fixed_singular_values", ndim=3
        )
        if fixed_singular.shape[0] != level_count:
            raise ValueError("fixed decoder Jacobian levels do not match config")
        if int(fixed_singular.shape[1]) > self.jacobian_samples:
            raise ValueError(
                "fixed decoder Jacobian sample count exceeds configured cap"
            )
        actual_jacobian_samples = int(fixed_singular.shape[1])
        if (
            _integer(diagnostic.get("jacobian_samples"), label="jacobian_samples")
            != actual_jacobian_samples
        ):
            raise ValueError("Action Flow diagnostic jacobian_samples is inconsistent")
        if any(
            int(_tensor(diagnostic, key, ndim=2).shape[0]) != actual_jacobian_samples
            for _, key in jacobian_specs
        ):
            raise ValueError("Action Flow Jacobian sample counts do not align")
        decoder_statistics["fixed"] = {}
        for index, label in enumerate(self.noise_labels):
            summary = self._jacobian_summary(fixed_singular[index])
            decoder_statistics["fixed"][label] = summary
            prefix = f"Valid/ActionFlow/DecoderJacobian/fixed/{label}"
            for metric_name in (
                "spectral_norm_mean",
                "smallest_singular_mean",
                "frobenius_norm_mean",
                "condition_number_mean",
                "numerical_rank_mean",
            ):
                add_metric(
                    f"{prefix}/{metric_name}", source_label, summary[metric_name]
                )
        computed["decoder_jacobian_statistics"] = decoder_statistics

        noise_statistics = self._decoded_noise_summary(decoded_noise)
        computed["decoded_full_noise_statistics"] = noise_statistics
        for metric_name in (
            "token_radius_mean",
            "token_radius_std",
            "token_radius_max",
            "chunk_rms_radius_mean",
            "chunk_rms_radius_std",
        ):
            add_metric(
                f"Valid/ActionFlow/DecodedFullNoise/{metric_name}",
                source_label,
                noise_statistics[metric_name],
            )
        for coordinate in range(action_dim):
            for metric_name in (
                "coordinate_mean",
                "coordinate_std",
                "coordinate_second_moment",
            ):
                add_metric(
                    f"Valid/ActionFlow/DecodedFullNoise/{metric_name}/dim_{coordinate:02d}",
                    source_label,
                    noise_statistics[metric_name][coordinate],
                )

        reconstruction_mse = (reconstruction - target).square().mean(dim=(-2, -1))
        add_metric(
            "Valid/ActionFlow/CleanReconstructionMSE",
            source_label,
            reconstruction_mse.mean(),
        )
        trajectory_latent_mse = (
            (trajectory - clean.unsqueeze(0)).square().mean(dim=(-2, -1))
        )
        trajectory_decoded_mse = (
            (decoded_trajectory - target.unsqueeze(0)).square().mean(dim=(-2, -1))
        )
        reconstruction_native_mse = None
        trajectory_decoded_native_mse = None
        fixed_decoded_native_mse = None
        if native_error_fn is not None:
            reconstruction_native_mse = self._native_error_values(
                native_error_fn,
                reconstruction,
                target,
                expected_shape=(batch_size,),
                label="clean reconstruction",
            )
            add_metric(
                "Valid/ActionFlow/CleanReconstructionNativeMSE",
                source_label,
                reconstruction_native_mse.mean(),
            )
            trajectory_decoded_native_mse = self._native_error_values(
                native_error_fn,
                decoded_trajectory,
                target.unsqueeze(0).expand_as(decoded_trajectory),
                expected_shape=(inference_steps + 1, batch_size),
                label="denoising trajectory",
            )
            fixed_decoded_native_mse = {
                key: self._native_error_values(
                    native_error_fn,
                    value,
                    target.unsqueeze(0).expand_as(value),
                    expected_shape=(level_count, batch_size),
                    label=key,
                )
                for key, value in fixed_decoded.items()
            }
        trajectory_times = torch.linspace(
            1.0,
            0.0,
            inference_steps + 1,
            device=trajectory.device,
            dtype=torch.float32,
        )
        for step, time in enumerate(trajectory_times.tolist()):
            label = _noise_label(time)
            add_metric(
                f"Valid/ActionFlow/DenoisingTrajectory/LatentMSE/{label}",
                source_label,
                trajectory_latent_mse[step].mean(),
            )
            add_metric(
                f"Valid/ActionFlow/DenoisingTrajectory/DecodedMSE/{label}",
                source_label,
                trajectory_decoded_mse[step].mean(),
            )
            if trajectory_decoded_native_mse is not None:
                add_metric(
                    f"Valid/ActionFlow/DenoisingTrajectory/DecodedNativeMSE/{label}",
                    source_label,
                    trajectory_decoded_native_mse[step].mean(),
                )

        fixed_metrics: dict[str, dict[str, torch.Tensor]] = {}
        final_cosine = []
        clean_flat = clean.reshape(batch_size, -1)
        clean_norm = torch.linalg.vector_norm(clean_flat, dim=1)
        if bool((clean_norm <= 0.0).any()):
            raise ValueError("final-latent cosine has a zero clean-latent norm")
        for index, label in enumerate(self.noise_labels):
            values = {
                "state_latent_mse": (
                    fixed_latents["latent/fixed_states"][index] - clean
                )
                .square()
                .mean(dim=(-2, -1)),
                "predicted_clean_latent_mse": (
                    fixed_latents["latent/predicted_clean"][index] - clean
                )
                .square()
                .mean(dim=(-2, -1)),
                "final_latent_mse": (fixed_latents["latent/fixed_final"][index] - clean)
                .square()
                .mean(dim=(-2, -1)),
                "state_decoded_mse": (
                    fixed_decoded["decoded/fixed_states"][index] - target
                )
                .square()
                .mean(dim=(-2, -1)),
                "predicted_clean_decoded_mse": (
                    fixed_decoded["decoded/predicted_clean"][index] - target
                )
                .square()
                .mean(dim=(-2, -1)),
                "final_decoded_mse": (
                    fixed_decoded["decoded/fixed_final"][index] - target
                )
                .square()
                .mean(dim=(-2, -1)),
            }
            if fixed_decoded_native_mse is not None:
                values.update(
                    {
                        "state_decoded_native_mse": fixed_decoded_native_mse[
                            "decoded/fixed_states"
                        ][index],
                        "predicted_clean_decoded_native_mse": (
                            fixed_decoded_native_mse["decoded/predicted_clean"][index]
                        ),
                        "final_decoded_native_mse": fixed_decoded_native_mse[
                            "decoded/fixed_final"
                        ][index],
                    }
                )
            final = fixed_latents["latent/fixed_final"][index].reshape(batch_size, -1)
            final_norm = torch.linalg.vector_norm(final, dim=1)
            if bool((final_norm <= 0.0).any()):
                raise ValueError(f"final-latent cosine has a zero norm at {label}")
            cosine = (final * clean_flat).sum(dim=1) / (final_norm * clean_norm)
            if not bool(torch.isfinite(cosine).all()):
                raise ValueError(f"final-latent cosine is non-finite at {label}")
            values["final_latent_cosine"] = cosine
            final_cosine.append(cosine)
            fixed_metrics[label] = values
            for metric_name, metric_values in values.items():
                prefix = (
                    "Valid/ActionFlow/Alignment/FinalLatentCosine"
                    if metric_name == "final_latent_cosine"
                    else f"Valid/ActionFlow/FixedLevel/{metric_name}"
                )
                add_metric(f"{prefix}/{label}", source_label, metric_values.mean())
                add_metric(
                    f"{prefix}Std/{label}",
                    source_label,
                    metric_values.std(unbiased=False),
                )
        computed["fixed_level_metrics"] = fixed_metrics
        computed["final_latent_cosine_by_condition"] = torch.stack(final_cosine)
        computed["trajectory_times"] = trajectory_times
        computed["trajectory_latent_mse_by_condition"] = trajectory_latent_mse
        computed["trajectory_decoded_mse_by_condition"] = trajectory_decoded_mse
        computed["clean_reconstruction_mse_by_condition"] = reconstruction_mse
        if reconstruction_native_mse is not None:
            computed["clean_reconstruction_native_mse_by_condition"] = (
                reconstruction_native_mse
            )
            computed["trajectory_decoded_native_mse_by_condition"] = (
                trajectory_decoded_native_mse
            )

        alignment = {}
        if self.capture_activations:
            encoder_activations = _tensor(
                diagnostic, "activation/encoder_blocks", ndim=4
            ).float()
            encoder_indices = _tensor(
                diagnostic, "activation/encoder_block_indices", ndim=1
            ).long()
            field_activations = _tensor(
                diagnostic, "activation/field_blocks", ndim=5
            ).float()
            field_indices = _tensor(
                diagnostic, "activation/field_block_indices", ndim=1
            ).long()
            if encoder_activations.shape[:3] != (
                encoder_indices.numel(),
                batch_size,
                horizon,
            ) or field_activations.shape[:4] != (
                level_count,
                field_indices.numel(),
                batch_size,
                horizon,
            ):
                raise ValueError("Action Flow activation tensor shapes do not align")
            encoder_positions = {
                int(layer): position for position, layer in enumerate(encoder_indices)
            }
            field_positions = {
                int(layer): position for position, layer in enumerate(field_indices)
            }
            if (
                len(encoder_positions) != encoder_indices.numel()
                or len(field_positions) != field_indices.numel()
            ):
                raise ValueError("Action Flow activation block indices must be unique")
            if not 2 <= self.cknna_k < batch_size:
                raise ValueError(
                    "Action Flow CKNNA k is incompatible with returned paired rows"
                )
            for encoder_layer, field_layer in self.activation_layer_map:
                if (
                    encoder_layer not in encoder_positions
                    or field_layer not in field_positions
                ):
                    raise ValueError(
                        "configured Action Flow activation layer is missing from model output"
                    )
                pair_label = f"encoder_{encoder_layer:02d}__field_{field_layer:02d}"
                left = encoder_activations[encoder_positions[encoder_layer]].mean(dim=1)
                alignment[pair_label] = {}
                for level_index, level_label in enumerate(self.noise_labels):
                    right = field_activations[
                        level_index, field_positions[field_layer]
                    ].mean(dim=1)
                    cka = self._centered_linear_cka(left, right)
                    cknna = self._cknna(left, right, topk=self.cknna_k)
                    add_metric(
                        f"Valid/ActionFlow/Alignment/CKA/{pair_label}/{level_label}",
                        source_label,
                        cka,
                    )
                    add_metric(
                        f"Valid/ActionFlow/Alignment/CKNNA/{pair_label}/{level_label}",
                        source_label,
                        cknna,
                    )
                    alignment[pair_label][level_label] = {
                        "cka": cka,
                        "cknna": cknna,
                    }
        computed["alignment"] = alignment

        return self._cpu_tree(
            {
                "source_id": str(source_id),
                "diagnostic": diagnostic,
                "computed": computed,
            }
        )

    @staticmethod
    def _write_immutable_artifact(destination: Path, payload: Mapping[str, Any]) -> str:
        sidecar = Path(f"{destination}.sha256")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.temporary"
        )
        sidecar_temporary = sidecar.with_name(
            f".{sidecar.name}.{os.getpid()}.temporary"
        )
        for path in (destination, sidecar, temporary, sidecar_temporary):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        try:
            torch.save(dict(payload), temporary)
            artifact_sha256 = _sha256_file(temporary)
            sidecar_payload = {
                "algorithm": "sha256",
                "artifact": destination.name,
                "sha256": artifact_sha256,
                "identity_sha256": payload["identity_sha256"],
            }
            sidecar_temporary.write_text(
                json.dumps(sidecar_payload, sort_keys=True, indent=2) + "\n"
            )
            os.link(temporary, destination)
            try:
                os.link(sidecar_temporary, sidecar)
            except Exception:
                destination.unlink()
                raise
        finally:
            temporary.unlink(missing_ok=True)
            sidecar_temporary.unlink(missing_ok=True)
        return artifact_sha256

    def run(
        self,
        *,
        model: Any,
        batch: Mapping[Any, Mapping[str, Any]],
        batch_idx: int,
        rank: int,
        epoch: int,
        global_step: int,
        precision: Any,
        source_labels: Mapping[Any, str],
        cuda_devices: Sequence[int] = (),
        native_error_fns: Mapping[Any, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.should_run():
            return {}
        if not hasattr(model, "forward_action_flow_diagnostics"):
            raise AttributeError(
                "Action Flow diagnostics require "
                "model.forward_action_flow_diagnostics(...)"
            )
        if set(source_labels) != set(batch):
            raise ValueError("Action Flow diagnostic source labels do not match batch")
        labels = [str(source_labels[source]) for source in batch]
        if any(not label for label in labels) or len(labels) != len(set(labels)):
            raise ValueError("Action Flow diagnostic source labels must be unique")
        if self.native_error_enabled:
            if not isinstance(native_error_fns, Mapping) or set(
                native_error_fns
            ) != set(batch):
                raise ValueError(
                    "enabled Action Flow native error requires one adapter per source"
                )
            if any(not callable(native_error_fns[source]) for source in batch):
                raise TypeError("Action Flow native error adapters must be callable")
        elif native_error_fns is not None:
            raise ValueError(
                "native error adapters were supplied without a configured contract"
            )
        rank = int(rank)
        if not 0 <= rank < len(self.noise_seeds):
            raise ValueError(f"No Action Flow diagnostic noise seed for rank {rank}")
        noise_seed = self.noise_seeds[rank]
        with torch.random.fork_rng(devices=list(cuda_devices)):
            torch.manual_seed(noise_seed)
            with torch.inference_mode(False):
                diagnostics = model.forward_action_flow_diagnostics(
                    batch,
                    raw_noise_levels=self.noise_levels,
                    noise_seed=noise_seed,
                    max_samples=self.max_samples,
                    jacobian_samples=self.jacobian_samples,
                    capture_activations=self.capture_activations,
                )
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(batch):
            raise ValueError(
                "Action Flow diagnostic model outputs do not match batch sources"
            )

        metrics: dict[str, torch.Tensor] = {}
        macro: dict[str, list[torch.Tensor]] = defaultdict(list)

        def add_metric(base: str, source_label: str, value: torch.Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.detach().float()
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError(
                    f"invalid Action Flow diagnostic metric {base}/{source_label}"
                )
            metrics[f"{base}/{source_label}"] = value
            macro[base].append(value)

        artifact_sources = {}
        for source_id in batch:
            label = str(source_labels[source_id])
            artifact_sources[label] = self._analyze_source(
                diagnostics[source_id],
                source_id=source_id,
                source_label=label,
                add_metric=add_metric,
                expected_noise_seed=noise_seed,
                native_error_fn=(
                    None if native_error_fns is None else native_error_fns[source_id]
                ),
            )
        for base, values in macro.items():
            metrics[base] = torch.stack(values).mean()

        destination = (
            self.artifact_root
            / f"epoch-{int(epoch)}-step-{int(global_step)}"
            / f"rank-{rank}-batch-{int(batch_idx)}.pt"
        )
        statistics = {
            "latent_covariance": "rows_are_condition_times_horizon_tokens",
            "effective_rank": "exp_shannon_entropy_normalized_eigenvalues",
            "decoder_jacobian": ("full_chunk_jacobian_singular_values_per_condition"),
            "trajectory_error": "paired_diagnostic_not_distributional_score",
            "activation_pooling": "mean_horizon_tokens_then_paired_conditions",
            "cka": "biased_centered_linear_cka",
            "cknna": "unbiased_hsic_intersection_knn",
            "cknna_k": self.cknna_k if self.capture_activations else None,
        }
        if self.native_error is not None:
            statistics["native_action_error"] = self.native_error
        payload = {
            "schema_version": 1,
            "metric": "ActionFlowValidationDiagnostics",
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "rank": rank,
            "batch_idx": int(batch_idx),
            "precision": None if precision is None else str(precision),
            "noise_seed": noise_seed,
            "noise_seed_bank_sha256": self.noise_seed_bank_sha256,
            "raw_noise_levels": self.noise_levels,
            "noise_labels": self.noise_labels,
            "sampler": {
                "name": "reverse_euler",
                "fixed_level_coupling": (
                    "one_paired_clean_latent_and_one_base_gaussian_per_condition"
                ),
                "off_grid_rule": (
                    "partial_step_to_next_lower_canonical_grid_then_fixed_steps"
                ),
            },
            "statistics": statistics,
            "validation_view": self.validation_view,
            "provenance": self.provenance,
            "sources": artifact_sources,
        }
        self._write_immutable_artifact(destination, payload)
        self.batches_done += 1
        return metrics
