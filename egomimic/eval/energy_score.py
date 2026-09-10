"""Deterministic, block-balanced Energy Score for stochastic action chunks."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping

import torch

DEFAULT_PLANAR_BLOCKS = ((0, 2), (2, 4), (4, 5))

USOCKET_ENERGY_DISTANCE_TYPE = "usocket_normalized_xy_wrapped_theta_v1"
USOCKET_NATIVE_DECODER = "egomimic.pipeline.pushshapes.USocketRotVecNativeDecoder"
USOCKET_ENERGY_DISTANCE_CONFIG = {
    "type": USOCKET_ENERGY_DISTANCE_TYPE,
    "complete_normalized_chunk_shape": (16, 4),
    "normalized_translation_indices": (0, 1),
    "native_theta_index": 2,
    "rotation_scale_radians": math.pi,
    "semantic_weights": {"translation": 0.5, "rotation": 0.5},
    "native_decoder": USOCKET_NATIVE_DECODER,
}


def normalize_usocket_energy_distance_config(value: Mapping) -> dict:
    """Validate and canonicalize the one supported typed USocket distance.

    This intentionally accepts no undeclared knobs.  A metric configuration is
    part of an immutable validation identity, so a spelling-compatible but
    semantically different mapping must fail rather than silently change a
    curve.
    """
    if not isinstance(value, Mapping):
        raise TypeError("USocket EnergyScore distance must be a mapping")
    expected_keys = set(USOCKET_ENERGY_DISTANCE_CONFIG)
    actual_keys = {str(key) for key in value}
    if actual_keys != expected_keys:
        raise ValueError(
            "USocket EnergyScore distance keys differ: "
            f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
    weights = value["semantic_weights"]
    if not isinstance(weights, Mapping):
        raise TypeError("USocket EnergyScore semantic_weights must be a mapping")
    expected_weight_keys = {"translation", "rotation"}
    actual_weight_keys = {str(key) for key in weights}
    if actual_weight_keys != expected_weight_keys:
        raise ValueError(
            "USocket EnergyScore semantic weight keys differ: "
            f"expected {sorted(expected_weight_keys)}, got "
            f"{sorted(actual_weight_keys)}"
        )
    normalized = {
        "type": str(value["type"]),
        "complete_normalized_chunk_shape": tuple(
            int(item) for item in value["complete_normalized_chunk_shape"]
        ),
        "normalized_translation_indices": tuple(
            int(item) for item in value["normalized_translation_indices"]
        ),
        "native_theta_index": int(value["native_theta_index"]),
        "rotation_scale_radians": float(value["rotation_scale_radians"]),
        "semantic_weights": {
            "translation": float(weights.get("translation", float("nan"))),
            "rotation": float(weights.get("rotation", float("nan"))),
        },
        "native_decoder": str(value["native_decoder"]),
    }
    if normalized != USOCKET_ENERGY_DISTANCE_CONFIG:
        raise ValueError(
            "unsupported USocket EnergyScore distance contract: "
            f"expected {USOCKET_ENERGY_DISTANCE_CONFIG!r}, got {normalized!r}"
        )
    return copy.deepcopy(normalized)


def usocket_energy_distance_metadata(config: Mapping) -> dict:
    """Return the exact human- and machine-readable distance definition."""
    normalized = normalize_usocket_energy_distance_config(config)
    translation_weight = normalized["semantic_weights"]["translation"]
    rotation_weight = normalized["semantic_weights"]["rotation"]
    return {
        "type": normalized["type"],
        "space": "normalized_xy_plus_decoded_native_theta",
        "formula": (
            "0.5*rms(normalized_xy_delta_over_horizon_and_coordinates)"
            "+0.5*rms(wrap_radians(native_theta_delta)/pi_over_horizon)"
        ),
        "coverage": "all_16_steps_and_all_semantic_action_dofs",
        "complete_normalized_chunk_shape": normalized[
            "complete_normalized_chunk_shape"
        ],
        "normalized_action_representation": "x_y_cos_theta_sin_theta",
        "native_action_representation": "x_y_theta_radians",
        "normalizer": "bound_train_only_evaluator_normalizer",
        "native_decoder": normalized["native_decoder"],
        "translation": {
            "normalized_indices": normalized["normalized_translation_indices"],
            "weight": translation_weight,
            "reduction": "rms_over_horizon_and_coordinates",
        },
        "rotation": {
            "native_theta_index": normalized["native_theta_index"],
            "wrap_period_radians": 2.0 * math.pi,
            "scale_radians": normalized["rotation_scale_radians"],
            "weight": rotation_weight,
            "reduction": "rms_over_horizon",
        },
        "combine": "equal_semantic_weighted_sum",
    }


def semantic_chunk_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    blocks=DEFAULT_PLANAR_BLOCKS,
) -> torch.Tensor:
    """Average equal-weight RMS distances over declared semantic blocks."""
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError(
            f"action chunk shapes disagree: {left.shape[-2:]} vs {right.shape[-2:]}"
        )
    width = left.shape[-1]
    normalized_blocks = tuple(tuple(map(int, block)) for block in blocks)
    covered = [index for start, end in normalized_blocks for index in range(start, end)]
    if len(covered) != width or set(covered) != set(range(width)):
        raise ValueError(
            "semantic blocks must partition every action channel exactly once; "
            f"got {normalized_blocks} for width {width}"
        )
    terms = []
    for start, end in normalized_blocks:
        if not 0 <= int(start) < int(end) <= width:
            raise ValueError(f"invalid semantic block {(start, end)} for width {width}")
        error = (
            left[..., :, int(start) : int(end)] - right[..., :, int(start) : int(end)]
        )
        terms.append(error.square().mean(dim=(-2, -1)).sqrt())
    if not terms:
        raise ValueError("at least one semantic block is required")
    return torch.stack(terms).mean(dim=0)


def usocket_xy_theta_chunk_distance(
    normalized_left: torch.Tensor,
    normalized_right: torch.Tensor,
    native_left: torch.Tensor,
    native_right: torch.Tensor,
    *,
    config: Mapping,
) -> torch.Tensor:
    """Balanced H16 distance using normalized XY and decoded circular theta."""
    contract = normalize_usocket_energy_distance_config(config)
    expected_shape = contract["complete_normalized_chunk_shape"]
    if (
        tuple(normalized_left.shape[-2:]) != expected_shape
        or tuple(normalized_right.shape[-2:]) != expected_shape
    ):
        raise ValueError(
            "typed USocket EnergyScore requires complete normalized chunks with "
            f"shape {expected_shape}, got {normalized_left.shape[-2:]} and "
            f"{normalized_right.shape[-2:]}"
        )
    horizon, _ = expected_shape
    theta_index = contract["native_theta_index"]
    for label, normalized, native in (
        ("left", normalized_left, native_left),
        ("right", normalized_right, native_right),
    ):
        if tuple(native.shape[-2:]) != (horizon, 3):
            raise ValueError(
                f"typed USocket EnergyScore {label} native chunk has wrong shape: "
                f"{native.shape[-2:]}"
            )
        if tuple(native.shape[:-2]) != tuple(normalized.shape[:-2]):
            raise ValueError(
                f"typed USocket EnergyScore {label} normalized/native leading "
                f"shapes differ: {normalized.shape[:-2]} != {native.shape[:-2]}"
            )

    indices = contract["normalized_translation_indices"]
    translation_error = (
        normalized_left[..., :, indices] - normalized_right[..., :, indices]
    )
    translation = translation_error.square().mean(dim=(-2, -1)).sqrt()

    theta_error = native_left[..., :, theta_index] - native_right[..., :, theta_index]
    wrapped_theta_error = torch.atan2(torch.sin(theta_error), torch.cos(theta_error))
    rotation = (
        (wrapped_theta_error / contract["rotation_scale_radians"])
        .square()
        .mean(dim=-1)
        .sqrt()
    )
    weights = contract["semantic_weights"]
    return weights["translation"] * translation + weights["rotation"] * rotation


def energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    blocks=DEFAULT_PLANAR_BLOCKS,
    *,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute per-condition Energy Score using ordered distinct sample pairs.

    ``samples`` is ``(K, B, H, D)`` and ``target`` is ``(B, H, D)``. Returned
    scalar values are macro-averages across the B frozen validation conditions;
    ``*_by_condition`` values retain one diagnostic per condition.
    """
    if samples.ndim != 4 or target.ndim != 3 or samples.shape[1:] != target.shape:
        raise ValueError(
            f"expected samples (K,B,H,D) and target (B,H,D), got "
            f"{samples.shape} and {target.shape}"
        )
    sample_count = samples.shape[0]
    if sample_count < 2:
        raise ValueError("Energy Score requires at least two samples")
    distance = (
        (lambda left, right: semantic_chunk_distance(left, right, blocks))
        if distance_fn is None
        else distance_fn
    )
    accuracy_distances = distance(samples, target.unsqueeze(0))
    if tuple(accuracy_distances.shape) != (sample_count, target.shape[0]):
        raise ValueError(
            "Energy Score distance returned the wrong accuracy shape: "
            f"{tuple(accuracy_distances.shape)}"
        )
    accuracy_by_condition = accuracy_distances.mean(dim=0)
    pair_distance = distance(samples[:, None], samples[None, :])
    if tuple(pair_distance.shape) != (
        sample_count,
        sample_count,
        target.shape[0],
    ):
        raise ValueError(
            "Energy Score distance returned the wrong pairwise shape: "
            f"{tuple(pair_distance.shape)}"
        )
    mask = ~torch.eye(sample_count, dtype=torch.bool, device=samples.device)
    diversity_by_condition = (
        pair_distance[mask].reshape(-1, target.shape[0]).mean(dim=0)
    )
    score_by_condition = accuracy_by_condition - 0.5 * diversity_by_condition
    return {
        "score": score_by_condition.mean(),
        "accuracy": accuracy_by_condition.mean(),
        "diversity": diversity_by_condition.mean(),
        "score_by_condition": score_by_condition,
        "accuracy_by_condition": accuracy_by_condition,
        "diversity_by_condition": diversity_by_condition,
    }
