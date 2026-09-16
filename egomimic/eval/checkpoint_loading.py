"""Strict checkpoint-state selection for Pipeline algorithms."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch

_NETS_PREFIX = "nets."


def _checkpoint_state_dict(checkpoint: dict) -> Mapping[str, torch.Tensor]:
    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint has no state_dict")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint state_dict is not a mapping")
    return state_dict


def _extract_nets_state(
    source: Mapping[str, torch.Tensor], *, source_name: str
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(source, Mapping):
        raise TypeError(f"checkpoint {source_name} is not a mapping")
    return OrderedDict(
        (str(key)[len(_NETS_PREFIX) :], value)
        for key, value in source.items()
        if str(key).startswith(_NETS_PREFIX)
    )


def extract_pipeline_nets_state(checkpoint: dict, use_ema: bool = False):
    """Extract the exact current Pipeline parameter namespace."""
    if use_ema:
        if "ema_state_dict" not in checkpoint:
            raise KeyError("checkpoint has no EMA state")
        source = checkpoint["ema_state_dict"]
        source_name = "ema_state_dict"
    else:
        source = _checkpoint_state_dict(checkpoint)
        source_name = "state_dict"
    extracted = _extract_nets_state(source, source_name=source_name)
    if not extracted:
        raise ValueError(f"{source_name} contains no Pipeline nets parameters")
    return extracted


def _require_exact_keys(actual, expected, *, label: str) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)[:8]} "
            f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )


def _require_compatible_tensors(actual, expected, *, label: str) -> None:
    """Reject incompatible state before ``load_state_dict`` can mutate a model."""

    incompatible = []
    for key in expected:
        source_value = actual[key]
        target_value = expected[key]
        if not isinstance(source_value, torch.Tensor):
            incompatible.append(f"{key}: source is {type(source_value).__name__}")
        elif tuple(source_value.shape) != tuple(target_value.shape):
            incompatible.append(
                f"{key}: source shape {tuple(source_value.shape)} != "
                f"target shape {tuple(target_value.shape)}"
            )
    if incompatible:
        raise ValueError(f"{label} tensor mismatch: " + "; ".join(incompatible[:8]))


def strict_load_pipeline_checkpoint(algo, checkpoint: dict, use_ema: bool = False):
    """Strictly load current online state, optionally overlaying current EMA."""
    online = extract_pipeline_nets_state(checkpoint)
    expected = algo.nets.state_dict()
    _require_exact_keys(online, expected, label="Pipeline checkpoint")
    _require_compatible_tensors(online, expected, label="Pipeline checkpoint")

    state = OrderedDict(online)
    if use_ema:
        averaged = extract_pipeline_nets_state(checkpoint, use_ema=True)
        parameters = dict(algo.nets.named_parameters())
        parameter_keys = set(parameters)
        _require_exact_keys(averaged, parameter_keys, label="EMA parameter")
        _require_compatible_tensors(averaged, parameters, label="EMA parameter")
        state.update((key, averaged[key]) for key in parameter_keys)

    algo.nets.load_state_dict(state, strict=True)
    return algo
