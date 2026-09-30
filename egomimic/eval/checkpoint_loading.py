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


def strict_load_pipeline_checkpoint(algo, checkpoint: dict, use_ema: bool = False):
    """Strictly load current online state, optionally overlaying current EMA."""
    online = extract_pipeline_nets_state(checkpoint)
    expected = algo.nets.state_dict()
    _require_exact_keys(online, expected, label="Pipeline checkpoint")

    state = OrderedDict(online)
    if use_ema:
        averaged = extract_pipeline_nets_state(checkpoint, use_ema=True)
        parameter_keys = set(dict(algo.nets.named_parameters()))
        _require_exact_keys(averaged, parameter_keys, label="EMA parameter")
        state.update((key, averaged[key]) for key in parameter_keys)

    algo.nets.load_state_dict(state, strict=True)
    return algo


def init_pipeline_weights_from_checkpoint(algo, checkpoint: dict, use_ema: bool = False):
    """Weights-only initialisation for fine-tuning.

    Loads a checkpoint's Pipeline weights into a freshly built model and returns the
    tensors it could NOT carry over. Unlike :func:`strict_load_pipeline_checkpoint`
    this tolerates a *shape* mismatch on individual tensors, because a fine-tune may
    change the action layout -- e.g. a 14-D cartesian chunk to a 16-D arc token, which
    reshapes only the flow denoiser's ``proj_u`` / ``proj_d``. Those tensors keep their
    freshly initialised values and are reported so the caller can log them. A
    *key-set* mismatch is still a hard error: it means a different architecture.

    This deliberately restores no optimizer, scheduler or ``global_step``. Passing the
    checkpoint as ``cfg.ckpt_path`` instead would make Lightning resume the whole
    training state, which for a 210k-step checkpoint under a shorter fine-tune budget
    stops immediately.
    """
    online = extract_pipeline_nets_state(checkpoint, use_ema=use_ema)
    expected = algo.nets.state_dict()
    _require_exact_keys(online, expected, label="Fine-tune checkpoint")

    state = OrderedDict()
    reinitialised = []
    for key, want in expected.items():
        got = online[key]
        if tuple(got.shape) == tuple(want.shape):
            state[key] = got
        else:
            state[key] = want
            reinitialised.append((key, tuple(got.shape), tuple(want.shape)))

    algo.nets.load_state_dict(state, strict=True)
    return algo, reinitialised
