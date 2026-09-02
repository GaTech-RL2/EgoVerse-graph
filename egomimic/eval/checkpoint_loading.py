"""Strict checkpoint-state selection for Pipeline algorithms."""

from __future__ import annotations

from collections import OrderedDict

import torch


def extract_pipeline_nets_state(checkpoint: dict, use_ema: bool = False):
    """Return one unambiguous ``algo.nets`` state, rejecting partial aliases."""
    source_name = "ema_state_dict" if use_ema else "state_dict"
    if source_name not in checkpoint:
        raise KeyError(f"checkpoint has no {source_name}")
    extracted = OrderedDict()
    for raw_key, value in checkpoint[source_name].items():
        key = str(raw_key)
        if key.startswith("model.nets."):
            key = key[len("model.nets.") :]
        elif key.startswith("nets."):
            key = key[len("nets.") :]
        else:
            continue
        if key in extracted and not torch.equal(extracted[key], value):
            raise ValueError(f"conflicting checkpoint aliases for {key!r}")
        extracted[key] = value
    if not extracted:
        raise ValueError(f"{source_name} contains no Pipeline nets parameters")
    return extracted


def strict_load_pipeline_checkpoint(algo, checkpoint: dict, use_ema: bool = False):
    """Strictly load online state, optionally overlaying parameter-only EMA."""
    online = extract_pipeline_nets_state(checkpoint, use_ema=False)
    expected = set(algo.nets.state_dict())
    if set(online) != expected:
        raise ValueError(
            "Pipeline checkpoint key mismatch: "
            f"missing={sorted(expected - set(online))[:8]} "
            f"unexpected={sorted(set(online) - expected)[:8]}"
        )
    state = online
    if use_ema:
        averaged = extract_pipeline_nets_state(checkpoint, use_ema=True)
        parameter_keys = set(dict(algo.nets.named_parameters()))
        if set(averaged) != parameter_keys:
            raise ValueError(
                "EMA parameter key mismatch: "
                f"missing={sorted(parameter_keys - set(averaged))[:8]} "
                f"unexpected={sorted(set(averaged) - parameter_keys)[:8]}"
            )
        state = OrderedDict(online)
        # EMA intentionally contains parameters only. Registered buffers must
        # retain the checkpoint's online values, never their initialized values.
        state.update(averaged)
    algo.nets.load_state_dict(state, strict=True)
    return algo
