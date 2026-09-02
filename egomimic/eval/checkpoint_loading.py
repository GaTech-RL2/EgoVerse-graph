"""Strict checkpoint-state selection for Pipeline algorithms."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch


def _extract_prefixed_state(
    source: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...],
    *,
    source_name: str,
):
    """Strip equivalent wrapper prefixes while rejecting conflicting aliases."""
    if not isinstance(source, Mapping):
        raise TypeError(f"checkpoint {source_name} is not a mapping")
    extracted = OrderedDict()
    origins = {}
    for raw_key, value in source.items():
        key = str(raw_key)
        prefix = next((item for item in prefixes if key.startswith(item)), None)
        if prefix is None:
            continue
        normalized_key = key[len(prefix) :]
        if normalized_key in extracted:
            if not torch.equal(extracted[normalized_key], value):
                raise ValueError(
                    f"conflicting checkpoint aliases in {source_name} for "
                    f"{normalized_key!r}: "
                    f"{origins[normalized_key]!r} and {key!r}"
                )
            continue
        extracted[normalized_key] = value
        origins[normalized_key] = key
    return extracted


def _checkpoint_state_dict(checkpoint: dict):
    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint has no state_dict")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint state_dict is not a mapping")
    return state_dict


def _extract_ema_representation(checkpoint: dict, state_dict):
    """Return the sole EMA tree and whether it contains registered buffers."""
    registered = _extract_prefixed_state(
        state_dict,
        ("model.ema_nets.", "ema_nets."),
        source_name="registered EMA state_dict",
    )
    has_callback_ema = "ema_state_dict" in checkpoint
    callback = None
    if has_callback_ema:
        callback = _extract_prefixed_state(
            checkpoint["ema_state_dict"],
            ("model.nets.", "nets."),
            source_name="ema_state_dict",
        )

    if registered and has_callback_ema:
        raise ValueError(
            "checkpoint contains both registered EMA state and ema_state_dict"
        )
    if registered:
        return registered, "registered EMA state_dict", True
    if has_callback_ema:
        if not callback:
            raise ValueError("ema_state_dict contains no Pipeline nets parameters")
        return callback, "ema_state_dict", False
    return None, None, False


def extract_pipeline_nets_state(checkpoint: dict, use_ema: bool = False):
    """Return one unambiguous ``algo.nets`` state, rejecting alias conflicts."""
    state_dict = _checkpoint_state_dict(checkpoint)
    if use_ema:
        extracted, _, _ = _extract_ema_representation(checkpoint, state_dict)
        if extracted is None:
            raise KeyError("checkpoint has no EMA state")
        return extracted

    extracted = _extract_prefixed_state(
        state_dict,
        ("model.nets.", "nets."),
        source_name="state_dict",
    )
    if not extracted:
        raise ValueError("state_dict contains no Pipeline nets parameters")
    return extracted


def _rewrite_state_prefixes(
    state: Mapping[str, torch.Tensor],
    prefix_rewrites: Mapping[str, str] | None,
    *,
    source_name: str,
):
    """Apply one unambiguous prefix rewrite per key without losing entries."""
    if not prefix_rewrites:
        return OrderedDict(state)

    rewrites = []
    for raw_source, raw_destination in prefix_rewrites.items():
        if not isinstance(raw_source, str) or not isinstance(raw_destination, str):
            raise TypeError("checkpoint prefix rewrites must map strings to strings")
        if not raw_source:
            raise ValueError("checkpoint rewrite source prefixes must be non-empty")
        rewrites.append((raw_source, raw_destination))

    rewritten = OrderedDict()
    origins = {}
    for key, value in state.items():
        matches = [rewrite for rewrite in rewrites if key.startswith(rewrite[0])]
        if len(matches) > 1:
            sources = [source for source, _ in matches]
            raise ValueError(
                f"ambiguous {source_name} prefix rewrite for {key!r}: {sources}"
            )
        if matches:
            source, destination = matches[0]
            destination_key = destination + key[len(source) :]
        else:
            destination_key = key
        if destination_key in rewritten:
            raise ValueError(
                f"{source_name} prefix rewrite collision for {destination_key!r}: "
                f"{origins[destination_key]!r} and {key!r}"
            )
        rewritten[destination_key] = value
        origins[destination_key] = key
    return rewritten


def strict_load_pipeline_checkpoint(
    algo,
    checkpoint: dict,
    use_ema: bool = False,
    prefix_rewrites: Mapping[str, str] | None = None,
):
    """Strictly load online state or one exact supported EMA representation."""
    state_dict = _checkpoint_state_dict(checkpoint)
    online = _rewrite_state_prefixes(
        _extract_prefixed_state(
            state_dict,
            ("model.nets.", "nets."),
            source_name="state_dict",
        ),
        prefix_rewrites,
        source_name="state_dict",
    )
    if not online:
        raise ValueError("state_dict contains no Pipeline nets parameters")
    expected = set(algo.nets.state_dict())
    if set(online) != expected:
        raise ValueError(
            "Pipeline checkpoint key mismatch: "
            f"missing={sorted(expected - set(online))[:8]} "
            f"unexpected={sorted(set(online) - expected)[:8]}"
        )
    state = online
    if use_ema:
        averaged, ema_source_name, registered_ema = _extract_ema_representation(
            checkpoint, state_dict
        )
        if averaged is None:
            raise KeyError("checkpoint has no EMA state")
        averaged = _rewrite_state_prefixes(
            averaged,
            prefix_rewrites,
            source_name=ema_source_name,
        )
        parameter_keys = set(dict(algo.nets.named_parameters()))
        required_ema_keys = expected if registered_ema else parameter_keys
        if set(averaged) != required_ema_keys:
            mismatch_name = (
                "registered EMA state_dict key mismatch"
                if registered_ema
                else "EMA parameter key mismatch"
            )
            raise ValueError(
                f"{mismatch_name}: "
                f"missing={sorted(required_ema_keys - set(averaged))[:8]} "
                f"unexpected={sorted(set(averaged) - required_ema_keys)[:8]}"
            )
        if registered_ema:
            # The legacy wrapper evaluated its registered EMA ModuleDict,
            # including that tree's buffers, so reproduce the complete state.
            state = averaged
        else:
            state = OrderedDict(online)
            # Callback EMA is parameter-only; retain checkpointed online buffers.
            state.update((key, averaged[key]) for key in parameter_keys)
    algo.nets.load_state_dict(state, strict=True)
    return algo
