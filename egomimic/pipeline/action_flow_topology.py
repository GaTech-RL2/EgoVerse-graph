"""Structural discovery for Action Flow pipeline components."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def resolve_action_flow_topology(model: Any) -> tuple[Any, Any, Any]:
    """Find one encoder, field, and decoder from declarative stage contracts."""

    pipeline = getattr(model, "pipeline", None)
    stages = tuple(getattr(pipeline, "stages", ()))
    if not stages:
        raise RuntimeError("Action Flow requires a staged pipeline")

    def writes(stage, mode: str) -> tuple[str, ...]:
        contract = getattr(stage, "contract", None)
        if not callable(contract):
            return ()
        _, values = contract(mode)
        return tuple(values)

    encoder_stages = tuple(
        stage
        for stage in stages
        if isinstance(getattr(stage, "encoder", None), nn.Module)
        and any(key.endswith("/clean_latent") for key in writes(stage, "train"))
    )
    field_stages = tuple(
        stage
        for stage in stages
        if isinstance(getattr(stage, "field", None), nn.Module)
        and any(key.endswith("/predicted_velocity") for key in writes(stage, "train"))
        and any(key.endswith("/generated_latent") for key in writes(stage, "inference"))
    )
    decoder_stages = tuple(
        stage
        for stage in stages
        if isinstance(getattr(stage, "decoder", None), nn.Module)
        and any(key.endswith("/reconstruction") for key in writes(stage, "train"))
    )
    counts = (len(encoder_stages), len(field_stages), len(decoder_stages))
    if counts != (1, 1, 1):
        raise RuntimeError(
            "Action Flow graph must have exactly one clean-latent encoder, "
            "conditional field, and content decoder; "
            f"found {counts}"
        )
    return encoder_stages[0], field_stages[0], decoder_stages[0]
