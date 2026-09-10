"""Small tensor-tree helpers shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def clone_inference_tensors(value: Any) -> Any:
    """Clone inference tensors into ordinary tensors for AD-sensitive paths."""

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return type(value)(
            (key, clone_inference_tensors(item)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(clone_inference_tensors(item) for item in value)
    if isinstance(value, list):
        return [clone_inference_tensors(item) for item in value]
    return value


def cuda_devices(value: Any) -> list[int]:
    """Return CUDA device indices found recursively in a tensor tree."""

    devices = set()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            if item.device.type == "cuda" and item.device.index is not None:
                devices.add(int(item.device.index))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return sorted(devices)
