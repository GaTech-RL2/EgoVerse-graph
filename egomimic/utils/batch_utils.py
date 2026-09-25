"""Batch compatible graph inputs without combining source-specific metadata."""

from collections import defaultdict
from collections.abc import Mapping

import torch


def batch_size(value):
    """Find the leading sample dimension in a batch or nested model input."""
    if torch.is_tensor(value) and value.ndim:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            if torch.is_tensor(item) and item.ndim >= 2:
                return int(item.shape[0])
        for item in value.values():
            try:
                return batch_size(item)
            except ValueError:
                pass
    raise ValueError("Batch must contain a tensor with a leading sample dimension")


def batch_signature(value):
    if torch.is_tensor(value):
        if not value.ndim:
            raise ValueError("Batched tensors must have a leading sample dimension")
        return (torch.Tensor, tuple(value.shape[1:]), value.dtype, value.device)
    if isinstance(value, Mapping):
        return tuple((key, batch_signature(value[key])) for key in sorted(value))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return (list, str)
    if value is None:
        return None
    raise TypeError(f"Unsupported model batch value: {type(value).__name__}")


def concatenate_batches(values):
    first = values[0]
    if len(values) == 1:
        return first
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, Mapping):
        return {
            key: concatenate_batches([value[key] for value in values]) for key in first
        }
    if isinstance(first, list):
        return [item for value in values for item in value]
    if first is None:
        return None
    raise TypeError(f"Unsupported model batch value: {type(first).__name__}")


def compatible_groups(batches):
    groups = defaultdict(list)
    for source, value in batches.items():
        _validate_size(value, batch_size(value))
        groups[batch_signature(value)].append(source)
    return groups.values()


def _validate_size(value, size):
    if torch.is_tensor(value):
        if not value.ndim or value.shape[0] != size:
            raise ValueError("All model inputs must share a leading sample dimension")
    elif isinstance(value, Mapping):
        for item in value.values():
            _validate_size(item, size)
    elif isinstance(value, list) and len(value) != size:
        raise ValueError("Prompt lists must match the sample count")


def split_batches(value, sizes):
    if torch.is_tensor(value):
        if not value.ndim or value.shape[0] != sum(sizes):
            raise ValueError(
                "Grouped model output must preserve the leading sample dimension"
            )
        return value.split(sizes)
    if isinstance(value, Mapping):
        parts = {key: split_batches(item, sizes) for key, item in value.items()}
        return [
            {key: items[i] for key, items in parts.items()} for i in range(len(sizes))
        ]
    raise TypeError("Grouped model outputs must be tensors or mappings of tensors")


def map_batches(inputs, forward):
    """Call a shared network once per compatible shape group; retain gradients."""
    result = {}
    for group in compatible_groups(inputs):
        sizes = [batch_size(inputs[source]) for source in group]
        output = forward(concatenate_batches([inputs[source] for source in group]))
        result.update(zip(group, split_batches(output, sizes)))
    return result


def sample_mean(losses, sizes):
    if not losses or len(losses) != len(sizes) or min(sizes) < 1:
        raise ValueError("Expected nonempty losses with positive aligned sample counts")
    return sum(loss * size for loss, size in zip(losses, sizes)) / sum(sizes)
