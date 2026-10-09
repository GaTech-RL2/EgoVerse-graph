"""Explicit per-embodiment prefix dimensions shared by loss and metrics."""

from collections.abc import Mapping

from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


def normalize_active_action_dimensions(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "Active action dimensions must be a nonempty embodiment mapping"
        )
    result = {}
    for name, count in value.items():
        label = str(name).lower()
        if get_embodiment_id(label) is None:
            raise ValueError(f"Unknown action-dimension embodiment {label!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Active action dimensions must be positive integers")
        if label in result:
            raise ValueError(f"Duplicate action-dimension embodiment {label!r}")
        result[label] = count
    return result


def active_action_prefix(value, label, dimensions):
    if not dimensions:
        return value
    label = str(label).lower()
    if label not in dimensions:
        raise ValueError(f"No active action dimensions for {label!r}")
    count = dimensions[label]
    if value.ndim < 2 or not 0 < count <= value.shape[-1]:
        raise ValueError(
            f"Active action dimensions {count} exceed tensor shape {tuple(value.shape)}"
        )
    return value[..., :count]


def batch_embodiment_name(batch):
    if "embodiment" not in batch:
        raise ValueError("Masked action loss requires homogeneous embodiment metadata")
    identity = int(resolve_homogeneous_scalar(batch["embodiment"], label="embodiment"))
    name = get_embodiment(identity)
    if name is None:
        raise ValueError(f"Unknown action-dimension embodiment id {identity}")
    return name.lower()
