"""Restore a complete exported data contract without opening training data."""

import json
from pathlib import Path

import torch

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def load_normalizer(path):
    payload = json.loads(Path(path).read_text())
    state = payload.get("normalizer_state", payload)
    required = {
        "norm_mode",
        "embodiments",
        "key_types",
        "zarr_keys",
        "shapes",
        "norm_stats",
    }
    if not required <= state.keys():
        raise ValueError(
            "Export a full normalizer_state with trainHydra norm_stats_only=true and the training recipe"
        )
    for name in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        state[name] = {int(key): value for key, value in state[name].items()}
    state["embodiments"] = [int(key) for key in state["embodiments"]]
    normalizer = MultiDataset.from_state(state)
    for embodiment in normalizer.embodiments:
        for key, stats in normalizer.norm_stats[embodiment].items():
            for values in stats.values():
                tensor = torch.as_tensor(values)
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Nonfinite normalization statistics: {embodiment}/{key}"
                    )
                torch.broadcast_to(tensor, tuple(normalizer.key_shape(key, embodiment)))
    return normalizer
