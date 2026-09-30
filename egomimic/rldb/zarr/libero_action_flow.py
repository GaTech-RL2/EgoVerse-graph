"""LIBERO normalization with a frozen 1% episode split and train-only limits.

Unlike the released OAT comparison normalizer, this Action Flow variant never
fits action or proprio limits from held-out validation episodes. Task IDs stay
integer-valued for the conditional observation adapter.
"""

import hashlib

import numpy as np
import torch
import zarr

from egomimic.rldb.zarr.libero_dataset import (
    EMBODIMENT,
    LiberoDataset,
    LiberoNormalizer,
    validation_mask,
)


class _LogicalReplayKeyNormalization:
    """Normalize the keys emitted by `_ReplayEpisode`, not the Zarr keys."""

    def normalize(self, data, embodiment_id):
        stats = self.norm_stats.get(embodiment_id, {})
        out = dict(data)
        for key, value in data.items():
            if key in stats and torch.is_tensor(value):
                out[key] = self._apply_norm_one(value, stats[key])
        return out

    def unnormalize(self, data, embodiment_id):
        stats = self.norm_stats.get(embodiment_id, {})
        out = dict(data)
        for key, value in data.items():
            if key in stats and torch.is_tensor(value):
                out[key] = self._apply_unnorm_one(value, stats[key])
        return out


class LiberoActionFlowDataset(_LogicalReplayKeyNormalization, LiberoDataset):
    """Split-selected replay with exact logical-key normalization."""


class LiberoActionFlowNormalizer(_LogicalReplayKeyNormalization, LiberoNormalizer):
    def infer_norm_from_dataset(self, dataset, dataset_name, **kwargs):
        del dataset_name
        if kwargs.get("precomputed_norm_path") is not None:
            raise ValueError("train-only LIBERO limits must be fit for this split")
        if dataset.val_ratio != 0.01:
            raise ValueError("Action Flow LIBERO requires a 1% episode split")
        group = zarr.open_group(dataset.resolver.folder_path, mode="r")
        ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
        valid = validation_mask(len(ends), dataset.val_ratio, dataset.split_seed)
        starts = np.r_[0, ends[:-1]]
        arrays = group["data"]
        digest = hashlib.sha256(ends.tobytes())
        for key in ("action", "task_uid"):
            array = arrays[key]
            for start in range(0, array.shape[0], 4096):
                digest.update(np.ascontiguousarray(array[start:start + 4096]).tobytes())
        self.context = {
            "dataset_sha256": digest.hexdigest(),
            "suite": dataset.resolver.suite,
            "split_seed": int(dataset.split_seed),
            "val_ratio": float(dataset.val_ratio),
            "normalization_scope": "training_episodes_only",
            "training_episode_count": int((~valid).sum()),
            "validation_episode_count": int(valid.sum()),
        }
        stats = self.norm_stats.setdefault(EMBODIMENT, {})
        for key, info in dataset.resolver.key_map.items():
            if key == "task_uid":
                continue
            if info["key_type"] == "camera_keys":
                # RGB byte range is fixed independently of the split.
                stats[key] = {
                    "min": np.zeros(3, dtype=np.float32),
                    "max": np.full(3, 255, dtype=np.float32),
                    "scale": np.full(3, 2 / 255, dtype=np.float32),
                    "offset": -np.ones(3, dtype=np.float32),
                }
                continue
            array = arrays[info["zarr_key"]]
            low = np.full(array.shape[-1], np.inf, dtype=np.float32)
            high = -low.copy()
            for first, last, is_valid in zip(starts, ends, valid):
                if is_valid:
                    continue
                for offset in range(int(first), int(last), 4096):
                    values = np.asarray(
                        array[offset:min(offset + 4096, int(last))],
                        dtype=np.float32,
                    ).reshape(-1, array.shape[-1])
                    if not np.isfinite(values).all():
                        raise ValueError(f"non-finite training normalization data: {key}")
                    low = np.minimum(low, values.min(axis=0))
                    high = np.maximum(high, values.max(axis=0))
            if not np.isfinite(low).all() or not np.isfinite(high).all():
                raise ValueError(f"no training samples for normalization: {key}")
            span = high - low
            constant = span < 1e-4
            scale = 2 / np.where(constant, 2, span)
            offset = np.where(constant, -low, -1 - scale * low)
            stats[key] = {"min": low, "max": high, "scale": scale, "offset": offset}
