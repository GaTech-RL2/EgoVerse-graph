"""Read pinned, decoded episode caches through the shared MultiDataset API."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import EpisodeResolver


@lru_cache(maxsize=192)
def _array(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class NumpyEpisode(torch.utils.data.Dataset):
    def __init__(self, folder_path, row, key_map, transform_list):
        self.episode_path = Path(folder_path) / row["episode_id"]
        self.metadata = row
        self.embodiment = row["embodiment"]
        self.total_frames = row["total_frames"]
        self.key_map = key_map
        self.transform = transform_list
        self.intrinsics = None

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        if index < 0 or index >= self.total_frames:
            raise IndexError(index)
        result = {}
        for key, spec in self.key_map.items():
            values = _array(str(self.episode_path / (spec["zarr_key"] + ".npy")))
            horizon = spec.get("horizon")
            if horizon is None:
                value = np.array(values[index], copy=True)
            else:
                frames = np.minimum(np.arange(index, index + int(horizon)), self.total_frames - 1)
                value = np.array(values[frames], copy=True)
            if spec.get("key_type") == "camera_keys":
                # Preserve the historical JPEG reader's RGB -> CHW / 255 contract.
                value = np.moveaxis(value, -1, -3) / 255.0
            result[key] = value
        for transform in self.transform or []:
            result = transform.transform(result)
        result = {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)
                  if isinstance(v, np.ndarray) else v for k, v in result.items()}
        result.update(embodiment=get_embodiment_id(self.embodiment),
            intrinsics=torch.full((3, 4), float("nan")),
            episode_hash=self.metadata["episode_id"], frame_index=int(index))
        return result


class NumpyEpisodeResolver(EpisodeResolver):
    def __init__(self, folder_path, manifest_sha256, embodiment, split,
                 key_map, transform_list, sources=None):
        super().__init__(Path(folder_path), key_map, transform_list)
        self.manifest_sha256 = manifest_sha256
        self.embodiment = embodiment
        if split not in {"train", "valid"}:
            raise ValueError("An explicit train/valid split is required")
        self.split = split
        self.sources = set(sources or [])

    def resolve(self, filters=None):
        if filters is not None:
            raise ValueError("Pinned cache manifests cannot be filtered implicitly")
        raw = (self.folder_path / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest_sha256:
            raise ValueError("Decoded cache manifest hash mismatch")
        rows = [e for e in json.loads(raw)["episodes"] if e["embodiment"] == self.embodiment
                and e["split"] == self.split and (not self.sources or e["source"] in self.sources)]
        if not rows:
            raise ValueError("Empty pinned dataset selection")
        datasets = {}
        for row in rows:
            if row["episode_id"] in datasets:
                raise ValueError("Duplicate cache episode ID")
            # Source alignment belongs to the loader, before tokenization.
            transforms = []
            import copy
            key_map = copy.deepcopy(self.key_map)
            offset = row["action_target_offset_obs2"]
            expected_offset = 1 if row["observation_alignment"] == "pre_step" else 2
            if offset != expected_offset:
                raise ValueError("Inconsistent observation alignment")
            for transform in self.transform_list or []:
                item = copy.deepcopy(transform)
                if hasattr(item, "start") and hasattr(item, "horizon"):
                    item.start = offset
                    for key in item.keys:
                        key_map[key]["horizon"] = item.horizon + offset
                transforms.append(item)
            datasets[row["episode_id"]] = NumpyEpisode(self.folder_path, row, key_map, transforms)
        return datasets
