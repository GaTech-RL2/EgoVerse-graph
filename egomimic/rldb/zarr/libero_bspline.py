"""Episode-level BSP fitting over the existing decoded LIBERO replay loader."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr
from filelock import FileLock

from egomimic.models.bspline import BSplineCodec, FittedTrajectory
from egomimic.rldb.zarr.libero_dataset import (
    EMBODIMENT,
    LiberoNormalizer,
    LiberoReplayResolver,
)


class LiberoBSplineResolver(LiberoReplayResolver):
    def __init__(self, *args, codec=None, cache_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.codec = BSplineCodec(**(codec or {}))
        self.cache_dir = Path(cache_dir or (self.folder_path + ".bspline"))

    def resolve(self):
        return {
            name: _BSplineEpisode(episode, self.codec, self.cache_dir)
            for name, episode in super().resolve().items()
        }


class _BSplineEpisode(torch.utils.data.Dataset):
    embodiment = "libero_panda"

    def __init__(self, episode, codec, cache_dir):
        self.episode, self.codec, self.cache_dir = episode, codec, cache_dir
        self.episode_path = episode.episode_path
        self.key_map = {
            **episode.key_map,
            "bspline": {
                "zarr_key": "bspline",
                "key_type": "action_keys",
                "horizon": codec.num_knots,
            },
        }
        self._fit = None

    def __len__(self):
        return len(self.episode)

    @property
    def trajectory(self):
        if self._fit is None:
            ep = self.episode
            actions = np.asarray(
                zarr.open_group(ep.path, mode="r")["data/action"][ep.start : ep.end],
                dtype=np.float64,
            )
            identity = hashlib.sha256(
                json.dumps(self.codec.context(), sort_keys=True).encode()
            )
            identity.update(str(actions.shape).encode())
            identity.update(actions.tobytes())
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / (identity.hexdigest() + ".npz")
            with FileLock(str(path) + ".lock"):
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        self._fit = FittedTrajectory(
                            saved["knots"],
                            saved["control_points"],
                            len(actions),
                            float(saved["max_error"]),
                        )
                else:
                    self._fit = self.codec.fit(actions)
                    with tempfile.NamedTemporaryFile(
                        dir=self.cache_dir, suffix=".npz", delete=False
                    ) as f:
                        temp = Path(f.name)
                        np.savez(
                            f,
                            knots=self._fit.knots,
                            control_points=self._fit.control_points,
                            max_error=self._fit.max_error,
                        )
                    try:
                        os.replace(temp, path)
                    finally:
                        temp.unlink(missing_ok=True)
        return self._fit

    def __getitem__(self, index):
        sample = self.episode[index]
        sample["bspline"] = torch.from_numpy(self.codec.encode(self.trajectory, index))
        return sample


class LiberoBSplineNormalizer(LiberoNormalizer):
    def infer_norm_from_dataset(self, dataset, dataset_name, **kwargs):
        # Keep raw-action/observation normalization identical to existing controls.
        super().infer_norm_from_dataset(dataset, dataset_name, **kwargs)
        low, high = np.full(8, np.inf), np.full(8, -np.inf)
        for episode in dataset.datasets.values():
            # Fit once, cache compact coefficients, and reduce one episode at a time.
            values = np.stack(
                [
                    episode.codec.encode(episode.trajectory, t)
                    for t in range(len(episode))
                ]
            )
            low = np.minimum(low, values.min(axis=(0, 1)))
            high = np.maximum(high, values.max(axis=(0, 1)))
        span = high - low
        constant = span < 1e-4
        scale = 2 / np.where(constant, 2, span)
        self.norm_stats[EMBODIMENT]["bspline"] = {
            "min": low.astype(np.float32),
            "max": high.astype(np.float32),
            "scale": scale.astype(np.float32),
            "offset": np.where(constant, -low, -1 - scale * low).astype(np.float32),
        }
        self.context["bspline"] = {
            **dataset.resolver.codec.context(),
            "parameter_norm_scope": "training_episodes",
            "fit_action_space": "native_delta_osc_including_gripper",
        }
