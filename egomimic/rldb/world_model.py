"""Prepared world-model data contract and temporal sampling.

Owns no simulator or tracker: upstream extraction must supply real tracks and
fixed-frame measured poses. Never reinterpret policy commands as pose targets.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from egomimic.rldb.world_model_preparation import (
    ArtifactVerificationMemo,
    load_normalization,
)


@contextmanager
def open_prepared_episode(path):
    """Keep historical NPZ support; HDF5 slices only selected frame chunks."""
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py

        with h5py.File(path, "r") as data:
            yield data
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            yield data
    else:
        raise ValueError("Prepared episodes must use NPZ or HDF5")


def points_to_reference(points, reference_from_source):
    """Transform [..., P, 3] points with calibrated [..., 4, 4] rigid transforms."""
    if points.shape[-1] != 3 or reference_from_source.shape[-2:] != (4, 4):
        raise ValueError("Expected XYZ and homogeneous rigid transforms")
    if not np.isfinite(reference_from_source).all():
        raise ValueError("Non-finite calibration")
    rotation = reference_from_source[..., :3, :3]
    if not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), np.eye(3), atol=1e-4):
        raise ValueError("Calibration rotation is not orthonormal")
    if not np.allclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("Calibration must preserve handedness")
    if not np.allclose(reference_from_source[..., 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous transform")
    return (
        points @ np.swapaxes(rotation, -1, -2) + reference_from_source[..., None, :3, 3]
    )


def _bounded_time_sampling(max_past_seconds, max_future_seconds):
    if max_past_seconds is None and max_future_seconds is None:
        return False
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
        for value in (max_past_seconds, max_future_seconds)
    ) or max_past_seconds + max_future_seconds <= 0:
        raise ValueError("Time bounds require two finite nonnegative seconds values with positive span")
    return True


def _physical_times(timestamps, length):
    times = np.asarray(timestamps, dtype=np.float64)
    if (
        times.shape != (length,)
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError("Physical timestamps must match episode length and strictly increase")
    return times - times[0]


def sample_times(
    length,
    rng,
    mode="continuous",
    min_frames=2,
    max_frames=4,
    discrete_frames=4,
    stride=1,
    anchor=None,
    timestamps=None,
    max_past_seconds=None,
    max_future_seconds=None,
):
    """Select recorded times; optional bounds are relative to the anchor time.

    Bounded continuous mode draws uniformly without replacement among recorded
    timestamps inside [anchor-past, anchor+future]. This is not a uniform draw
    over continuous seconds. Unbounded calls retain the historical index-based
    multiscale sampler and its RNG behavior for checkpoint-bound callers.
    """
    if (
        mode not in ("continuous", "discrete")
        or min_frames < 2
        or max_frames < min_frames
        or stride < 1
    ):
        raise ValueError("Invalid time sampler")
    bounded = _bounded_time_sampling(max_past_seconds, max_future_seconds)
    if bounded and mode != "continuous":
        raise ValueError("Physical-time bounds apply only to continuous sampling")
    if mode == "discrete":
        available = length - (discrete_frames - 1) * stride
        if discrete_frames < 2 or available <= 0:
            raise ValueError("Episode too short for complete discrete window")
        start = int(rng.integers(available)) if anchor is None else int(anchor)
        if not 0 <= start < available:
            raise ValueError("Invalid discrete anchor")
        return start + np.arange(discrete_frames) * stride
    if length < min_frames:
        raise ValueError("Episode too short")
    if bounded:
        times = _physical_times(timestamps, length)
        start = int(rng.integers(length)) if anchor is None else int(anchor)
        if not 0 <= start < length:
            raise ValueError("Invalid anchor")
        lo = np.searchsorted(times, times[start] - max_past_seconds, side="left")
        hi = np.searchsorted(times, times[start] + max_future_seconds, side="right")
        eligible = np.arange(lo, hi, dtype=np.int64)
        if len(eligible) < min_frames:
            raise ValueError("Too few recorded timestamps inside anchor time bounds")
        count = int(rng.integers(min_frames, min(max_frames, len(eligible)) + 1))
        others = rng.choice(eligible[eligible != start], size=count - 1, replace=False)
        return np.sort(np.r_[start, others]).astype(np.int64)
    count = int(rng.integers(min_frames, min(max_frames, length) + 1))
    start = int(rng.integers(length)) if anchor is None else int(anchor)
    if not 0 <= start < length:
        raise ValueError("Invalid anchor")
    selected = {start}
    # Cycle nearby / intermediate / episode-wide support; randomized first scale.
    scales = rng.permutation([1, max(2, length // 8), length])
    for i in range(count - 1):
        radius = int(scales[i % 3])
        candidates = np.array(
            [
                j
                for j in range(max(0, start - radius), min(length, start + radius + 1))
                if j not in selected
            ]
        )
        if not len(candidates):
            candidates = np.array([j for j in range(length) if j not in selected])
        selected.add(int(rng.choice(candidates)))
    return np.array(sorted(selected), dtype=np.int64)


def collate_world(samples):
    """Homogeneous embodiment batches; pad streams independently without leaking masks."""
    embodiments = {s["embodiment"] for s in samples}
    if len(embodiments) != 1:
        raise ValueError("Batch must be homogeneous in embodiment")
    embodiment = samples[0]["embodiment"]
    result = {}
    for name in ("image", "track", "proprio"):
        size = max(s["world"][name]["value"].shape[0] for s in samples)
        stream = {}
        for key in samples[0]["world"][name]:
            arrays = []
            for sample in samples:
                value = sample["world"][name][key]
                target = torch.zeros((size, *value.shape[1:]), dtype=value.dtype)
                target[: len(value)] = value
                arrays.append(target)
            stream[key] = torch.stack(arrays)
        result[name] = stream
    item = {"embodiment": embodiment, "world": result}
    if "sampling_cursor" in samples[0]:
        cursors = {sample["sampling_cursor"] for sample in samples}
        if len(cursors) != 1:
            raise ValueError("A resumed batch must have one source-draw cursor")
        item["sampling_cursor"] = cursors.pop()
    return {embodiment: item}


class PreparedWorldDataset(Dataset):
    """Read per-episode NPZ or sliced HDF5 caches with SHA-bound manifests.

    A flat global anchor index samples proportional to eligible frames/windows,
    NOT equally per embodiment. Use HomogeneousWorldBatchSampler for batches.
    """

    def __init__(
        self,
        manifest_path,
        split="train",
        mode="continuous",
        seed=42,
        min_frames=2,
        max_frames=4,
        discrete_frames=4,
        stride=1,
        forecasting_probability=0.5,
        max_past_seconds=None,
        max_future_seconds=None,
        verification_memo=None,
    ):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        m = self.manifest
        if m.get("schema") != "mot-world-prepared-v1":
            raise ValueError("Unsupported prepared-data schema")
        for key in (
            "dino_identity",
            "normalization",
            "pose_schemas",
            "track_provenance",
        ):
            if not m.get(key):
                raise ValueError(f"Missing {key}")
        for key in ("model", "revision", "image_size", "patch_grid", "feature_layer"):
            if not m["dino_identity"].get(key):
                raise ValueError(f"Missing DINO identity field {key}")
        for key in ("image_size", "patch_grid"):
            shape = m["dino_identity"][key]
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(x) is not int or x < 1 for x in shape)
            ):
                raise ValueError(f"DINO {key} must contain two positive integers")
        if (
            m["normalization"].get("scope") != "train_only"
            or not re.fullmatch(r"[0-9a-f]{64}", m["normalization"].get("sha256", ""))
            or type(m["normalization"].get("arrays_normalized")) is not bool
        ):
            raise ValueError("Require train-only normalization provenance")
        for schema in m["pose_schemas"].values():
            if (
                schema.get("dimension", 0) < 1
                or len(schema.get("fields", [])) != schema["dimension"]
            ):
                raise ValueError(
                    "Pose schemas require explicit field names for every coordinate"
                )
        for key in ("method", "reference_frame", "calibration_sha256"):
            if not m["track_provenance"].get(key):
                raise ValueError(f"Missing track provenance {key}")
        if m["track_provenance"].get("interpolated_endpoints", True):
            raise ValueError("Endpoint interpolation is not actual tracked supervision")
        verifier = verification_memo if verification_memo is not None else ArtifactVerificationMemo()
        self.normalization = load_normalization(m, self.path.parent, verifier)
        ids, paths = set(), set()
        for ep in m["episodes"]:
            identity = (ep["embodiment"], ep["episode_id"])
            physical = (self.path.parent / ep["path"]).resolve()
            if identity in ids or physical in paths:
                raise ValueError(
                    "Duplicate episode identity or physical path across splits"
                )
            ids.add(identity)
            paths.add(physical)
            if (
                ep.get("frame") != "fixed_reference"
                or ep.get("pose_kind") != "measured_pose"
            ):
                raise ValueError(
                    "Require measured poses and tracks in a fixed reference frame"
                )
            if ep["split"] not in ("train", "val"):
                raise ValueError("Explicit train/val episode assignment required")
            verifier.verify(physical, ep["sha256"],
                            trusted_fingerprint=ep.get("verified_fingerprint"))
        self.episodes = [ep for ep in m["episodes"] if ep["split"] == split]
        if not self.episodes:
            raise ValueError("No episodes in selected split")
        self.mode, self.seed, self.split = mode, seed, split
        self.sampling = dict(
            mode=mode,
            min_frames=min_frames,
            max_frames=max_frames,
            discrete_frames=discrete_frames,
            stride=stride,
        )
        bounded = _bounded_time_sampling(max_past_seconds, max_future_seconds)
        if bounded and mode != "continuous":
            raise ValueError("Physical-time bounds apply only to continuous sampling")
        if bounded:
            self.sampling.update(
                max_past_seconds=max_past_seconds,
                max_future_seconds=max_future_seconds,
            )
        if not 0 <= forecasting_probability <= 1:
            raise ValueError("Invalid forecasting probability")
        self.forecasting_probability = forecasting_probability
        self.anchor_indices = []
        for ep in self.episodes:
            if bounded:
                with open_prepared_episode(self.path.parent / ep["path"]) as saved:
                    times = _physical_times(saved["time"][:], ep["frames"])
                lo = np.searchsorted(times, times - max_past_seconds, side="left")
                hi = np.searchsorted(times, times + max_future_seconds, side="right")
                # An isolated frame cannot yield min_frames without leaving the
                # requested time range. Exclude that anchor instead of escaping.
                anchors = np.flatnonzero(hi - lo >= min_frames)
            else:
                count = ep["frames"] - (
                    (discrete_frames - 1) * stride if mode == "discrete" else 0
                )
                anchors = np.arange(max(0, count))
            self.anchor_indices.append(anchors)
        self.counts = [len(indices) for indices in self.anchor_indices]
        if any(c < 1 for c in self.counts):
            raise ValueError("Episode too short for temporal recipe")
        self.ends = np.cumsum(self.counts)
        self.rng = None
        self.worker_seed = None

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        draw = None
        if isinstance(index, tuple):
            index, cursor, slot = index
            draw = (int(cursor), int(slot))
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Training RNG advances on repeated visits; validation choices stay reproducible.
        worker_seed = torch.initial_seed()
        if self.rng is None or self.worker_seed != worker_seed:
            self.rng = np.random.default_rng((worker_seed + self.seed) % 2**63)
            self.worker_seed = worker_seed
        rng = (
            self.rng
            if self.split == "train"
            else np.random.default_rng(self.seed + index)
        )
        if draw is not None:
            if self.split != "train":
                raise ValueError("Checkpointed draw keys apply only to training")
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, *draw]))
        episode_index = int(np.searchsorted(self.ends, index, side="right"))
        ep = self.episodes[episode_index]
        offset = 0 if episode_index == 0 else int(self.ends[episode_index - 1])
        anchor = int(self.anchor_indices[episode_index][index - offset])
        with open_prepared_episode(self.path.parent / ep["path"]) as data:
            required = (
                "time",
                "image",
                "image_xy",
                "track",
                "track_xy",
                "track_camera",
                "track_confidence",
                "pose",
                "pose_valid",
            )
            if any(k not in data for k in required):
                raise ValueError("Prepared cache missing required arrays")
            full_time = data["time"][:]
            if (
                len(full_time) != ep["frames"]
                or not np.isfinite(full_time).all()
                or not (np.diff(full_time) > 0).all()
            ):
                raise ValueError(
                    "Timestamps must be finite, strictly increasing physical seconds"
                )
            indices = sample_times(
                ep["frames"], rng, anchor=anchor, timestamps=full_time, **self.sampling
            )
            times = full_time[indices].astype(np.float64)
            # Subtract origin before float32 conversion (important for epoch timestamps).
            times = (times - times[int(rng.integers(len(times)))]).astype(np.float32)
            image, track, pose = (
                data[k][indices].astype(np.float32) for k in ("image", "track", "pose")
            )
            if self.normalization is not None:
                values = []
                for name, value in (("image", image), ("track", track), (f"pose/{ep['embodiment']}", pose)):
                    mean, scale = self.normalization[name]
                    if value.shape[-1] != len(mean):
                        raise ValueError(f"Normalization schema width mismatch: {name}")
                    values.append((value - mean) / scale)
                image, track, pose = values
            if (
                image.ndim != 4
                or track.ndim != 3
                or track.shape[-1] != 3
                or pose.ndim != 2
            ):
                raise ValueError("Expected image[T,V,P,D], track[T,Q,3], pose[T,Dm]")
            if (
                pose.shape[-1]
                != self.manifest["pose_schemas"][ep["embodiment"]]["dimension"]
            ):
                raise ValueError("Pose schema width mismatch")
            f, v, p, _ = image.shape
            grid_h, grid_w = self.manifest["dino_identity"]["patch_grid"]
            if p != grid_h * grid_w:
                raise ValueError("DINO patch grid differs from cached image tokens")
            q = track.shape[1]
            if (
                data["image_xy"].shape != (p, 2)
                or data["track_xy"].shape != (q, 2)
                or data["track_camera"].shape != (q,)
                or data["track_confidence"].shape != (ep["frames"], q)
                or data["pose_valid"].shape != (ep["frames"],)
            ):
                raise ValueError(
                    "Prepared geometry/mask shapes differ from token arrays"
                )
            if not np.isfinite(image).all():
                raise ValueError("Non-finite DINO features")
            if rng.random() < self.forecasting_probability:
                cutoff = int(rng.integers(1, f))
                known = np.arange(f) < cutoff
                observed = {k: known.copy() for k in ("image", "track", "proprio")}
                # Independently omit observed proprio/track frames; never reveal future ones.
                for k in ("track", "proprio"):
                    observed[k] &= rng.random(f) < 0.8
            else:
                observed = {
                    k: rng.random(f) < 0.5 for k in ("image", "track", "proprio")
                }
                # Ensure context and a target without privileging either edge:
                # random conditioning must include backcasting/interpolation.
                if not observed["image"].any():
                    observed["image"][rng.integers(f)] = True
                if observed["image"].all():
                    observed["image"][rng.integers(f)] = False

            def base(values, repeats, name):
                n = f * repeats
                return {
                    "value": values.reshape(n, -1),
                    "time": np.repeat(times, repeats),
                    "index": np.repeat(np.arange(f), repeats).astype(np.float32),
                    "valid": np.ones(n, bool),
                    "target_valid": np.ones(n, bool),
                    "observed": np.repeat(observed[name], repeats),
                }

            streams = {
                "image": base(image, v * p, "image"),
                "track": base(track, q, "track"),
                "proprio": base(pose, 1, "proprio"),
            }
            streams["image"].update(
                xy=np.tile(data["image_xy"][:].astype(np.float32), (f * v, 1)),
                camera=np.tile(np.repeat(np.arange(v), p), f),
            )
            confidence = (
                data["track_confidence"][indices].reshape(-1).astype(np.float32)
            )
            streams["track"].update(
                xy=np.tile(data["track_xy"][:].astype(np.float32), (f, 1)),
                camera=np.tile(data["track_camera"][:].astype(np.int64), f),
                point_id=np.tile(np.arange(q), f),
                confidence=confidence,
            )
            streams["track"]["target_valid"] = (confidence > 0) & np.isfinite(
                track.reshape(-1, 3)
            ).all(-1)
            streams["track"]["observed"] &= streams["track"]["target_valid"]
            streams["proprio"]["target_valid"] = data["pose_valid"][indices].astype(
                bool
            )
            streams["proprio"]["observed"] &= streams["proprio"]["target_valid"]
            sample = {
                "embodiment": ep["embodiment"],
                "world": {
                    name: {
                        key: torch.from_numpy(np.array(value, copy=True))
                        for key, value in s.items()
                    }
                    for name, s in streams.items()
                },
            }
            if draw is not None:
                sample["sampling_cursor"] = draw[0] + 1
            return sample


class HomogeneousWorldBatchSampler:
    """Choose an embodiment proportional to eligible anchors, then draw its batch."""

    def __init__(self, dataset, batch_size=1, seed=42, shuffle=True, checkpointable=False,
                 validation_batches_per_embodiment=None):
        if batch_size < 1:
            raise ValueError("Positive batch size required")
        if validation_batches_per_embodiment is not None:
            if type(validation_batches_per_embodiment) is not int or validation_batches_per_embodiment < 1:
                raise ValueError("validation_batches_per_embodiment must be a positive integer or None")
            if shuffle or getattr(dataset, "split", None) != "val":
                raise ValueError("Per-embodiment batch caps require unshuffled validation data")
        self.validation_batches_per_embodiment = validation_batches_per_embodiment
        self.dataset, self.batch_size, self.seed, self.shuffle = (
            dataset,
            batch_size,
            seed,
            shuffle,
        )
        self.epoch = 0
        self.checkpointable = checkpointable
        self.consumed_batches = 0
        self.ranges = {}
        start = 0
        for ep, count in zip(dataset.episodes, dataset.counts):
            self.ranges.setdefault(ep["embodiment"], []).append(
                np.arange(start, start + count)
            )
            start += count
        self.groups = {k: np.concatenate(v) for k, v in self.ranges.items()}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.shuffle:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        return sum(
            min((len(v) + self.batch_size - 1) // self.batch_size,
                self.validation_batches_per_embodiment)
            if self.validation_batches_per_embodiment is not None
            else (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.groups.values()
        )

    def __iter__(self):
        if not self.shuffle:
            for group in self.groups.values():
                end = len(group)
                if self.validation_batches_per_embodiment is not None:
                    end = min(end, self.validation_batches_per_embodiment * self.batch_size)
                for i in range(0, end, self.batch_size):
                    yield group[i : i + self.batch_size].tolist()
            return
        if self.checkpointable:
            groups = list(self.groups.values())
            probabilities = np.array([len(v) for v in groups], float)
            probabilities /= probabilities.sum()
            start = self.consumed_batches
            end = (start // len(self) + 1) * len(self)
            for cursor in range(start, end):
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, cursor]))
                group = groups[int(rng.choice(len(groups), p=probabilities))]
                chosen = rng.choice(group, self.batch_size, replace=True)
                yield [(int(index), cursor, slot) for slot, index in enumerate(chosen)]
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = list(self.groups.values())
        probabilities = np.array([len(v) for v in groups], float)
        probabilities /= probabilities.sum()
        for _ in range(len(self)):
            group = groups[int(rng.choice(len(groups), p=probabilities))]
            yield rng.choice(group, self.batch_size, replace=True).tolist()
        self.epoch += 1
