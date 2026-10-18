"""Native RGB patches with optional proprioception and verified command labels.

RGB remains uint8 in its prepared immutable cache. Selected frames are converted
to [-1, 1] and rearranged into full raw-pixel patches at load time. This adapter
does not resize images, average patches, or consume encoder features or tracks.
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from egomimic.rldb.world_model import (
    HomogeneousWorldBatchSampler,
    _physical_times,
    open_prepared_episode,
)
from egomimic.rldb.world_model_preparation import (
    ArtifactVerificationMemo,
    training_identities,
)


def collate_pixel_proprio(samples):
    """Pad declared streams independently, preserving per-sample task labels."""
    if not samples or len({s["embodiment"] for s in samples}) != 1:
        raise ValueError("Batch must be nonempty and homogeneous in embodiment")
    names = set(samples[0]["world"])
    if names not in ({"image"}, {"image", "proprio"}, {"image", "proprio", "action"}):
        raise ValueError("Expected image stream with optional proprio/action labels")
    if any(set(s["world"]) != names for s in samples):
        raise ValueError("Cannot mix objectives in a batch")
    result = {}
    for name in sorted(names):
        size = max(len(s["world"][name]["value"]) for s in samples)
        fields = set(samples[0]["world"][name])
        if any(set(s["world"][name]) != fields for s in samples):
            raise ValueError("Inconsistent stream fields")
        result[name] = {}
        for key in fields:
            values = []
            for sample in samples:
                value = sample["world"][name][key]
                target = torch.zeros((size, *value.shape[1:]), dtype=value.dtype)
                target[: len(value)] = value
                values.append(target)
            result[name][key] = torch.stack(values)
    kinds = [s.get("task_kind") for s in samples]
    allowed = {"forecast", "completion", "direct_action"}
    if any(kind not in allowed for kind in kinds):
        raise ValueError(
            "Every sample requires its explicit observation-mask task kind"
        )
    embodiment = samples[0]["embodiment"]
    item = dict(embodiment=embodiment, world=result, task_kind=kinds)
    if any("pixel_metadata" in s for s in samples):
        if not all("pixel_metadata" in s for s in samples):
            raise ValueError(
                "Pixel provenance metadata must be present for every sample"
            )
        item["pixel_metadata"] = [s["pixel_metadata"] for s in samples]
    if any("sampling_cursor" in s for s in samples):
        cursors = {s.get("sampling_cursor") for s in samples}
        if None in cursors or len(cursors) != 1:
            raise ValueError("A resumed batch must have one source-draw cursor")
        item["sampling_cursor"] = cursors.pop()
    if any("validation_batch_index" in s for s in samples):
        indices = [s.get("validation_batch_index") for s in samples]
        if (
            any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != 1
        ):
            raise ValueError("A validation batch requires one global batch index")
        item["validation_batch_index"] = indices
    return {embodiment: item}


@dataclass(frozen=True)
class PixelValidationIndex:
    """Keep the original validation draw identity after rank sharding."""

    index: int
    batch_index: int


class DistributedPixelTrainingBatchSampler(HomogeneousWorldBatchSampler):
    """Split each deterministic global draw into rank-local homogeneous batches.

    The inherited source distribution remains sampling with replacement. Ranks
    consume distinct draw slots, not copies of rank zero's draw. A slot may
    independently draw the same anchor as another slot, as in the original
    sampler, but its frame/mask RNG remains independently keyed by global slot.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        seed,
        *,
        world_size,
        rank,
        balance_training_work=False,
    ):
        _validate_rank(world_size, rank)
        self.world_size, self.rank = world_size, rank
        self.per_rank_batch_size = batch_size
        self.balance_training_work = balance_training_work
        super().__init__(
            dataset,
            batch_size * world_size,
            seed,
            shuffle=True,
            checkpointable=True,
        )

    def __iter__(self):
        start = self.rank * self.per_rank_batch_size
        for batch in super().__iter__():
            if self.balance_training_work:
                yield balance_global_pixel_draws(
                    self.dataset, batch, self.per_rank_batch_size, self.world_size
                )[self.rank]
            else:
                yield batch[start : start + self.per_rank_batch_size]


def balance_global_pixel_draws(dataset, draws, batch_size, world_size):
    """Assign the same global draw keys to fixed-size ranks by estimated work.

    Longest-first assignment is used only when it lowers the largest rank's
    summed squared token count. In particular B=1 and equal-length discrete
    batches retain their original assignment. No sample is added, removed, or
    moved to another optimizer update. Model noise remains rank-seeded, so a
    reassigned clip has the same source frames/masks but different model noise.
    """
    if len(draws) != batch_size * world_size:
        raise ValueError("Work balancing requires one complete global batch")
    original = [draws[i * batch_size : (i + 1) * batch_size] for i in range(world_size)]
    if batch_size == 1 or world_size == 1:
        return original
    work = {draw: dataset.predicted_token_work(draw[0], draw[1:]) for draw in draws}
    assigned = [[] for _ in range(world_size)]
    totals = [0] * world_size
    for draw in sorted(draws, key=lambda draw: (-work[draw], draw[2])):
        rank = min(
            (rank for rank in range(world_size) if len(assigned[rank]) < batch_size),
            key=lambda rank: (totals[rank], rank),
        )
        assigned[rank].append(draw)
        totals[rank] += work[draw]
    if max(totals) >= max(sum(work[draw] for draw in batch) for batch in original):
        return original
    return [sorted(batch, key=lambda draw: draw[2]) for batch in assigned]


def _validate_rank(world_size, rank):
    if (
        type(world_size) is not int
        or world_size < 1
        or type(rank) is not int
        or not 0 <= rank < world_size
    ):
        raise ValueError("Invalid distributed sampling topology")


class DistributedPixelValidationBatchSampler:
    """Contiguous exact shards, without dropped or padded validation examples.

    Equal local lengths prevent DDP forward collectives from hanging at epoch
    boundaries. An incompatible total fails explicitly instead of duplicating
    validation samples. The unsharded loader is still used for offline audits.
    """

    def __init__(self, sampler, *, world_size, rank):
        _validate_rank(world_size, rank)
        self.sampler, self.world_size, self.rank = sampler, world_size, rank
        self.global_batches = len(sampler)
        if not self.global_batches or self.global_batches % world_size:
            raise ValueError(
                "Validation batch count must be positive and divisible by world_size "
                "for exact distributed coverage without padding or dropping"
            )

    def __len__(self):
        return self.global_batches // self.world_size

    def __iter__(self):
        start = self.rank * len(self)
        for batch_index, batch in enumerate(
            islice(iter(self.sampler), start, start + len(self)), start=start
        ):
            yield [PixelValidationIndex(int(index), batch_index) for index in batch]


class EvenlySpacedEpisodeBatchSampler:
    """Deterministic held-out anchor coverage for every episode and embodiment."""

    def __init__(self, dataset, anchors_per_episode, batch_size=1):
        if dataset.split != "val":
            raise ValueError("Episode-spanning subsets are for validation only")
        if (
            type(anchors_per_episode) is not int
            or anchors_per_episode < 1
            or type(batch_size) is not int
            or batch_size < 1
        ):
            raise ValueError("Positive integer anchor and batch counts required")
        self.groups, self.batch_size = {}, batch_size
        offset = 0
        for ep, count in zip(dataset.episodes, dataset.counts):
            selected = np.linspace(
                0, count - 1, min(anchors_per_episode, count), dtype=np.int64
            )
            self.groups.setdefault(ep["embodiment"], []).extend(
                (selected + offset).tolist()
            )
            offset += count

    def __len__(self):
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.groups.values()
        )

    def __iter__(self):
        for indices in self.groups.values():
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]


def sample_full_episode_times(length, rng, min_frames=2, max_frames=128, anchor=None):
    """Uniform frame-count draw and uniform other frames over the whole episode.

    The globally sampled anchor is always retained. No fixed time window or
    nearby-frame preference is imposed. Recorded frames are not interpolated.
    """
    if (
        any(type(v) is not int for v in (length, min_frames, max_frames))
        or min_frames < 2
        or max_frames < min_frames
        or length < min_frames
    ):
        raise ValueError("Invalid full-episode frame count")
    if anchor is None:
        anchor = int(rng.integers(length))
    if not 0 <= anchor < length:
        raise ValueError("Invalid full-episode anchor")
    count = int(rng.integers(min_frames, min(max_frames, length) + 1))
    candidates = np.arange(length)
    other = rng.choice(candidates[candidates != anchor], count - 1, replace=False)
    return np.sort(np.r_[anchor, other]).astype(np.int64)


def sample_discrete_window(length, rng, frames, stride, anchor):
    """A complete raw-frame window containing anchor, grouped without overlap.

    Images use each group's first frame. Each proprio/action token contains all
    ``stride`` consecutive recorded states/commands in that group.
    """
    if (
        any(type(v) is not int for v in (length, frames, stride, anchor))
        or frames < 2
        or stride < 1
        or length < frames * stride
        or not 0 <= anchor < length
    ):
        raise ValueError("Invalid complete discrete window")
    span = frames * stride
    lower, upper = max(0, anchor - span + 1), min(anchor, length - span)
    start = int(rng.integers(lower, upper + 1))
    return np.arange(start, start + span, dtype=np.int64).reshape(frames, stride)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _coordinate_schema(schema, label):
    dimension, fields = schema.get("dimension"), schema.get("fields", [])
    if (
        type(dimension) is not int
        or dimension < 1
        or len(fields) != dimension
        or any(not isinstance(v, str) or not v for v in fields)
        or len(set(fields)) != dimension
    ):
        raise ValueError(
            f"{label} requires an explicit unique field for every coordinate"
        )


def read_selected_rgb_frames(rgb, indices):
    """Read complete selected frames without HDF5's costly point selection.

    Prepared HDF5 RGB is chunked by frame and camera. Basic integer reads keep
    those chunks intact; stacking preserves the exact requested frame order and
    dtype. NPZ members are NumPy arrays and retain their existing indexing path.
    """
    if isinstance(rgb, np.ndarray):
        return rgb[indices]
    import h5py

    if not isinstance(rgb, h5py.Dataset):
        raise TypeError("Expected an HDF5 RGB dataset or a NumPy RGB array")
    indices = np.asarray(indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise TypeError("Selected RGB frames must be a one-dimensional integer array")
    if not indices.size:
        return np.empty((0, *rgb.shape[1:]), dtype=rgb.dtype)
    return np.stack([rgb[int(index)] for index in indices], axis=0)


def patchify_rgb(rgb, patch_size):
    """Return [T,V,patches,3*P*P] with row-major patches and HWC within each patch."""
    if rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[-1] != 3:
        raise ValueError("Pixel input must be uint8 RGB[T,V,H,W,3]")
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    frames, cameras, height, width, channels = rgb.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    patches = rgb.reshape(
        frames,
        cameras,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.transpose(0, 1, 2, 4, 3, 5, 6).reshape(
        frames, cameras, height // patch_size * (width // patch_size), 3 * patch_size**2
    )
    return patches.astype(np.float32) / np.float32(127.5) - np.float32(1)


def load_pose_normalization(manifest, root, verifier):
    contract = manifest.get("normalization", {})
    if (
        contract.get("scope") != "train_only"
        or contract.get("arrays_normalized") is not False
        or not contract.get("path")
        or not _sha(contract.get("sha256"))
    ):
        raise ValueError(
            "Pixel caches require bound train-only normalization of raw poses"
        )
    path = root / contract["path"]
    verifier.verify(path, contract["sha256"])
    saved = json.loads(path.read_text())
    required = {f"pose/{emb}" for emb in manifest["pose_schemas"]}
    if (
        saved.get("schema") != "pixel-proprio-normalization-v1"
        or saved.get("scope") != "train_only"
        or saved.get("train_episodes") != training_identities(manifest["episodes"])
        or set(saved.get("statistics", {})) != required
    ):
        raise ValueError(
            "Pixel normalization must contain only poses fitted to exact training episodes"
        )
    result = {}
    for key, item in saved["statistics"].items():
        mean, std = (
            np.asarray(item["mean"], np.float32),
            np.asarray(item["std"], np.float32),
        )
        dim = manifest["pose_schemas"][key[5:]]["dimension"]
        if (
            mean.shape != (dim,)
            or std.shape != (dim,)
            or item.get("count", 0) < 1
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ValueError(f"Invalid pose normalization: {key}")
        result[key] = mean, std
    return result


class PreparedPixelProprioDataset(Dataset):
    """Image-only/conditioned visual prediction or current-frame causal action BC.

    Image-only batches retain the same frozen manifest and RGB verification but
    never open the optional pose/command caches or pose normalization artifact.
    """

    def __init__(
        self,
        manifest_path,
        split="train",
        task="visual_diffusion",
        mode="continuous",
        seed=42,
        min_frames=2,
        max_frames=128,
        discrete_frames=4,
        stride=1,
        forecasting_probability=0.5,
        max_past_seconds=None,
        max_future_seconds=None,
        verification_memo=None,
        sampling_strategy="full_episode_uniform",
        image_only=False,
    ):
        if task not in (
            "visual_diffusion",
            "visual_state_action",
            "direct_action",
        ) or split not in (
            "train",
            "val",
        ):
            raise ValueError(
                "Require a declared pixel objective and explicit train/val split"
            )
        if type(image_only) is not bool or (image_only and task != "visual_diffusion"):
            raise ValueError("image_only requires visual_diffusion and a boolean flag")
        self.image_only = image_only
        self.path, self.task, self.split, self.seed, self.mode = (
            Path(manifest_path),
            task,
            split,
            seed,
            mode,
        )
        self.manifest = json.loads(self.path.read_text())
        manifest = self.manifest
        if manifest.get("schema") != "pixel-proprio-prepared-v1":
            raise ValueError("Unsupported pixel/proprio prepared schema")
        identity = manifest.get("pixel_identity", {})
        size, patch = identity.get("image_size"), identity.get("patch_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(v) is not int or v < 1 for v in size)
            or type(patch) is not int
            or patch < 1
            or any(v % patch for v in size)
            or identity.get("channels") != 3
            or identity.get("patch_order") != "row_major_hwc"
            or identity.get("normalization") != "uint8_to_minus_one_one"
        ):
            raise ValueError(
                "Require explicit divisible RGB dimensions, patch size/order, and pixel normalization"
            )
        self.image_size, self.patch_size = tuple(size), patch
        self.grid = tuple(v // patch for v in size)
        self.image_dim = 3 * patch**2
        yy, xx = np.meshgrid(
            (np.arange(self.grid[0]) + 0.5) / self.grid[0] * 2 - 1,
            (np.arange(self.grid[1]) + 0.5) / self.grid[1] * 2 - 1,
            indexing="ij",
        )
        self.image_xy = np.stack((xx, yy), -1).reshape(-1, 2).astype(np.float32)
        if not manifest.get("episodes"):
            raise ValueError("Require explicit episode identities")
        if not image_only:
            if not manifest.get("pose_schemas"):
                raise ValueError("Require measured pose schemas")
            for schema in manifest["pose_schemas"].values():
                _coordinate_schema(schema, "Pose schema")
        self.verifier = verification_memo or ArtifactVerificationMemo()
        ids, paths, rgbs = set(), set(), set()
        self.action_alignment = {}
        for ep in manifest["episodes"]:
            key = (ep["embodiment"], ep["episode_id"])
            rgb = (self.path.parent / ep["rgb_path"]).resolve()
            if key in ids or rgb in rgbs:
                raise ValueError(
                    "Duplicate pixel episode identity or physical source across splits"
                )
            ids.add(key)
            rgbs.add(rgb)
            if (
                ep.get("split") not in ("train", "val")
                or type(ep.get("frames")) is not int
                or ep["frames"] < 1
                or not ep.get("cameras")
                or len(set(ep["cameras"])) != len(ep["cameras"])
                or not _sha(ep.get("rgb_sha256"))
            ):
                raise ValueError(
                    "Pixel episode requires exact RGB identities, cameras and split"
                )
            if not image_only:
                path = (self.path.parent / ep["path"]).resolve()
                if path in paths:
                    raise ValueError("Duplicate measured-pose source across splits")
                paths.add(path)
                if (
                    ep["embodiment"] not in manifest["pose_schemas"]
                    or ep.get("frame") != "fixed_reference"
                    or ep.get("pose_kind") != "measured_pose"
                    or not _sha(ep.get("sha256"))
                ):
                    raise ValueError("Require exact measured-pose identity and frame")
                self.verifier.verify(
                    path,
                    ep["sha256"],
                    trusted_fingerprint=ep.get("verified_fingerprint"),
                )
            self.verifier.verify(
                rgb,
                ep["rgb_sha256"],
                trusted_fingerprint=ep.get("rgb_verified_fingerprint"),
            )
            if task in ("direct_action", "visual_state_action"):
                self._validate_actions(ep)
        self.normalization = (
            {}
            if image_only
            else load_pose_normalization(manifest, self.path.parent, self.verifier)
        )
        self.episodes = [ep for ep in manifest["episodes"] if ep["split"] == split]
        if not self.episodes:
            raise ValueError("No episodes in selected split")
        if (
            (sampling_strategy, mode)
            not in (
                ("full_episode_uniform", "continuous"),
                ("discrete_window", "continuous"),
                ("discrete_window", "discrete"),
            )
            or max_past_seconds is not None
            or max_future_seconds is not None
        ):
            raise ValueError(
                "Pixel sampling requires continuous full_episode_uniform or discrete_window and no time bounds"
            )
        if (
            type(min_frames) is not int
            or type(max_frames) is not int
            or min_frames < 2
            or max_frames < min_frames
        ):
            raise ValueError("Invalid full-episode frame count")
        if not 0 <= forecasting_probability <= 1:
            raise ValueError("Invalid forecasting probability")
        if (
            type(discrete_frames) is not int
            or discrete_frames < 2
            or type(stride) is not int
            or stride < 1
        ):
            raise ValueError("Invalid discrete frame count or stride")
        if task == "direct_action" and sampling_strategy != "full_episode_uniform":
            raise ValueError(
                "Direct action baseline requires single-frame continuous observations"
            )
        self.forecasting_probability = forecasting_probability
        self.sampling = dict(min_frames=min_frames, max_frames=max_frames)
        self.sampling_strategy = sampling_strategy
        self.discrete_frames, self.stride = discrete_frames, stride
        self.chunk_size = stride if mode == "discrete" else 1
        self.anchor_indices = []
        for ep in self.episodes:
            if task == "direct_action":
                anchors = np.arange(ep["frames"])
            else:
                anchors = (
                    np.arange(ep["frames"])
                    if ep["frames"]
                    >= (
                        discrete_frames * stride
                        if sampling_strategy == "discrete_window"
                        else min_frames
                    )
                    else np.array([], np.int64)
                )
            self.anchor_indices.append(anchors)
        self.counts = [len(indices) for indices in self.anchor_indices]
        if any(c < 1 for c in self.counts):
            raise ValueError("Episode too short for temporal recipe")
        self.ends = np.cumsum(self.counts)
        self.rng = self.worker_seed = None

    def _validate_actions(self, ep):
        schema = self.manifest.get("action_schemas", {}).get(ep["embodiment"])
        provenance, source = (
            ep.get("action_provenance", {}),
            ep.get("native_source", {}),
        )
        if (
            schema is None
            or not provenance.get("path")
            or not _sha(provenance.get("sha256"))
        ):
            raise ValueError(
                "Direct action BC requires verified real command labels for every embodiment/episode"
            )
        _coordinate_schema(schema, "Action schema")
        if (
            schema.get("semantics") != "action_command"
            or schema.get("value_space") != "native_commands"
        ):
            raise ValueError("Action schemas must declare native action commands")
        path = self.path.parent / provenance["path"]
        self.verifier.verify(path, provenance["sha256"])
        proof = json.loads(path.read_text())
        if (
            proof.get("schema") != "pixel-proprio-action-alignment-v1"
            or proof.get("relation") != "action_t_applied_to_state_t"
            or not _sha(source.get("source_demo_sha256"))
            or proof.get("source_demo_sha256") != source["source_demo_sha256"]
            or not source.get("episode")
            or proof.get("source_episode") != source["episode"]
            or any(
                not _sha(proof.get(k))
                for k in (
                    "producer_sha256",
                    "raw_export_metadata_sha256",
                    "raw_payload_sha256",
                )
            )
            or any(
                not isinstance(proof.get("basis", {}).get(k), str)
                or not proof["basis"][k].strip()
                for k in (
                    "cache_index_source",
                    "action_index_source",
                    "state_action_relation_source",
                )
            )
        ):
            raise ValueError(
                "Incomplete or conflicting pixel action alignment provenance"
            )
        n = ep["frames"]
        indices = {
            k: np.asarray(proof.get(k))
            for k in ("cache_frame_index", "source_state_index", "source_action_index")
        }
        if (
            any(v.shape != (n,) or v.dtype.kind not in "iu" for v in indices.values())
            or not np.array_equal(indices["cache_frame_index"], np.arange(n))
            or not np.array_equal(
                indices["source_state_index"], indices["source_action_index"]
            )
            or np.any(indices["source_state_index"] < 0)
            or not (np.diff(indices["source_state_index"]) > 0).all()
        ):
            raise ValueError("Pixel action/state/cache index alignment mismatch")
        with open_prepared_episode(self.path.parent / ep["path"]) as data:
            if (
                "action" not in data
                or "action_valid" not in data
                or data["action"].shape != (n, schema["dimension"])
                or data["action_valid"].shape != (n,)
            ):
                raise ValueError("Pixel cache lacks declared real action arrays")
            values, valid = data["action"][:], data["action_valid"][:]
            if (
                valid.dtype != np.bool_
                or not valid.all()
                or not np.isfinite(values).all()
            ):
                raise ValueError("Direct action labels must all be finite and valid")
        self.action_alignment[(ep["embodiment"], ep["episode_id"])] = proof

    def __len__(self):
        return int(self.ends[-1])

    def _episode_anchor(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ei = int(np.searchsorted(self.ends, index, side="right"))
        offset = 0 if ei == 0 else int(self.ends[ei - 1])
        return self.episodes[ei], int(self.anchor_indices[ei][index - offset])

    def _sample_frame_chunks(self, length, anchor, rng):
        """One RNG contract shared by actual data reads and metadata-only work preview."""
        if self.task == "direct_action":
            return np.array([[anchor]])
        if self.sampling_strategy == "discrete_window":
            chunks = sample_discrete_window(
                length, rng, self.discrete_frames, self.stride, anchor
            )
            return chunks[:, :1] if self.mode == "continuous" else chunks
        return sample_full_episode_times(length, rng, anchor=anchor, **self.sampling)[
            :, None
        ]

    def predicted_token_work(self, index, draw):
        """Pure preview of image-attention work for the exact checkpointed draw.

        The preview reads manifest metadata only and never advances a worker's
        RNG. Squared image-token count estimates attention cost; it is not a
        measured GPU time or a reason to change the selected frames.
        """
        if (
            self.split != "train"
            or len(draw) != 2
            or any(type(value) is not int or value < 0 for value in draw)
        ):
            raise ValueError("Work preview requires a training cursor/global-slot draw")
        episode, anchor = self._episode_anchor(index)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, *draw]))
        frames = len(self._sample_frame_chunks(episode["frames"], anchor, rng))
        return (frames * len(episode["cameras"]) * self.grid[0] * self.grid[1]) ** 2

    def __getitem__(self, index):
        draw = None
        validation_batch_index = None
        if isinstance(index, PixelValidationIndex):
            if self.split != "val" or index.batch_index < 0:
                raise ValueError("Global validation indices apply only to validation")
            validation_batch_index, index = index.batch_index, index.index
        if isinstance(index, tuple):
            index, cursor, slot = index
            draw = int(cursor), int(slot)
        ep, anchor = self._episode_anchor(index)
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
        with ExitStack() as stack:
            rgb_data = stack.enter_context(
                open_prepared_episode(self.path.parent / ep["rgb_path"])
            )
            if any(k not in rgb_data for k in ("rgb", "time")):
                raise ValueError("Missing required raw RGB arrays")
            full_time = rgb_data["time"][:]
            _physical_times(full_time, ep["frames"])
            if not self.image_only:
                data = stack.enter_context(
                    open_prepared_episode(self.path.parent / ep["path"])
                )
                if any(k not in data for k in ("time", "pose", "pose_valid")):
                    raise ValueError("Missing required measured proprioception arrays")
                if not np.array_equal(full_time, data["time"][:]):
                    raise ValueError(
                        "RGB and measured-proprioception timestamps disagree"
                    )
            if rgb_data["rgb"].dtype != np.uint8 or rgb_data["rgb"].shape != (
                ep["frames"],
                len(ep["cameras"]),
                *self.image_size,
                3,
            ):
                raise ValueError(
                    "Native RGB shape/dtype differs from declared pixel geometry/cameras"
                )
            chunk_indices = self._sample_frame_chunks(ep["frames"], anchor, rng)
            indices = chunk_indices[:, 0]
            raw_indices = chunk_indices.reshape(-1)
            time = np.asarray(full_time[indices], np.float64)
            origin = 0 if self.task == "direct_action" else int(rng.integers(len(time)))
            physical_times = time.tolist()
            physical_origin = float(time[origin])
            time = (time - time[origin]).astype(np.float32)
            image = patchify_rgb(
                read_selected_rgb_frames(rgb_data["rgb"], indices), self.patch_size
            )
            f, cameras, patches, _ = image.shape
            if not self.image_only:
                dim = self.manifest["pose_schemas"][ep["embodiment"]]["dimension"]
                if data["pose"].shape != (ep["frames"], dim) or data[
                    "pose_valid"
                ].shape != (ep["frames"],):
                    raise ValueError(
                        "Measured proprioception shape differs from schema"
                    )
                pose, pose_valid = (
                    np.asarray(data["pose"][raw_indices], np.float32),
                    data["pose_valid"][raw_indices],
                )
                if (
                    pose_valid.dtype != np.bool_
                    or not np.isfinite(pose[pose_valid]).all()
                ):
                    raise ValueError(
                        "Measured proprioception validity or values are invalid"
                    )
                mean, std = self.normalization[f"pose/{ep['embodiment']}"]
                pose = (pose - mean) / std
                pose[~pose_valid] = 0
                pose = pose.reshape(f, self.chunk_size * dim)
                pose_valid = pose_valid.reshape(f, self.chunk_size).all(axis=1)
            # Preserve the established RNG consumption even when proprio is
            # omitted: completion's image-mask repairs and later draws must
            # select identical RGB frames/masks across conditioning variants.
            if self.task == "direct_action":
                task_kind = "direct_action"
                observed = {name: np.ones(f, bool) for name in ("image", "proprio")}
            elif rng.random() < self.forecasting_probability:
                task_kind = "forecast"
                known = np.arange(f) < int(rng.integers(1, f))
                observed = {
                    "image": known.copy(),
                    "proprio": known & (rng.random(f) < 0.8),
                }
            else:
                task_kind = "completion"
                observed = {name: rng.random(f) < 0.5 for name in ("image", "proprio")}
                if not observed["image"].any():
                    observed["image"][rng.integers(f)] = True
                if observed["image"].all():
                    observed["image"][rng.integers(f)] = False

            def base(value, repeats, known):
                n = f * repeats
                return dict(
                    value=value.reshape(n, -1),
                    time=np.repeat(time, repeats),
                    index=np.repeat(np.arange(f), repeats).astype(np.float32),
                    valid=np.ones(n, bool),
                    target_valid=np.ones(n, bool),
                    observed=np.repeat(known, repeats),
                )

            streams = {"image": base(image, cameras * patches, observed["image"])}
            streams["image"].update(
                xy=np.tile(self.image_xy, (f * cameras, 1)),
                camera=np.tile(np.repeat(np.arange(cameras), patches), f),
            )
            if not self.image_only:
                streams["proprio"] = base(pose, 1, observed["proprio"])
                streams["proprio"]["target_valid"] = pose_valid
                streams["proprio"]["observed"] &= pose_valid
            if self.task in ("direct_action", "visual_state_action"):
                action = np.asarray(data["action"][raw_indices], np.float32).reshape(
                    f, -1
                )
                streams["action"] = base(action, 1, np.zeros(f, bool))
                if task_kind == "forecast":
                    # Past command queries can see later observed context. They
                    # remain real valid labels, but are not causal BC targets.
                    streams["action"]["target_valid"] &= ~observed["image"]
        sample = dict(
            embodiment=ep["embodiment"],
            task_kind=task_kind,
            pixel_metadata=dict(
                episode_id=ep["episode_id"],
                camera_names=list(ep["cameras"]),
                frame_indices=indices.tolist(),
                timestamps_seconds=physical_times,
                time_origin_seconds=physical_origin,
                chunk_size=self.chunk_size,
                sampling_strategy=self.sampling_strategy,
                image_size=list(self.image_size),
                patch_size=self.patch_size,
                normalization="uint8_to_minus_one_one",
                patch_order="row_major_hwc",
            ),
            world={
                name: {
                    key: torch.from_numpy(np.array(value, copy=True))
                    for key, value in stream.items()
                }
                for name, stream in streams.items()
            },
        )
        if not self.image_only:
            sample["pixel_metadata"].update(
                pose_base_dimension=dim,
                proprio_frame_indices=chunk_indices.tolist(),
                proprio_timestamps_seconds=full_time[chunk_indices].tolist(),
            )
        if draw is not None:
            sample["sampling_cursor"] = draw[0] + 1
        if validation_batch_index is not None:
            sample["validation_batch_index"] = validation_batch_index
        return sample


__all__ = [
    "PreparedPixelProprioDataset",
    "collate_pixel_proprio",
    "patchify_rgb",
    "EvenlySpacedEpisodeBatchSampler",
]
