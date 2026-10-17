"""OAT replay buffers exposed through EgoVerse's MultiDataset/data module.

Preserves upstream episode splits, frame padding, action alignment and affine
normalization. No OAT Dataset, Workspace or training loop is imported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from egomimic.benchmarks.libero.catalog import validate_task_coverage
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

OBS_KEYS = (
    "agentview_rgb",
    "robot0_eye_in_hand_rgb",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "task_uid",
)
EMBODIMENT = get_embodiment_id("libero_panda")


def keymap(n_obs_steps=2, horizon=32, norm_mode=False):
    del norm_mode  # Pixel channel limits are part of OAT's data normalization.
    result = {
        "actions": {"zarr_key": "action", "key_type": "action_keys", "horizon": horizon}
    }
    if n_obs_steps:
        for key in OBS_KEYS:
            result[key] = {
                "zarr_key": key,
                "horizon": n_obs_steps,
                "key_type": "camera_keys" if key.endswith("rgb") else "proprio_keys",
            }
    return result


def validation_mask(n_episodes, val_ratio=0.1, seed=42):
    if not 0 <= val_ratio < 1 or n_episodes < 1:
        raise ValueError("Invalid episode count or validation ratio")
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio:
        count = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
        mask[np.random.default_rng(seed).choice(n_episodes, count, replace=False)] = (
            True
        )
    return mask


class LiberoReplayResolver:
    embodiment = "libero_panda"

    def validate_task_ids(self, task_ids):
        validate_task_coverage(task_ids, self.suite)

    def __init__(
        self,
        folder_path,
        key_map,
        suite,
        decoded_cache=False,
        action_dim=7,
        embodiment="libero_panda",
        gripper_state_dim=2,
        action_representation="native",
    ):
        self.folder_path = str(folder_path)
        self.key_map = dict(key_map)
        self.suite = suite
        self.decoded_cache = bool(decoded_cache)
        self._decoded = None
        self.action_dim = int(action_dim)
        self.embodiment = str(embodiment)
        self.embodiment_id = get_embodiment_id(self.embodiment)
        self.gripper_state_dim = int(gripper_state_dim)
        self.action_representation = str(action_representation)
        if self.action_representation not in ("native", "wrist_hand_keypoints"):
            raise ValueError("Unknown LIBERO action representation")
        if self.action_dim <= 0 or self.gripper_state_dim <= 0:
            raise ValueError("Action and hand-state dimensions must be positive")
        if self.embodiment == "libero_panda" and (
            (self.action_dim, self.gripper_state_dim) != (7, 2)
            or self.action_representation != "native"
        ):
            raise ValueError(
                "Original Panda contract requires action7 and gripper state2"
            )
        if self.embodiment == "libero_panda_sharpa":
            expected = 28 if self.action_representation == "native" else 69
            if (self.action_dim, self.gripper_state_dim) != (expected, 22):
                raise ValueError(
                    f"Sharpa contract requires action{expected} and hand state22"
                )

    def open_arrays(self):
        keys = {info["zarr_key"] for info in self.key_map.values()}
        if self.decoded_cache:
            if self._decoded is None:
                from egomimic.rldb.zarr.decoded_replay import prepare_decoded_replay

                self._decoded = prepare_decoded_replay(self.folder_path, keys)
            return self._decoded.open()
        data = zarr.open_group(self.folder_path, mode="r")["data"]
        return {key: data[key] for key in keys}

    def resolve(self):
        if self.action_representation == "wrist_hand_keypoints":
            manifest = json.loads(
                (Path(self.folder_path) / "benchmark_manifest.json").read_text()
            )
            if (
                not manifest.get("complete")
                or manifest.get("action_dim") != 69
                or manifest.get("action_representation") != self.action_representation
                or manifest.get("keypoint_frame") != "wrist_relative_Quest_XYZ_m"
                or manifest.get("observation_alignment")
                != "causal_pre_action_raw_lossless"
            ):
                raise ValueError("Keypoint representation/frame manifest differs")
        group = zarr.open_group(self.folder_path, mode="r")
        ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
        if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(np.r_[0, ends]) <= 0):
            raise ValueError("episode_ends must be strictly increasing and nonempty")
        data = group["data"]
        if data["action"].shape != (ends[-1], self.action_dim):
            raise ValueError(
                f"{self.embodiment} action must be (frames,{self.action_dim})"
            )
        if "robot0_gripper_qpos" in data and data["robot0_gripper_qpos"].shape != (
            ends[-1],
            self.gripper_state_dim,
        ):
            raise ValueError(
                f"{self.embodiment} hand-state shape differs from declared contract"
            )
        for info in self.key_map.values():
            if data[info["zarr_key"]].shape[0] != ends[-1]:
                raise ValueError("Replay array lengths differ")
        self.validate_task_ids(np.unique(data["task_uid"][:]))
        if self.decoded_cache:
            self.open_arrays()
        return {
            f"episode_{index:06d}": _ReplayEpisode(
                self.folder_path,
                int(start),
                int(end),
                self.key_map,
                index,
                decoded=self._decoded,
                embodiment=self.embodiment,
            )
            for index, (start, end) in enumerate(zip(np.r_[0, ends[:-1]], ends))
        }


class _ReplayEpisode(torch.utils.data.Dataset):
    embodiment = "libero_panda"

    def __init__(
        self, path, start, end, key_map, index, decoded=None, embodiment="libero_panda"
    ):
        self.path, self.start, self.end = path, start, end
        self.key_map = key_map
        self.episode_path = Path(path) / f"episode_{index:06d}"
        self._arrays = None
        self._decoded = decoded
        self.embodiment = str(embodiment)
        self.embodiment_id = get_embodiment_id(self.embodiment)

    def __len__(self):
        return self.end - self.start

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = None
        return state

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        if self._arrays is None:
            if self._decoded is not None:
                self._arrays = self._decoded.open()
            else:
                group = zarr.open_group(self.path, mode="r")["data"]
                # Reuse handles for the uncached reference path.
                self._arrays = {
                    info["zarr_key"]: group[info["zarr_key"]]
                    for info in self.key_map.values()
                }
        result = {}
        for key, info in self.key_map.items():
            horizon = int(info["horizon"])
            # Observations end at t; a_t is the FIRST target. Pad within this episode.
            offsets = (
                np.arange(horizon) if key == "actions" else np.arange(1 - horizon, 1)
            )
            indices = np.clip(idx + offsets, 0, len(self) - 1) + self.start
            array = self._arrays[info["zarr_key"]]
            values = np.asarray(
                array[indices] if self._decoded is not None else array.oindex[indices]
            )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {key} in {self.episode_path}")
            result[key] = torch.from_numpy(np.ascontiguousarray(values)).float()
        result.update(
            embodiment=self.embodiment_id,
            episode_hash=self.episode_path.name,
            frame_index=idx,
        )
        return result


class LiberoDataset(MultiDataset):
    NORMALIZE_KEY_TYPES = ("proprio_keys", "action_keys", "camera_keys")

    @classmethod
    def _from_resolver(cls, resolver, mode="train", valid_ratio=0.1, split_seed=42):
        leaves = resolver.resolve()
        mask = validation_mask(len(leaves), valid_ratio, split_seed)
        if mode not in ("train", "valid", "total"):
            raise ValueError("mode must be train, valid or total")
        selected = {
            key: value
            for i, (key, value) in enumerate(leaves.items())
            if mode == "total" or bool(mask[i]) == (mode == "valid")
        }
        if not selected:
            raise ValueError(f"Empty {mode} split")
        dataset = cls(
            datasets=selected, mode="total", valid_ratio=0, bounds_check=False
        )
        dataset.resolver = resolver
        dataset.split_seed, dataset.val_ratio = int(split_seed), float(valid_ratio)
        dataset.split_mode = mode
        return dataset

    def __getitem__(self, idx, _attempts=None):
        # Benchmarks fail on corrupt samples; never replace a validation frame.
        name, offset = self.index_map[idx]
        sample = self.datasets[name][offset]
        return self.normalize(sample, sample["embodiment"])

    def _apply_norm_one(self, tensor, stats):
        scale = torch.as_tensor(
            stats["scale"], device=tensor.device, dtype=torch.float32
        )
        offset = torch.as_tensor(
            stats["offset"], device=tensor.device, dtype=torch.float32
        )
        return tensor.float() * scale + offset

    def _apply_unnorm_one(self, tensor, stats):
        scale = torch.as_tensor(
            stats["scale"], device=tensor.device, dtype=torch.float32
        )
        offset = torch.as_tensor(
            stats["offset"], device=tensor.device, dtype=torch.float32
        )
        return (tensor.float() - offset) / scale


class LiberoNormalizer(LiberoDataset):
    """OAT limits fit over all replay frames, including the validation split.

    This scope deliberately matches the released baseline. The common stats
    and episode split are recorded with both methods' checkpoints.
    """

    def __init__(self, state=None, norm_mode="oat_limits", **kwargs):
        if norm_mode != "oat_limits":
            raise ValueError("OAT comparisons require norm_mode=oat_limits")
        super().__init__(
            state={} if state is None else state, norm_mode=norm_mode, **kwargs
        )
        self.context = (state or {}).get("benchmark_context", {})

    def infer_norm_from_dataset(self, dataset, dataset_name, **kwargs):
        del dataset_name
        if kwargs.get("precomputed_norm_path") is not None:
            raise ValueError(
                "LIBERO fits exact channel limits; use its saved normalizer_state for evaluation"
            )
        group = zarr.open_group(dataset.resolver.folder_path, mode="r")
        arrays = dataset.resolver.open_arrays()
        digest = hashlib.sha256(np.asarray(group["meta/episode_ends"][:]).tobytes())
        # Hash actions and task identity in replay order, independent of path/horizon.
        for key in ("action", "task_uid"):
            # Tokenizer keymaps omit task_uid but its identity is still hashed.
            array = arrays[key] if key in arrays else group["data"][key]
            for start in range(0, array.shape[0], 4096):
                digest.update(
                    np.ascontiguousarray(array[start : start + 4096]).tobytes()
                )
        self.context = {
            "dataset_sha256": digest.hexdigest(),
            "suite": dataset.resolver.suite,
            "split_seed": dataset.split_seed,
            "val_ratio": dataset.val_ratio,
        }
        stats = self.norm_stats.setdefault(
            get_embodiment_id(dataset.resolver.embodiment), {}
        )
        observation_digest = hashlib.sha256()
        for key, info in dataset.resolver.key_map.items():
            array = arrays[info["zarr_key"]]
            if key != "actions":
                observation_digest.update(key.encode())
            low = np.full(array.shape[-1], np.inf, dtype=np.float32)
            high = -low.copy()
            # Bound memory for image arrays as well as state arrays.
            for start in range(0, array.shape[0], 64):
                values = np.asarray(
                    array[start : start + 64], dtype=np.float32
                ).reshape(-1, array.shape[-1])
                if key != "actions":
                    observation_digest.update(np.ascontiguousarray(values).tobytes())
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite normalization data in {key}")
                low = np.minimum(low, values.min(axis=0))
                high = np.maximum(high, values.max(axis=0))
            span = high - low
            constant = span < 1e-4
            scale = 2 / np.where(constant, 2, span)
            offset = np.where(constant, -low, -1 - scale * low)
            stats[key] = {"min": low, "max": high, "scale": scale, "offset": offset}
        self.context["observations_sha256"] = observation_digest.hexdigest()

    def to_state(self):
        state = super().to_state()
        state["benchmark_context"] = self.context
        return state

    def tokenizer_context(self):
        return {
            **{
                key: value
                for key, value in self.context.items()
                if key != "observations_sha256"
            },
            "action_scale": self.norm_stats[EMBODIMENT]["actions"]["scale"].tolist(),
            "action_offset": self.norm_stats[EMBODIMENT]["actions"]["offset"].tolist(),
        }

    def assert_tokenizer_context(self, reference):
        if self.tokenizer_context() != reference:
            raise ValueError(
                "Tokenizer and policy dataset/split/action-normalizer differ"
            )


class LiberoMultiEmbodimentNormalizer(LiberoDataset):
    """Separate native affine limits fit only on each selected training split.

    Historical LiberoNormalizer keeps its all-replay baseline contract. This
    adapter records a distinct metric/data identity and never mixes hand widths.
    """

    def __init__(self, state=None, norm_mode="libero_train_limits", **kwargs):
        if norm_mode != "libero_train_limits":
            raise ValueError("Joint native LIBERO requires train-only affine limits")
        super().__init__(
            state={} if state is None else state, norm_mode=norm_mode, **kwargs
        )
        self.contexts = dict((state or {}).get("native_embodiment_contexts", {}))

    def infer_norm_from_dataset(self, dataset, dataset_name, **kwargs):
        if dataset.split_mode != "train":
            raise ValueError("Normalization may only fit selected training episodes")
        if kwargs.get("precomputed_norm_path") is not None:
            raise ValueError("Use a complete saved multi-embodiment data context")
        resolver = dataset.resolver
        identity = resolver.embodiment_id
        if str(identity) in self.contexts:
            raise ValueError("One immutable native source per embodiment is required")
        arrays = resolver.open_arrays()
        stats = self.norm_stats.setdefault(identity, {})
        digest = hashlib.sha256()
        selected = sorted(dataset.datasets.items())
        for name, _ in selected:
            digest.update(name.encode())
        for key, info in resolver.key_map.items():
            array = arrays[info["zarr_key"]]
            low = np.full(array.shape[-1], np.inf, dtype=np.float32)
            high = -low.copy()
            digest.update(key.encode())
            for _, episode in selected:
                for start in range(episode.start, episode.end, 64):
                    values = np.asarray(
                        array[start : min(start + 64, episode.end)], dtype=np.float32
                    )
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"Non-finite training normalization in {dataset_name}/{key}"
                        )
                    digest.update(np.ascontiguousarray(values).tobytes())
                    flat = values.reshape(-1, array.shape[-1])
                    low = np.minimum(low, flat.min(axis=0))
                    high = np.maximum(high, flat.max(axis=0))
            span = high - low
            constant = span < 1e-4
            scale = 2 / np.where(constant, 2, span)
            offset = np.where(constant, -low, -1 - scale * low)
            stats[key] = {"min": low, "max": high, "scale": scale, "offset": offset}
        self.contexts[str(identity)] = {
            "embodiment": resolver.embodiment,
            "action_dim": resolver.action_dim,
            "action_representation": resolver.action_representation,
            "hand_state_dim": resolver.gripper_state_dim,
            "suite": resolver.suite,
            "split_seed": dataset.split_seed,
            "val_ratio": dataset.val_ratio,
            "normalization_scope": "selected_training_episodes_only",
            "train_episode_ids": [name for name, _ in selected],
            "train_content_sha256": digest.hexdigest(),
        }

    def to_state(self):
        state = super().to_state()
        state["native_embodiment_contexts"] = self.contexts
        return state

    def tokenizer_context(self):
        return {
            "native_embodiments": {
                identity: {
                    **context,
                    "action_scale": np.asarray(
                        self.norm_stats[int(identity)]["actions"]["scale"]
                    ).tolist(),
                    "action_offset": np.asarray(
                        self.norm_stats[int(identity)]["actions"]["offset"]
                    ).tolist(),
                }
                for identity, context in self.contexts.items()
            }
        }

    def assert_tokenizer_context(self, reference):
        if self.tokenizer_context() != reference:
            raise ValueError("Native embodiment/split/normalizer identity differs")
