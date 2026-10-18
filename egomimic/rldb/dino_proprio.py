"""DINO and measured-proprioception caches, with optional verified action labels.

Point tracks are neither required nor loaded. Action commands are never inferred
from measured poses: direct BC requires a separate, SHA-bound alignment receipt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from egomimic.rldb.world_model import (
    _bounded_time_sampling,
    _physical_times,
    open_prepared_episode,
    sample_times,
)
from egomimic.rldb.world_model_preparation import (
    ArtifactVerificationMemo,
    training_identities,
)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _coordinate_schema(schema, label):
    dimension = schema.get("dimension")
    fields = schema.get("fields", [])
    if (
        type(dimension) is not int
        or dimension < 1
        or len(fields) != dimension
        or any(not isinstance(field, str) or not field for field in fields)
        or len(set(fields)) != dimension
    ):
        raise ValueError(
            f"{label} requires explicit unique fields for every coordinate"
        )


def load_dino_proprio_normalization(manifest, root, verifier):
    """Use only image/pose statistics, including from historical three-stream files."""
    contract = manifest["normalization"]
    if (
        contract.get("scope") != "train_only"
        or not _sha(contract.get("sha256"))
        or type(contract.get("arrays_normalized")) is not bool
    ):
        raise ValueError("Require train-only normalization provenance")
    if contract["arrays_normalized"] and not contract.get("path"):
        return None
    if not contract.get("path"):
        raise ValueError(
            "Raw caches require bound image/proprio normalization statistics"
        )
    path = root / contract["path"]
    verifier.verify(path, contract["sha256"])
    saved = json.loads(path.read_text())
    if (
        saved.get("schema")
        not in ("mot-world-normalization-v1", "dino-proprio-normalization-v1")
        or saved.get("scope") != "train_only"
        or saved.get("train_episodes") != training_identities(manifest["episodes"])
    ):
        raise ValueError(
            "Normalization statistics do not match training episode identities"
        )
    arrays = {}
    required = {"image"} | {f"pose/{name}" for name in manifest["pose_schemas"]}
    for key in required:
        item = saved.get("statistics", {}).get(key)
        if item is None:
            raise ValueError(f"Missing normalization statistics: {key}")
        mean, std = (
            np.asarray(item["mean"], np.float32),
            np.asarray(item["std"], np.float32),
        )
        if (
            mean.ndim != 1
            or not len(mean)
            or mean.shape != std.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
            or item.get("count", 0) < 1
        ):
            raise ValueError(f"Invalid normalization statistics: {key}")
        if (
            key.startswith("pose/")
            and len(mean) != manifest["pose_schemas"][key[5:]]["dimension"]
        ):
            raise ValueError(f"Normalization schema width mismatch: {key}")
        arrays[key] = mean, std
    return arrays if not contract["arrays_normalized"] else None


def collate_dino_proprio(samples):
    """Pad only the declared streams; action values remain label-only tensors."""
    if not samples or len({s["embodiment"] for s in samples}) != 1:
        raise ValueError("Batch must be nonempty and homogeneous in embodiment")
    names = set(samples[0]["world"])
    if names not in ({"image", "proprio"}, {"image", "proprio", "action"}):
        raise ValueError("Expected DINO/proprio with optional action labels")
    if any(set(s["world"]) != names for s in samples):
        raise ValueError("Cannot mix training objectives in a batch")
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
                padded = torch.zeros((size, *value.shape[1:]), dtype=value.dtype)
                padded[: len(value)] = value
                values.append(padded)
            result[name][key] = torch.stack(values)
    embodiment = samples[0]["embodiment"]
    kinds = [sample.get("task_kind") for sample in samples]
    allowed = {"direct_action"} if "action" in names else {"forecast", "completion"}
    if any(kind not in allowed for kind in kinds):
        raise ValueError(
            "Every sample requires its explicit observation-mask task kind"
        )
    item = {"embodiment": embodiment, "world": result, "task_kind": kinds}
    if any("sampling_cursor" in s for s in samples):
        cursors = {s.get("sampling_cursor") for s in samples}
        if None in cursors or len(cursors) != 1:
            raise ValueError("A resumed batch must have one source-draw cursor")
        item["sampling_cursor"] = cursors.pop()
    return {embodiment: item}


class EvenlySpacedEpisodeBatchSampler:
    """Fixed held-out anchors spanning every episode, with homogeneous batches."""

    def __init__(self, dataset, anchors_per_episode, batch_size=1):
        if getattr(dataset, "split", None) != "val":
            raise ValueError("Episode-spanning anchor subsets are for validation only")
        if type(anchors_per_episode) is not int or anchors_per_episode < 1:
            raise ValueError(
                "validation_anchors_per_episode must be a positive integer"
            )
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive batch size required")
        self.batch_size = batch_size
        self.groups = {}
        offset = 0
        for episode, count in zip(dataset.episodes, dataset.counts):
            selected = np.linspace(
                0, count - 1, min(anchors_per_episode, count), dtype=np.int64
            )
            self.groups.setdefault(episode["embodiment"], []).extend(
                (selected + offset).tolist()
            )
            offset += count

    def __len__(self):
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )

    def __iter__(self):
        for indices in self.groups.values():
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]


class PreparedDinoProprioDataset(Dataset):
    """DINO + proprio diffusion/readout, or same-frame causal action BC.

    Direct BC reads exactly one observed frame and its aligned action command.
    The visual objective retains the configured recorded-time sampler and
    observation masks. A global anchor index makes source draws frame-weighted.
    """

    def __init__(
        self,
        manifest_path,
        split="train",
        task="visual_diffusion",
        action_manifest_path=None,
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
        if task not in ("visual_diffusion", "direct_action"):
            raise ValueError("Unknown DINO/proprio objective")
        if task == "direct_action" and not action_manifest_path:
            raise ValueError(
                "Direct action BC requires a verified action alignment manifest"
            )
        if split not in ("train", "val"):
            raise ValueError("Explicit train/val split required")
        self.path, self.task = Path(manifest_path), task
        self.manifest = json.loads(self.path.read_text())
        manifest = self.manifest
        if manifest.get("schema") not in (
            "mot-world-prepared-v1",
            "dino-proprio-prepared-v1",
        ):
            raise ValueError("Unsupported DINO/proprio prepared schema")
        for key in ("dino_identity", "normalization", "pose_schemas", "episodes"):
            if not manifest.get(key):
                raise ValueError(f"Missing {key}")
        identity = manifest["dino_identity"]
        for key in ("model", "revision", "image_size", "patch_grid", "feature_layer"):
            if not identity.get(key):
                raise ValueError(f"Missing DINO identity field {key}")
        for key in ("image_size", "patch_grid"):
            value = identity[key]
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(type(x) is not int or x < 1 for x in value)
            ):
                raise ValueError(f"DINO {key} requires two positive integers")
        for schema in manifest["pose_schemas"].values():
            _coordinate_schema(schema, "Pose schema")
        self.verifier = verification_memo or ArtifactVerificationMemo()
        self.normalization = load_dino_proprio_normalization(
            manifest, self.path.parent, self.verifier
        )
        ids, paths = set(), set()
        for ep in manifest["episodes"]:
            key = (ep["embodiment"], ep["episode_id"])
            physical = (self.path.parent / ep["path"]).resolve()
            if key in ids or physical in paths:
                raise ValueError(
                    "Duplicate episode identity or physical path across splits"
                )
            ids.add(key)
            paths.add(physical)
            if (
                ep.get("frame") != "fixed_reference"
                or ep.get("pose_kind") != "measured_pose"
                or ep["embodiment"] not in manifest["pose_schemas"]
            ):
                raise ValueError(
                    "Require measured proprioception in a declared fixed reference frame"
                )
            if (
                ep.get("split") not in ("train", "val")
                or type(ep.get("frames")) is not int
                or ep["frames"] < 1
            ):
                raise ValueError("Explicit split and positive frame count required")
            if not _sha(ep.get("sha256")):
                raise ValueError("Require feature-cache SHA256")
            self.verifier.verify(
                physical,
                ep["sha256"],
                trusted_fingerprint=ep.get("verified_fingerprint"),
            )
        self.action_path = Path(action_manifest_path) if action_manifest_path else None
        self.action_manifest, self.actions, self.action_alignment = None, {}, {}
        if task == "direct_action":
            self._load_actions()
        self.episodes = [ep for ep in manifest["episodes"] if ep["split"] == split]
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
        if (
            mode not in ("continuous", "discrete")
            or min_frames < 2
            or max_frames < min_frames
            or stride < 1
        ):
            raise ValueError("Invalid time sampler")
        if bounded and mode != "continuous":
            raise ValueError("Physical-time bounds apply only to continuous sampling")
        if bounded:
            self.sampling.update(
                max_past_seconds=max_past_seconds, max_future_seconds=max_future_seconds
            )
        if not 0 <= forecasting_probability <= 1:
            raise ValueError("Invalid forecasting probability")
        self.forecasting_probability = forecasting_probability
        self.anchor_indices = []
        for ep in self.episodes:
            if task == "direct_action":
                anchors = np.arange(ep["frames"])
            elif bounded:
                with open_prepared_episode(self.path.parent / ep["path"]) as saved:
                    times = _physical_times(saved["time"][:], ep["frames"])
                lo = np.searchsorted(times, times - max_past_seconds, side="left")
                hi = np.searchsorted(times, times + max_future_seconds, side="right")
                anchors = np.flatnonzero(hi - lo >= min_frames)
            else:
                count = ep["frames"] - (
                    (discrete_frames - 1) * stride if mode == "discrete" else 0
                )
                anchors = (
                    np.arange(max(0, count))
                    if ep["frames"] >= min_frames
                    else np.array([], np.int64)
                )
            self.anchor_indices.append(anchors)
        self.counts = [len(a) for a in self.anchor_indices]
        if any(c < 1 for c in self.counts):
            raise ValueError("Episode too short for temporal recipe")
        self.ends = np.cumsum(self.counts)
        self.rng = self.worker_seed = None

    def _load_actions(self):
        self.action_manifest = json.loads(self.action_path.read_text())
        action = self.action_manifest
        if action.get("schema") != "dino-proprio-actions-v1":
            raise ValueError("Unsupported action alignment schema")
        schemas = action.get("action_schemas", {})
        for schema in schemas.values():
            _coordinate_schema(schema, "Action schema")
            if (
                schema.get("semantics") != "action_command"
                or schema.get("value_space") != "native_commands"
            ):
                raise ValueError("Action schema must declare native action commands")
        for item in action.get("episodes", []):
            key = (item["embodiment"], item["episode_id"])
            if key in self.actions:
                raise ValueError("Duplicate action episode identity")
            self.actions[key] = item
        expected = {
            (ep["embodiment"], ep["episode_id"]) for ep in self.manifest["episodes"]
        }
        if set(self.actions) != expected:
            raise ValueError(
                "Action manifest must match every prepared episode exactly"
            )
        for ep in self.manifest["episodes"]:
            item = self.actions[(ep["embodiment"], ep["episode_id"])]
            source = ep.get("native_source", {})
            proof = item.get("alignment", {})
            if (
                ep["embodiment"] not in schemas
                or item.get("feature_sha256") != ep["sha256"]
                or not _sha(source.get("source_demo_sha256"))
                or item.get("source_demo_sha256") != source["source_demo_sha256"]
                or not source.get("episode")
                or item.get("source_episode") != source["episode"]
                or proof.get("relation") != "action_t_applied_to_state_t"
                or proof.get("cache_frame_mapping") != "explicit_source_indices"
                or not _sha(proof.get("producer_sha256"))
                or not _sha(proof.get("source_alignment_sha256"))
                or not proof.get("source_alignment_path")
            ):
                raise ValueError(
                    "Action alignment proof must bind source commands to exact feature frames"
                )
            # A digest string is not evidence of source alignment. Require the
            # actual bound receipt and inspect its source identities and mapping.
            # This authenticates the exporter-recorded mapping, not the truth of
            # an unknown controller convention; that remains an upstream audit.
            receipt_path = self.action_path.parent / proof["source_alignment_path"]
            self.verifier.verify(receipt_path, proof["source_alignment_sha256"])
            receipt = json.loads(receipt_path.read_text())
            basis = receipt.get("basis", {})
            if (
                receipt.get("schema") != "dino-proprio-source-alignment-v1"
                or any(
                    receipt.get(k) != item[k]
                    for k in ("feature_sha256", "source_demo_sha256", "source_episode")
                )
                or receipt.get("relation") != proof["relation"]
                or receipt.get("producer_sha256") != proof["producer_sha256"]
                or any(
                    not isinstance(basis.get(k), str) or not basis[k].strip()
                    for k in (
                        "cache_index_source",
                        "action_index_source",
                        "state_action_relation_source",
                    )
                )
            ):
                raise ValueError(
                    "Action source alignment receipt has incomplete or conflicting provenance"
                )
            self.action_alignment[(ep["embodiment"], ep["episode_id"])] = receipt
            self.verifier.verify(
                self.action_path.parent / item["path"],
                item["sha256"],
                trusted_fingerprint=item.get("verified_fingerprint"),
            )
            # Fail during dataset setup, before optimization, for any malformed
            # alignment or invalid command. These arrays are tiny relative to DINO.
            with open_prepared_episode(self.path.parent / ep["path"]) as cache:
                time = cache["time"][:]
            self._read_actions(ep, np.arange(ep["frames"]), time)

    def _read_actions(self, ep, indices, cache_time):
        item = self.actions[(ep["embodiment"], ep["episode_id"])]
        receipt = self.action_alignment[(ep["embodiment"], ep["episode_id"])]
        schema = self.action_manifest["action_schemas"][ep["embodiment"]]
        with open_prepared_episode(self.action_path.parent / item["path"]) as data:
            names = (
                "action",
                "time",
                "valid",
                "cache_frame_index",
                "source_state_index",
                "source_action_index",
            )
            if any(name not in data for name in names):
                raise ValueError("Action sidecar missing explicit alignment arrays")
            n = ep["frames"]
            if data["action"].shape != (n, schema["dimension"]) or any(
                data[k].shape != (n,) for k in names[1:]
            ):
                raise ValueError("Action alignment array shape mismatch")
            for name in names[3:]:
                recorded = np.asarray(receipt.get(name))
                if (
                    np.asarray(data[name][:]).dtype.kind not in "iu"
                    or recorded.dtype.kind not in "iu"
                    or recorded.shape != (n,)
                ):
                    raise ValueError("Source alignment indices must be integers")
                if not np.array_equal(data[name][:], recorded):
                    raise ValueError(
                        "Action alignment arrays differ from the bound source receipt"
                    )
            state, command = (
                data["source_state_index"][:],
                data["source_action_index"][:],
            )
            if (
                not np.array_equal(data["cache_frame_index"][:], np.arange(n))
                or not np.array_equal(state, command)
                or np.any(state < 0)
                or not (np.diff(state) > 0).all()
                or not np.array_equal(data["time"][:], cache_time)
            ):
                raise ValueError("Action/state/cache-frame alignment mismatch")
            value = np.asarray(data["action"][indices], np.float32)
            valid = np.asarray(data["valid"][indices])
            if (
                valid.dtype != np.bool_
                or not valid.all()
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "Direct action BC requires finite valid commands at every cache frame"
                )
        return value

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        draw = None
        if isinstance(index, tuple):
            index, cursor, slot = index
            draw = int(cursor), int(slot)
        if not 0 <= index < len(self):
            raise IndexError(index)
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
        ei = int(np.searchsorted(self.ends, index, side="right"))
        ep = self.episodes[ei]
        offset = 0 if ei == 0 else int(self.ends[ei - 1])
        anchor = int(self.anchor_indices[ei][index - offset])
        with open_prepared_episode(self.path.parent / ep["path"]) as data:
            if any(
                k not in data
                for k in ("time", "image", "image_xy", "pose", "pose_valid")
            ):
                raise ValueError("Prepared cache missing required DINO/proprio arrays")
            full_time = data["time"][:]
            _physical_times(full_time, ep["frames"])
            indices = (
                np.array([anchor])
                if self.task == "direct_action"
                else sample_times(
                    ep["frames"],
                    rng,
                    anchor=anchor,
                    timestamps=full_time,
                    **self.sampling,
                )
            )
            time = np.asarray(full_time[indices], np.float64)
            origin = 0 if self.task == "direct_action" else int(rng.integers(len(time)))
            time = (time - time[origin]).astype(np.float32)
            image, pose = (
                np.asarray(data[k][indices], np.float32) for k in ("image", "pose")
            )
            if (
                image.ndim != 4
                or pose.ndim != 2
                or data["image"].shape[0] != ep["frames"]
                or data["pose"].shape[0] != ep["frames"]
                or pose.shape[-1]
                != self.manifest["pose_schemas"][ep["embodiment"]]["dimension"]
            ):
                raise ValueError("Expected image[T,V,P,D] and measured pose[T,Dm]")
            f, cameras, patches, _ = image.shape
            if patches != np.prod(self.manifest["dino_identity"]["patch_grid"]):
                raise ValueError("DINO patch grid differs from cached image tokens")
            if data["image_xy"].shape != (patches, 2) or data["pose_valid"].shape != (
                ep["frames"],
            ):
                raise ValueError(
                    "Prepared geometry/mask shapes differ from token arrays"
                )
            pose_valid = np.asarray(data["pose_valid"][indices], bool)
            if not np.isfinite(image).all() or not np.isfinite(pose[pose_valid]).all():
                raise ValueError("Non-finite valid DINO/proprio values")
            if self.normalization is not None:
                im, scale = self.normalization["image"]
                pm, pscale = self.normalization[f"pose/{ep['embodiment']}"]
                if image.shape[-1] != len(im):
                    raise ValueError("Normalization schema width mismatch: image")
                image, pose = (image - im) / scale, (pose - pm) / pscale
            # Absent measurements cannot contaminate token projections/losses.
            pose[~pose_valid] = 0
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

            def base(value, repeats, observed_frames):
                n = f * repeats
                return dict(
                    value=value.reshape(n, -1),
                    time=np.repeat(time, repeats),
                    index=np.repeat(np.arange(f), repeats).astype(np.float32),
                    valid=np.ones(n, bool),
                    target_valid=np.ones(n, bool),
                    observed=np.repeat(observed_frames, repeats),
                )

            streams = {
                "image": base(image, cameras * patches, observed["image"]),
                "proprio": base(pose, 1, observed["proprio"]),
            }
            xy = np.asarray(data["image_xy"][:], np.float32)
            if not np.isfinite(xy).all():
                raise ValueError("Non-finite image patch coordinates")
            streams["image"].update(
                xy=np.tile(xy, (f * cameras, 1)),
                camera=np.tile(np.repeat(np.arange(cameras), patches), f),
            )
            streams["proprio"]["target_valid"] = pose_valid
            streams["proprio"]["observed"] &= pose_valid
            if self.task == "direct_action":
                action = self._read_actions(ep, indices, full_time)
                streams["action"] = base(action, 1, np.zeros(f, bool))
        sample = {
            "embodiment": ep["embodiment"],
            "task_kind": task_kind,
            "world": {
                name: {
                    key: torch.from_numpy(np.array(value, copy=True))
                    for key, value in stream.items()
                }
                for name, stream in streams.items()
            },
        }
        if draw is not None:
            sample["sampling_cursor"] = draw[0] + 1
        return sample
