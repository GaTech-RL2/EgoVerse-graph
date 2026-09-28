"""Immutable HDF5 episodes and an allowlisted, episode-local HPT window adapter."""

import json
import os
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.models.stems.byte_language import encode_bytes

OBSERVATION_KEYS = {"external_rgb", "wrist_rgb", "proprioception", "timestamps"}
STUDENT_KEYS = {
    "observations.images.external_rgb",
    "observations.images.wrist_rgb",
    "observations.state.proprioception",
    "language",
    "embodiment",
}


def write_episode(path, *, observations, actions, instruction, provenance, audit):
    """Row t pairs the observation before the executed command with action t."""
    if set(observations) != OBSERVATION_KEYS:
        raise ValueError("Episode observation keys differ from the student allowlist")
    encode_bytes([instruction])
    actions = np.asarray(actions)
    count = len(actions)
    if (
        actions.dtype != np.float32
        or actions.shape != (count, 7)
        or not 1 <= count <= 150
        or not np.isfinite(actions).all()
        or np.abs(actions).max() > 1
    ):
        raise ValueError(
            "Executed actions must be finite bounded float32 [T,7], T<=150"
        )
    for key in ("external_rgb", "wrist_rgb"):
        image = np.asarray(observations[key])
        if image.dtype != np.uint8 or image.shape != (count, 224, 224, 3):
            raise ValueError("Raw cameras must be uint8 [T,224,224,3]")
    state = np.asarray(observations["proprioception"])
    timestamps = np.asarray(observations["timestamps"])
    if (
        state.shape != (count, 9)
        or not np.isfinite(state).all()
        or timestamps.shape != (count,)
        or not np.isfinite(timestamps).all()
    ):
        raise ValueError("Invalid proprioception/timestamp arrays")
    if count > 1 and not np.allclose(np.diff(timestamps), 0.1, atol=1e-6, rtol=0):
        raise ValueError(
            "Observation/action rows must be consecutive 10 Hz control steps"
        )
    required = {
        "episode_id",
        "scene_hash",
        "task_hash",
        "teacher_hash",
        "initial_state_hash",
        "phase",
        "arm",
        "round",
        "stage",
        "partition",
        "accepted",
    }
    if set(provenance) != required or type(provenance["accepted"]) is not bool:
        raise ValueError("Episode provenance must contain the complete declared fields")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with h5py.File(temporary, "x") as output:
        output.attrs["schema_version"] = "astrapush-episode-1"
        output.attrs["instruction"] = instruction
        output.attrs["provenance"] = json.dumps(
            provenance, sort_keys=True, allow_nan=False
        )
        output.attrs["alignment"] = "observation_t_before_executed_action_t"
        for key, values in observations.items():
            output.create_dataset(
                f"observations/{key}",
                data=values,
                compression="gzip",
                compression_opts=1,
            )
        output.create_dataset("actions", data=actions)
        output.create_dataset(
            "audit/json",
            data=json.dumps(audit, sort_keys=True, allow_nan=False),
            dtype=h5py.string_dtype("utf-8"),
        )
        output.flush()
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    receipt = {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "steps": count,
        "provenance": provenance,
    }
    publish_json(path.with_suffix(".json"), receipt)
    return receipt


class ProprioceptionStats:
    def __init__(self, mean, std, source_hashes):
        self.mean, self.std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
        if (
            self.mean.shape != (9,)
            or self.std.shape != (9,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std < 0).any()
        ):
            raise ValueError(
                "Commissioning normalization requires finite nine-dimensional statistics"
            )
        self.std = np.maximum(self.std, 1e-6)
        self.source_hashes = tuple(source_hashes)

    @classmethod
    def fit_commissioning(cls, records):
        if len(records) != 30 or len({r["sha256"] for r in records}) != 30:
            raise ValueError(
                "Normalization requires thirty distinct commissioning episodes"
            )
        values = []
        for record in records:
            with verified_episode(record) as episode:
                provenance = json.loads(episode.attrs["provenance"])
                if (
                    provenance["phase"] != "commissioning"
                    or provenance["partition"] != "training"
                    or not provenance["accepted"]
                ):
                    raise ValueError(
                        "Normalization may use only accepted training-side commissioning"
                    )
                values.append(episode["observations/proprioception"][:])
        state = np.concatenate(values).astype(np.float64)
        return cls(state.mean(0), state.std(0), [r["sha256"] for r in records])

    def normalize(self, value):
        return ((np.asarray(value, dtype=np.float64) - self.mean) / self.std).astype(
            np.float32
        )

    def snapshot(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "source_hashes": list(self.source_hashes),
            "minimum_std": 1e-6,
        }


def verified_episode(record):
    if file_hash(record["path"]) != record["sha256"]:
        raise ValueError("Episode hash differs from its frozen manifest")
    return h5py.File(record["path"], "r")


class EpisodeWindows(Dataset):
    """Explicit training manifest only; no filesystem glob or privileged features."""

    def __init__(self, records, normalization):
        self.records, self.normalization = list(records), normalization
        self.lengths, self.provenance = [], []
        if (
            len({r["sha256"] for r in self.records}) != len(self.records)
            or not self.records
        ):
            raise ValueError("Training manifest must be nonempty and duplicate-free")
        for record in self.records:
            with verified_episode(record) as episode:
                if (
                    episode.attrs["schema_version"] != "astrapush-episode-1"
                    or episode.attrs["alignment"]
                    != "observation_t_before_executed_action_t"
                ):
                    raise ValueError(
                        "Episode schema or action alignment is incompatible"
                    )
                provenance = json.loads(episode.attrs["provenance"])
                if provenance != record["provenance"]:
                    raise ValueError("Episode/manifest provenance differs")
                if (
                    provenance["partition"] != "training"
                    or provenance["phase"] not in {"seed", "round"}
                    or not provenance["accepted"]
                ):
                    raise ValueError(
                        "Only accepted imitation training episodes can enter replay"
                    )
                self.lengths.append(len(episode["actions"]))
                self.provenance.append(provenance)
        self.offsets = np.cumsum([0, *self.lengths])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        frame = int(index - self.offsets[episode_index])
        with h5py.File(self.records[episode_index]["path"], "r") as episode:
            row = {
                key: episode[f"observations/{key}"][frame] for key in OBSERVATION_KEYS
            }
            instruction = episode.attrs["instruction"]
            actions = np.zeros((10, 7), np.float32)
            count = min(10, self.lengths[episode_index] - frame)
            actions[:count] = episode["actions"][frame : frame + count]
        return {
            "external_rgb": row["external_rgb"],
            "wrist_rgb": row["wrist_rgb"],
            "proprioception": self.normalization.normalize(row["proprioception"]),
            "instruction": instruction,
            "actions": actions,
            "action_valid": np.arange(10) < count,
        }


def collate_windows(rows):
    allowed = {
        "external_rgb",
        "wrist_rgb",
        "proprioception",
        "instruction",
        "actions",
        "action_valid",
    }
    if not rows or any(set(row) != allowed for row in rows):
        raise ValueError("Student batch contains missing or privileged keys")
    batch = {}
    for key in ("external_rgb", "wrist_rgb"):
        images = (
            torch.from_numpy(np.stack([row[key] for row in rows]))
            .permute(0, 3, 1, 2)
            .float()
        )
        batch[f"observations.images.{key}"] = (images / 127.5 - 1)[:, None, None]
    batch["observations.state.proprioception"] = torch.from_numpy(
        np.stack([row["proprioception"] for row in rows])
    )[:, None]
    batch["language"] = encode_bytes([row["instruction"] for row in rows])
    batch["embodiment"] = ["libero_push"] * len(rows)
    batch["actions"] = torch.from_numpy(np.stack([row["actions"] for row in rows]))
    batch["action_valid"] = torch.from_numpy(
        np.stack([row["action_valid"] for row in rows])
    )
    return {"libero_push": batch}


class EpisodeBalancedReplay:
    """Exact 2+2 episode-balanced draws, with serializable RNG for round resume."""

    def __init__(self, dataset, *, seed, arm=None, round_index=None):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        if arm is None:
            if any(
                p["phase"] != "seed" or p["stage"] != "S1" for p in dataset.provenance
            ):
                raise ValueError("Warm start requires only common S1 seed data")
            self.current, self.history = list(range(len(dataset.records))), []
        else:
            if arm not in {"A", "U"} or round_index not in {1, 2, 3, 4}:
                raise ValueError("Invalid arm/round")
            self.current, self.history = [], []
            for i, p in enumerate(dataset.provenance):
                if p["phase"] == "seed":
                    self.history.append(i)
                elif p["arm"] != arm or p["round"] > round_index:
                    raise ValueError("Replay includes another arm or future round")
                elif p["round"] == round_index:
                    self.current.append(i)
                else:
                    self.history.append(i)
            if not self.current or not self.history:
                raise ValueError("Round replay requires current and historical pools")

    def next_indices(self):
        pools = [self.current] * (2 if self.history else 4) + [self.history] * (
            2 if self.history else 0
        )
        result = []
        for pool in pools:
            episode = int(self.rng.choice(pool))
            frame = int(self.rng.integers(self.dataset.lengths[episode]))
            result.append(int(self.dataset.offsets[episode] + frame))
        return result

    def state_dict(self):
        return {
            "rng": self.rng.bit_generator.state,
            "current": self.current,
            "history": self.history,
            "episode_hashes": [r["sha256"] for r in self.dataset.records],
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(
            state.get(k) != expected[k]
            for k in ("current", "history", "episode_hashes")
        ):
            raise ValueError("Replay state belongs to a different frozen data manifest")
        self.rng.bit_generator.state = state["rng"]
