"""Episode-safe goal-conditioned replay for native chunk Bellman backups.

Sampling follows DQC's CGCDataset (see third_party/dqc/LICENSE). File loading
uses OGBench's public adapter, including oracle goal representations. Random
streams are derived from (training seed, update) so checkpoint resume and
paired representations consume identical transition/goal draws.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch.utils.data import IterableDataset


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_verified(url, path, expected_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            while block := response.read(8 * 1024 * 1024):
                out.write(block)
        temporary.replace(path)
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"dataset hash mismatch: {path}")
    return digest


class GoalReplay:
    def __init__(self, data, backup_horizon=25, discount=0.999,
                 p_current=0.2, p_future=0.5, p_random=0.3):
        if not np.isclose(p_current + p_future + p_random, 1):
            raise ValueError("goal probabilities must sum to one")
        self.data = data
        self.horizon, self.discount = int(backup_horizon), float(discount)
        self.p_current, self.p_future = float(p_current), float(p_future)
        self.terminals = np.flatnonzero(data["terminals"] > 0)
        if not len(self.terminals) or self.terminals[-1] != len(data["actions"]) - 1:
            raise ValueError("replay must end at an explicit episode boundary")
        starts = np.r_[0, self.terminals[:-1] + 1]
        valid = [np.arange(start, terminal + 1 - self.horizon)
                 for start, terminal in zip(starts, self.terminals)]
        self.valid_indices = np.concatenate(valid)
        if not len(self.valid_indices):
            raise ValueError("no complete chunk windows")

    def sample(self, batch_size, rng, indices=None, goal_indices=None):
        idx = (rng.choice(self.valid_indices, size=batch_size) if indices is None
               else np.asarray(indices, dtype=np.int64))
        ends = self.terminals[np.searchsorted(self.terminals, idx)]
        if np.any(idx + self.horizon > ends):
            raise ValueError("chunk crosses an episode boundary")
        if goal_indices is None:
            random_goals = rng.choice(self.valid_indices, size=len(idx))
            distance = rng.rand(len(idx))
            future = np.round(np.minimum(idx + 1, ends) * distance + ends * (1 - distance)).astype(int)
            goals = np.where(rng.rand(len(idx)) < self.p_future / (1 - self.p_current), future, random_goals)
            goals = np.where(rng.rand(len(idx)) < self.p_current, idx, goals)
        else:
            goals = np.asarray(goal_indices, dtype=np.int64)
        offsets = goals - idx
        horizon = np.where((offsets >= 0) & (offsets < self.horizon), offsets, self.horizon)
        successes = (horizon < self.horizon).astype(np.float32)
        chunk_indices = idx[:, None] + np.arange(self.horizon)
        representations = self.data.get("oracle_reps", self.data["observations"])
        result = {
            "observations": self.data["observations"][idx],
            "high_value_goals": representations[goals],
            "high_value_next_observations": self.data["observations"][idx + horizon],
            "high_value_action_chunks": np.clip(self.data["actions"][chunk_indices], -1 + 1e-5, 1 - 1e-5),
            "high_value_backup_horizon": horizon.astype(np.float32),
            "high_value_rewards": (self.discount ** horizon * successes).astype(np.float32),
            "high_value_masks": 1 - successes,
            "policy_valid": np.ones(len(idx), dtype=np.float32),
        }
        return {key: torch.from_numpy(np.asarray(value, dtype=np.float32)) for key, value in result.items()}


class ShardedGoalReplay(IterableDataset):
    """One complete pre-batched sample per optimizer update (no worker RNG)."""
    def __init__(self, env_name, shards, cache_dir, seed, batch_size, steps,
                 replace_interval=1000, start_step=0, **replay_options):
        super().__init__()
        if not shards:
            raise ValueError("empty dataset manifest")
        self.env_name, self.shards = env_name, list(shards)
        self.cache_dir = Path(cache_dir)
        self.seed, self.batch_size, self.steps = int(seed), int(batch_size), int(steps)
        self.replace_interval, self.start_step = int(replace_interval), int(start_step)
        self.replay_options = replay_options
        self.receipts = {}
        self._shard_index, self._replay = None, None
        self._env = None

    def load_shard(self, index):
        import ogbench
        if self._shard_index == index:
            return self._replay
        spec = self.shards[index]
        path = self.cache_dir / Path(spec["url"]).name
        digest = download_verified(spec["url"], path, spec.get("sha256"))
        # The public loader expects a val companion even during dataset_only.
        val_path = path.with_name(path.stem + "-val.npz")
        val_digest = download_verified(spec["validation_url"], val_path, spec.get("validation_sha256"))
        if self._env is None:
            self._env = ogbench.make_env_and_datasets(self.env_name, env_only=True)
        self._replay = None
        data, _ = ogbench.make_env_and_datasets(self.env_name, dataset_path=str(path),
                                               compact_dataset=True, dataset_only=True,
                                               cur_env=self._env)
        self._replay = GoalReplay(data, **self.replay_options)
        self._shard_index = index
        self.receipts[index] = {"url": spec["url"], "sha256": digest,
                               "validation_sha256": val_digest,
                               "transitions": int(len(data["actions"])),
                               "valid_chunk_starts": int(len(self._replay.valid_indices))}
        return self._replay

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("ShardedGoalReplay requires num_workers=0 for exact update alignment")
        for step in range(self.start_step, self.steps):
            index = (step // self.replace_interval) % len(self.shards)
            replay = self.load_shard(index)
            rng = np.random.RandomState(np.random.SeedSequence([self.seed, step]).generate_state(1)[0])
            yield {"ogbench": replay.sample(self.batch_size, rng)}
