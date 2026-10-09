"""Homogeneous batches with exact proportional exposure of source windows.

The existing multi-source CombinedLoader(max_size_cycle) exposes every source
equally per optimizer step and cycles smaller corpora. This sampler instead
visits each eligible window exactly once per epoch, without discarding a tail
or requiring equal action widths. It is intended for a ConcatDataset whose
source order and lengths match ``source_lengths``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping


class ProportionalHomogeneousBatchSampler:
    """Yield global ConcatDataset indices, with one source per batch.

    Every source index appears once per complete epoch. Batch order and local
    index order are deterministically shuffled using ``seed + epoch``. The
    final partial batch of each source is retained, so exposure is exact even
    when corpus lengths are not divisible by batch size.
    """

    def __init__(
        self, source_lengths: Mapping[str, int], batch_size: int, seed: int = 42
    ):
        if not source_lengths or len(source_lengths) < 2:
            raise ValueError("Proportional co-training requires at least two sources")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        self.source_lengths = dict(source_lengths)
        if any(not isinstance(name, str) or not name for name in self.source_lengths):
            raise ValueError("Every source needs a nonempty string name")
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in self.source_lengths.values()
        ):
            raise ValueError("Every source needs a positive integer window count")
        self.batch_size = batch_size
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0
        self.offsets = {}
        offset = 0
        for name, n in self.source_lengths.items():
            self.offsets[name] = offset
            offset += n
        self.total_windows = offset

    def __len__(self) -> int:
        return sum(math.ceil(n / self.batch_size) for n in self.source_lengths.values())

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def state_dict(self) -> dict:
        """Save the active epoch so Lightning can skip completed batches on resume."""
        return {
            "schema_version": 1,
            "source_lengths": self.source_lengths,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "active_epoch": max(0, self.epoch - 1),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported proportional sampler state")
        if (
            state.get("source_lengths") != self.source_lengths
            or state.get("batch_size") != self.batch_size
            or state.get("seed") != self.seed
        ):
            raise ValueError("Proportional sampler resume identity changed")
        self.set_epoch(state["active_epoch"])

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for name, n in self.source_lengths.items():
            offset = self.offsets[name]
            indices = list(range(offset, offset + n))
            rng.shuffle(indices)
            batches.extend(
                indices[i : i + self.batch_size] for i in range(0, n, self.batch_size)
            )
        rng.shuffle(batches)
        if not 0 <= self.start_batch <= len(batches):
            raise ValueError("Proportional sampler resume batch is out of range")
        start_batch = self.start_batch
        self.start_batch = 0
        self.epoch += 1
        yield from batches[start_batch:]

    def source_for_index(self, index: int) -> str:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.total_windows
        ):
            raise IndexError(index)
        for name, offset in reversed(tuple(self.offsets.items())):
            if index >= offset:
                return name
        raise AssertionError("Unreachable")
