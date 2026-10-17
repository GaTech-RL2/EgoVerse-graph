"""Prepared world-model caches through the existing trainHydra data boundary."""

import hashlib

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from egomimic.pl_utils.data_context import DataContext
from egomimic.rldb.world_model import (
    HomogeneousWorldBatchSampler,
    PreparedWorldDataset,
    collate_world,
)


class WorldModelDataModule(LightningDataModule):
    def __init__(
        self,
        manifest_path,
        batch_size=1,
        num_workers=0,
        mode="continuous",
        seed=42,
        min_frames=2,
        max_frames=4,
        discrete_frames=4,
        stride=1,
        forecasting_probability=0.5,
        max_past_seconds=None,
        max_future_seconds=None,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.batch_size, self.num_workers = batch_size, num_workers
        self.settings = dict(
            mode=mode,
            seed=seed,
            min_frames=min_frames,
            max_frames=max_frames,
            discrete_frames=discrete_frames,
            stride=stride,
            forecasting_probability=forecasting_probability,
        )
        # Preserve historical unbounded checkpoint data contexts. Configured
        # bounds become part of strict context identity for new bounded runs.
        if max_past_seconds is not None or max_future_seconds is not None:
            self.settings.update(
                max_past_seconds=max_past_seconds,
                max_future_seconds=max_future_seconds,
            )
        self.datasets = {}

    def setup(self, stage=None):
        if not self.datasets:
            self.datasets = {
                s: PreparedWorldDataset(self.manifest_path, split=s, **self.settings)
                for s in ("train", "val")
            }

    def prepare_context(self, *, mode, normalization, normalizer, restored_state=None):
        del mode
        if normalizer is not None or normalization not in (None, "prepared"):
            raise ValueError(
                "World-model caches already carry train-only normalization"
            )
        self.setup()
        dataset = self.datasets["train"]
        state = {
            "kind": "mot-world-prepared-v1",
            "manifest": dataset.manifest,
            "manifest_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
            "sampling": self.settings,
        }
        if restored_state is not None and restored_state != state:
            raise ValueError("World-model checkpoint/data context mismatch")
        return DataContext(
            normalizer=None,
            sample_schema={"world": "three_token_streams"},
            validation_groups=tuple(
                sorted({e["embodiment"] for e in dataset.episodes})
            ),
            state=state,
        )

    def configure_evaluation(self, requirements):
        if requirements.complete_episodes or set(requirements.required_keys) - {
            "world",
            "embodiment",
        }:
            raise ValueError("Unsupported world-model evaluation requirements")

    def frame_counts(self):
        return [
            (s, ep["embodiment"], ep["frames"])
            for s, dataset in self.datasets.items()
            for ep in dataset.episodes
        ]

    def _loader(self, split):
        self.setup()
        if self.trainer is not None and self.trainer.world_size != 1:
            raise ValueError(
                "Distributed source sharding is not implemented; use one device"
            )
        dataset = self.datasets[split]
        return DataLoader(
            dataset,
            batch_sampler=HomogeneousWorldBatchSampler(
                dataset,
                self.batch_size,
                self.settings["seed"],
                shuffle=split == "train",
            ),
            collate_fn=collate_world,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")
