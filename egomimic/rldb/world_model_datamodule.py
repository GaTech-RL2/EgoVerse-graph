"""Prepared world-model caches through the existing trainHydra data boundary."""

import hashlib
from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from egomimic.pl_utils.data_context import DataContext
from egomimic.rldb.world_model import (
    HomogeneousWorldBatchSampler,
    PreparedWorldDataset,
    collate_world,
)
from egomimic.rldb.world_model_preparation import ArtifactVerificationMemo


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
        checkpointable_sampling=False,
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
        self.checkpointable_sampling = bool(checkpointable_sampling)
        if self.checkpointable_sampling and self.num_workers != 0:
            raise ValueError("Exact MoT sampling resume currently requires num_workers=0")
        self.train_sampler = None
        self.consumed_batches = 0

    def setup(self, stage=None):
        if not self.datasets:
            # Reuse finalized fingerprints only for unchanged exact files;
            # otherwise hash once per setup and share that proof across splits.
            verification = ArtifactVerificationMemo()
            self.datasets = {
                s: PreparedWorldDataset(self.manifest_path, split=s,
                                        verification_memo=verification, **self.settings)
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
        if self.checkpointable_sampling:
            state["sampling_resume"] = {"schema": "mot-sampling-state-v1", "batch_size": self.batch_size}
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
        sampler = HomogeneousWorldBatchSampler(
                dataset,
                self.batch_size,
                self.settings["seed"],
                shuffle=split == "train",
                checkpointable=self.checkpointable_sampling and split == "train",
            )
        if split == "train":
            self.train_sampler = sampler
            sampler.consumed_batches = self.consumed_batches
        # Iterator construction must not advance the model's global Torch RNG.
        generator = None
        if self.checkpointable_sampling:
            import torch
            generator = torch.Generator().manual_seed(self.settings["seed"])
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_world,
            num_workers=self.num_workers,
            generator=generator,
        )

    def acknowledge_batch(self, cursor):
        if not self.checkpointable_sampling:
            return
        if cursor != self.consumed_batches + 1:
            raise ValueError("Nonconsecutive world-model source draw")
        self.consumed_batches = cursor
        if self.train_sampler is not None:
            self.train_sampler.consumed_batches = cursor

    def state_dict(self):
        if not self.checkpointable_sampling:
            return {}
        return {"schema": "mot-sampling-state-v1", "consumed_batches": self.consumed_batches,
                "manifest_sha256": hashlib.sha256(Path(self.manifest_path).read_bytes()).hexdigest(),
                "settings": self.settings, "batch_size": self.batch_size}

    def load_state_dict(self, state_dict):
        if not self.checkpointable_sampling:
            if state_dict:
                raise ValueError("Checkpointed sampling requires its original opt-in configuration")
            return
        expected = self.state_dict()
        for key in ("schema", "manifest_sha256", "settings", "batch_size"):
            if state_dict.get(key) != expected[key]:
                raise ValueError(f"World-model sampling restore mismatch: {key}")
        cursor = state_dict.get("consumed_batches")
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Invalid consumed source-draw cursor")
        self.consumed_batches = cursor
        if self.train_sampler is not None:
            self.train_sampler.consumed_batches = cursor

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")
