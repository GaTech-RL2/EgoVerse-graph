"""DINO/proprio objectives through the shared prepared-data training boundary."""

import hashlib

import torch
from torch.utils.data import DataLoader

from egomimic.pl_utils.data_context import DataContext
from egomimic.rldb.dino_proprio import (
    EvenlySpacedEpisodeBatchSampler,
    PreparedDinoProprioDataset,
    collate_dino_proprio,
)
from egomimic.rldb.world_model import HomogeneousWorldBatchSampler
from egomimic.rldb.world_model_datamodule import WorldModelDataModule
from egomimic.rldb.world_model_preparation import ArtifactVerificationMemo


class DinoProprioDataModule(WorldModelDataModule):
    """Reuse the checkpointable source sampler, never the three-stream dataset."""

    def __init__(
        self,
        manifest_path,
        task="visual_diffusion",
        action_manifest_path=None,
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
        validation_batches_per_embodiment=None,
        validation_anchors_per_episode=None,
    ):
        if task not in ("visual_diffusion", "direct_action"):
            raise ValueError("Unknown DINO/proprio objective")
        if task == "direct_action" and not action_manifest_path:
            raise ValueError(
                "Direct action BC requires a verified action alignment manifest"
            )
        if task == "visual_diffusion" and action_manifest_path is not None:
            raise ValueError("Visual/proprio prediction does not consume action labels")
        if validation_anchors_per_episode is not None:
            if (
                type(validation_anchors_per_episode) is not int
                or validation_anchors_per_episode < 1
            ):
                raise ValueError(
                    "validation_anchors_per_episode must be a positive integer or None"
                )
            if validation_batches_per_embodiment is not None:
                raise ValueError(
                    "Per-episode validation anchors and prefix batch caps are mutually exclusive"
                )
        super().__init__(
            manifest_path,
            batch_size,
            num_workers,
            mode,
            seed,
            min_frames,
            max_frames,
            discrete_frames,
            stride,
            forecasting_probability,
            max_past_seconds,
            max_future_seconds,
            checkpointable_sampling,
            validation_batches_per_embodiment,
        )
        self.validation_anchors_per_episode = validation_anchors_per_episode
        self.settings.update(task=task)
        if action_manifest_path is not None:
            self.settings["action_manifest_path"] = str(action_manifest_path)

    def setup(self, stage=None):
        if not self.datasets:
            verifier = ArtifactVerificationMemo()
            self.datasets = {
                split: PreparedDinoProprioDataset(
                    self.manifest_path,
                    split=split,
                    verification_memo=verifier,
                    **self.settings,
                )
                for split in ("train", "val")
            }

    def prepare_context(self, *, mode, normalization, normalizer, restored_state=None):
        del mode
        if normalizer is not None or normalization not in (None, "prepared"):
            raise ValueError("DINO/proprio caches carry bound train-only normalization")
        self.setup()
        dataset = self.datasets["train"]
        state = {
            "kind": "dino-proprio-prepared-v1",
            "manifest": dataset.manifest,
            "manifest_sha256": hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
            "sampling": self.settings,
        }
        if dataset.action_manifest is not None:
            state["action_manifest"] = dataset.action_manifest
            state["action_manifest_sha256"] = hashlib.sha256(
                dataset.action_path.read_bytes()
            ).hexdigest()
        if self.checkpointable_sampling:
            state["sampling_resume"] = {
                "schema": "mot-sampling-state-v1",
                "batch_size": self.batch_size,
            }
        if self.validation_batches_per_embodiment is not None:
            state["validation_sampling"] = {
                "batches_per_embodiment": self.validation_batches_per_embodiment
            }
        if self.validation_anchors_per_episode is not None:
            state["validation_sampling"] = {
                "anchors_per_episode": self.validation_anchors_per_episode,
                "selection": "deterministic_evenly_spaced_eligible_anchors",
            }
        if restored_state is not None and restored_state != state:
            raise ValueError("DINO/proprio checkpoint/data context mismatch")
        return DataContext(
            normalizer=None,
            sample_schema={
                "world": (
                    "image_proprio_action_labels"
                    if dataset.task == "direct_action"
                    else "image_proprio"
                )
            },
            validation_groups=tuple(
                sorted({ep["embodiment"] for ep in dataset.episodes})
            ),
            state=state,
        )

    def _loader(self, split):
        self.setup()
        if self.trainer is not None and self.trainer.world_size != 1:
            raise ValueError(
                "Distributed source sharding is not implemented; use one device"
            )
        if split == "val" and self.validation_anchors_per_episode is not None:
            sampler = EvenlySpacedEpisodeBatchSampler(
                self.datasets[split],
                self.validation_anchors_per_episode,
                self.batch_size,
            )
        else:
            sampler = HomogeneousWorldBatchSampler(
                self.datasets[split],
                self.batch_size,
                self.settings["seed"],
                shuffle=split == "train",
                checkpointable=self.checkpointable_sampling and split == "train",
                validation_batches_per_embodiment=(
                    self.validation_batches_per_embodiment if split == "val" else None
                ),
            )
        if split == "train":
            self.train_sampler = sampler
            sampler.consumed_batches = self.consumed_batches
        generator = (
            torch.Generator().manual_seed(self.settings["seed"])
            if self.checkpointable_sampling
            else None
        )
        return DataLoader(
            self.datasets[split],
            batch_sampler=sampler,
            collate_fn=collate_dino_proprio,
            num_workers=self.num_workers,
            generator=generator,
        )

    def state_dict(self):
        state = super().state_dict()
        if state:
            state["validation_anchors_per_episode"] = (
                self.validation_anchors_per_episode
            )
        if state and "action_manifest_path" in self.settings:
            from pathlib import Path

            state["action_manifest_sha256"] = hashlib.sha256(
                Path(self.settings["action_manifest_path"]).read_bytes()
            ).hexdigest()
        return state

    def load_state_dict(self, state_dict):
        if (
            self.checkpointable_sampling
            and state_dict.get("validation_anchors_per_episode")
            != self.validation_anchors_per_episode
        ):
            raise ValueError(
                "DINO/proprio sampling restore mismatch: validation_anchors_per_episode"
            )
        if state_dict.get("action_manifest_sha256") != self.state_dict().get(
            "action_manifest_sha256"
        ):
            raise ValueError(
                "DINO/proprio sampling restore mismatch: action_manifest_sha256"
            )
        super().load_state_dict(state_dict)
