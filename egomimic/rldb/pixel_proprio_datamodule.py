"""Configurable native-pixel/proprio data through the shared training boundary."""

import hashlib

import torch
from torch.utils.data import DataLoader

from egomimic.pl_utils.data_context import DataContext
from egomimic.rldb.pixel_proprio import (
    DistributedPixelTrainingBatchSampler,
    DistributedPixelValidationBatchSampler,
    EvenlySpacedEpisodeBatchSampler,
    PreparedPixelProprioDataset,
    collate_pixel_proprio,
)
from egomimic.rldb.world_model import HomogeneousWorldBatchSampler
from egomimic.rldb.world_model_datamodule import WorldModelDataModule
from egomimic.rldb.world_model_preparation import ArtifactVerificationMemo


class PixelProprioDataModule(WorldModelDataModule):
    def __init__(
        self,
        manifest_path,
        task="visual_diffusion",
        batch_size=1,
        num_workers=0,
        mode="continuous",
        seed=42,
        min_frames=2,
        max_frames=128,
        discrete_frames=4,
        stride=1,
        forecasting_probability=0.5,
        max_past_seconds=None,
        max_future_seconds=None,
        checkpointable_sampling=False,
        validation_batches_per_embodiment=None,
        validation_anchors_per_episode=None,
        sampling_strategy="full_episode_uniform",
        validation_sampling_strategy=None,
        validation_discrete_frames=None,
        validation_stride=None,
        image_only=False,
        distributed_world_size=1,
        validation_batch_size=None,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        balance_training_work=False,
        couple_state_observations=False,
    ):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if validation_batch_size is not None and (
            type(validation_batch_size) is not int or validation_batch_size < 1
        ):
            raise ValueError("validation_batch_size must be a positive integer or None")
        if type(pin_memory) is not bool or type(persistent_workers) is not bool:
            raise ValueError("pin_memory and persistent_workers must be booleans")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if prefetch_factor is not None and (
            type(prefetch_factor) is not int or prefetch_factor < 1 or num_workers == 0
        ):
            raise ValueError(
                "prefetch_factor requires a positive integer and num_workers > 0"
            )
        if type(distributed_world_size) is not int or distributed_world_size < 1:
            raise ValueError("distributed_world_size must be a positive integer")
        if distributed_world_size > 1 and not checkpointable_sampling:
            raise ValueError(
                "Distributed pixel loading requires checkpointable_sampling"
            )
        if type(balance_training_work) is not bool or (
            balance_training_work and not checkpointable_sampling
        ):
            raise ValueError(
                "balance_training_work requires a boolean and checkpointable_sampling"
            )
        self.balance_training_work = balance_training_work
        self.distributed_world_size = distributed_world_size
        if task not in ("visual_diffusion", "visual_state_action", "direct_action"):
            raise ValueError("Unknown pixel/proprio objective")
        if type(image_only) is not bool or (image_only and task != "visual_diffusion"):
            raise ValueError("image_only requires visual_diffusion and a boolean flag")
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
            # The legacy parent forbids workers for exact resume. Pixel draws
            # instead key every frame/mask RNG by (cursor, global draw slot), so
            # worker scheduling and discarded prefetched batches cannot change
            # the first unconsumed draw. Leave the old dataset guard untouched.
            0,
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
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self._explicit_validation_batch_size = validation_batch_size is not None
        self.validation_batch_size = (
            batch_size if validation_batch_size is None else validation_batch_size
        )
        self.settings.update(task=task, sampling_strategy=sampling_strategy)
        # Preserve historical conditioned checkpoint contexts when disabled.
        if image_only:
            self.settings["image_only"] = True
        if type(couple_state_observations) is not bool:
            raise ValueError("couple_state_observations must be boolean")
        if couple_state_observations:
            if image_only:
                raise ValueError("Coupled state observations require image and proprio")
            self.settings["couple_state_observations"] = True
        self.validation_anchors_per_episode = validation_anchors_per_episode
        self.validation_settings = {
            key: value
            for key, value in dict(
                sampling_strategy=validation_sampling_strategy,
                discrete_frames=validation_discrete_frames,
                stride=validation_stride,
            ).items()
            if value is not None
        }

    def preflight_configuration(self):
        """Check both declared pixel samplers without reading a prepared cache."""
        result = super().preflight_configuration()
        for split, settings in (
            ("train", self.settings),
            ("val", self.settings | self.validation_settings),
        ):
            strategy = settings["sampling_strategy"]
            if (
                (strategy, settings["mode"])
                not in (
                    ("full_episode_uniform", "continuous"),
                    ("discrete_window", "continuous"),
                    ("discrete_window", "discrete"),
                )
                or settings.get("max_past_seconds") is not None
                or settings.get("max_future_seconds") is not None
            ):
                raise ValueError(f"Invalid {split} pixel sampling strategy/time bounds")
            for key, minimum in (("discrete_frames", 2), ("stride", 1)):
                value = settings[key]
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{split} {key} must be an integer >= {minimum}")
            if (
                settings["task"] == "direct_action"
                and strategy != "full_episode_uniform"
            ):
                raise ValueError("Direct action requires full_episode_uniform sampling")
        return result | {
            "validation_sampling": dict(self.validation_settings),
            "distributed_world_size": self.distributed_world_size,
            "validation_batch_size": self.validation_batch_size,
        }

    def setup(self, stage=None):
        if not self.datasets:
            verifier = ArtifactVerificationMemo()
            self.datasets = {
                split: PreparedPixelProprioDataset(
                    self.manifest_path,
                    split=split,
                    verification_memo=verifier,
                    **(
                        self.settings
                        | (self.validation_settings if split == "val" else {})
                    ),
                )
                for split in ("train", "val")
            }

    def configure_evaluation(self, requirements):
        if requirements.complete_episodes or set(requirements.required_keys) - {
            "world",
            "embodiment",
            "task_kind",
            "pixel_metadata",
        }:
            raise ValueError("Unsupported pixel/proprio evaluation requirements")

    def prepare_context(self, *, mode, normalization, normalizer, restored_state=None):
        del mode
        if normalizer is not None or normalization not in (None, "prepared"):
            raise ValueError(
                "Pixel adapter owns train-only pose normalization and fixed RGB scaling"
            )
        self.setup()
        dataset = self.datasets["train"]
        state = dict(
            kind="pixel-image-prepared-v1"
            if dataset.image_only
            else "pixel-proprio-prepared-v1",
            manifest=dataset.manifest,
            manifest_sha256=hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
            sampling=self.settings,
            validation_overrides=self.validation_settings,
        )
        if self.checkpointable_sampling:
            state["sampling_resume"] = {
                "schema": "mot-sampling-state-v1",
                "batch_size": self.batch_size,
            }
        if self.distributed_world_size > 1:
            state["distributed_sampling"] = self._distributed_sampling_identity()
        if self._explicit_validation_batch_size:
            state["batching"] = self._batching_identity()
        if self.balance_training_work:
            state["training_work_balance"] = self._work_balance_identity()
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
            raise ValueError("Pixel/proprio checkpoint/data context mismatch")
        return DataContext(
            normalizer=None,
            sample_schema={
                "world": "image"
                if dataset.image_only
                else "image_proprio_action_labels"
                if dataset.task in ("direct_action", "visual_state_action")
                else "image_proprio"
            },
            validation_groups=tuple(
                sorted({ep["embodiment"] for ep in dataset.episodes})
            ),
            state=state,
        )

    def _loader(self, split):
        self.setup()
        rank = 0
        if self.trainer is not None:
            if self.trainer.world_size != self.distributed_world_size:
                raise ValueError(
                    "Trainer world_size differs from bound distributed_world_size"
                )
            rank = self.trainer.global_rank
        batch_size = self.batch_size if split == "train" else self.validation_batch_size
        if split == "val" and self.validation_anchors_per_episode is not None:
            sampler = EvenlySpacedEpisodeBatchSampler(
                self.datasets[split],
                self.validation_anchors_per_episode,
                batch_size,
            )
        elif split == "train" and self.distributed_world_size > 1:
            sampler = DistributedPixelTrainingBatchSampler(
                self.datasets[split],
                batch_size,
                self.settings["seed"],
                world_size=self.distributed_world_size,
                rank=rank,
                balance_training_work=self.balance_training_work,
            )
        else:
            sampler = HomogeneousWorldBatchSampler(
                self.datasets[split],
                batch_size,
                self.settings["seed"],
                shuffle=split == "train",
                checkpointable=self.checkpointable_sampling and split == "train",
                validation_batches_per_embodiment=self.validation_batches_per_embodiment
                if split == "val"
                else None,
            )
        # Before Trainer attachment, inspection/reload must see all held-out
        # anchors. Runtime sharding preserves their original global IDs.
        if (
            split == "val"
            and self.trainer is not None
            and self.distributed_world_size > 1
        ):
            sampler = DistributedPixelValidationBatchSampler(
                sampler, world_size=self.distributed_world_size, rank=rank
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
            collate_fn=collate_pixel_proprio,
            num_workers=self.num_workers,
            generator=generator,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def state_dict(self):
        state = super().state_dict()
        if state:
            state["validation_anchors_per_episode"] = (
                self.validation_anchors_per_episode
            )
            state["validation_overrides"] = self.validation_settings
            if self.distributed_world_size > 1:
                state["distributed_sampling"] = self._distributed_sampling_identity()
            if self._explicit_validation_batch_size:
                state["batching"] = self._batching_identity()
            if self.balance_training_work:
                state["training_work_balance"] = self._work_balance_identity()
        return state

    def _batching_identity(self):
        # Only explicit split-specific batching adds a new identity field.
        # Historical configs keep their exact bound data/checkpoint contexts.
        # Worker/pinning/prefetch options are recorded in the resolved config,
        # but do not change a stateless source draw or its data identity.
        return {
            "schema": "pixel-split-batching-v1",
            "train_batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
        }

    def _distributed_sampling_identity(self):
        return {
            "schema": "pixel-distributed-sampling-v1",
            "world_size": self.distributed_world_size,
            "per_rank_batch_size": self.batch_size,
            "global_batch_size": self.batch_size * self.distributed_world_size,
            "training_sharding": "global_draw_rank_slots",
            "validation_sharding": "contiguous_global_batches_no_padding",
        }

    def _work_balance_identity(self):
        return {
            "schema": "pixel-global-batch-work-v1",
            "strategy": "longest_first_fixed_rank_capacity_if_lower_maximum",
            "estimate": "sampled_image_tokens_squared",
            "preserved": "global_draw_keys_frames_and_conditioning_masks",
            "model_noise": "rank_seeded_realization_changes_for_reassigned_clips",
        }

    def load_state_dict(self, state_dict):
        expected_balance = (
            self._work_balance_identity() if self.balance_training_work else None
        )
        if state_dict.get("training_work_balance") != expected_balance:
            raise ValueError("Pixel sampling restore mismatch: training work balance")
        expected_batching = (
            self._batching_identity() if self._explicit_validation_batch_size else None
        )
        if (
            self.checkpointable_sampling
            and state_dict.get("batching") != expected_batching
        ):
            raise ValueError(
                "Pixel sampling restore mismatch: train/validation batching"
            )
        expected_topology = (
            self._distributed_sampling_identity()
            if self.distributed_world_size > 1
            else None
        )
        if state_dict.get("distributed_sampling") != expected_topology:
            raise ValueError("Pixel sampling restore mismatch: distributed topology")
        if self.checkpointable_sampling and (
            state_dict.get("validation_anchors_per_episode")
            != self.validation_anchors_per_episode
            or state_dict.get("validation_overrides", {}) != self.validation_settings
        ):
            raise ValueError(
                "Pixel sampling restore mismatch: validation_anchors_per_episode"
            )
        super().load_state_dict(state_dict)
