import logging
from collections.abc import Mapping

import hydra
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, default_collate

from egomimic.eval.eval import EvaluationDataRequirements
from egomimic.rldb.weighted_dataset import WeightedDataset

logger = logging.getLogger(__name__)

# Name of the val group that keeps the pre-groups behaviour: its metrics stay
# `Valid/...`, so runs predating multi-group validation overlay on the same
# wandb charts. Every other group is namespaced (`Valid_{group}/...`).
DEFAULT_VALID_GROUP = "valid"


def _is_dataset_leaf(value):
    return not isinstance(value, Mapping) or "_target_" in value


def as_valid_groups(valid_datasets: dict, *, layout: str = "auto") -> dict:
    """Normalize source/group structure without interpreting source names.

    Auto accepts dataset objects or Hydra target mappings as leaves. Adapters
    whose dataset objects themselves implement Mapping must declare a layout.
    """
    if layout not in {"auto", "flat", "grouped"}:
        raise ValueError("validation layout must be auto, flat or grouped")
    if not valid_datasets:
        return {}
    leaves = [_is_dataset_leaf(v) for v in valid_datasets.values() if v is not None]
    if layout == "flat" or (layout == "auto" and all(leaves)):
        return {DEFAULT_VALID_GROUP: dict(valid_datasets)}
    if layout == "auto" and any(leaves):
        raise ValueError("valid_datasets mixes dataset leaves and validation groups")
    groups = {}
    for name, members in valid_datasets.items():
        if members is None:
            continue
        if not isinstance(members, Mapping):
            raise ValueError(f"val group {name!r} must map source -> dataset")
        if any(not _is_dataset_leaf(v) for v in members.values()):
            raise ValueError(
                f"val group {name!r} contains nested groups; only one level is supported"
            )
        groups[name] = dict(members)
    return groups


def _params_for_group(params: dict, group_name: str, source_names) -> dict:
    group = params.get(group_name)
    if isinstance(group, Mapping) and set(group).issubset(source_names):
        return group
    return params


def _episode_id_at(dataset: Dataset, index: int) -> str:
    capability = getattr(dataset, "episode_id_at", None)
    if not callable(capability):
        raise TypeError(
            f"Complete-episode validation requires episode_id_at(index); "
            f"{type(dataset).__name__} does not declare that capability"
        )
    return str(capability(index))


class EpisodeLimitedDataset(Dataset):
    """Ordered prefix containing complete episodes from a map-backed dataset."""

    def __init__(self, dataset: Dataset, max_episodes: int):
        max_episodes = int(max_episodes)
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        self.dataset = dataset
        self.max_episodes = max_episodes
        selected: set[str] = set()
        self.indices: list[int] = []
        for index in range(len(dataset)):
            episode = _episode_id_at(dataset, index)
            if episode not in selected:
                if len(selected) >= max_episodes:
                    break
                selected.add(episode)
            self.indices.append(index)
        if not self.indices:
            raise ValueError("cannot limit an empty validation dataset")
        self.episode_ids = tuple(sorted(selected))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    @property
    def norm_stats(self):
        return self.dataset.norm_stats

    def set_norm_stats_from(self, source) -> None:
        self.dataset.set_norm_stats_from(source)

    def __getattr__(self, name):
        # Preserve dataset metadata used by diagnostics and downstream probes.
        return getattr(self.dataset, name)


class MultiDataModuleWrapper(LightningDataModule):
    """
    Build dictionary-based multi-source loaders with Lightning CombinedLoader.

    Uses hydra to instantiate DataLoader objects and then wraps them in a combined loader
    """

    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict,
        train_dataloader_params: dict,
        valid_dataloader_params: dict,
        valid_episode_limit: int | None = None,
        force_valid_order: bool = False,
        validation_layout: str = "auto",
        collate_fn=None,
        train_loader_mode: str = "max_size_cycle",
        valid_loader_mode: str = "max_size_cycle",
        source_fps: float | None = None,
        dataset_weights: dict | None = None,
        weighted_dataloader_params: dict | None = None,
        samples_per_epoch: int | None = None,
        sampling_seed: int = 42,
    ):
        """
        Args:
            train_datasets: dictionary of train datasets
            valid_datasets: dictionary of valid datasets
            train_dataloader_params: dictionary of train dataloader parameters
            valid_dataloader_params: dictionary of valid dataloader parameters

        The collate function stacks ordinary values and preserves variable-length
        list-valued fields for whichever configured stage consumes them.
        """
        super().__init__()
        # Drop `None` slots so downstream iteration sites don't need null guards.
        # `None` entries arise when an inheriting data config opts out of a
        # dataset defined in a base (e.g. `aria_bimanual: null`).
        self.train_datasets = {k: v for k, v in train_datasets.items() if v is not None}
        # `valid_datasets` may be flat ({source: dataset}) or grouped
        # ({group: {source: dataset}}); normalise to the grouped form and drop
        # `None` slots inside each group.
        self.valid_groups = {
            group: {k: v for k, v in members.items() if v is not None}
            for group, members in as_valid_groups(
                valid_datasets, layout=validation_layout
            ).items()
        }
        self.valid_groups = {g: m for g, m in self.valid_groups.items() if m}
        self.valid_episode_limit = (
            None if valid_episode_limit is None else int(valid_episode_limit)
        )
        if self.valid_episode_limit is not None and self.valid_episode_limit < 1:
            raise ValueError("valid_episode_limit must be positive")
        self.force_valid_order = bool(
            force_valid_order or self.valid_episode_limit is not None
        )
        if self.valid_episode_limit is not None:
            self.valid_groups = {
                group: {
                    source: EpisodeLimitedDataset(dataset, self.valid_episode_limit)
                    for source, dataset in members.items()
                }
                for group, members in self.valid_groups.items()
            }
        # Positional: Lightning hands `validation_step` a `dataloader_idx` that
        # indexes this list, and that is how the evaluator recovers the group
        # name for its metric prefix.
        self.valid_group_names = list(self.valid_groups.keys())
        # Kept for callers that expect the flat attribute; it is the default
        # group when present, else the first. Anything that must touch ALL of
        # validation uses `iter_valid_datasets` instead.
        self.valid_datasets = self.valid_groups.get(
            DEFAULT_VALID_GROUP,
            next(iter(self.valid_groups.values()), {}),
        )
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params
        self.collate_fn = collate_fn or annotation_collate
        self.train_loader_mode = train_loader_mode
        self.valid_loader_mode = valid_loader_mode
        self.source_fps = source_fps
        self.weighted_dataset = (
            WeightedDataset(self.train_datasets, dataset_weights)
            if dataset_weights is not None
            else None
        )
        self.weighted_dataloader_params = weighted_dataloader_params
        self.samples_per_epoch = samples_per_epoch
        self.sampling_seed = sampling_seed
        if self.weighted_dataset is not None and not weighted_dataloader_params:
            raise ValueError("dataset_weights requires weighted_dataloader_params")
        if self.weighted_dataset is None and (
            weighted_dataloader_params is not None or samples_per_epoch is not None
        ):
            raise ValueError("Weighted loader settings require dataset_weights")

    def iter_valid_datasets(self):
        """Yield ``(group, source, dataset)`` for EVERY val dataset.

        Use this for anything that must touch all of validation -- norm-stats
        wiring above all. ``self.valid_datasets`` is only a back-compat alias
        for a SINGLE group, so iterating it silently skips every other group,
        leaving those datasets unnormalised while the evaluator unnormalises
        their samples anyway.
        """
        for group, members in self.valid_groups.items():
            for source, dataset in members.items():
                yield group, source, dataset

    def configure_evaluation(self, requirements: EvaluationDataRequirements):
        if not isinstance(requirements, EvaluationDataRequirements):
            raise TypeError("Evaluator must return EvaluationDataRequirements")
        self.force_valid_order |= requirements.ordered or requirements.complete_episodes
        if (
            requirements.source_fps is not None
            and self.source_fps != requirements.source_fps
        ):
            raise ValueError(
                f"Evaluator requires source_fps={requirements.source_fps}; data declares {self.source_fps}"
            )
        for group, source, dataset in self.iter_valid_datasets():
            if not len(dataset):
                raise ValueError(f"Validation dataset {group}/{source} is empty")
            if requirements.complete_episodes or requirements.max_episodes is not None:
                _episode_id_at(dataset, 0)
                if requirements.complete_episodes:
                    frame_at = getattr(dataset, "frame_index_at", None)
                    length_at = getattr(dataset, "episode_length_at", None)
                    if not callable(frame_at) or not callable(length_at):
                        raise TypeError(
                            "Complete episodes require frame_index_at(index) and episode_length_at(index)"
                        )
                    finished, current, expected = set(), None, 0
                    for index in range(len(dataset)):
                        episode = _episode_id_at(dataset, index)
                        if episode != current:
                            if current is not None and expected != length_at(index - 1):
                                raise ValueError(
                                    f"Validation episode is missing its tail: {group}/{source}/{current}"
                                )
                            if episode in finished:
                                raise ValueError(
                                    f"Validation episodes are interleaved: {group}/{source}"
                                )
                            finished.add(episode)
                            current, expected = episode, 0
                        if frame_at(index) != expected:
                            raise ValueError(
                                f"Validation episode is incomplete or unordered: {group}/{source}/{episode}, expected frame {expected}"
                            )
                        expected += 1
                    if expected != length_at(len(dataset) - 1):
                        raise ValueError(
                            f"Validation episode is missing its tail: {group}/{source}/{current}"
                        )
            required_keys = {
                requirements.sample_id_key,
                requirements.frame_index_key,
            } - {None}
            required_keys.update(requirements.required_keys)
            if required_keys:
                missing = required_keys - set(dataset[0])
                if missing:
                    raise ValueError(
                        f"Validation dataset {group}/{source} lacks metadata {sorted(missing)}"
                    )
            params = _params_for_group(
                self.valid_dataloader_params, group, set(self.valid_groups[group])
            )
            options = params.get(source, {})
            if self.force_valid_order and any(
                options.get(key) is not None
                for key in ("sampler", "anchor_sampler", "group_balance_sampler")
            ):
                raise ValueError(
                    f"Ordered validation does not accept an undeclared sampler: {group}/{source}"
                )
            if requirements.complete_episodes and options.get("drop_last", False):
                raise ValueError(
                    f"Complete episodes require drop_last=False: {group}/{source}"
                )
            if requirements.max_episodes is not None:
                self.valid_groups[group][source] = EpisodeLimitedDataset(
                    dataset, requirements.max_episodes
                )
        self.valid_datasets = self.valid_groups.get(
            DEFAULT_VALID_GROUP, next(iter(self.valid_groups.values()), {})
        )

    def _make_loader(self, dataset, params, *, default_shuffle):
        params = dict(params)
        sampler = params.pop("sampler", None)
        anchor = params.pop("anchor_sampler", None)
        balance = params.pop("group_balance_sampler", None)
        if sum(value is not None for value in (sampler, anchor, balance)) > 1:
            raise ValueError("Configure only one sampler per source")
        if anchor is not None:
            from egomimic.rldb.zarr.e1_anchor_sampler import build_anchor_sampler

            sampler = build_anchor_sampler(dataset, **dict(anchor))
        if balance is not None:
            from egomimic.rldb.zarr.group_balance_sampler import (
                build_group_balance_sampler,
            )

            trainer = self.trainer
            sampler = build_group_balance_sampler(
                dataset,
                **dict(balance),
                num_replicas=trainer.world_size if trainer is not None else 1,
                rank=trainer.global_rank if trainer is not None else 0,
            )
        collate = params.pop("collate_fn", self.collate_fn)
        if isinstance(collate, Mapping):
            collate = hydra.utils.instantiate(collate)
        if isinstance(sampler, Mapping):
            sampler = hydra.utils.instantiate(sampler, dataset=dataset)
        shuffle = params.pop("shuffle", default_shuffle)
        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False if sampler is not None else shuffle,
            collate_fn=collate,
            **params,
        )

    def train_dataloader(self):
        if self.weighted_dataset is not None:
            params = dict(self.weighted_dataloader_params)
            if any(
                key in params
                for key in ("shuffle", "sampler", "batch_sampler", "collate_fn")
            ):
                raise ValueError("The weighted loader owns sampling and collation")
            trainer = self.trainer
            sampler = self.weighted_dataset.sampler(
                num_samples=self.samples_per_epoch,
                seed=self.sampling_seed,
                num_replicas=trainer.world_size if trainer is not None else 1,
                rank=trainer.global_rank if trainer is not None else 0,
            )
            return DataLoader(
                self.weighted_dataset,
                sampler=sampler,
                collate_fn=weighted_collate,
                **params,
            )
        iterables = dict()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config train_dataloader_params."
                )
            dataset_params = dict(dataset_params)
            iterables[dataset_name] = self._make_loader(
                dataset, dataset_params, default_shuffle=True
            )

        return CombinedLoader(iterables, self.train_loader_mode)

    def _val_loader_for_group(self, group_name: str) -> CombinedLoader:
        group_params = _params_for_group(
            self.valid_dataloader_params, group_name, set(self.valid_groups[group_name])
        )
        iterables = dict()
        for dataset_name, dataset in self.valid_groups[group_name].items():
            dataset_params = group_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name} in val group {group_name!r}. Please add {dataset_name} into your data config valid_dataloader_params."
                )
            dataset_params = dict(dataset_params)
            requested_shuffle = dataset_params.pop("shuffle", False)
            if self.force_valid_order and requested_shuffle:
                logger.warning(
                    "Forcing shuffle=False for ordered validation group %s, source %s",
                    group_name,
                    dataset_name,
                )
            shuffle = False if self.force_valid_order else requested_shuffle
            iterables[dataset_name] = self._make_loader(
                dataset, dataset_params, default_shuffle=shuffle
            )

        return CombinedLoader(iterables, self.valid_loader_mode)

    def val_dataloader(self):
        """One CombinedLoader per val group.

        A single group returns the bare CombinedLoader, which is exactly what
        this method returned before groups existed -- so single-group runs keep
        their old dataloader topology and `dataloader_idx` stays 0. Multiple
        groups return a list, and Lightning then runs the val loops back to
        back in list order, passing the group's index as `dataloader_idx`.
        """
        loaders = [self._val_loader_for_group(g) for g in self.valid_group_names]
        if len(loaders) == 1:
            return loaders[0]
        return loaders


def _extract_list_keys(batch):
    """Pop all list-valued keys from *batch* samples and return them separately.

    This lets ``default_collate`` handle tensors / numbers while variable-length
    annotation lists (``key_type == "annotation_keys"``) are preserved as
    ``list[list[str]]``.
    """
    list_keys = {k for k in batch[0] if isinstance(batch[0][k], list)}
    return {k: [sample.pop(k) for sample in batch] for k in list_keys}


def _extract_keys(batch, keys):
    return {k: [sample.pop(k) for sample in batch] for k in keys}


def annotation_collate(batch):
    """Collate that preserves variable-length list-valued keys (e.g. annotation_keys)."""
    batch = [dict(sample) for sample in batch]
    extracted = _extract_list_keys(batch)
    collated = default_collate(batch)
    collated.update(extracted)
    return collated


def weighted_collate(batch):
    """Keep each dataset's schema intact until model-side homogeneous batching."""
    by_dataset = {}
    for name, sample in batch:
        by_dataset.setdefault(name, []).append(sample)
    return {name: annotation_collate(samples) for name, samples in by_dataset.items()}
