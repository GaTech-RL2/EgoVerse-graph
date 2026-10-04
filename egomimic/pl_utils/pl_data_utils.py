import logging
from bisect import bisect_right
from functools import partial
from pathlib import Path

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Dataset, default_collate

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.proportional_batch_sampler import ProportionalHomogeneousBatchSampler

logger = logging.getLogger(__name__)

# Name of the val group that keeps the pre-groups behaviour: its metrics stay
# `Valid/...`, so runs predating multi-group validation overlay on the same
# wandb charts. Every other group is namespaced (`Valid_{group}/...`).
DEFAULT_VALID_GROUP = "valid"


def _is_embodiment_name(name) -> bool:
    """True when *name* names an embodiment the loader keys can carry.

    This is what separates the two shapes `valid_datasets` accepts. Note the
    pipeline runner itself treats source keys as opaque -- it never resolves an
    embodiment -- so this lookup is used ONLY to disambiguate config shape here,
    never to route a batch.
    """
    try:
        get_embodiment_id(str(name))
    except (KeyError, AttributeError):
        return False
    return True


def as_valid_groups(valid_datasets: dict) -> dict:
    """Normalise `valid_datasets` to `{group_name: {source: dataset}}`.

    Two accepted shapes:

      * FLAT -- `{source: dataset}`. Every key is an embodiment name, so the
        whole mapping becomes the single `DEFAULT_VALID_GROUP` group. This is
        the historical shape and stays identical in behaviour.
      * GROUPED -- `{group_name: {source: dataset}}`. Keys are arbitrary labels
        (`valid`, `newtask`, ...), each mapping to a per-source dict. Lightning
        runs one val loop per group, in insertion order.

    The shapes are told apart by whether the top-level keys are embodiment
    names rather than by inspecting value types -- hydra hands us instantiated
    objects for the flat shape and DictConfig/dict for the grouped one, and
    that distinction is fragile across omegaconf versions.
    """
    if not valid_datasets:
        return {}

    keys = list(valid_datasets.keys())
    if all(_is_embodiment_name(key) for key in keys):
        return {DEFAULT_VALID_GROUP: dict(valid_datasets)}

    mixed = [key for key in keys if _is_embodiment_name(key)]
    if mixed:
        raise ValueError(
            "valid_datasets mixes embodiment keys with val-group keys "
            f"({mixed} look like embodiments, "
            f"{[key for key in keys if key not in mixed]} do not). Use either "
            "{source: dataset} or {group: {source: dataset}}, not both."
        )

    groups = {}
    for group_name, members in valid_datasets.items():
        try:
            members = dict(members)
        except TypeError as exc:
            raise ValueError(
                f"val group {group_name!r} must map source -> dataset, got "
                f"{type(members).__name__}."
            ) from exc
        bad = [key for key in members if not _is_embodiment_name(key)]
        if bad:
            raise ValueError(
                f"val group {group_name!r} has non-embodiment keys {bad}. A "
                "group's keys name datasets, so they must be embodiment names; "
                "nesting groups inside groups is not supported."
            )
        groups[group_name] = members
    return groups


def _params_for_group(valid_dataloader_params: dict, group_name: str) -> dict:
    """Per-group dataloader params, falling back to a single shared block.

    `valid_dataloader_params` may be keyed by source (one block shared by every
    group) or by group name (a block per group). The former is the historical
    shape.
    """
    if not valid_dataloader_params:
        return {}
    if all(_is_embodiment_name(key) for key in valid_dataloader_params):
        return valid_dataloader_params
    return valid_dataloader_params.get(group_name, {})


def _episode_id_at(dataset: Dataset, index: int) -> str:
    """Resolve an episode id without loading a sample or decoding images."""

    index_map = getattr(dataset, "index_map", None)
    datasets = getattr(dataset, "datasets", None)
    if index_map is not None and datasets is not None:
        dataset_name, local_index = index_map[index]
        return _episode_id_at(datasets[dataset_name], int(local_index))

    episode_path = getattr(dataset, "episode_path", None)
    if episode_path is None:
        raise TypeError(
            "limit_val_episodes requires a dataset exposing either index_map/"
            "datasets or episode_path; got "
            f"{type(dataset).__name__}"
        )
    name = Path(episode_path).name
    return name[:-5] if name.endswith(".zarr") else name


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
            for group, members in as_valid_groups(valid_datasets).items()
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
        self.collate_fn = annotation_collate

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

    def train_dataloader(self):
        iterables = dict()
        for dataset_name, dataset in self.train_datasets.items():
            dataset_params = self.train_dataloader_params.get(dataset_name)
            if dataset_params is None or len(dataset_params) == 0:
                raise ValueError(
                    f"No dataloader params found for dataset {dataset_name}. Please add {dataset_name} into your data config train_dataloader_params."
                )
            dataset_params = dict(dataset_params)
            sampler_cfg = dataset_params.pop("anchor_sampler", None)
            if sampler_cfg:
                from omegaconf import OmegaConf

                from egomimic.rldb.zarr.e1_anchor_sampler import build_anchor_sampler

                if OmegaConf.is_config(sampler_cfg):
                    sampler_cfg = OmegaConf.to_container(sampler_cfg, resolve=True)
                sampler = build_anchor_sampler(dataset, **dict(sampler_cfg))
                dataset_params.pop("shuffle", None)
                iterables[dataset_name] = DataLoader(
                    dataset,
                    sampler=sampler,
                    collate_fn=self.collate_fn,
                    **dataset_params,
                )
                continue
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=True,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")

    def _val_loader_for_group(self, group_name: str) -> CombinedLoader:
        group_params = _params_for_group(self.valid_dataloader_params, group_name)
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
                    "Forcing shuffle=False for ordered validation group %s, "
                    "source %s",
                    group_name,
                    dataset_name,
                )
            shuffle = False if self.force_valid_order else requested_shuffle
            iterables[dataset_name] = DataLoader(
                dataset,
                shuffle=shuffle,
                collate_fn=self.collate_fn,
                **dataset_params,
            )

        return CombinedLoader(iterables, "max_size_cycle")

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


class _SourceTaggedConcatDataset(Dataset):
    """Expose flat index space while retaining each sample's embodiment name."""

    def __init__(self, datasets: dict[str, Dataset]):
        self.names = tuple(datasets)
        self.concatenated = ConcatDataset(tuple(datasets.values()))

    def __len__(self):
        return len(self.concatenated)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        source = self.names[bisect_right(self.concatenated.cumulative_sizes, idx)]
        return source, self.concatenated[idx]


def _proportional_collate(tagged_samples):
    if not tagged_samples:
        raise ValueError('Empty proportional batch')
    source = tagged_samples[0][0]
    if any(name != source for name, _ in tagged_samples):
        raise ValueError('Proportional batch mixed action spaces')
    return {source: annotation_collate([sample for _, sample in tagged_samples])}


def _single_source_validation_collate(samples, *, source):
    """Give Lightning a plain DataLoader batch with the pipeline source key."""
    return {source: annotation_collate(samples)}


class StatefulProportionalDataLoader(DataLoader):
    """Checkpoint delivered batches, not sampler-prefetched batches.

    Lightning saves and restores the dataloader's state mid-epoch. A plain
    DataLoader does not expose that state, even when its datamodule does.
    """

    def __init__(self, *args, batch_sampler: ProportionalHomogeneousBatchSampler, **kwargs):
        super().__init__(*args, batch_sampler=batch_sampler, **kwargs)
        self._active_epoch = 0
        self._next_batch = 0

    def state_dict(self) -> dict:
        epoch = self._active_epoch
        next_batch = self._next_batch
        if next_batch == len(self):
            epoch += 1
            next_batch = 0
        return {
            'schema_version': 2,
            'source_lengths': dict(self.batch_sampler.source_lengths),
            'batch_size': self.batch_sampler.batch_size,
            'seed': self.batch_sampler.seed,
            'active_epoch': epoch,
            'next_batch': next_batch,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get('schema_version') != 2:
            raise ValueError('Unsupported proportional dataloader state')
        if (
            state.get('source_lengths') != self.batch_sampler.source_lengths
            or state.get('batch_size') != self.batch_sampler.batch_size
            or state.get('seed') != self.batch_sampler.seed
        ):
            raise ValueError('Proportional dataloader resume identity changed')
        epoch = state.get('active_epoch')
        next_batch = state.get('next_batch')
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('Invalid proportional dataloader epoch')
        if isinstance(next_batch, bool) or not isinstance(next_batch, int) or not 0 <= next_batch < len(self):
            raise ValueError('Invalid proportional dataloader batch offset')
        self._active_epoch = epoch
        self._next_batch = next_batch

    def __iter__(self):
        self.batch_sampler.set_epoch(self._active_epoch)
        self.batch_sampler.start_batch = self._next_batch
        for batch in super().__iter__():
            self._next_batch += 1
            yield batch
        self._active_epoch += 1
        self._next_batch = 0


class ProportionalMultiDataModuleWrapper(MultiDataModuleWrapper):
    """Co-train sources by window count, not CombinedLoader max-size cycling.

    Each complete epoch visits every eligible frame-window once in homogeneous
    batches. Validation uses one plain DataLoader per group: nesting
    CombinedLoaders inside Lightning's multi-loader list yields tuples, not
    pipeline mappings.
    The exact source order, counts, seed, batch size, and partial-tail policy
    must be recorded in the resolved run bundle.
    """

    def __init__(
        self,
        *args,
        proportional_train_batch_size: int,
        proportional_train_num_workers: int,
        proportional_train_seed: int = 42,
        proportional_pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(proportional_train_num_workers, bool) or proportional_train_num_workers < 0:
            raise ValueError('proportional_train_num_workers must be nonnegative')
        self.proportional_train_batch_size = proportional_train_batch_size
        self.proportional_train_num_workers = int(proportional_train_num_workers)
        self.proportional_train_seed = int(proportional_train_seed)
        self.proportional_pin_memory = bool(proportional_pin_memory)
        self.proportional_batch_sampler = None
        self.proportional_data_loader = None
        self._sampler_resume_state = None

    def train_dataloader(self):
        lengths = {name: len(ds) for name, ds in self.train_datasets.items()}
        tagged = _SourceTaggedConcatDataset(self.train_datasets)
        sampler = ProportionalHomogeneousBatchSampler(
            lengths, self.proportional_train_batch_size, self.proportional_train_seed
        )
        self.proportional_batch_sampler = sampler
        loader = StatefulProportionalDataLoader(
            tagged,
            batch_sampler=sampler,
            num_workers=self.proportional_train_num_workers,
            pin_memory=self.proportional_pin_memory,
            collate_fn=_proportional_collate,
        )
        if self._sampler_resume_state is not None:
            loader.load_state_dict(self._sampler_resume_state)
            self._sampler_resume_state = None
        self.proportional_data_loader = loader
        return loader

    def val_dataloader(self):
        loaders = []
        for group_name in self.valid_group_names:
            members = self.valid_groups[group_name]
            if len(members) != 1:
                raise ValueError('Proportional validation groups require one native action space each')
            source, dataset = next(iter(members.items()))
            params = _params_for_group(self.valid_dataloader_params, group_name).get(source)
            if not params:
                raise ValueError(f'Missing validation loader params for {group_name}/{source}')
            params = dict(params)
            if params.pop('shuffle', False):
                raise ValueError('Proportional validation must be deterministic')
            loaders.append(DataLoader(
                dataset, shuffle=False,
                collate_fn=partial(_single_source_validation_collate, source=source),
                **params,
            ))
        return loaders[0] if len(loaders) == 1 else loaders

    def state_dict(self) -> dict:
        loader = self.proportional_data_loader
        if loader is None:
            return dict(self._sampler_resume_state or {})
        return loader.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            raise ValueError('Proportional co-train checkpoint has no sampler state')
        self._sampler_resume_state = dict(state_dict)
        if self.proportional_data_loader is not None:
            self.proportional_data_loader.load_state_dict(state_dict)


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
    extracted = _extract_list_keys(batch)
    collated = default_collate(batch)
    collated.update(extracted)
    return collated
