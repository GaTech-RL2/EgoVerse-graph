"""Data-owned episode selection and evenly spaced diagnostic sampling."""

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.data_module import ZarrDataModule
from egomimic.rldb.zarr.zarr_dataset_multi import EvenStrideDataset, MultiDataset


class SelectedZarrDataModule(ZarrDataModule):
    """Keep an explicit YAML selection namespace out of loader arguments.

    Interpolations pass these options to dataset factories. The module itself
    does not interpret tasks, source names, hashes or sampling modes.
    """

    def __init__(self, selection, **kwargs):
        self.selection = selection
        super().__init__(**kwargs)


def selected_episodes(
    resolver,
    *,
    selection_mode,
    filters=None,
    episode_hashes=None,
    frames_per_episode=128,
    stride=None,
    **dataset_options,
):
    if selection_mode not in {"random", "pairs", "custom"}:
        raise ValueError("selection_mode must be random, pairs, or custom")
    if selection_mode == "random":
        return MultiDataset._from_resolver(resolver, filters=filters, **dataset_options)
    if not episode_hashes or len(set(episode_hashes)) != len(episode_hashes):
        raise ValueError(
            f"{selection_mode} selection requires nonempty unique episode hashes"
        )
    base = MultiDataset._from_resolver(
        resolver,
        filters=DatasetFilter(episode_hashes=episode_hashes),
        **dataset_options,
    )
    if stride is not None:
        return EvenStrideDataset(base, stride=stride)
    return EvenStrideDataset(base, frames_per_episode=frames_per_episode)
