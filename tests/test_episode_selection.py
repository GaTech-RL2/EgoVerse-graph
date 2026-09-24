"""Selection stays in the data adapter and cannot change pinned membership."""

from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.selection import selected_episodes
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from tests.fixtures.synthetic_episodes import write_episode
from tests.test_retained_recipe_steps import CONFIGS


@pytest.mark.parametrize("mode", ["random", "pairs", "custom"])
def test_selected_sources_restore_data_context_and_real_frame_indices(mode, tmp_path):
    for vendor in ("eva", "aria"):
        for i in range(3):
            write_episode(tmp_path / vendor, vendor, T=8, H=32, W=32, seed=i)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=["data=cotrain_pi_latent", f"data.selection.mode={mode}"],
        )
    cfg.data.selection.pair_hashes.eva = ["eva_00", "eva_01"]
    cfg.data.selection.pair_hashes.aria = ["aria_00", "aria_01"]
    cfg.data.selection.custom_hashes = deepcopy(cfg.data.selection.pair_hashes)
    cfg.data.selection.frames_per_episode = 3
    for domain, vendor in [("eva_bimanual", "eva"), ("human_bimanual", "aria")]:
        ds = cfg.data.train_datasets[domain]
        with open_dict(ds):
            ds.resolver._target_ = (
                "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
            )
            del ds.resolver.require_annotations
            ds.resolver.folder_path = str(tmp_path / vendor)
            ds.filters = None
            ds.bounds_check = False
    dm = instantiate(cfg.data, _recursive_=False)
    context = dm.prepare_context(mode="train", normalization={"num_workers": 0})
    for ds in dm.train_datasets.values():
        assert ds.norm_stats is context.normalizer.norm_stats
        vendor = "eva" if ds[0]["embodiment"] == 6 else "aria"
        count = 3 if mode == "random" else 2
        assert set(ds.datasets) == {f"{vendor}_{i:02d}" for i in range(count)}
        if mode != "random":
            assert len(ds) == 6
            assert ds.base.norm_stats is context.normalizer.norm_stats
            assert [ds.frame_index_at(i) for i in range(3)] == [0, 4, 7]
            assert not isinstance(ds, MultiDataset)


def test_pins_reject_incomplete_resolver_results_and_empty_selection():
    class IncompleteResolver:
        def resolve(self, filters):
            return {}

    with pytest.raises(ValueError, match="Pinned episodes did not load"):
        MultiDataset._from_resolver(
            IncompleteResolver(),
            filters=DatasetFilter(episode_hashes=["corrupt"]),
            mode="total",
        )
    with pytest.raises(ValueError, match="nonempty"):
        selected_episodes(
            IncompleteResolver(), selection_mode="pairs", episode_hashes=[]
        )
