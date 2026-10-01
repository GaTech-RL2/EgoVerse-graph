"""Opaque mixture source names never override the data-owned normalization identity."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from egomimic.rldb.zarr.data_module import ZarrDataModule, _digest
from tests.test_data_context_contract import _state


class PoolDataset:
    embodiment = 7
    key_map = {"action": {"zarr_key": "action", "key_type": "action_keys"}}

    def __init__(self, value, resolver):
        self.value = value
        self.norm_stats = None

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {
            "embodiment": 7,
            "action": np.full((2, 3), self.value, dtype=np.float32),
        }

    def set_norm_stats_from(self, normalizer):
        self.norm_stats = normalizer.norm_stats


def test_same_embodiment_statistics_pool_both_sources_and_restore(tmp_path):
    datasets = {
        name: {
            "_target_": __name__ + ".PoolDataset",
            "value": value,
            "resolver": {"key_map": {}, "transform_list": []},
        }
        for name, value in (("a", 0), ("b", 10))
    }

    def module():
        return ZarrDataModule(datasets, {}, {}, {})

    with pytest.raises(ValueError, match="pooled statistics"):
        module().prepare_context(mode="train", normalization={})
    context = module().prepare_context(
        mode="normalization",
        normalization={
            "save_cache_dir": str(tmp_path),
            "sample_frac": 1,
            "num_workers": 0,
        },
    )
    stats = context.normalizer.norm_stats[7]["action"]
    np.testing.assert_allclose(stats["quantile_1"], 0)
    np.testing.assert_allclose(stats["quantile_99"], 10)
    restored = module().prepare_context(
        mode="train",
        normalization={
            "precomputed_norm_path": str(tmp_path / "norm_stats/norm_stats.json"),
        },
    )
    assert restored.snapshot()["sha256"] == context.snapshot()["sha256"]
    assert restored.snapshot()["preprocessing"] == context.snapshot()["preprocessing"]


@pytest.mark.parametrize(
    "names",
    [
        ("yam_bimanual", "eva_bimanual"),
        ("abc_yam_bimanual", "rl2_yam_bimanual"),
        ("a", "b"),
    ],
)
def test_mixture_sources_share_only_their_actual_normalization_identity(names):
    state = _state()
    module = ZarrDataModule(
        train_datasets=OmegaConf.create(
            {
                name: {"_target_": "tests.test_data_context_contract.TinyDataset"}
                for name in names
            }
        ),
        valid_datasets={},
        train_dataloader_params={},
        valid_dataloader_params={},
        dataset_weights={name: 1 for name in names},
        weighted_dataloader_params={"batch_size": 2},
    )
    context = module.prepare_context(
        mode="train",
        normalization={},
        restored_state={
            "kind": "zarr-normalizer-v1",
            "normalizer_state": state,
            "sha256": _digest(state),
        },
    )
    assert context.normalizer.embodiments == {7}
    for dataset in module.train_datasets.values():
        assert dataset.norm_stats is context.normalizer.norm_stats
    assert set(module.weighted_dataset.datasets) == set(names)
