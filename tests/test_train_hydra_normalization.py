"""Normalization-only entry points must preserve every embodiment's cache."""
import json

import numpy as np
from omegaconf import OmegaConf
import zarr

from egomimic import trainHydra


def test_reload_two_embodiment_cache_at_same_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(trainHydra, "load_env", lambda: None)
    datasets = {}
    for name, width in [("pushshapes_sim_u_socket", 3), ("pushshapes_sim_chain_gripper", 4)]:
        folder = tmp_path / name
        group = zarr.open_group(str(folder / "episode_0.zarr"), mode="w")
        group.create_array("actions", data=np.arange(12*width).reshape(12, width).astype(float))
        group.create_array("observations.state", data=np.arange(72).reshape(12, 6).astype(float))
        group.attrs.update(total_frames=12, embodiment=name, features={
            "actions": {"dtype": "float64"}, "observations.state": {"dtype": "float64"}})
        datasets[name] = {
            "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver",
            "resolver": {
                "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver",
                "folder_path": str(folder),
                "key_map": {
                    "_target_": "egomimic.rldb.embodiment.pushshapes.get_planar_keymap",
                    "action_horizon": 4, "observation_horizon": 2,
                    "action_target_offset": 1, "norm_mode": True,
                },
                "transform_list": {
                    "_target_": "egomimic.rldb.embodiment.pushshapes.get_planar_paper_transform_list",
                    "action_horizon": 4, "action_target_offset": 1,
                },
            },
            "mode": "train", "valid_ratio": 0, "bounds_check": False,
        }
    cfg = OmegaConf.create({
        "mode": "train", "seed": 42, "model": {"pipeline": {}},
        "norm_stats_only": True, "paths": {"output_dir": str(tmp_path)},
        "data": {
            "_target_": "egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper",
            "train_datasets": datasets, "valid_datasets": {},
            "train_dataloader_params": {}, "valid_dataloader_params": {},
        },
        "norm_stats": {"norm_mode": "quantile", "sample_frac": 1,
                       "num_workers": 0, "save_cache_dir": str(tmp_path / "cache"),
                       "precomputed_norm_path": None},
    })
    trainHydra.train(cfg)
    path = tmp_path / "cache/norm_stats/norm_stats.json"
    original = json.loads(path.read_text())["stats"]
    assert len(original) == 2
    cfg.norm_stats.precomputed_norm_path = str(path.parent)
    trainHydra.train(cfg)
    assert json.loads(path.read_text())["stats"] == original
