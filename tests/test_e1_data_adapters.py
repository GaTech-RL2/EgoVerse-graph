"""The campaign adapters preserve domain routing and graph loader behavior."""

from types import SimpleNamespace

import numpy as np
import torch
import zarr

from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
from egomimic.rldb.zarr.e1_anchor_sampler import anchor_weights
from egomimic.rldb.zarr.e1_resolvers import (
    LocalFolderEpisodeResolverWithEmbodimentOverride,
    S3EpisodeResolverWithEmbodimentOverride,
)
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver


def test_local_resolver_keeps_image_size_and_overrides_stale_embodiment(tmp_path):
    (tmp_path / "episode.zarr").mkdir()

    def leaf(path, **kwargs):
        return SimpleNamespace(episode_path=path, embodiment="eva_bimanual", **kwargs)

    resolver = LocalFolderEpisodeResolverWithEmbodimentOverride(
        tmp_path, embodiment_override="yam_bimanual", image_hw=(480, 640)
    )
    resolver._dataset_class = leaf
    datasets = resolver.resolve()
    assert set(datasets) == {"episode"}
    assert datasets["episode"].embodiment == "yam_bimanual"
    assert datasets["episode"].image_hw == (480, 640)


def test_s3_override_changes_only_the_resolved_domain(tmp_path, monkeypatch):
    leaf = SimpleNamespace(embodiment="eva_bimanual", image_hw=(480, 640))
    monkeypatch.setattr(
        S3EpisodeResolver, "resolve", lambda *_args, **_kwargs: {"episode": leaf}
    )
    resolver = S3EpisodeResolverWithEmbodimentOverride(
        tmp_path, embodiment_override="yam_bimanual"
    )
    assert resolver.resolve()["episode"] is leaf
    assert leaf.embodiment == "yam_bimanual"
    assert leaf.image_hw == (480, 640)


def test_progress_weights_reduce_slow_episode_overrepresentation(tmp_path):
    class Leaf:
        def __init__(self, path):
            self.episode_path = path

        def __len__(self):
            return 80

    leaves = {}
    for name, speed in (("slow", 0.1), ("fast", 0.3)):
        path = tmp_path / name
        group = zarr.open_group(path, mode="w")
        pose = np.zeros((80, 7))
        pose[:, 0] = speed * np.arange(80) / 30
        pose[:, 3] = 1
        for hand in ("left", "right"):
            group.create_array(f"{hand}.obs_wrist_pose", data=pose)
        leaves[name] = Leaf(path)
    dataset = SimpleNamespace(
        datasets=leaves,
        index_map=[(name, i) for name in leaves for i in range(80)],
    )
    uniform = anchor_weights(dataset, alpha=1, fc_hz=3, fs_hz=30, pose_key="wrist")
    np.testing.assert_array_equal(uniform, np.ones(160))
    progress = anchor_weights(dataset, alpha=0, fc_hz=3, fs_hz=30, pose_key="wrist")
    np.testing.assert_allclose(
        progress[80:].mean() / progress[:80].mean(), 3.0, rtol=1e-6
    )


def test_anchor_sampler_uses_graph_collation_without_changing_config(monkeypatch):
    dataset = [{"x": torch.tensor([i]), "annotation_keys": [str(i)]} for i in range(4)]
    sampler = torch.utils.data.SequentialSampler(dataset)
    monkeypatch.setattr(
        "egomimic.rldb.zarr.e1_anchor_sampler.build_anchor_sampler",
        lambda actual_dataset, **_kwargs: sampler,
    )
    params = {
        "train": {
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": True,
            "anchor_sampler": {"alpha": 0.2},
        }
    }
    module = MultiDataModuleWrapper(
        train_datasets={"train": dataset},
        valid_datasets={"human_bimanual": dataset},
        train_dataloader_params=params,
        valid_dataloader_params={"human_bimanual": {"batch_size": 2}},
    )
    loader = module.train_dataloader().iterables["train"]
    assert loader.sampler is sampler
    batch = next(iter(loader))
    assert batch["annotation_keys"] == [["0"], ["1"]]
    torch.testing.assert_close(batch["x"], torch.tensor([[0], [1]]))
    assert params["train"]["shuffle"] is True
    assert params["train"]["anchor_sampler"] == {"alpha": 0.2}
