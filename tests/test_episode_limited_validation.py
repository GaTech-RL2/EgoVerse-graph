from pathlib import Path

from torch.utils.data import Dataset, SequentialSampler

from egomimic.pl_utils.pl_data_utils import (
    EpisodeLimitedDataset,
    MultiDataModuleWrapper,
)


class _Leaf(Dataset):
    def __init__(self, episode: str, length: int):
        self.episode_path = Path(f"{episode}.zarr")
        self.length = length

    def __len__(self):
        return self.length

    def episode_id_at(self, index):
        return self.episode_path.stem

    def __getitem__(self, index):
        return {
            "episode_hash": self.episode_path.stem,
            "frame_index": index,
        }


class _Multi(Dataset):
    def __init__(self):
        self.datasets = {
            "episode-b": _Leaf("episode-b", 2),
            "episode-a": _Leaf("episode-a", 3),
        }
        self.index_map = [
            (name, index)
            for name, dataset in self.datasets.items()
            for index in range(len(dataset))
        ]

    def __len__(self):
        return len(self.index_map)

    def episode_id_at(self, index):
        name, local_index = self.index_map[index]
        return self.datasets[name].episode_id_at(local_index)

    def __getitem__(self, index):
        name, local_index = self.index_map[index]
        return self.datasets[name][local_index]


def test_episode_limit_keeps_complete_ordered_prefix():
    limited = EpisodeLimitedDataset(_Multi(), max_episodes=1)
    assert len(limited) == 2
    assert [limited[index]["episode_hash"] for index in range(len(limited))] == [
        "episode-b",
        "episode-b",
    ]


def test_open_loop_datamodule_forces_ordered_validation():
    dataset = _Multi()
    datamodule = MultiDataModuleWrapper(
        train_datasets={},
        valid_datasets={"human_bimanual": dataset},
        train_dataloader_params={},
        valid_dataloader_params={"human_bimanual": {"batch_size": 2, "shuffle": True}},
        valid_episode_limit=1,
        force_valid_order=True,
    )
    loader = datamodule.val_dataloader().iterables["human_bimanual"]
    assert isinstance(loader.sampler, SequentialSampler)
    assert len(datamodule.valid_datasets["human_bimanual"]) == 2
