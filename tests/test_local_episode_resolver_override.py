from __future__ import annotations

from egomimic.rldb.zarr import zarr_dataset_multi as dataset_module


class _FakeZarrEpisode:
    def __init__(self, _path):
        self.metadata = {
            "total_frames": 1,
            "embodiment": "PUSHSHAPES_SIM",
            "features": {},
        }

    def _collect_keys(self):
        return []

    @property
    def intrinsics(self):
        return None


def test_local_resolver_applies_override_before_leaf_initialization(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset_module, "ZarrEpisode", _FakeZarrEpisode)
    episode = tmp_path / "episode.zarr"
    episode.mkdir()
    resolver = dataset_module.LocalEpisodeResolverWithEmbodimentOverride(
        folder_path=tmp_path,
        key_map={},
        embodiment_override="pushshapes_sim_u_socket",
    )

    datasets = resolver._load_zarr_datasets(tmp_path, {"episode"})

    assert datasets["episode"].embodiment == "pushshapes_sim_u_socket"
