from pathlib import Path

import numpy as np
import pytest

from egomimic.rldb.zarr import zarr_dataset_multi as dm
from egomimic.rldb.zarr.planar_retiming import PlanarCommandRetiming


class Episode:
    def __init__(self, path):
        self.metadata = {
            "total_frames": 35,
            "fps": 30.0,
            "embodiment": "PUSHSHAPES_SIM_U_SOCKET",
            "features": {},
        }
        self.intrinsics = None
        self.actions = np.stack([np.arange(35), np.zeros(35), np.zeros(35)], -1)

    def _collect_keys(self):
        return ["actions"]

    def read(self, requests):
        return {k: self.actions[a:b].copy() for k, (a, b) in requests.items()}


def test_virtual_views_have_equal_rates_and_never_pad(monkeypatch):
    monkeypatch.setattr(dm, "ZarrEpisode", Episode)
    t = PlanarCommandRetiming((1, 1.25, 1.5, 1.75, 2))
    ds = dm.ZarrDataset(
        Path("fake.zarr"), {"actions": {"zarr_key": "actions", "horizon": 31}}, [t]
    )
    assert len(ds) == 25
    for i in range(len(ds)):
        batch = ds[i]
        anchor, view = divmod(i, 5)
        assert batch["frame_index"] == anchor
        assert batch["retiming_rate"].item() == t.rates[view]
        assert batch["actions"][0, 0] == anchor
        assert batch["actions"][-1, 0] == anchor + 15 * t.rates[view]
    with pytest.raises(IndexError):
        ds[25]


def test_loader_rejects_wrong_raw_horizon(monkeypatch):
    monkeypatch.setattr(dm, "ZarrEpisode", Episode)
    with pytest.raises(ValueError, match="horizon"):
        dm.ZarrDataset(
            Path("fake.zarr"),
            {"actions": {"zarr_key": "actions", "horizon": 16}},
            [PlanarCommandRetiming((1, 2))],
        )


class HistoryEpisode(Episode):
    reads = []

    def _collect_keys(self):
        return ["actions", "state"]

    def read(self, requests):
        output = {}
        for key, (start, end) in requests.items():
            self.reads.append((key, start, end))
            values = self.actions if key == "actions" else np.arange(35)[:, None]
            output[key] = (values[start] if end is None else values[start:end]).copy()
        return output


@pytest.mark.parametrize("anchor", [0, 1, 3, 4])
@pytest.mark.parametrize("view", range(5))
def test_causal_history_preserves_views_current_actions_and_episode_boundary(
    monkeypatch, anchor, view
):
    monkeypatch.setattr(dm, "ZarrEpisode", HistoryEpisode)
    key_map = {
        "actions": {"zarr_key": "actions", "horizon": 31},
        "observations.state.test": {"zarr_key": "state", "key_type": "proprio_keys"},
    }
    transform = PlanarCommandRetiming((1, 1.25, 1.5, 1.75, 2))
    single = dm.ZarrDataset(Path("fake.zarr"), key_map, [transform])
    history = dm.ZarrDataset(
        Path("fake.zarr"), key_map, [transform], observation_history_steps=2
    )
    assert len(history) == len(single) == 25
    expected = single[anchor * 5 + view]
    HistoryEpisode.reads = []
    result = history[anchor * 5 + view]
    frames = [max(0, anchor - 1), anchor]
    assert result["observations.state.test"].tolist() == [[frames[0]], [frames[1]]]
    assert result["observation_frame_indices"].tolist() == frames
    assert result["frame_index"] == anchor
    assert result["actions"].equal(expected["actions"])
    assert result["retiming_rate"].equal(expected["retiming_rate"])
    assert all(
        start <= anchor and end is None
        for key, start, end in HistoryEpisode.reads
        if key == "state"
    )


def test_history_invalid_length_rejected_before_episode_read(monkeypatch):
    for steps in [0, -1, 2.5, True]:
        with pytest.raises(ValueError, match="positive integer"):
            dm.ZarrDataset(Path("missing.zarr"), {}, observation_history_steps=steps)


def test_history_resolver_forwards_option_to_native_leaf():
    from egomimic.rldb.zarr.embodiment_resolvers import (
        LocalFolderEpisodeResolverWithEmbodimentOverride,
    )

    resolver = LocalFolderEpisodeResolverWithEmbodimentOverride(
        Path("fake"), observation_history_steps=2
    )
    assert resolver._dataset_init_kwargs()["observation_history_steps"] == 2
