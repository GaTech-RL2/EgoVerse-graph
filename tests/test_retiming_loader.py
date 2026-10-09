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
