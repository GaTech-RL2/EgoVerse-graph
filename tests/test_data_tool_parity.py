"""Actual HDF5/Zarr and no-model recorded-video workflows."""

import hashlib
import json
import subprocess
import sys
from copy import deepcopy

import av
import h5py
import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.scripts.eva_process.eva_to_zarr import convert_episode
from egomimic.scripts.viz_language import visualize_data
from tests.fixtures.synthetic_episodes import write_episode
from tests.test_retained_recipe_steps import CONFIGS


def test_hdf5_conversion_preserves_source_and_refuses_overwrite(tmp_path):
    source = tmp_path / "1780000000000.hdf5"
    actions = np.tile(np.array([0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.7] * 2), (4, 1))
    with h5py.File(source, "w") as handle:
        for root in ("observations", "actions"):
            handle[f"{root}/eepose"] = actions
            handle[f"{root}/joints"] = actions
        for camera in ("front_img_1", "left_wrist_img", "right_wrist_img"):
            handle[f"observations/images/{camera}"] = np.full(
                (4, 64, 64, 3), 127, np.uint8
            )
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    target, preview = convert_episode(source, tmp_path / "out", "episode", "both", 30)
    assert preview is None
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    data = zarr.open(str(target), mode="r")
    assert data.attrs["total_frames"] == 4
    assert data.attrs["embodiment"] == "eva_bimanual"
    assert data["left.cmd_ee_pose"].shape[-1] == 7
    np.testing.assert_allclose(data["right.cmd_gripper"][:4], 0.7)
    assert np.isfinite(data["left.cmd_ee_pose"][:4]).all()
    assert set(data.attrs["intrinsics"]) == {"front_1"}
    assert len(bytes(data["images.front_1"][0])) > 0
    metadata = (target / "zarr.json").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        convert_episode(source, tmp_path / "out", "episode", "both", 30)
    assert (target / "zarr.json").read_bytes() == metadata


def test_local_conversion_imports_without_cloud_or_models():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'boto3','botocore','sqlalchemy','openpi'} or fullname.startswith('egomimic.models'):
            raise AssertionError('Unnecessary dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from egomimic.scripts.eva_process.eva_to_zarr import convert_episode
from egomimic.rldb.zarr.hdf5_to_zarr import convert_hdf5_to_zarr
from egomimic.scripts.viz_language import visualize_data
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_recorded_visualization_uses_native_units_and_complete_videos(
    tmp_path, monkeypatch
):
    root = tmp_path / "episodes"
    for i in range(2):
        write_episode(root, "eva", T=8, H=64, W=64, seed=i)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name="viz_language", overrides=["data=eva_pi_lang"])
    with open_dict(cfg):
        cfg.output_dir = str(tmp_path / "visualization")
        cfg.evaluator.max_episodes = 1
        cfg.evaluator.sample_every = 2
        ds = deepcopy(cfg.data.train_datasets.eva_bimanual)
        ds.resolver._target_ = (
            "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
        )
        ds.resolver.folder_path = str(root)
        ds.resolver.key_map.annotation_key = "annotations"
        ds.filters = None
        ds.mode = "total"
        ds.bounds_check = False
        cfg.data.valid_datasets = {"eva_bimanual": ds}
        cfg.data.valid_dataloader_params = {
            "eva_bimanual": {"batch_size": 3, "num_workers": 0}
        }
        cfg.data.train_datasets = {
            "must_not_be_opened": {"_target_": "builtins.int", "invalid": "error"}
        }

    def no_fitting(*args, **kwargs):
        raise AssertionError(
            "Recorded-data visualization must not fit a model normalizer"
        )

    monkeypatch.setattr(MultiDataset, "infer_norm_from_dataset", no_fitting)
    import egomimic.rldb.embodiment.embodiment as renderer

    painted = []
    original = renderer._viz_annotations

    def paint(*args, **kwargs):
        painted.append(kwargs["annotations"])
        return original(*args, **kwargs)

    monkeypatch.setattr(renderer, "_viz_annotations", paint)
    receipt = json.loads(visualize_data(cfg).read_text())
    assert len(receipt["videos"]) == 1
    with av.open(receipt["videos"][0]) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == 8
        assert float(video.streams.video[0].average_rate) == 30
    assert len(receipt["samples"]) == 4
    assert {row["frame"] for row in receipt["samples"]} == {0, 2, 4, 6}
    assert any("pick up the red cube" in str(text) for text in painted)
    assert any("place it in the bin" in str(text) for text in painted)
    with np.load(receipt["samples"][0]["arrays"], allow_pickle=False) as arrays:
        assert arrays["actions_cartesian"].shape == (100, 14)
        assert np.isfinite(arrays["actions_cartesian"]).all()
    with pytest.raises(FileExistsError):
        visualize_data(cfg)


def test_data_audit_records_exact_frames_and_configured_slices(tmp_path):
    from types import SimpleNamespace

    import torch

    from egomimic.eval.data_field_audit import DataFieldAudit

    audit = DataFieldAudit(
        {
            "opaque": {
                "first_zero": {"key": "a", "condition": "all_zero", "columns": [0, 2]},
                "invalid": {"key": "a", "condition": "nonfinite"},
            }
        }
    )
    values = torch.ones(3, 5, 4)
    values[0, :, :2] = 0
    values[1, 2, 2] = float("nan")
    audit.trainer = SimpleNamespace(default_root_dir=str(tmp_path))
    audit.set_validation_group("group")
    audit.on_validation_start()
    audit.on_validation_step(
        {
            "opaque": {
                "a": values,
                "episode_hash": ["a", "b", "c"],
                "frame_index": [7, 11, 13],
            }
        },
        0,
    )
    audit.on_validation_end()
    rows = [
        json.loads(row)
        for row in (tmp_path / "data-audit-rows.jsonl").read_text().splitlines()
    ]
    assert {(r["check"], r["episode"], r["frame"]) for r in rows} == {
        ("first_zero", "a", 7),
        ("invalid", "b", 11),
    }
    receipt = json.loads((tmp_path / "data-audit.json").read_text())
    assert receipt["counts"]["group"]["opaque"]["first_zero"] == {
        "frames": 3,
        "matches": 1,
    }
