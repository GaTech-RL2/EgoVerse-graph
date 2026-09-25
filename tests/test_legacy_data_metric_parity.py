"""Numerical references executed in the immutable EgoVerse source checkout."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from scripts.integration.reference_probe import probe
from tests.fixtures.synthetic_episodes import write_moving_episode

ROOT = Path(__file__).parents[1]
REFERENCES = Path(__file__).parent / "fixtures/legacy_pipeline"
CASES = [
    (v, "cartesian", f)
    for v in ("eva", "aria", "mecka", "scale")
    for f in ("camframe", "eef_frame")
] + [("aria", "keypoints", f) for f in ("camframe", "eef_frame")]


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    root = tmp_path_factory.mktemp("moving-legacy-fixtures")
    for vendor in ("eva", "aria", "mecka", "scale"):
        for seed in range(3):
            write_moving_episode(root / vendor, vendor, seed=seed)
    return root


@pytest.mark.parametrize("vendor,representation,frame", CASES)
def test_preprocessing_normalization_and_headline_metrics_match_legacy(
    episodes, vendor, representation, frame
):
    name = f"{vendor}-{representation}-{frame}"
    archive = REFERENCES / (name + ".npz")
    reference = json.loads(archive.with_suffix(".json").read_text())
    assert reference["source_commit"] == "ec5c903c067bf1b29bbff781bc414c6df2bcf1f2"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == reference["archive_sha256"]
    )
    actual, metadata = probe(episodes, ROOT, vendor, representation, frame, "graph")
    assert metadata["model_keys"] == reference["model_keys"]
    assert metadata["normalization_frames"] == reference["normalization_frames"]
    with np.load(archive, allow_pickle=False) as expected:
        assert set(actual) == set(expected.files)
        for key, value in actual.items():
            assert value.shape == expected[key].shape, key
            if "observations.images." in key:
                # JPEG decoder rounding can differ between platform wheels;
                # compare routing/shape and pixels to one quantization unit.
                np.testing.assert_allclose(
                    value, expected[key], rtol=0, atol=1 / 255, err_msg=key
                )
            else:
                np.testing.assert_allclose(
                    value, expected[key], rtol=2e-5, atol=2e-6, err_msg=key
                )


@pytest.mark.parametrize("kind", ["pooled", "pertoken"])
def test_retained_language_schedule_matches_all_source_epochs(kind):
    expected = json.loads(
        (Path(__file__).parent / "fixtures/legacy_language_schedule.json").read_text()
    )
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"model=hpt_bc_pickplace_qwen_{kind}"],
        )
    assert cfg.model.scheduler_interval == "epoch"
    optimizer = instantiate(cfg.model.optimizer)(
        params=[torch.nn.Parameter(torch.ones(1))]
    )
    scheduler = instantiate(cfg.model.scheduler)(optimizer=optimizer)
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(expected["total_epochs"]):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values == pytest.approx(expected["values"], rel=1e-12, abs=1e-15)
