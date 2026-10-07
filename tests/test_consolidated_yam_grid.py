"""The entire retained grid binds its actual preprocessing, independent of names."""

import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egomimic.pl_utils.data_context import DataContext
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.data_module import _preprocessing_contract
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

GRID = sorted((CONFIGS / "experiment/yam_arc_grid").glob("*.yaml"))


def test_complete_grid_is_retained():
    inventory = json.loads(
        (Path(__file__).parent / "fixtures/yam-grid-source-inventory.json").read_text()
    )
    expected = set(inventory["retained_recipes"]) | set(inventory["added_recipes"])
    assert {path.stem for path in GRID} == expected


@pytest.mark.parametrize("path", GRID, ids=lambda p: p.stem)
def test_grid_data_matches_declared_contract_and_rejects_frame_drift(path):
    with compose_for_audit(path) as cfg:
        required = OmegaConf.to_container(cfg.model.data_requirements, resolve=True)
        snapshot = {"preprocessing": {}}
        for dataset in cfg.data.train_datasets.values():
            identity = str(get_embodiment_id(dataset.resolver.embodiment_override))
            actual = _preprocessing_contract(dataset, cfg.data.source_fps)
            if identity in snapshot["preprocessing"]:
                assert snapshot["preprocessing"][identity] == actual
            snapshot["preprocessing"][identity] = actual
        context = DataContext(None, {}, (), snapshot)
        context.validate_requirements(required)
        for identity in required["preprocessing"]:
            changed = copy.deepcopy(snapshot)
            changed["preprocessing"][identity]["source_fps"] = 15
            with pytest.raises(ValueError, match="source_fps"):
                DataContext(None, {}, (), changed).validate_requirements(required)
