"""The entire retained grid binds its actual preprocessing, independent of names."""

import copy

import pytest
from omegaconf import OmegaConf

from egomimic.pl_utils.data_context import DataContext
from egomimic.rldb.zarr.data_module import _preprocessing_contract
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

GRID = sorted((CONFIGS / "experiment/yam_arc_grid").glob("*.yaml"))


def test_complete_grid_is_retained():
    assert len(GRID) == 64


@pytest.mark.parametrize("path", GRID, ids=lambda p: p.stem)
def test_grid_data_matches_declared_contract_and_rejects_frame_drift(path):
    with compose_for_audit(path) as cfg:
        required = OmegaConf.to_container(cfg.model.data_requirements, resolve=True)
        for dataset in cfg.data.train_datasets.values():
            actual = _preprocessing_contract(dataset, cfg.data.source_fps)
            snapshot = {"preprocessing": {"7": actual}}
            context = DataContext(None, {}, (), snapshot)
            context.validate_requirements(required)
            changed = copy.deepcopy(snapshot)
            changed["preprocessing"]["7"]["source_fps"] = 15
            with pytest.raises(ValueError, match="source_fps"):
                DataContext(None, {}, (), changed).validate_requirements(required)
