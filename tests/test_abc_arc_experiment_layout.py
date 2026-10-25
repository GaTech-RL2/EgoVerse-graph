"""The experiment menu: nine real task leaves, the ABC-lab menu under abc_arc, and the migration map."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

ROOT = Path(__file__).resolve().parents[1]
REAL_LEAVES = sorted(
    f"real/{task}_{mode}"
    for task in ("stationary", "earbuds", "towels")
    for mode in ("time", "duration", "velocity")
)
ABC_LEAVES = sorted(
    p.relative_to(CONFIGS / "experiment").with_suffix("").as_posix()
    for p in (CONFIGS / "experiment/abc_arc").rglob("*.yaml")
)


def test_real_is_exactly_the_nine_task_leaves():
    found = sorted(
        p.relative_to(CONFIGS / "experiment").with_suffix("").as_posix()
        for p in (CONFIGS / "experiment/real").rglob("*.yaml")
    )
    assert found == REAL_LEAVES


def test_retired_menus_are_gone():
    for folder in ("abc_arc/cotrain", "yam_arc_grid", "real/arcdur_abstime"):
        assert not list((CONFIGS / "experiment" / folder).rglob("*.yaml"))
    pools = {p.stem for p in (CONFIGS / "data/abc_arc").glob("*.yaml")}
    assert {"stationary", "earbuds", "towels", "stationery_yam_loader"} <= pools
    assert not {
        p
        for p in pools
        if p.startswith(("stationery_s", "earbuds_", "towels_", "organize_"))
    }
    shared = {
        p.stem for p in (CONFIGS / "experiment/_shared/frozen_yam").glob("*.yaml")
    }
    assert shared == {"base", "stattempo", "stattempo_openloop", "hpt300_openloop"}


def test_migration_targets_exist_or_are_retired():
    mapping = json.loads((ROOT / "docs/experiment_config_migration.json").read_text())
    for target in mapping.values():
        if target is None:  # retired on 2026-10-22; recoverable from git history
            continue
        assert any(
            (CONFIGS / f"experiment/{target}{suffix}").is_file()
            for suffix in (".yaml", ".json")
        ), target
    assert sum(v is None for v in mapping.values()) == 18


@pytest.mark.parametrize("leaf", REAL_LEAVES + ABC_LEAVES)
def test_public_recipe_is_a_complete_run_entrypoint(leaf, monkeypatch):
    monkeypatch.setenv("EGOVERSE_DINOV3_CHECKPOINT", "/tmp/dinov3.pth")
    monkeypatch.setenv("EGOVERSE_CLIP_CHECKPOINT", "/tmp/clip")
    with compose_for_audit(CONFIGS / f"experiment/{leaf}.yaml") as cfg:
        for section in ("model", "data", "evaluator", "trainer", "callbacks", "logger"):
            assert cfg[section] is not None
            OmegaConf.to_container(cfg[section], resolve=True, throw_on_missing=True)
