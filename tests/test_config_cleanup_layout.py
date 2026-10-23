"""Protect the compact config menu without removing distinct protocols."""

import json

from omegaconf import OmegaConf

from scripts.audit_hydra_configs import CONFIGS


def test_deleted_experiment_menus_stay_deleted():
    for folder in (
        "abc",
        "abc_arc/human_bc",
        "arc_real/arcdur_abstime",
        "real",
        "sim",
        "_shared",
    ):
        assert not list((CONFIGS / "experiment" / folder).rglob("*.yaml"))


def test_real_recipes_never_inherit_another_experiment():
    paths = list((CONFIGS / "experiment/arc_real").rglob("*.yaml"))
    assert len(paths) == 9
    for path in paths:
        config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        assert all(
            not isinstance(entry, str) or not entry.startswith("/experiment/")
            for entry in config["defaults"]
        ), path


def test_models_are_grouped_and_have_reusable_bases():
    assert not list((CONFIGS / "model").glob("*.yaml"))
    for family in ("base", "hpt", "pi"):
        assert list((CONFIGS / "model" / family).glob("*.yaml"))
    assert not list((CONFIGS / "model/pi05").glob("*.yaml"))


def test_evaluator_names_describe_behavior_not_models():
    for path in (CONFIGS / "evaluator").rglob("*.yaml"):
        assert not path.stem.startswith(("eval_hpt", "eval_pi"))


def test_every_selector_migration_points_to_a_retained_config():
    receipt = CONFIGS.parents[1] / "docs/config_cleanup_migration.json"
    mapping = json.loads(receipt.read_text())["renamed"]
    for old, new in mapping.items():
        assert not (CONFIGS / f"{old}.yaml").exists()
        assert (CONFIGS / f"{new}.yaml").is_file()


def test_no_identical_yaml_config_bodies():
    bodies = {}
    for path in sorted(CONFIGS.rglob("*.yaml")):
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        body = json.dumps(value, sort_keys=True)
        assert body not in bodies, f"Duplicate configs: {bodies.get(body)}, {path}"
        bodies[body] = path
