"""Explicit objective identity and single-variable resolved recipe regression."""

import copy
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from typed_libero_profile_v3 import (  # noqa: E402 - task tool path initialized above
    COMMON,
    PHASES,
    validate_flat_config,
)

from egomimic.benchmarks.libero.native_launch_profiles import (  # noqa: E402 - task tool path initialized above
    AV0_PROFILES,
    PROFILES,
    profile_for_argv,
    profile_for_config,
)


def recipe(profile, monkeypatch):
    monkeypatch.setenv(profile.replay_environment, "/unchanged/replay")
    with initialize_config_dir(
        version_base="1.3", config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=["hydra/launcher=basic", "+experiment=" + profile.experiment],
            return_hydra_config=True,
        )
        cfg.hydra.runtime.output_dir = "/unchanged/output"
        HydraConfig.instance().set_config(cfg)
        OmegaConf.set_struct(cfg, False)
        del cfg["hydra"]
        return OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize("suite", list(PROFILES))
def test_only_resolved_model_velocity_weight_changes(suite, monkeypatch):
    base = recipe(PROFILES[suite], monkeypatch)
    ablated = recipe(AV0_PROFILES[suite], monkeypatch)
    assert profile_for_config(base) == PROFILES[suite]
    assert profile_for_config(ablated) == AV0_PROFILES[suite]
    assert (
        profile_for_argv(["+experiment=" + AV0_PROFILES[suite].experiment])
        == AV0_PROFILES[suite]
    )
    expected = copy.deepcopy(base["model"])
    expected["pipeline"]["stages"][8]["action_velocity_weight"] = 0.0
    assert ablated["model"] == expected
    expected_data = copy.deepcopy(base["data"])
    expected_data["run_provenance"]["objective"]["action_velocity_weight"] = 0.0
    expected_data["run_provenance"]["ablation"] = "action_velocity_off"
    assert ablated["data"] == expected_data
    for key in ("trainer", "normalizer", "callbacks", "evaluator", "seed"):
        assert ablated[key] == base[key], key
    assert ablated["run_provenance"]["objective"] == {
        **base["run_provenance"]["objective"],
        "action_velocity_weight": 0.0,
    }
    bad = copy.deepcopy(ablated)
    bad["run_provenance"]["objective"]["action_velocity_weight"] = 1.0
    with pytest.raises(ValueError, match="objective provenance"):
        profile_for_config(bad)
    bad = copy.deepcopy(base)
    bad["run_provenance"]["ablation"] = "action_velocity_off"
    with pytest.raises(ValueError, match="ablation provenance"):
        profile_for_config(bad)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_typed_flat_guard_rejects_opposite_objective(weight):
    flat = {
        **COMMON,
        **PHASES["smoke"],
        "model.pipeline.stages.8.action_velocity_weight": weight,
    }
    for idx, owner in [(4, "encoder"), (6, "field")]:
        for key, value in [("hidden_dim", 240), ("depth", 12), ("num_heads", 8)]:
            flat[f"model.pipeline.stages.{idx}.{owner}.backbone.{key}"] = value
    flat.update(
        {
            "data.train_source_names": ["libero_panda"],
            "data.valid_source_names": ["libero_panda"],
            "typed_profile.velocity_augmentation": False,
            "typed_profile.homogeneous": "not_applicable_single_source",
            "typed_profile.checkpoint_policy": "dit-half",
        }
    )
    assert validate_flat_config(flat, "smoke", action_velocity_weight=weight)
    with pytest.raises(ValueError, match="resolved profile mismatch"):
        validate_flat_config(flat, "smoke", action_velocity_weight=1.0 - weight)


def verifier_functions():
    # Execute exact pure verifier functions without importing GPU/model dependencies.
    import ast
    import math
    from collections.abc import Mapping, Sequence
    from typing import Any

    source = ast.parse(
        (ROOT / "scripts/train/verify_action_flow_training_smoke.py").read_text()
    )
    names = {
        "SmokeVerificationError",
        "_require",
        "_metric",
        "_complete_row",
        "_validate_history",
        "_validate_checkpoint_loss_schedule",
    }
    constants = {
        "LEGACY_METHOD",
        "GRAPH_METHOD",
        "LIKELIHOOD_METHOD",
        "STOPGRAD_METHOD",
        "STOPGRAD_UNITE_METHOD",
        "SOURCE_LABEL",
    }
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    namespace = dict(
        math=math, Any=Any, Mapping=Mapping, Sequence=Sequence, OmegaConf=OmegaConf
    )
    for node in ast.parse(
        (ROOT / "tools/validate_action_flow_config.py").read_text()
    ).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    namespace[target.id] = ast.literal_eval(node.value)
    for node in source.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    namespace[target.id] = ast.literal_eval(node.value)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(ROOT / "scripts/train/verify_action_flow_training_smoke.py"),
            "exec",
        ),
        namespace,
    )
    fixture = ast.parse(
        (ROOT / "tests/test_verify_action_flow_training_smoke.py").read_text()
    )
    import types

    namespace["MODULE"] = types.SimpleNamespace(**namespace)
    row_node = next(
        node
        for node in fixture.body
        if isinstance(node, ast.FunctionDef) and node.name == "_history_row"
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[row_node], type_ignores=[])),
            "existing_history_fixture",
            "exec",
        ),
        namespace,
    )
    return namespace


def test_disabled_velocity_must_be_zero_weight_in_schedule_and_total():
    v = verifier_functions()
    row = v["_history_row"]()
    row["Train/ActionFlow/Schedule/EffectiveActionVelocityWeight"] = 0.0
    for suffix in ("", "/" + v["SOURCE_LABEL"]):
        row[f"Train/ActionFlow/TotalLoss{suffix}_step"] = 2.5
    row["Valid/ActionFlow/TotalLoss"] = 3.0
    assert (
        v["_validate_history"]({2: row}, action_velocity_weight=0.0)["train_step"] == 2
    )
    bad = dict(row)
    bad["Train/ActionFlow/TotalLoss_step"] = 3.75
    with pytest.raises(v["SmokeVerificationError"], match="weighted total"):
        v["_validate_history"]({2: bad}, action_velocity_weight=0.0)
    bad = dict(row)
    bad["Train/ActionFlow/Schedule/EffectiveActionVelocityWeight"] = 1.0
    with pytest.raises(v["SmokeVerificationError"], match="joint smoke step"):
        v["_validate_history"]({2: bad}, action_velocity_weight=0.0)
    stages = [{} for _ in range(9)]
    stages[8] = {"action_velocity_weight": 0.0}
    cfg = OmegaConf.create(
        {
            "name": AV0_PROFILES["libero10"].name,
            "benchmark": {"suite": "libero10"},
            "data": {
                mode + "_datasets": {
                    "libero_panda": {"resolver": {"suite": "libero10"}}
                }
                for mode in ("train", "valid")
            },
            "run_provenance": {
                "ablation": "action_velocity_off",
                "objective": {"action_velocity_weight": 0.0},
            },
            "model": {
                "reconstruction_only_warmup_steps": 0,
                "pipeline": {"stages": stages},
            },
        }
    )
    schedule = {
        "joint_objective_begins_at_global_step": 0,
        "reconstruction_only_optimizer_steps": 0,
        "joint_flow_weight": 1.0,
        "joint_reconstruction_weight": 1.0,
        "joint_action_velocity_weight": 0.0,
        "schema_version": 1,
    }
    assert (
        v["_validate_checkpoint_loss_schedule"](
            schedule, cfg, reconstruction_weight=1.0, flow_weight=1.0
        )
        == 0
    )
    with pytest.raises(v["SmokeVerificationError"], match="loss schedule"):
        v["_validate_checkpoint_loss_schedule"](
            {**schedule, "joint_action_velocity_weight": 1.0},
            cfg,
            reconstruction_weight=1.0,
            flow_weight=1.0,
        )

    # A nonnative config cannot weaken the historical checkpoint guard by setting stage7=0.
    nonnative = OmegaConf.create(
        {
            "name": "unrelated",
            "model": {
                "reconstruction_only_warmup_steps": 0,
                "pipeline": {"stages": [{}] * 7 + [{"action_velocity_weight": 0.0}]},
            },
        }
    )
    with pytest.raises(
        v["_validate_checkpoint_loss_schedule"].__globals__["SmokeVerificationError"],
        match="loss schedule",
    ):
        v["_validate_checkpoint_loss_schedule"](
            schedule, nonnative, reconstruction_weight=1.0, flow_weight=1.0
        )
    assert (
        v["_validate_checkpoint_loss_schedule"](
            {**schedule, "joint_action_velocity_weight": 1.0},
            nonnative,
            reconstruction_weight=1.0,
            flow_weight=1.0,
        )
        == 0
    )
    baseline = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    baseline.name = PROFILES["libero10"].name
    baseline.run_provenance.objective.action_velocity_weight = 1.0
    baseline.run_provenance.ablation = None
    baseline.model.pipeline.stages[8].action_velocity_weight = 1.0
    with pytest.raises(v["SmokeVerificationError"], match="loss schedule"):
        v["_validate_checkpoint_loss_schedule"](
            schedule, baseline, reconstruction_weight=1.0, flow_weight=1.0
        )
