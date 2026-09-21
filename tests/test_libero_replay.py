import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from egomimic.benchmarks.libero.replay import (
    candidate_id,
    candidates_from_spec,
    rank_candidates,
    rebase_demo_xml,
    reconstruct_episode,
    summarize,
    validate_controls,
    validate_spec,
)


@pytest.fixture
def spec():
    path = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/benchmark/libero_arc_replay.yaml"
    )
    return yaml.safe_load(path.read_text())


def test_splits_are_disjoint_and_all_candidates_match_policy_shape(spec):
    validate_spec(spec)
    assert len(candidates_from_spec(spec)) == 125
    spec["confirmation_demos"].append(spec["calibration_demos"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_spec(spec)


def test_replay_preserves_episode_length_and_dense_commands(spec):
    actions = np.random.default_rng(3).uniform(-0.2, 0.2, (73, 7))
    actions[:, 6] = -1
    actions[24:, 6] = 1
    candidate = dict(num_waypoints=36, max_translation=None, max_rotation_degrees=None)
    decoded, metrics = reconstruct_episode(actions, candidate, spec)
    np.testing.assert_allclose(decoded, actions, atol=3e-6)
    assert decoded.shape == actions.shape
    assert metrics["execution_coverage"] == 1
    assert metrics["waypoints_per_executed_action"] == pytest.approx(5 * 36 / 73)


def test_short_horizon_cannot_cheat_by_replanning_early(spec):
    actions = np.zeros((32, 7))
    actions[:, 0] = 0.2
    candidate = dict(num_waypoints=16, max_translation=0.02, max_rotation_degrees=None)
    decoded, metrics = reconstruct_episode(actions, candidate, spec)
    # 2 represented commands in each unchanged 16-command execution interval.
    assert metrics["execution_coverage"] == pytest.approx(4 / 32)
    np.testing.assert_allclose(decoded[[0, 1, 16, 17], 0], 0.2, atol=1e-6)
    np.testing.assert_allclose(decoded[2:16, 0], 0, atol=1e-6)


def test_selection_uses_paired_success_not_equal_aggregate_counts(spec):
    candidates = {
        "small": {"num_waypoints": 8},
        "large": {"num_waypoints": 16},
    }
    base = {"eef_rmse_vs_raw_metres": 0, "action_mse": 0}
    rows = [
        {
            "candidates": {
                key: {**base, "success": success} for key, success in outcome.items()
            }
        }
        for outcome in [
            {"raw": True, "small": False, "large": True},
            {"raw": False, "small": True, "large": False},
        ]
    ]
    summary = summarize(rows, ["raw", *candidates])
    assert summary["small"]["success_rate"] == summary["large"]["success_rate"]
    assert rank_candidates(summary, candidates, spec) == ["large"]
    with pytest.raises(RuntimeError, match="Raw demonstration"):
        validate_controls(summary, spec)


def test_fidelity_is_secondary_to_retention_and_token_rate(spec):
    summary = {
        "small": {
            "retention": 1,
            "successes": 10,
            "success_rate": 1,
            "raw_successes": 10,
            "episodes": 10,
            "action_mse": 0.01,
        }
    }
    summary["large"] = {**summary["small"], "action_mse": 0.001}
    assert rank_candidates(
        summary, {"small": {"num_waypoints": 8}, "large": {"num_waypoints": 16}}, spec
    ) == ["small", "large"]
    controls = {"raw": {"success_rate": 1}, "dense": {"retention": 0.8}}
    with pytest.raises(RuntimeError, match="Dense ARC"):
        validate_controls(controls, spec)


def test_model_relocation_changes_only_asset_paths(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "texture.png").write_bytes(b"texture")
    suite = tmp_path / "robosuite"
    suite.mkdir()
    (suite / "robot.stl").write_bytes(b"mesh")
    xml = '<mujoco><option timestep="0.002"/><asset><texture file="/old/chiliocosm/assets/texture.png"/><mesh file="/old/robosuite/robot.stl"/></asset></mujoco>'
    updated = rebase_demo_xml(xml, suite, assets)
    assert 'timestep="0.002"' in updated
    assert str(assets / "texture.png") in updated
    assert str(suite / "robot.stl") in updated
    with pytest.raises(FileNotFoundError):
        rebase_demo_xml(xml.replace("robot.stl", "missing.stl"), suite, assets)


def test_candidate_names_distinguish_rotation_horizon_and_metric_radius(spec):
    candidate = dict(num_waypoints=16, max_translation=0.1, max_rotation_degrees=24)
    assert candidate_id(candidate) == "R24_D0.1_M16"
    bad = copy.deepcopy(spec)
    bad["waypoints"] = [33]
    with pytest.raises(ValueError, match="multiple of four"):
        candidates_from_spec(bad)
