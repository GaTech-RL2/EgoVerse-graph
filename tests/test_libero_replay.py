import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from egomimic.benchmarks.libero.replay import (
    candidate_id,
    candidates_from_spec,
    demo_task_definition,
    load_calibration_parent,
    rank_candidates,
    read_demo,
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
    assert len(candidates_from_spec(spec)) == 336
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


def test_selection_reports_paired_losses_and_can_require_strict_retention(spec):
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
    assert summary["small"]["lost_raw_successes"] == 1
    assert summary["small"]["gained_successes"] == 1
    assert rank_candidates(summary, candidates, spec) == ["small", "large"]
    spec["minimum_retention"] = 1.0
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
    controls = {
        "raw": {"success_rate": 1},
        "raw_repeat": {"max_state_l2_vs_raw_mean": 0},
        "dense": {"retention": 0.8, "max_action_mse": 1.0e-4},
    }
    with pytest.raises(RuntimeError, match="Dense ARC"):
        validate_controls(controls, spec)


def test_numerical_control_distinguishes_roundoff_from_reset_nondeterminism(spec):
    controls = {
        "raw": {"success_rate": 1},
        "raw_repeat": {"max_state_l2_vs_raw_mean": 0},
        "dense": {"retention": 0.8, "max_action_mse": 1.0e-14},
    }
    validate_controls(controls, spec)
    controls["raw_repeat"]["max_state_l2_vs_raw_mean"] = 1.0e-8
    with pytest.raises(RuntimeError, match="not repeatable"):
        validate_controls(controls, spec)


def test_replay_commands_use_training_precision_and_preserve_source(tmp_path):
    import h5py

    path = tmp_path / "task_demo.hdf5"
    actions = np.full((7, 7), 0.1, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = "task.bddl"
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("states", data=np.zeros((7, 10)))
        demo.attrs["model_file"] = "<mujoco/>"
    demo = read_demo(path, 0)
    assert demo["actions"].dtype == np.float32
    np.testing.assert_array_equal(demo["actions"], actions.astype(np.float32))
    np.testing.assert_array_equal(demo["source_actions"], actions)
    assert not np.array_equal(demo["source_actions"], demo["actions"])


@pytest.mark.parametrize("waypoints", [4, 8, 16, 24, 32, 36])
def test_candidate_support_shapes_allow_policy_forward_and_backward(waypoints):
    import torch

    from egomimic.models.denoising_nets import ConditionalUnet1D

    model = ConditionalUnet1D(
        input_dim=11,
        cond_dim=276,
        ac_latent_seq=1,
        diffusion_step_embed_dim=16,
        down_dims=[8, 16, 32],
    )
    inputs = torch.randn(2, waypoints, 11, requires_grad=True)
    output = model(inputs, torch.tensor([1, 3]), torch.randn(2, 276))
    assert output.shape == inputs.shape
    output.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()


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


def test_legacy_task_binding_preserves_goal_and_saved_physics():
    bddl = """(:objects new_salad_dressing_1 - new_salad_dressing)
    (:goal (And (In new_salad_dressing_1 wooden_tray_1_contain_region)))
    ; region: table_new_salad_dressing_init_region
    """
    xml = '<mujoco><worldbody><body name="salad_dressing_1_main" pos="0 0 0"><geom size="0.1"/></body></worldbody></mujoco>'
    updated, aliases = demo_task_definition(bddl, xml)
    assert updated == bddl.replace("new_salad_dressing_1", "salad_dressing_1").replace(
        "- new_salad_dressing)", "- salad_dressing)"
    )
    assert aliases == {
        "new_salad_dressing_1": "salad_dressing_1",
        "new_salad_dressing": "salad_dressing",
    }
    # Current recordings retain their original task definition unchanged.
    current_xml = xml.replace("salad_dressing_1_main", "new_salad_dressing_1_main")
    assert demo_task_definition(bddl, current_xml) == (bddl, {})
    assert demo_task_definition(bddl, "<mujoco/>") == (bddl, {})


def test_candidate_names_distinguish_rotation_horizon_and_metric_radius(spec):
    candidate = dict(num_waypoints=16, max_translation=0.1, max_rotation_degrees=24)
    assert candidate_id(candidate) == "R24_D0.1_M16"
    bad = copy.deepcopy(spec)
    bad["waypoints"] = [33]
    with pytest.raises(ValueError, match="multiple of four"):
        candidates_from_spec(bad)


def test_success_first_prefers_more_successes_before_fewer_tokens(spec):
    summary = {
        "small": {
            "retention": 1,
            "successes": 10,
            "success_rate": 0.5,
            "raw_successes": 10,
            "episodes": 20,
            "action_mse": 0.01,
        }
    }
    summary["large"] = {**summary["small"], "successes": 11, "success_rate": 0.55}
    candidates = {"small": {"num_waypoints": 8}, "large": {"num_waypoints": 16}}
    assert rank_candidates(summary, candidates, spec) == ["small", "large"]
    spec["selection_objective"] = "success_then_tokens"
    assert rank_candidates(summary, candidates, spec) == ["large", "small"]


@pytest.mark.parametrize(
    "invalid", [None, "suite", "codec", "dependency", "precision", "incomplete"]
)
def test_refinement_reuses_only_compatible_calibration(
    tmp_path, monkeypatch, spec, invalid
):
    import importlib.metadata
    import io
    import json

    from egomimic.benchmarks.libero.cluster import DATA_REVISION, digest

    root = Path(__file__).parents[1]
    candidate = dict(num_waypoints=16, max_translation=None, max_rotation_degrees=None)
    key = candidate_id(candidate)
    summary = {
        "raw": {"success_rate": 1, "episodes": 20},
        "raw_repeat": {"max_state_l2_vs_raw_mean": 0, "episodes": 20},
        "dense": {"max_action_mse": 0, "episodes": 20},
        key: {"episodes": 19 if invalid == "incomplete" else 20},
    }
    versions = {
        name: "1.0.0" for name in ("numpy", "scipy", "mujoco", "robosuite", "libero")
    }
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "2.0.0" if invalid == "dependency" else "1.0.0",
    )
    parent_spec = copy.deepcopy(spec)
    if invalid == "precision":
        parent_spec["action_dtype"] = "float64"
    values = {
        "runtime.json": {"suite": "libero_goal" if invalid == "suite" else "libero_10"},
        "spec.json": parent_spec,
        "environment-audit.json": {
            "codec_sha256": "wrong"
            if invalid == "codec"
            else digest(root / "egomimic/rldb/zarr/libero_arc.py"),
            "versions": versions,
        },
        "data-source.json": {"revision": DATA_REVISION},
        "screen.json": {
            "candidates": {key: candidate},
            "execution_coverage": {key: 1.0},
        },
        "calibration.json": summary,
    }
    requested = []

    class Storage:
        def get_object(self, Bucket, Key):
            name = Key.rsplit("/", 1)[1]
            requested.append(name)
            return {"Body": io.BytesIO(json.dumps(values[name]).encode())}

    if invalid:
        with pytest.raises(ValueError):
            load_calibration_parent(
                Storage(), "parent-run", "libero_10", spec, tmp_path
            )
    else:
        eligible, recovered = load_calibration_parent(
            Storage(), "parent-run", "libero_10", spec, tmp_path
        )
        assert eligible == {key: candidate} and recovered == summary
        assert json.loads((tmp_path / "calibration-parent.json").read_text())[
            "reused_only_calibration"
        ]
    assert not {"selection.json", "confirmation.json", "result.json"} & set(requested)
