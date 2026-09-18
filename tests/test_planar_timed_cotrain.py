"""Timed co-training must preserve causal anchors, clocks and both domains."""
import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.pushshapes import PlanarArcTimedNativeDecoder
from egomimic.eval.planar_rollout import PlanarTimedArcExecutionSelector
from egomimic.rldb.embodiment.pushshapes import get_planar_arc_timed_transform_list
from egomimic.rldb.zarr.planar_arc import TokenizePlanarArcTimed

ROOT = Path(__file__).resolve().parents[1]


def roundtrip(actions, mode="duration", waypoints=16):
    token = TokenizePlanarArcTimed(min_distance_unit=80, resampled_vector_length=waypoints,
                                  timing_mode=mode).tokenize(actions)
    decoded = PlanarArcTimedNativeDecoder(waypoints, len(actions), actions.shape[1], mode)(token)
    return token, decoded


@pytest.mark.parametrize("mode", ["duration", "velocity"])
@pytest.mark.parametrize("m", [16, 56])
def test_linear_timed_codec_retains_causal_anchor_and_grip(mode, m):
    t = np.arange(80) / 30
    actions = np.column_stack((20 + 20*t, 30 + 10*t, .15*t, .1 + .1*t))
    _, decoded = roundtrip(actions, mode, m)
    np.testing.assert_allclose(decoded[:, :2], actions[:, :2], atol=1e-10)
    np.testing.assert_allclose(decoded[:, 3], actions[:, 3], atol=1e-10)
    np.testing.assert_allclose(decoded[:, 2], actions[:, 2], atol=1e-6)


def test_duration_holds_and_stationary_grip_change():
    actions = np.zeros((16, 4))
    actions[:, :2] = [30, 40]
    actions[:, 3] = np.linspace(0, 1, 16)
    token, decoded = roundtrip(actions)
    assert np.all(token[:-1, 2] > 0)
    np.testing.assert_allclose(decoded, actions, atol=1e-12)


def test_duration_motion_wait_then_motion():
    actions = np.zeros((31, 4))
    actions[:, 0] = np.r_[np.arange(11), np.repeat(10., 10), np.arange(11., 21.)]
    _, decoded = roundtrip(actions, waypoints=16)
    np.testing.assert_allclose(decoded[10:21, 0], 10., atol=1e-10)
    np.testing.assert_allclose(decoded[:, 0], actions[:, 0], atol=1e-10)


@pytest.mark.parametrize("mode", ["duration", "velocity"])
def test_alignment_slices_past_action_before_tokenization(mode):
    raw = np.column_stack((np.arange(81), np.zeros(81), np.zeros(81), np.ones(81)))
    batch = {"actions": raw.copy()}
    for transform in get_planar_arc_timed_transform_list(raw_action_horizon=80,
            action_target_offset=1, resampled_vector_length=16, timing_mode=mode):
        batch = transform.transform(batch)
    decoded = PlanarArcTimedNativeDecoder(16, 80, 4, mode)(batch["actions"])
    np.testing.assert_allclose(decoded[0], raw[1])


@pytest.mark.parametrize("mode", ["duration", "velocity"])
def test_bad_predicted_timing_cannot_teleport_geometry(mode):
    tokens = np.zeros((16, 7)); tokens[:, 0] = np.arange(16); tokens[:, 3] = 1
    decoded = PlanarArcTimedNativeDecoder(16, 80, 4, mode)(tokens)
    np.testing.assert_allclose(decoded[0], [0, 0, 0, 0])
    assert decoded[1, 0] < .02


@pytest.mark.parametrize("rotation_active,expected", [(False, 22), (True, 7)])
def test_half_support_selection_uses_earliest_active_clock(rotation_active, expected):
    tokens = np.zeros((16, 7))
    tokens[:, 0] = np.arange(16)
    tokens[:, 2] = .1
    angle = np.arange(16) * (.01 if rotation_active else 0)
    tokens[:, 3], tokens[:, 4] = np.cos(angle), np.sin(angle)
    tokens[:, 5] = .03
    result = PlanarTimedArcExecutionSelector().select(
        tokens, PlanarArcTimedNativeDecoder(16, 80, 4, "duration"))
    assert result["selected_waypoint_count"] == 8
    assert result["execution_steps"] == expected
    assert result["execution_start_index"] == 0


def test_half_support_stationary_grip_uses_translation_duration():
    actions = np.zeros((80, 4)); actions[:, 3] = np.linspace(0, 1, 80)
    tokens, _ = roundtrip(actions)
    result = PlanarTimedArcExecutionSelector().select(
        tokens, PlanarArcTimedNativeDecoder(16, 80, 4, "duration"))
    assert result["clocks"]["translation_grip"]["active"]
    assert not result["clocks"]["rotation"]["active"]
    assert result["execution_steps"] == 37


@pytest.mark.parametrize("recipe", ["paper_dp"] + [f"arc_{mode}_D80_M{m}_R26deg_paper"
        for mode in ["duration", "stacked"] for m in [16, 56]])
def test_five_cotrain_recipes_use_same_data_and_network(recipe):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")):
        cfg = compose(config_name="train_zarr_cartesian", overrides=[
            f"+experiment=pusht/planar_v2_cotrain_obstacle_{recipe}"])
    assert cfg.planar.observation_horizon == 2 and cfg.planar.action_target_offset == 1
    assert cfg.planar.arc_waypoint_sampling == "uniform"
    assert cfg.trainer.limit_train_batches == 1.0
    assert cfg.trainer.max_steps == cfg.model.scheduler.max_steps == 240000
    assert cfg.ckpt_path is None
    assert cfg.run_provenance.action_contract.rollout_action_chunk_start_index == 0
    assert cfg.planar.batch_size * cfg.launch_params.gpus_per_node * len(cfg.data.train_datasets) == 128
    assert set(cfg.data.train_datasets) == {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
    assert cfg.run_provenance.dataset_count == 7919
    assert cfg.eval_checkpoint.use_ema is True
    stage = cfg.model.pipeline.stages[3]
    assert list(stage.policy.model.down_dims) == [512, 1024, 2048]
    assert stage.action_dim == (5 if recipe == "paper_dp" else 7)
    if recipe != "paper_dp":
        assert cfg.planar.raw_action_horizon == 80
        for spec in cfg.evaluator.native_decoders.values():
            decoder = instantiate(spec)
            assert decoder.action_horizon == 80
    manifest = json.loads((ROOT / cfg.run_provenance.split_manifest_path).read_text())
    for domain, item in manifest["domains"].items():
        assert not set(item["train_ids"]) & set(item["valid_ids"])
        assert cfg.data.train_datasets[domain].resolver.expected_episode_count == item["count"]
