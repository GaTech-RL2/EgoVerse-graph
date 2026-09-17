"""Control-rate history, normalization and native execution regressions."""

import hashlib

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.eval.planar_rollout import (
    PlanarActionQueue,
    PlanarGraphPolicy,
    PlanarArcExecutionSelector,
    load_planar_graph_policy,
)
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.pushshapes import PlanarArcTrajectoryNativeDecoder
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.rldb.zarr.planar_arc import lambda_for_radius, planar_step_distance

DOMAIN = "pushshapes_sim_chain_gripper"


class EchoPlanarStage(Stage):
    reads = ("state_agent_obj",)
    writes = ("pred_action",)

    def __init__(self):
        super().__init__()
        self.output = torch.nn.Parameter(torch.zeros(32, 5))
        self.seen = []

    def forward(self, batch):
        self.seen.append(batch["state_agent_obj"].detach().clone())
        batch["pred_action"] = self.output.unsqueeze(0)
        return batch


def normalizer():
    mean = np.zeros((32, 5))
    mean[:16] = [50, 60, 1, 0, 0.6]
    mean[17:, 0] = 1 / 30
    return MultiDataset.from_state(
        {
            "norm_mode": "zscore",
            "embodiments": [20],
            "key_types": {
                20: {"state_agent_obj": "proprio_keys", "actions": "action_keys"}
            },
            "zarr_keys": {
                20: {"state_agent_obj": "state_agent_obj", "actions": "actions"}
            },
            "shapes": {20: {"state_agent_obj": [2, 6], "actions": [32, 5]}},
            "norm_stats": {
                20: {
                    "state_agent_obj": {
                        "mean": np.full(6, 2.0),
                        "std": np.full(6, 2.0),
                    },
                    "actions": {"mean": mean, "std": np.ones((32, 5))},
                }
            },
        }
    )


def observation(step):
    return {
        "agent_pos": np.array([step, 0.0]),
        "agent_angle": np.array([0.0]),
        "object_pose": np.zeros(3),
        "image": np.full((4, 4, 3), step, dtype=np.uint8),
    }


def policy():
    stage = EchoPlanarStage()
    graph = PipelineAlgo([stage], device="cpu")
    decoder = PlanarArcTrajectoryNativeDecoder(16, 4, 40)
    return PlanarGraphPolicy(
        graph=graph,
        normalizer=normalizer(),
        decoder=decoder,
        embodiment_name=DOMAIN,
        observation_horizon=2,
        token_shape=(32, 5),
        native_shape=(40, 4),
    ), stage


@pytest.mark.parametrize("execution_horizon", [8, 20, 40])
def test_replans_use_adjacent_control_frames_and_unnormalize_once(execution_horizon):
    value, stage = policy()
    queue = PlanarActionQueue(value, execution_horizon=execution_horizon)
    queue.reset(observation(0))
    for step in range(execution_horizon + 1):
        action = queue.next_action()
        np.testing.assert_allclose(action, [50, 60, 0, 0.6], atol=1e-5)
        queue.observe(observation(step + 1))
    assert len(stage.seen) == 2
    np.testing.assert_allclose(stage.seen[0][0, :, 0], [-1, -1], atol=1e-6)
    expected = (np.array([execution_horizon - 1, execution_horizon]) - 2) / 2
    np.testing.assert_allclose(stage.seen[1][0, :, 0], expected, atol=1e-5)


def test_observation_gaps_duplicates_and_missing_execution_feedback_are_rejected():
    value, _ = policy()
    with pytest.raises(RuntimeError, match="reset state"):
        value.predict_native_actions()
    value.observe(observation(0), step=0)
    for step in (0, 2, 40):
        with pytest.raises(ValueError, match="consecutive"):
            value.observe(observation(step), step=step)
    queue = PlanarActionQueue(value, execution_horizon=40)
    queue.reset(observation(0))
    queue.next_action()
    with pytest.raises(RuntimeError, match="last executed action"):
        queue.next_action()
    queue.observe(observation(1))
    with pytest.raises(RuntimeError, match="No executed action"):
        queue.observe(observation(2))
    queue.reset(observation(0))
    np.testing.assert_allclose(queue.next_action(), [50, 60, 0, 0.6], atol=1e-5)


@pytest.mark.parametrize("execution_horizon", [20, 40])
def test_native_queue_starts_at_zero_and_does_not_reuse_old_tail(execution_horizon):
    class IndexedPolicy:
        native_shape = (40, 4)

        def reset(self):
            self.prediction = 0
            self.observed = []

        def observe(self, observation, *, step):
            self.observed.append(step)

        def predict_native_actions(self):
            self.prediction += 1
            return np.repeat(
                (100 * self.prediction + np.arange(40))[:, None], 4, axis=1
            )

    value = IndexedPolicy()
    queue = PlanarActionQueue(value, execution_horizon=execution_horizon)
    queue.reset({})
    executed = []
    for _ in range(2 * execution_horizon + 1):
        executed.append(queue.next_action()[0])
        queue.observe({})
    assert executed == (
        list(range(100, 100 + execution_horizon))
        + list(range(200, 200 + execution_horizon))
        + [300]
    )
    assert value.observed == list(range(2 * execution_horizon + 2))


def arc_token(x, durations, theta=None, grip=None):
    count = len(x)
    token = torch.zeros((count * 2, 5), dtype=torch.float64)
    token[:count, 0] = torch.as_tensor(x)
    angles = torch.zeros(count) if theta is None else torch.as_tensor(theta)
    token[:count, 2] = torch.cos(angles)
    token[:count, 3] = torch.sin(angles)
    if grip is not None:
        token[:count, 4] = torch.as_tensor(grip)
    token[count:-1, 0] = torch.as_tensor(durations)
    token[-1, 0] = 999  # Padding must never create an extra interval.
    return token


def test_distance_prefix_counts_supports_then_uses_their_nonuniform_durations():
    decoder = PlanarArcTrajectoryNativeDecoder(6, 4, 40, dt=0.1)
    token = arc_token([0, 3, 7, 19, 21, 40], [0.1, 0.2, 0.3, 0.4, 0.5])
    distance = PlanarArcExecutionSelector(mode="distance_fraction", distance_budget=40)
    waypoints = PlanarArcExecutionSelector(mode="waypoint_fraction")
    selected = distance.select(token, decoder)
    assert selected["selected_waypoint_count"] == 4
    assert selected["selected_distance"] == 19
    assert selected["execution_steps"] == 7  # t=0 ... 0.6 inclusive.
    assert waypoints.select(token, decoder)["execution_steps"] == 4
    # Same M, D and native horizon: different waypoint spacing changes count.
    token[:6, 0] = torch.tensor([0, 3, 7, 19, 20, 40])
    selected = distance.select(token, decoder)
    assert selected["selected_waypoint_count"] == 5
    assert selected["execution_steps"] == 11
    # Half the configured D, not half this shortened prediction's total length.
    token[:6, 0] = torch.tensor([0, 1, 2, 3, 4, 10])
    assert distance.select(token, decoder)["selected_waypoint_count"] == 6


def test_distance_prefix_uses_the_tokenizer_rotation_metric_across_angle_seam():
    decoder = PlanarArcTrajectoryNativeDecoder(4, 4, 40, rotation_radius=3)
    token = arc_token([0, 1, 2, 3], [0.1, 0.1, 0.1], [3.0, -3.0, -2.5, -2.0])
    distances, _ = decoder.waypoint_schedule(token)
    theta = np.arctan2(token[:4, 3].numpy(), token[:4, 2].numpy())
    expected = np.r_[0, np.cumsum(planar_step_distance(
        token[:4, :2].numpy(), theta, lambda_for_radius(3)
    ))]
    np.testing.assert_allclose(distances.numpy(), expected, atol=1e-7)
    selector = PlanarArcExecutionSelector(mode="distance_fraction", distance_budget=3)
    assert selector.select(token, decoder)["selected_waypoint_count"] == 1


def test_distance_boundary_holds_and_zero_distance_grip_use_duration_clock():
    decoder = PlanarArcTrajectoryNativeDecoder(4, 4, 40, dt=0.1)
    selector = PlanarArcExecutionSelector(mode="distance_fraction", distance_budget=40)
    token = arc_token([0, 20, 20, 30], [0.1, 0.3, 0.1], grip=[0, 0, 1, 1])
    selected = selector.select(token, decoder)
    assert selected["selected_waypoint_count"] == 3
    assert selected["execution_steps"] == 5
    assert decoder.decode(token)[0, 4, 3] == pytest.approx(1)
    token[:4, 0] = 0
    selected = selector.select(token, decoder)
    assert selected["selected_waypoint_count"] == 4
    assert selected["execution_steps"] == 6


def test_waypoint_time_prefix_rounds_down_and_caps_at_the_native_horizon():
    decoder = PlanarArcTrajectoryNativeDecoder(4, 4, 40, dt=0.1)
    selector = PlanarArcExecutionSelector(mode="waypoint_fraction")
    token = arc_token([0, 1, 2, 3], [0.25, 0.1, 0.1])
    assert selector.select(token, decoder)["execution_steps"] == 3
    token[4, 0] = 0  # Moving without duration uses the decoder's stalled rule.
    selected = selector.select(token, decoder)
    assert selected["execution_steps"] == 40
    assert selected["native_horizon_clipped"]
    assert selected["selected_endpoint_seconds"] == pytest.approx(4.1)


def test_variable_prefix_replans_with_adjacent_history_and_discards_old_tail():
    value, stage = policy()
    # Install physical predictions via the normalizer's inverse. Predict six
    # supports within D/2 initially, then only two within D/2 at the next replan.
    selector = PlanarArcExecutionSelector(mode="distance_fraction", distance_budget=42)
    queue = PlanarActionQueue(value, execution_selector=selector)
    queue.reset(observation(0))
    token = arc_token(np.arange(16) * 4, [1 / 30] * 15)
    with torch.no_grad():
        stage.output.copy_(value.normalizer.normalize({"actions": token}, 20)["actions"])
    actual = []
    for step in range(6):
        actual.append(queue.next_action()[0])
        queue.observe(observation(step + 1))
    assert queue.last_execution["selected_waypoint_count"] == 6
    assert queue.last_execution["execution_steps"] == 6
    token[:16, 0] = torch.arange(16) * 11 + 100
    with torch.no_grad():
        stage.output.copy_(value.normalizer.normalize({"actions": token}, 20)["actions"])
    for step in range(6, 9):
        actual.append(queue.next_action()[0])
        queue.observe(observation(step + 1))
    np.testing.assert_allclose(actual, [0, 4, 8, 12, 16, 20, 100, 111, 100], atol=1e-4)
    assert queue.prediction_count == 3
    assert queue.last_execution["replan_control_step"] == 8
    np.testing.assert_allclose(stage.seen[1][0, :, 0], [1.5, 2], atol=1e-5)
    np.testing.assert_allclose(stage.seen[2][0, :, 0], [2.5, 3], atol=1e-5)


def test_load_restores_exact_current_checkpoint_and_exported_normalizer(tmp_path):
    value, _ = policy()
    normalizer().cache_stats(str(tmp_path))
    cfg = OmegaConf.create(
        {
            "model": {
                "pipeline": {
                    "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                    "stages": [
                        {"_target_": "tests.test_planar_rollout.EchoPlanarStage"}
                    ],
                }
            },
            "planar": {
                "observation_horizon": 2,
                "action_target_offset": 1,
                "action_horizon": 32,
                "action_dims": {DOMAIN: 5},
                "eval_native_decoder": {
                    "_target_": "egomimic.pipeline.pushshapes.PlanarArcTrajectoryNativeDecoder",
                    "resampled_vector_length": 16,
                    "native_action_dim": 4,
                    "raw_action_horizon": 40,
                },
            },
            "data": {"train_datasets": {DOMAIN: {}}},
            "norm_stats": {"norm_mode": "zscore"},
            "run_provenance": {
                "action_contract": {"rollout_action_chunk_start_index": 0}
            },
        }
    )
    config = tmp_path / "training.yaml"
    OmegaConf.save(cfg, config)
    checkpoint = tmp_path / "fresh.ckpt"
    torch.save(
        {
            "state_dict": {
                f"nets.{key}": tensor
                for key, tensor in value.graph.nets.state_dict().items()
            },
            "hyper_parameters": {
                "config_tree": {
                    "model": cfg.model,
                    "run_provenance": cfg.run_provenance,
                }
            },
        },
        checkpoint,
    )
    norm = tmp_path / "norm_stats/norm_stats.json"
    options = {
        "checkpoint_path": checkpoint,
        "config_path": config,
        "normalizer_path": norm,
        "embodiment_name": DOMAIN,
        "device": "cpu",
        "use_ema": False,
    }
    for key, path in (
        ("checkpoint", checkpoint),
        ("config", config),
        ("normalizer", norm),
    ):
        options[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = load_planar_graph_policy(**options)
    restored.observe(observation(0), step=0)
    assert restored.predict_native_actions().shape == (40, 4)
    with pytest.raises(KeyError, match="EMA"):
        load_planar_graph_policy(**{**options, "use_ema": True})
    with pytest.raises(ValueError, match="SHA-256"):
        load_planar_graph_policy(**{**options, "normalizer_sha256": "0" * 64})
    cfg.model.pipeline.stages = []
    OmegaConf.save(cfg, config)
    options["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Checkpoint model differs"):
        load_planar_graph_policy(**options)
