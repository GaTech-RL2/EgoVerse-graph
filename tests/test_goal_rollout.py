"""Matched reset RNG and early termination of decoded native chunks."""
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from egomimic.eval.goal_rollout import evaluate_goals, evaluation_action_space


class GoalFromActionSpace:
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.unwrapped = self
        self.task_infos = [None, None]
        self.data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2))

    def reset(self, seed, options):
        self.steps = 0
        self.data.qpos[:] = np.random.default_rng(seed).normal(size=2)
        # Models the separate Gym space RNG used by Cube's reset settling.
        self.goal = self.action_space.sample()
        return self.data.qpos.copy(), {"goal": self.goal}

    def step(self, action):
        self.steps += 1
        return self.data.qpos.copy(), 0., self.steps == 2, False, {"success": True}


class FixedChunk:
    def __init__(self):
        self.nets = torch.nn.Linear(2, 2)
        self.device = "cpu"

    def forward_eval(self, batch):
        return {"ogbench": {"pred_action": torch.zeros(1, 5, 2), "action_lengths": torch.tensor([5])}}


class FreshActionSpace(GoalFromActionSpace):
    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    @action_space.setter
    def action_space(self, value):
        pass


@pytest.mark.parametrize("environment", [GoalFromActionSpace, FreshActionSpace])
def test_goal_reset_spaces_are_seeded_and_native_execution_stops_at_done(tmp_path, environment):
    records = []
    for label in ["first", "second"]:
        env, policy = environment(), FixedChunk()
        # Distinct prior draws cannot change the canonical evaluation reset.
        env.action_space.seed(18 if label == "first" else 53)
        env.action_space.sample()
        output = tmp_path / label
        scores = evaluate_goals(policy, env, output, episodes=3, seed_start=700)
        assert scores["episodes"] == 6 and scores["success"] == 1
        records.append([json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()])
        assert policy.nets.training
        assert type(env) is environment
        for row in records[-1]:
            assert row["native_steps"] == 2 and row["replans"] == 1
    assert records[0] == records[1]


@pytest.mark.parametrize("domain", ["cube-triple", "cube-quadruple"])
def test_real_cube_reset_goal_is_identical_across_environment_instances(domain):
    ogbench = pytest.importorskip("ogbench")
    states = []
    for previous_seed in [4, 19]:
        env = ogbench.make_env_and_datasets(domain + "-play-oraclerep-v0", env_only=True)
        original_class = type(env.unwrapped)
        try:
            with evaluation_action_space(env) as space:
                space.seed(previous_seed)
                space.sample()
                np.random.seed(2050000)
                space.seed(2050000)
                _, info = env.reset(seed=2050000, options={"task_id": 5, "render_goal": False})
                physics = env.unwrapped._data
                states.append((physics.qpos.copy(), physics.qvel.copy(), info["goal"].copy()))
            assert type(env.unwrapped) is original_class
        finally:
            env.close()
    for left, right in zip(states[0], states[1]):
        np.testing.assert_array_equal(left, right)
