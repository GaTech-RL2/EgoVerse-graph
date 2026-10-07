"""Failed model replacement clears stale history before the current policy resumes."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from egomimic.robot.interface import ARM_OFFSET
from egomimic.robot.rollout import run_rollout


class HistoryPolicy:
    action_type = "joints"

    def __init__(self):
        self.history = []
        self.predict_histories = []
        self.resets = 0

    def reset(self):
        self.history.clear()
        self.resets += 1

    def predict(self, obs):
        self.predict_histories.append(tuple(self.history))
        self.history.append(obs["frame"])
        plan = np.zeros((2, 14))
        plan[:, [0, 7]] = 0.01
        plan[:, [6, 13]] = 0.5
        return plan


class Robot:
    arms = ("left", "right")
    camera_res = {}

    def __init__(self):
        self.frame = 0
        self.q = np.zeros(14)
        self.q[[6, 13]] = 0.5

    def get_obs(self):
        self.frame += 1
        return {"joint_positions": self.q.copy(), "frame": self.frame}

    def set_joints(self, command, arm):
        offset = ARM_OFFSET[arm]
        self.q[offset : offset + 7] = command


class View:
    def __init__(self, bundle):
        self.keys = iter([None, None, None, "c", "q"])
        self.bundles = iter([None, bundle, None, None, None])
        self.statuses = []

    def update(self, obs):
        return next(self.keys)

    def take_model_selection_request(self):
        return next(self.bundles, None)

    def set_status(self, status):
        self.statuses.append(status)

    def close(self):
        pass


@pytest.mark.parametrize("failure", ["load_error", "wrong_action_representation"])
def test_failed_model_selection_clears_old_policy_history_before_resume(
    monkeypatch, failure
):
    policy = HistoryPolicy()
    bundle = SimpleNamespace(
        checkpoint=Path("/new.ckpt"),
        training_config=Path("/new.yaml"),
        normalizer_path=Path("/new-normalizer.pt"),
        inference_config=None,
    )
    view = View(bundle)
    observations_at_load = []

    def replace(config):
        observations_at_load.append(tuple(policy.history))
        if failure == "load_error":
            raise ValueError("Invalid checkpoint contract")
        return SimpleNamespace(action_type="cartesian")

    monkeypatch.setattr("egomimic.robot.rollout.load_policy", replace)
    monkeypatch.setattr("egomimic.robot.rollout.time.sleep", lambda _: None)
    steps = run_rollout(
        Robot(),
        policy,
        {
            "frequency": 30,
            "max_steps": 20,
            "max_joint_velocity": 1,
            "preview": {"enabled": False},
            "policy": {"kind": "graph", "checkpoint": "/old.ckpt"},
        },
        view=view,
    )

    assert steps == 2
    assert observations_at_load == [()]  # cleared before potentially slow/failed load
    assert policy.resets == 2  # initial reset and model-change reset
    assert policy.predict_histories == [(), ()]  # resumed old policy sees fresh history
    assert any("Model load failed" in status for status in view.statuses)
