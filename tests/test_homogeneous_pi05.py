from copy import deepcopy
from types import MethodType

import numpy as np
import torch
from hydra.utils import instantiate

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from tests.test_pi05_graph import batch, config, normalizer
from tests.test_pi05_graph import tiny_openpi as tiny_openpi


def test_pi05_batches_normalized_embodiments_and_duplicate_source_domains(tiny_openpi):
    cfg = config(["model=pi05/pi0.5_cotrain_eva_aria_6d"])
    graph = instantiate(cfg.model.pipeline)
    norm = normalizer()
    human = get_embodiment_id("human_bimanual")
    norm.embodiments.add(human)
    norm.key_types[human] = dict(norm.key_types[6])
    norm.zarr_keys[human] = dict(norm.zarr_keys[6])
    norm.shapes[human] = {
        "actions_cartesian": (100, 18),
        "observations.state.ee_pose": (18,),
    }
    norm.norm_stats[human] = {
        key: {"quantile_1": np.zeros(shape), "quantile_99": np.full(shape, 4.0)}
        for key, shape in norm.shapes[human].items()
    }
    graph.bind_data_context(normalizer=norm)
    policy = graph.pipeline.stages[0].backend.nets["policy"]
    calls = []

    def unreduced(self, observation, action):
        calls.append((len(action), observation.state.shape, observation.image_masks))
        return (action - self.weight).square()  # OpenPI's real loss is per sample.

    policy.forward = MethodType(unreduced, policy)
    robot = batch()["opaque-input"]
    human_batch = dict(
        robot,
        embodiment=torch.tensor([human] * 3),
        base_0_rgb=torch.rand(3, 3, 32, 32),
        annotations=[["human"]] * 3,
    )
    human_batch["observations.state.ee_pose"] = torch.zeros(3, 18)
    human_batch["actions_cartesian"] = torch.ones(3, 100, 18) * 0.1
    values = {"robot-a": robot, "human": human_batch, "robot-b": deepcopy(robot)}
    grouped = graph.forward_training(values)
    assert calls[0][0] == 7 and calls[0][1] == (7, 32) and len(calls) == 1
    grouped_loss = graph.compute_losses(grouped, values)["loss"]
    grouped_loss.backward()
    gradient = policy.weight.grad.clone()
    policy.weight.grad = None
    calls.clear()
    graph.homogeneous_training = False
    reference = graph.forward_training(values)
    assert [call[0] for call in calls] == [2, 3, 2]
    loss = graph.compute_losses(reference, values)["loss"]
    loss.backward()
    torch.testing.assert_close(loss, grouped_loss)
    torch.testing.assert_close(gradient, policy.weight.grad)
    for source in values:
        torch.testing.assert_close(
            grouped[source]["loss/pi05"], reference[source]["loss/pi05"]
        )
