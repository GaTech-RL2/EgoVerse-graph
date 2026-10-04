"""The paired native4 validation distance must respect the theta seam."""

import math

import pytest
import torch

from egomimic.eval.energy_score import (
    CHAIN_NATIVE4_ENERGY_DISTANCE_CONFIG,
    chain_native4_chunk_distance,
    normalize_chain_native4_energy_distance_config,
)
from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.pipeline.pushshapes import (
    ChainGripperNative4Decoder,
    ChainGripperPointsNativeDecoder,
)


def test_native4_distance_covers_grip_and_wraps_theta():
    left = torch.zeros(2, 16, 4)
    right = torch.zeros_like(left)
    native_left = left.clone()
    native_right = right.clone()
    native_left[..., 2] = math.pi - 0.01
    native_right[..., 2] = -math.pi + 0.01
    near_seam = chain_native4_chunk_distance(
        left, right, native_left, native_right,
        config=CHAIN_NATIVE4_ENERGY_DISTANCE_CONFIG,
    )
    assert torch.all(near_seam < 0.01)
    right[..., 3] = 1.0
    with_grip = chain_native4_chunk_distance(
        left, right, native_left, native_right,
        config=CHAIN_NATIVE4_ENERGY_DISTANCE_CONFIG,
    )
    assert torch.all(with_grip > near_seam + 0.3)


def test_native4_distance_rejects_missing_channel_or_changed_contract():
    with pytest.raises(ValueError):
        normalize_chain_native4_energy_distance_config(
            {**CHAIN_NATIVE4_ENERGY_DISTANCE_CONFIG, "rotation_scale_radians": 1.0}
        )
    wrong = torch.zeros(1, 16, 3)
    with pytest.raises(ValueError, match="complete aligned"):
        chain_native4_chunk_distance(
            wrong, wrong, wrong, wrong,
            config=CHAIN_NATIVE4_ENERGY_DISTANCE_CONFIG,
        )


def test_evaluator_routes_typed_native4_and_generic_points6():
    class IdentityNormalizer:
        def unnormalize(self, values, embodiment_id):
            assert embodiment_id in (20, 21)
            return values

    evaluator = PlanarActionEval(
        energy_score_enabled=False,
        native_decoders={
            "pushshapes_sim_chain_gripper_native4": ChainGripperNative4Decoder(),
            "pushshapes_sim_chain_gripper": ChainGripperPointsNativeDecoder(),
        },
        semantic_blocks_by_embodiment={
            "pushshapes_sim_chain_gripper_native4": ((0, 2), (2, 3), (3, 4)),
            "pushshapes_sim_chain_gripper": ((0, 2), (2, 4), (4, 6)),
        },
        energy_score_distances_by_embodiment={
            "pushshapes_sim_chain_gripper_native4": CHAIN_NATIVE4_ENERGY_DISTANCE_CONFIG,
            "pushshapes_sim_chain_gripper": None,
        },
    )
    evaluator.normalizer = IdentityNormalizer()
    native_samples = torch.zeros(32, 1, 16, 4)
    native_samples[..., 2] = math.pi - 0.01
    native_target = torch.zeros(1, 16, 4)
    native_target[..., 2] = -math.pi + 0.01
    native_values = evaluator._energy_values(
        native_samples,
        native_target,
        21,
        "pushshapes_sim_chain_gripper_native4",
    )
    assert float(native_values["accuracy"]) < 0.01
    points_values = evaluator._energy_values(
        torch.zeros(32, 1, 16, 6),
        torch.zeros(1, 16, 6),
        20,
        "pushshapes_sim_chain_gripper",
    )
    assert float(points_values["accuracy"]) == 0.0
    with pytest.raises(KeyError, match="no EnergyScore distance"):
        evaluator._energy_values(native_samples, native_target, 21, "unknown")
