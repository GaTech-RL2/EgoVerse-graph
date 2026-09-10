"""The grip channel must survive tokenization, or the corpus fix is undone.

The articulated corpus exists because the previous one never wrote the engage
channel -- "every grasp is a shove". A width-6 ARC token reintroduces exactly
that defect one layer up, so these tests assert the channel is live end to end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from egomimic.rldb.embodiment.articulated_arc import (
    ArticulatedArcDurationGripNativeDecoder,
    ArticulatedArcStackedGripNativeDecoder,
)
from egomimic.rldb.zarr.planar_arc import (
    PLANAR_ARC_GRIP_DIM,
    PadPlanarAction,
    TokenizeArcDurationGrip,
    TokenizeArcVelocityStackedGrip,
    TokenizeUSocketArcVelocityStacked,
)

D, M, H, DT = 80.0, 56, 16, 1.0 / 30.0
R = math.radians(26.0)

VELOCITY = (TokenizeArcVelocityStackedGrip, ArticulatedArcStackedGripNativeDecoder)
DURATION = (TokenizeArcDurationGrip, ArticulatedArcDurationGripNativeDecoder)


def _reach_grasp_carry(seed: int = 0, *, grip: bool = True) -> np.ndarray:
    """A trajectory shaped like the corpus: approach, engage, transport."""
    t = np.linspace(0.0, 1.0, 80)
    x = 60.0 * t + 8.0 * np.sin(3.0 * t + seed)
    y = 25.0 * np.sin(2.2 * t + seed) + 3.0 * t
    theta = 0.9 * t + 0.15 * np.sin(5.0 * t + seed)
    if not grip:
        return np.column_stack([x, y, theta])
    engage = (t > 0.42).astype(float)
    engage = np.convolve(engage, np.ones(5) / 5.0, mode="same")  # ramped, as collected
    return np.column_stack([x, y, theta, engage])


def _common5(native: np.ndarray) -> np.ndarray:
    batch = PadPlanarAction(["actions"]).transform({"actions": native.copy()})
    return np.asarray(batch["actions"], dtype=np.float64)


def _tokenizer(cls):
    return cls(
        min_distance_unit=D,
        resampled_vector_length=M,
        dt=DT,
        rotation_distance_unit=R,
    )


@pytest.mark.parametrize("cls,_dec", [VELOCITY, DURATION])
def test_token_is_width_seven_and_grip_is_live(cls, _dec):
    token = _tokenizer(cls).tokenize(_common5(_reach_grasp_carry()))
    assert token.shape == (M, PLANAR_ARC_GRIP_DIM)
    grip = token[:, 6]
    # A constant channel would mean the tokenizer dropped the engage event.
    assert grip.max() > 0.9, "grip never reaches engaged in the token"
    assert grip.min() < 0.1, "grip never reaches released in the token"


@pytest.mark.parametrize("cls,_dec", [VELOCITY, DURATION])
def test_three_dof_tool_pads_to_constant_zero_grip(cls, _dec):
    """u_socket, triangle and scoop have no grip channel; zero is correct."""
    token = _tokenizer(cls).tokenize(_common5(_reach_grasp_carry(grip=False)))
    assert np.allclose(token[:, 6], 0.0)


def test_adding_grip_leaves_the_geometry_bit_identical():
    """Width 7 must be width 6 plus a column, or the ablation is confounded."""
    actions = _common5(_reach_grasp_carry())
    six = _tokenizer(TokenizeUSocketArcVelocityStacked).tokenize(actions)
    seven = _tokenizer(TokenizeArcVelocityStackedGrip).tokenize(actions)
    assert np.array_equal(six, seven[:, :6])


@pytest.mark.parametrize("cls,decoder_cls", [VELOCITY, DURATION])
def test_round_trip_recovers_pose_and_grip(cls, decoder_cls):
    tokenizer = _tokenizer(cls)
    decoder = decoder_cls(
        resampled_vector_length=M, action_horizon=H, native_action_dim=4, dt=DT
    )
    for seed in range(8):
        native = _reach_grasp_carry(seed)
        token = tokenizer.tokenize(_common5(native))
        out = decoder.decode(
            torch.as_tensor(token, dtype=torch.float32).unsqueeze(0)
        )[0].numpy()
        assert out.shape == (H, 4)
        reference = native[:H]
        assert np.abs(out[:, :2] - reference[:, :2]).max() < 0.5
        assert np.abs(out[:, 3] - reference[:, 3]).max() < 0.05


@pytest.mark.parametrize("cls,decoder_cls", [VELOCITY, DURATION])
def test_decoded_grip_stays_in_the_simulator_range(cls, decoder_cls):
    """Diffusion samples land outside [0, 1]; the sim reads grip as a level."""
    decoder = decoder_cls(
        resampled_vector_length=M, action_horizon=H, native_action_dim=4, dt=DT
    )
    token = _tokenizer(cls).tokenize(_common5(_reach_grasp_carry()))
    noisy = torch.as_tensor(token, dtype=torch.float32).unsqueeze(0)
    noisy[..., 6] = noisy[..., 6] * 3.0 - 1.0  # push it well outside [0, 1]
    out = decoder.decode(noisy)[0].numpy()
    assert out[:, 3].min() >= 0.0 and out[:, 3].max() <= 1.0


def test_every_articulated_embodiment_resolves_to_an_id():
    """get_embodiment_id raises KeyError for an unregistered override."""
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    roster = (
        "u_socket", "gripper", "chain_gripper", "suction", "umi",
        "triangle", "scoop", "flipper", "spring",
    )
    ids = {e: get_embodiment_id(f"pushshapes_sim_{e}") for e in roster}
    assert len(set(ids.values())) == len(roster), f"duplicate ids: {ids}"
    assert ids["u_socket"] == 19 and ids["chain_gripper"] == 20  # pinned
