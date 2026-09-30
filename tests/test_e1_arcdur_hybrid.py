"""arcdurhyb: an E1 duration token whose rotation keeps its own clock (M, 18).

One synthetic arm per case, round-tripped tokenize -> detokenize against the raw 30 Hz chunk.
The plain arcdur token slaves rotation (and the gripper) to the translation clock; the hybrid
token must track an in-place wrist turn and a gripper change during a translation hold.
"""

import numpy as np

from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1

DT = 1 / 30


def _tok(mode):
    return TokenizeBimanualArcLengthE1(
        action_key="a", output_action_key="a", min_distance_unit=0.40,
        resampled_vector_length=100, dt=DT, velocity_norm="path", velocity_mode=mode,
    )


def _chunk():
    """Left arm: moves 0.2 m in x over frames 0-30, then turns yaw 60 deg in place over 30-70 (any turn inside the window)
    while the gripper closes, then holds. Right arm: fully still, gripper opens over 20-60."""
    T = 100
    c = np.zeros((T, 14))
    s = np.clip(np.arange(T) / 30.0, 0, 1)
    c[:, 0] = 0.2 * s
    c[:, 3] = np.deg2rad(60) * np.clip((np.arange(T) - 30) / 40.0, 0, 1)
    c[:, 6] = 1.0 - np.clip((np.arange(T) - 30) / 40.0, 0, 1)
    c[:, 7:10] = [0.1, -0.3, 0.2]
    c[:, 13] = np.clip((np.arange(T) - 20) / 40.0, 0, 1)
    return c


def _roundtrip(mode, chunk):
    tok = _tok(mode)
    arc = tok.transform({"a": chunk.copy()})["a"]
    return arc, tok.detokenize(arc, action_horizon=len(chunk))


def test_shape_and_columns():
    arc, _ = _roundtrip("durhyb", _chunk())
    assert arc.shape == (100, 18)
    assert np.all(arc[:, 14:] >= 0)
    assert arc[:, 16].sum() > 0  # left arm rotates: rotation clock carries time
    assert arc[:, 17].sum() == 0  # right arm never rotates
    assert arc[:, 15].sum() > 0  # right arm translation hold still times its gripper


def test_hybrid_tracks_rotation_and_gripper_the_plain_token_loses():
    c = _chunk()
    _, dec_h = _roundtrip("durhyb", c)
    _, dec_p = _roundtrip("dur", c)
    yaw_h = np.abs(dec_h[:, 3] - c[:, 3]).max()
    yaw_p = np.abs(dec_p[:, 3] - c[:, 3]).max()
    assert yaw_h < np.deg2rad(2) < yaw_p  # plain arcdur loses the in-place turn
    assert np.abs(dec_h[:, 0:3] - c[:, 0:3]).max() < 5e-3  # translation still exact
    assert np.abs(dec_h[:, 13] - c[:, 13]).max() < 0.05  # held arm's gripper keeps its timing
    assert np.abs(dec_p[:, 13] - c[:, 13]).max() > 0.5  # ... which the plain token drops
