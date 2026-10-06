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


# -- arcvelhyb (profhyb): the same token with interval speeds in place of durations ---------------

def _delayed_turn():
    """Left arm still for 1 s, then yaws 45 deg in place over 1 s; right arm translates 0.15 m from frame 0."""
    T = 100
    c = np.zeros((T, 14))
    c[:, 3] = np.deg2rad(45) * np.clip((np.arange(T) - 30) / 30.0, 0, 1)
    c[:, 7] = 0.15 * np.clip(np.arange(T) / 60.0, 0, 1)
    return c


def test_profhyb_columns_are_speeds_with_the_start_delay_row():
    arc, _ = _roundtrip("profhyb", _chunk())
    dur, _ = _roundtrip("durhyb", _chunk())
    assert arc.shape == (100, 18) and np.all(arc[:, 14:] >= 0)
    assert np.allclose(arc[:, :14], dur[:, :14])  # identical waypoints
    assert 0.1 < np.median(arc[:99, 14]) < 0.5  # left arm: 0.2 m in 1 s -> ~0.2 m/s
    assert np.allclose(arc[:, 15], 0)  # right arm translation hold: all zeros (no rare large value in row M-1)
    assert arc[:, 17].sum() == 0  # right arm never rotates


def test_profhyb_decodes_like_durhyb():
    for c in (_chunk(), _delayed_turn()):
        _, dec_v = _roundtrip("profhyb", c)
        _, dec_d = _roundtrip("durhyb", c)
        assert np.abs(dec_v - dec_d).max() < 1e-6
    _, dec = _roundtrip("profhyb", _delayed_turn())
    assert np.abs(dec[:30, 3]).max() < np.deg2rad(1)  # the delayed turn does not start early
    assert np.abs(dec[:, 3] - _delayed_turn()[:, 3]).max() < np.deg2rad(2)


# -- arcdurtri / arcveltri (durtri / proftri): the gripper gets its own stream ----------------------

def test_tri_shape_and_gripper_columns():
    arc, _ = _roundtrip("durtri", _chunk())
    assert arc.shape == (100, 20)
    assert np.all(arc[:, 14:] >= 0)
    assert arc[:, 18].sum() > 0 and arc[:, 19].sum() > 0  # both grippers move: both gripper clocks carry time
    still = _chunk()
    still[:, 6], still[:, 13] = 0.3, 0.7
    arc, dec = _roundtrip("durtri", still)
    assert np.allclose(arc[:, 18:], 0)  # gripper holds: all-zero columns
    assert np.allclose(dec[:, 6], 0.3) and np.allclose(dec[:, 13], 0.7)


def test_tri_gripper_keeps_its_own_timing():
    """Left gripper closes over frames 30-70, after the arm stops translating at frame 30: durhyb samples the
    gripper on the translation arc, whose last interval smears the close over frames 30-99; durtri times it on
    its own clock."""
    c = _chunk()
    _, dec_t = _roundtrip("durtri", c)
    _, dec_h = _roundtrip("durhyb", c)
    assert np.abs(dec_t[:, 6] - c[:, 6]).max() < 0.05
    assert np.abs(dec_h[:, 6] - c[:, 6]).max() > 0.3  # durhyb: 0.42 (one translation interval smears the close); durtri: 0.01
    assert np.abs(dec_t[:, 13] - c[:, 13]).max() < 0.05
    assert np.abs(dec_t[:, 0:3] - c[:, 0:3]).max() < 5e-3  # translation and rotation as in durhyb
    assert np.abs(dec_t[:, 3] - c[:, 3]).max() < np.deg2rad(2)


def test_proftri_decodes_like_durtri():
    for c in (_chunk(), _delayed_turn()):
        arc_v, dec_v = _roundtrip("proftri", c)
        arc_d, dec_d = _roundtrip("durtri", c)
        assert arc_v.shape == (100, 20) and np.allclose(arc_v[:, :14], arc_d[:, :14])
        assert np.abs(dec_v - dec_d).max() < 1e-6
    arc, _ = _roundtrip("proftri", _delayed_turn())
    assert np.allclose(arc[:, 18:], 0)  # no gripper motion: all zeros, no rare large value in row M-1
