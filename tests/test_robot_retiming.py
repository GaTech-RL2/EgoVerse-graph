import numpy as np
import pytest

from egomimic.rldb.zarr.physical_retiming import PhysicalWindowRetiming

RATES = (5.0, 2.5, 1 / 0.6, 1.25, 1.0)
FIELDS = {"left": "pose_wxyz", "right": "pose_wxyz", "grip": "hold"}


def transform(rates=RATES, enabled=True):
    return PhysicalWindowRetiming(
        rates,
        FIELDS,
        ("left", "right"),
        100,
        embodiment="robot",
        sample_views=len(rates),
        allow_robot_augmentation=enabled,
    )


@pytest.mark.parametrize("view,rate", enumerate(RATES))
def test_reciprocal_robot_future(view, rate):
    t = transform()
    assert t.required_frames == 496
    keys = {k: {"horizon": 100} for k in FIELDS}
    t.bind_episode({"fps": 30}, keys)
    assert all(v["horizon"] == 496 for v in keys.values())
    pose = np.zeros((496, 7))
    pose[:, 0] = np.arange(496)
    pose[:, 3] = 1
    grip = np.arange(496, dtype=float)
    obs = np.array([42])
    b = t.transform(
        {
            "left": pose,
            "right": pose.copy(),
            "grip": grip,
            "obs": obs,
            "_retiming_view": view,
        }
    )
    assert b["left"].shape == (100, 7)
    np.testing.assert_allclose(b["left"][:, 0], rate * np.arange(100), atol=1e-10)
    np.testing.assert_allclose(b["left"][:, 3], 1)
    np.testing.assert_allclose(
        b["grip"], np.floor(rate * np.arange(100) + 1e-10), atol=1
    )
    assert b["obs"] is obs
    assert abs(float(b["retiming_rate"][0]) - rate) < 1e-6
    if rate == 1:
        np.testing.assert_array_equal(b["left"], pose[:100])
    np.testing.assert_array_equal(pose[:, 0], np.arange(496))


def test_default_robot_stays_one_x():
    with pytest.raises(ValueError):
        transform(enabled=False)
    t = transform((1.0,), False)
    assert t.required_frames == 100


def test_reject_incomplete_future():
    t = transform()
    t.bind_episode({"fps": 30}, {k: {"horizon": 100} for k in FIELDS})
    pose = np.zeros((495, 7))
    pose[:, 3] = 1
    with pytest.raises(ValueError):
        t.transform(
            {"left": pose, "right": pose, "grip": np.zeros(495), "_retiming_view": 0}
        )


def test_human_speedup_remains_rejected():
    with pytest.raises(ValueError):
        PhysicalWindowRetiming(
            [2],
            FIELDS,
            ("left", "right"),
            100,
            embodiment="human",
            timestamp_key="clock",
            allow_robot_augmentation=True,
        )
