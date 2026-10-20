"""Rotation is per-arm SO(3) travel, never a sum of left/right rotation."""

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize("streams", ["full_separation", "translation_rotation"])
def test_rotation_cap_and_stationary_arm_are_independent(clock, streams):
    codec = TokenizeBimanualArcLengthCartesian(
        clock_mode=clock,
        streams=streams,
        min_distance_unit=1,
        rotation_distance_unit=0.5,
        resampled_vector_length=10,
    )
    raw = np.zeros((100, 14))
    raw[:, 0] = np.arange(100) * 0.01
    raw[:, 3], raw[:, 10] = np.arange(100) * 0.02, np.arange(100) * 0.01
    token = codec.transform({"actions_cartesian": raw})["actions_cartesian"]
    np.testing.assert_allclose(token[-1, [3, 10]], [0.5, 0.5])
    assert codec.control_steps(token) == 25
    decoded = codec.detokenize(token, 100)
    np.testing.assert_allclose(decoded[25, [3, 10]], [0.5, 0.25], atol=1e-9)
    np.testing.assert_allclose(decoded[50:, [3, 10]], 0.5, atol=1e-9)
    np.testing.assert_array_equal(decoded[:, 7:10], 0)
    raw[:, 10] = 0
    held = codec.transform({"actions_cartesian": raw})["actions_cartesian"]
    np.testing.assert_array_equal(held[:, 10:13], 0)
    assert codec.control_steps(held) == 25
