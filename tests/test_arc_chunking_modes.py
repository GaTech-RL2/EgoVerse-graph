"""Independent spatial budgets; execution races clocks, not token geometry."""

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian


@pytest.mark.parametrize("clock", ["velocity", "duration"])
@pytest.mark.parametrize("streams", ["full_separation", "translation_rotation"])
def test_spatial_caps_are_independent_but_execution_stops_at_fastest(clock, streams):
    c = TokenizeBimanualArcLengthCartesian(
        clock_mode=clock,
        streams=streams,
        min_distance_unit=0.4,
        rotation_distance_unit=0.6,
        resampled_vector_length=31,
    )
    time = np.arange(241) / 30.0
    raw = np.zeros((len(time), 14))
    raw[:, 0], raw[:, 7] = 0.2 * time, 0.1 * time
    raw[:, 3], raw[:, 10] = 0.05 * time, 0.1 * time
    token = c.transform({"actions_cartesian": raw})["actions_cartesian"]
    np.testing.assert_allclose(token[-1, [0, 7]], [0.4, 0.4], atol=1e-10)
    np.testing.assert_allclose(token[-1, [3, 10]], [0.4, 0.6], atol=1e-10)
    assert c.control_steps(token) == 60  # fast translation needs two seconds
    result = c.detokenize(token, 211)
    np.testing.assert_allclose(result[30, [0, 7]], [0.2, 0.1], atol=1e-9)
    np.testing.assert_allclose(result[180, [3, 10]], [0.3, 0.6], atol=1e-9)
    np.testing.assert_allclose(result[210, [0, 7]], [0.4, 0.4], atol=1e-9)
