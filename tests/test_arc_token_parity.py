"""Vectorized geometry helpers retain scalar-reference parity."""

import numpy as np

from egomimic.rldb.zarr import arc_length_tokenizer as arc_module
from egomimic.rldb.zarr.arc_length_tokenizer import _bracket_segment, _bracket_segments


def test_vectorized_brackets_are_bit_identical_to_scalar_reference():
    cumulative = np.array([0.0, 0.0, 0.2, 0.2, 0.7, 1.0], dtype=np.float64)
    targets = np.array([-0.1, 0.0, 0.1, 0.2, 0.25, 0.7, 0.9, 1.0, 1.2])
    indices, alpha = _bracket_segments(cumulative, targets)
    expected = [_bracket_segment(cumulative, float(target)) for target in targets]
    expected_indices = np.array([item[0] for item in expected])
    expected_alpha = np.array([item[1] for item in expected])
    assert np.array_equal(indices, expected_indices)
    assert np.array_equal(alpha, expected_alpha)


def test_batched_source_times_are_bit_identical_to_scalar_reference():
    cumulative = np.array([0.0, 0.0, 0.13, 0.41, 0.41, 0.92])
    targets = np.linspace(0.0, 0.92, 19)
    dt = 1.0 / 30.0
    expected = np.array(
        [
            (index + alpha) * dt
            for index, alpha in (
                _bracket_segment(cumulative, float(target)) for target in targets
            )
        ]
    )
    actual = arc_module._source_times_at_targets(cumulative, targets, dt)
    assert np.array_equal(actual, expected)


def test_batched_linear_interpolation_is_bit_identical_to_scalar_formula():
    values = np.array(
        [[0.0, 2.0], [0.25, -1.0], [0.8, 4.0], [1.0, 3.0]],
        dtype=np.float64,
    )
    cumulative = np.array([0.0, 0.2, 0.7, 1.0])
    targets = np.array([0.0, 0.11, 0.2, 0.51, 0.7, 0.91, 1.0])
    expected = []
    for target in targets:
        index, alpha = _bracket_segment(cumulative, float(target))
        expected.append((1.0 - alpha) * values[index] + alpha * values[index + 1])
    actual = arc_module._linear_at_targets(values, cumulative, targets)
    assert np.array_equal(actual, np.stack(expected))


def test_first_crossing_does_not_include_terminal_dwell():
    cumulative = np.array([0.0, 0.0, 0.2, 0.2, 0.7, 0.7])
    targets = np.array([0.0, 0.1, 0.2, 0.45, 0.7])
    indices, alpha = arc_module._first_crossing_brackets(cumulative, targets)
    np.testing.assert_allclose(indices + alpha, [0, 1.5, 2, 3.5, 4])
