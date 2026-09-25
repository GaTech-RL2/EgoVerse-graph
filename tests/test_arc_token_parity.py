from __future__ import annotations

import hashlib

import numpy as np
import pytest

from egomimic.rldb.zarr import arc_length_tokenizer as arc_module
from egomimic.rldb.zarr.arc_length_tokenizer import _bracket_segment, _bracket_segments
from tests.arc_token_parity_fixtures import ARC_CASES, tokenize_case


GOLDEN = {
    "joint_straight_m100_per_waypoint": (
        (200, 14),
        "b6f50da6488ce5ca514e86f410af4ab25de69af419cb1f40829eadb9ef009db1",
        "c5fa2238bec4d300f2489a899b04fcf34897282bf883966c8a88c17d9843bccb",
    ),
    "joint_curved_mean": (
        (14, 14),
        "fcaaa486e8925f70fa9985b8d1b6cbad0b2cc503065bca2a72c0d3cdaa09db26",
        "1c20ec6991e6494b77387836878baeb7e92320cf82dde0b527348b7676c35ce7",
    ),
    "joint_curved_duration_stationary_prefix": (
        (22, 14),
        "4bee282ba132a3e692983705d4d8f543e23f6f52601b7a45510f794818661623",
        "31bf6c456fe5f78b600c7633e0a7d004e95fa47ede3500ac36aa45268c0ffe55",
    ),
    "joint_hybrid_wrap": (
        (34, 14),
        "d38b48de7faeae3bc02ac7ab1284aa8f288e78cee57b62c5aa63f1e87fb44a94",
        "42703b775b107bd0293a7d37d023e574ce71918dd0a30f1bc10c9417712f310d",
    ),
    "race_left_first_fractional": (
        (38, 14),
        "ef00045cdf4ec354119fc5cd56dbdfb5209c2eb150763e948f667dcb0cd0d667",
        "94b3ebce5e6cc293fe0549109c718806ec1172445d4dcd9d956af439e5890e9c",
    ),
    "race_right_first_fractional": (
        (38, 14),
        "f9e26841f0df5b800c3cade07611f42e8ec79d817ff3d15adc0a7a00962acbe6",
        "789928e745c3f0a23cfe67534a31c3d69db2c3ba54bc5b64feea2bf5cf74b521",
    ),
    "race_simultaneous": (
        (30, 14),
        "b85aa61569c9bc977adf0c86e1122796773eee68322a998633ee311c47ea160e",
        "e20c3ab0ead0653ac6ed6e268bb3a63a1a190bd08e6ba4855f06a8d435c8308b",
    ),
    "race_no_crossing_short_tail": (
        (24, 14),
        "4f38de1eb22e1ac6c2a49ff1a136da54ca5ec7ffa5355198323fed472c5ba304",
        "c5a8af16351561e349c7bbcf5365054a37e41243fef45d6b2d783d9a32a908b3",
    ),
    "race_hybrid_one_stationary_arm": (
        (32, 14),
        "2123859d4f23b54f90aa951d74eeb055b13bfadd68a2f23a58455deb1f963aca",
        "832ce9c3741b2488b2e9f4814698a592a95761d760f060fde548f1f2701cf937",
    ),
}


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@pytest.mark.parametrize("case", ARC_CASES, ids=lambda case: case.name)
def test_arc_token_bytes_match_frozen_reference(case):
    token, preserved = tokenize_case(case)
    expected_shape, expected_token_sha, expected_preserved_sha = GOLDEN[case.name]

    assert token.dtype == np.dtype("float64")
    assert token.shape == expected_shape
    assert preserved.dtype == np.dtype("float64")
    assert preserved.shape == (case.preserve_rows, 14)
    assert _sha256(token) == expected_token_sha
    assert _sha256(preserved) == expected_preserved_sha


def test_m100_per_waypoint_layout_remains_200_by_14():
    case = next(c for c in ARC_CASES if c.name == "joint_straight_m100_per_waypoint")
    token, _ = tokenize_case(case)
    assert token.shape == (200, 14)


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
        expected.append(
            (1.0 - alpha) * values[index] + alpha * values[index + 1]
        )
    actual = arc_module._linear_at_targets(values, cumulative, targets)
    assert np.array_equal(actual, np.stack(expected))
