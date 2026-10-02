from __future__ import annotations

import hashlib

import numpy as np
import pytest

from egomimic.rldb.zarr import arc_length_tokenizer as arc_module
from egomimic.rldb.zarr.arc_length_tokenizer import _bracket_segment, _bracket_segments
from tests.arc_token_parity_fixtures import ARC_CASES, tokenize_case, tokenizer_for

# Frozen token bytes for the wide per-waypoint contract, regenerated after the
# per-arm rotation-budget and M×28 fixes. Stacked mode remains covered below as
# a compatibility layout and must decode to the same waypoints and rates.
GOLDEN = {
    "multistream_straight_m100_per_waypoint": (
        (100, 28),
        "eb7755d66f3963d47e218b5481d5f66a135d27d7373fa7d841ee2b6ccdbf70e8",
        "c5fa2238bec4d300f2489a899b04fcf34897282bf883966c8a88c17d9843bccb",
    ),
    "multistream_curved_mean": (
        (14, 14),
        "fcaaa486e8925f70fa9985b8d1b6cbad0b2cc503065bca2a72c0d3cdaa09db26",
        "1c20ec6991e6494b77387836878baeb7e92320cf82dde0b527348b7676c35ce7",
    ),
    "multistream_curved_duration_stationary_prefix": (
        (11, 28),
        "1e4498451fd1862fe32cbaebba739d25d061c7384b3c607186510885043b8681",
        "31bf6c456fe5f78b600c7633e0a7d004e95fa47ede3500ac36aa45268c0ffe55",
    ),
    "joint_distance_hybrid_wrap": (
        (17, 28),
        "3e0ed911a22b40302a6f055383aee3cf87aba37630738d4cfb005c8c13bc2698",
        "42703b775b107bd0293a7d37d023e574ce71918dd0a30f1bc10c9417712f310d",
    ),
    "race_left_first_fractional": (
        (19, 28),
        "fe21d26d2c493431fd63543c14e28c3d9630b33a92ba7cfe2daed38ab7cbcd3c",
        "94b3ebce5e6cc293fe0549109c718806ec1172445d4dcd9d956af439e5890e9c",
    ),
    "race_right_first_fractional": (
        (19, 28),
        "653b267e02f7311728b72a1dbfbdeb43e29eeae60e272b3d518b2808c85a3e7f",
        "789928e745c3f0a23cfe67534a31c3d69db2c3ba54bc5b64feea2bf5cf74b521",
    ),
    "race_simultaneous": (
        (15, 28),
        "da5708ba6c646edae205b415c6664a3837c526af92cb0d3392df47271dd02c1f",
        "e20c3ab0ead0653ac6ed6e268bb3a63a1a190bd08e6ba4855f06a8d435c8308b",
    ),
    "race_no_crossing_short_tail": (
        (12, 28),
        "67f71dbd4a5b979c6514aad7ae7bdc34fcfcbeb0207c9c42579be024776c909b",
        "c5a8af16351561e349c7bbcf5365054a37e41243fef45d6b2d783d9a32a908b3",
    ),
    "race_hybrid_one_stationary_arm": (
        (16, 28),
        "417e34b8b8329ce5b167afb989cadaa5393d6860dc46f9948d9dbb50f97a6fb0",
        "832ce9c3741b2488b2e9f4814698a592a95761d760f060fde548f1f2701cf937",
    ),
    "joint_distance_hybrid_m100_per_waypoint": (
        (100, 28),
        "de97c25088f7bdfeed482c3949ddbc895f9d06a7c5d36e16127aaaf8d544439b",
        "20fad4084cd4dbc67b35e2689dcbc4807aebbb05dc1696eff29d1f7905c852d0",
    ),
    "race_hybrid_m100_per_waypoint": (
        (100, 28),
        "9a9f63edeed205c1bb9f3dedfbe8b4975b2b0fa4feb8cf3a72305fce649cf6bf",
        "20fad4084cd4dbc67b35e2689dcbc4807aebbb05dc1696eff29d1f7905c852d0",
    ),
    "multistream_hybrid_m100_per_waypoint": (
        (100, 28),
        "29142fba56fc10a3ad91f23ce486cac86f4e6ac71373b6bb851284e0e4fa9b7a",
        "20fad4084cd4dbc67b35e2689dcbc4807aebbb05dc1696eff29d1f7905c852d0",
    ),
    "joint_distance_hybrid_no_crossing_short_tail": (
        (12, 28),
        "fe98ac40ccacaf1a97b5e2e09b222398f9c893f3833df14f5eedd19a4bb59052",
        "e8b29026f98dcfa1e942e82f1b76fc6d12f83b6f8aa8f9b7f451e42fc04d4ce2",
    ),
    "race_hybrid_no_crossing_short_tail": (
        (12, 28),
        "95c16cab554b15bb57e78f36cf166cc1d0b2f9b4f76791f8cb83b4d4448e8cf6",
        "e8b29026f98dcfa1e942e82f1b76fc6d12f83b6f8aa8f9b7f451e42fc04d4ce2",
    ),
    "multistream_hybrid_no_crossing_short_tail": (
        (12, 28),
        "95c16cab554b15bb57e78f36cf166cc1d0b2f9b4f76791f8cb83b4d4448e8cf6",
        "e8b29026f98dcfa1e942e82f1b76fc6d12f83b6f8aa8f9b7f451e42fc04d4ce2",
    ),
    "joint_distance_hybrid_one_stationary_arm": (
        (16, 28),
        "955e55ecb41b671b9925e096a006ac209eaf7f5850cd89c206a5021f4275e5b3",
        "832ce9c3741b2488b2e9f4814698a592a95761d760f060fde548f1f2701cf937",
    ),
    "multistream_hybrid_one_stationary_arm": (
        (16, 28),
        "1c7932634064c3a5f69f87a831af7eee398d84ad306f432cd75229c2e67297cd",
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


def test_m100_per_waypoint_default_layout_is_wide():
    case = next(
        c for c in ARC_CASES if c.name == "multistream_straight_m100_per_waypoint"
    )
    token, _ = tokenize_case(case)
    assert token.shape == (100, 28)


def test_wide_and_stacked_m100_tokens_encode_the_same_values():
    case = next(
        c for c in ARC_CASES if c.name == "multistream_straight_m100_per_waypoint"
    )
    wide, _ = tokenize_case(case)
    stacked_codec = tokenizer_for(case)
    stacked_codec.velocity_layout = "stacked"
    stacked = stacked_codec.transform({"raw": case.actions.copy()})["token"]
    np.testing.assert_array_equal(wide[:, :14], stacked[:100])
    np.testing.assert_array_equal(wide[:, 14:], stacked[100:])


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


def test_first_crossing_brackets_are_bit_identical_to_scalar_reference():
    from tests.arc_token_parity_fixtures import tokenizer_for

    # A plateau at the start, a zero-length segment, and a plateau at the end:
    # the three places the first-crossing rule and _bracket_segments disagree.
    cumulative = np.array([0.0, 0.0, 0.2, 0.2, 0.7, 0.7], dtype=np.float64)
    targets = np.array([-0.1, 0.0, 0.1, 0.2, 0.45, 0.7, 0.9])
    case = next(c for c in ARC_CASES if c.name == "race_hybrid_m100_per_waypoint")
    tokenizer = tokenizer_for(case)  # any per-arm mode selects the same rule

    indices, alpha = arc_module._first_crossing_brackets(cumulative, targets)
    expected = [
        tokenizer._translation_bracket(cumulative, float(target)) for target in targets
    ]
    assert np.array_equal(indices, np.array([item[0] for item in expected]))
    assert np.array_equal(alpha, np.array([item[1] for item in expected]))
