from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest

from egomimic.rldb.zarr import arc_length_tokenizer as arc_module
from egomimic.rldb.zarr.arc_length_tokenizer import _bracket_segment, _bracket_segments
from tests.arc_token_parity_fixtures import ARC_CASES, tokenize_case, tokenizer_for

# Source frozen from the canonical ARC tip (#197), not the integrated tokenizer.
# Run both implementations with the same NumPy/SciPy and inputs: SLERP and
# trigonometric functions need not produce identical bytes on macOS and Linux.
# This keeps bit-exact parity without weakening the comparison to a tolerance.
REFERENCE_DIR = Path(__file__).parent / "fixtures/arc_tokenizer_reference"
REFERENCE_SHA256 = "dcd74ea1e18cba1da718bfdccea8ae7743bc210ba06a21de9c8694e8000b425d"


@pytest.fixture(scope="module")
def frozen_reference():
    path = REFERENCE_DIR / "source.py.txt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    name = "frozen_arc_tokenizer_20507c6"
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves annotations through this.
    try:
        loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@pytest.mark.parametrize("case", ARC_CASES, ids=lambda case: case.name)
def test_arc_token_bytes_match_frozen_reference(case, frozen_reference):
    token, preserved = tokenize_case(case)
    expected_token, expected_preserved = tokenize_case(case, frozen_reference)
    metadata = json.loads((REFERENCE_DIR / "metadata.json").read_text())
    expected_shape = tuple(metadata["historical_hashes"][case.name]["shape"])

    assert token.dtype == np.dtype("float64")
    assert token.shape == expected_shape
    assert preserved.dtype == np.dtype("float64")
    assert preserved.shape == (case.preserve_rows, 14)
    assert expected_token.shape == expected_shape
    assert _sha256(token) == _sha256(expected_token)
    assert _sha256(preserved) == _sha256(expected_preserved)


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
