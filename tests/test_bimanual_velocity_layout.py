"""Wide is the default velocity layout, and both layouts round-trip.

"wide" puts the velocity rows beside the waypoints: (M, 28), columns 0..13
shape and 14..27 velocity. "stacked" puts them under: (2M, 14). Wide halves the
sequence length and gives each stream its own normalization statistics, so a
per-column normalizer no longer pools metres with metres per second.

Wide needs one velocity row per waypoint, so it is undefined for the "mean"
velocity mode. That is the only place stacked is still the default.
"""

import numpy as np
import pytest

from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARC_TOK_BIMANUAL_DIM,
    BIMANUAL_VELOCITY_LAYOUTS,
    TokenizeBimanualArcLengthCartesian,
    bimanual_arc_token_shape,
    default_bimanual_velocity_layout,
    validate_bimanual_velocity_layout,
)

M = 10
LAYOUTS = ("wide", "stacked")


def window(num_frames: int = 100) -> np.ndarray:
    """One synthetic (num_frames, 14) bimanual cartesian window."""
    raw = np.zeros((num_frames, 14), dtype=np.float64)
    frames = np.arange(num_frames, dtype=np.float64)
    raw[:, 0] = frames * 0.05  # left x
    raw[:, 7] = frames * 0.03  # right x
    raw[:, 3] = frames * 0.004  # left first euler angle
    raw[:, 10] = frames * 0.002  # right first euler angle
    raw[:, 6] = frames / num_frames  # left gripper
    raw[:, 13] = 1.0 - frames / num_frames  # right gripper
    return raw


def codec(velocity_layout=None, velocity_mode="per_waypoint"):
    # The independent rotation clock is only defined for per-waypoint velocities,
    # so the "mean" cases below build a plain translation-only codec.
    hybrid = velocity_mode == "per_waypoint"
    return TokenizeBimanualArcLengthCartesian(
        min_distance_unit=1.0,
        rotation_distance_unit=0.5 if hybrid else None,
        resampled_vector_length=M,
        velocity_mode=velocity_mode,
        arc_chunking_mode="race" if hybrid else None,
        velocity_layout=velocity_layout,
    )


def test_wide_is_the_default():
    assert default_bimanual_velocity_layout("per_waypoint") == "wide"
    assert default_bimanual_velocity_layout("duration") == "wide"
    assert codec().velocity_layout == "wide"
    assert BIMANUAL_VELOCITY_LAYOUTS[0] == "wide"


def test_mean_velocity_mode_still_defaults_to_stacked():
    # Wide has no meaning for a single whole-token velocity row.
    assert default_bimanual_velocity_layout("mean") == "stacked"
    assert codec(velocity_mode="mean").velocity_layout == "stacked"


def test_wide_with_mean_velocity_mode_is_refused():
    with pytest.raises(ValueError, match="one velocity row per waypoint"):
        validate_bimanual_velocity_layout("wide", "mean")
    with pytest.raises(ValueError, match="one velocity row per waypoint"):
        codec(velocity_layout="wide", velocity_mode="mean")


def test_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="velocity_layout must be one of"):
        validate_bimanual_velocity_layout("beside", "per_waypoint")


def test_token_shape_halves_the_rows_and_doubles_the_dim():
    assert bimanual_arc_token_shape(M, "per_waypoint", "stacked") == (
        2 * M,
        ARC_TOK_BIMANUAL_DIM,
    )
    assert bimanual_arc_token_shape(M, "per_waypoint", "wide") == (
        M,
        2 * ARC_TOK_BIMANUAL_DIM,
    )
    # Omitting the layout asks for the default, not for the historical layout.
    assert bimanual_arc_token_shape(M, "per_waypoint") == (M, 2 * ARC_TOK_BIMANUAL_DIM)
    assert bimanual_arc_token_shape(M, "mean") == (M + 1, ARC_TOK_BIMANUAL_DIM)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_transform_emits_the_declared_shape(layout):
    batch = codec(layout).transform({"actions_cartesian": window()})
    assert batch["actions_cartesian"].shape == bimanual_arc_token_shape(
        M, "per_waypoint", layout
    )


def test_wide_carries_the_same_numbers_as_stacked():
    """Wide is a reshape of stacked, not a different computation."""
    raw = window()
    stacked = codec("stacked").transform({"actions_cartesian": raw.copy()})[
        "actions_cartesian"
    ]
    wide = codec("wide").transform({"actions_cartesian": raw.copy()})[
        "actions_cartesian"
    ]
    assert np.allclose(wide[:, :ARC_TOK_BIMANUAL_DIM], stacked[:M])
    assert np.allclose(wide[:, ARC_TOK_BIMANUAL_DIM:], stacked[M:])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_detokenize_round_trips_either_layout(layout):
    raw = window()
    token = codec(layout).transform({"actions_cartesian": raw.copy()})[
        "actions_cartesian"
    ]
    chunk = codec(layout).detokenize(token, 30)
    assert chunk.shape == (30, ARC_TOK_BIMANUAL_DIM)
    assert np.isfinite(chunk).all()


def test_both_layouts_decode_to_the_same_chunk():
    raw = window()
    chunks = [
        codec(layout).detokenize(
            codec(layout).transform({"actions_cartesian": raw.copy()})[
                "actions_cartesian"
            ],
            30,
        )
        for layout in LAYOUTS
    ]
    assert np.allclose(chunks[0], chunks[1])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_detokenize_refuses_the_other_layouts_width(layout):
    other = "stacked" if layout == "wide" else "wide"
    token = codec(other).transform({"actions_cartesian": window()})["actions_cartesian"]
    with pytest.raises(ValueError, match="detokenize expects"):
        codec(layout).detokenize(token, 30)
