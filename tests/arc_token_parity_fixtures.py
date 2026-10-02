from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
)


@dataclass(frozen=True)
class ArcParityCase:
    name: str
    actions: np.ndarray
    distance: float
    waypoints: int
    velocity_mode: str
    arc_chunking_mode: str | None = None
    rotation_distance: float | None = None
    preserve_rows: int = 24


def _actions(
    rows: int,
    *,
    left_speed: float,
    right_speed: float,
    curve: float = 0.0,
    rotation: float = 0.0,
    stationary_prefix: int = 0,
    wrap: bool = False,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    progress = t.copy()
    if stationary_prefix:
        progress[:stationary_prefix] = 0.0
        progress[stationary_prefix:] = np.linspace(
            0.0, 1.0, rows - stationary_prefix, dtype=np.float64
        )

    out = np.zeros((rows, 14), dtype=np.float64)
    out[:, 0] = left_speed * progress
    out[:, 1] = curve * np.sin(np.pi * progress)
    out[:, 2] = 0.03 * progress**2
    out[:, 7] = -right_speed * progress
    out[:, 8] = curve * np.cos(np.pi * progress)
    out[:, 9] = -0.02 * progress**2
    out[:, 3] = rotation * progress
    out[:, 4] = -0.25 * rotation * progress
    out[:, 10] = -0.7 * rotation * progress
    out[:, 11] = 0.2 * rotation * progress
    if wrap:
        out[:, 5] = np.linspace(np.pi - 0.08, -np.pi + 0.12, rows)
        out[:, 12] = np.linspace(-np.pi + 0.05, np.pi - 0.09, rows)
    else:
        out[:, 5] = 0.3 * rotation * progress
        out[:, 12] = -0.15 * rotation * progress
    out[:, 6] = 0.5 + 0.4 * np.sin(2.0 * np.pi * t)
    out[:, 13] = 0.45 + 0.35 * np.cos(1.5 * np.pi * t)
    return out


ARC_CASES = (
    ArcParityCase(
        "multistream_straight_m100_per_waypoint",
        _actions(151, left_speed=0.31, right_speed=0.27),
        distance=0.40,
        waypoints=100,
        velocity_mode="per_waypoint",
    ),
    ArcParityCase(
        "multistream_curved_mean",
        _actions(73, left_speed=0.42, right_speed=0.18, curve=0.09, rotation=0.8),
        distance=0.33,
        waypoints=13,
        velocity_mode="mean",
    ),
    ArcParityCase(
        "multistream_curved_duration_stationary_prefix",
        _actions(
            61,
            left_speed=0.38,
            right_speed=0.24,
            curve=0.06,
            rotation=0.5,
            stationary_prefix=7,
        ),
        distance=0.29,
        waypoints=11,
        velocity_mode="duration",
    ),
    ArcParityCase(
        "joint_distance_hybrid_wrap",
        _actions(
            89,
            left_speed=0.36,
            right_speed=0.22,
            curve=0.05,
            rotation=1.1,
            stationary_prefix=5,
            wrap=True,
        ),
        distance=0.31,
        rotation_distance=0.42,
        waypoints=17,
        velocity_mode="per_waypoint",
        arc_chunking_mode="joint_distance",
    ),
    ArcParityCase(
        "race_left_first_fractional",
        _actions(71, left_speed=0.63, right_speed=0.21, curve=0.04, rotation=0.6),
        distance=0.40,
        rotation_distance=0.42,
        waypoints=19,
        velocity_mode="per_waypoint",
        arc_chunking_mode="race",
    ),
    ArcParityCase(
        "race_right_first_fractional",
        _actions(67, left_speed=0.19, right_speed=0.68, curve=0.03, rotation=0.4),
        distance=0.40,
        rotation_distance=0.42,
        waypoints=19,
        velocity_mode="per_waypoint",
        arc_chunking_mode="race",
    ),
    ArcParityCase(
        "race_simultaneous",
        _actions(55, left_speed=0.52, right_speed=0.52, rotation=0.2),
        distance=0.40,
        rotation_distance=0.42,
        waypoints=15,
        velocity_mode="per_waypoint",
        arc_chunking_mode="race",
    ),
    ArcParityCase(
        "race_no_crossing_short_tail",
        _actions(9, left_speed=0.08, right_speed=0.06, curve=0.01),
        distance=0.40,
        rotation_distance=0.42,
        waypoints=12,
        velocity_mode="per_waypoint",
        arc_chunking_mode="race",
    ),
    ArcParityCase(
        "race_hybrid_one_stationary_arm",
        _actions(47, left_speed=0.0, right_speed=0.57, curve=0.02, rotation=0.9),
        distance=0.40,
        rotation_distance=0.42,
        waypoints=16,
        velocity_mode="per_waypoint",
        arc_chunking_mode="race",
    ),
)


# The three modes the organize_stationary group compares, at the shape actually
# trained: hybrid rotation clock, M = 100, per-waypoint velocity. Without these
# the corpus would freeze nothing about the configuration that matters.
_PRODUCTION_SOURCE = _actions(
    151, left_speed=0.31, right_speed=0.27, curve=0.05, rotation=0.7
)

ARC_CASES = (
    ARC_CASES
    + tuple(
        ArcParityCase(
            f"{mode}_hybrid_m100_per_waypoint",
            _PRODUCTION_SOURCE,
            distance=0.40,
            rotation_distance=0.42,
            waypoints=100,
            velocity_mode="per_waypoint",
            arc_chunking_mode=mode,
        )
        for mode in ("joint_distance", "race", "multistream")
    )
    + tuple(
        # A source that never reaches D, where the per-arm first-crossing rule
        # and _bracket_segments' end clamping disagree most.
        ArcParityCase(
            f"{mode}_hybrid_no_crossing_short_tail",
            _actions(9, left_speed=0.08, right_speed=0.06, curve=0.01, rotation=0.3),
            distance=0.40,
            rotation_distance=0.42,
            waypoints=12,
            velocity_mode="per_waypoint",
            arc_chunking_mode=mode,
        )
        for mode in ("joint_distance", "race", "multistream")
    )
    + tuple(
        # One arm stationary, so the per-arm clocks diverge.
        ArcParityCase(
            f"{mode}_hybrid_one_stationary_arm",
            _actions(47, left_speed=0.0, right_speed=0.57, curve=0.02, rotation=0.9),
            distance=0.40,
            rotation_distance=0.42,
            waypoints=16,
            velocity_mode="per_waypoint",
            arc_chunking_mode=mode,
        )
        for mode in ("joint_distance", "multistream")
    )
)


def tokenizer_for(case: ArcParityCase, module=None):
    """Build the case's tokenizer, optionally from another revision's module.

    ``module`` lets the throughput benchmark construct the same case against a
    reference copy of this file loaded from an older commit, so before/after can
    be compared without two checkouts.
    """
    factory = (
        TokenizeBimanualArcLengthCartesian
        if module is None
        else module.TokenizeBimanualArcLengthCartesian
    )
    return factory(
        action_key="raw",
        output_action_key="token",
        min_distance_unit=case.distance,
        rotation_distance_unit=case.rotation_distance,
        resampled_vector_length=case.waypoints,
        preserve_action_key="preserved",
        preserve_action_rows=case.preserve_rows,
        velocity_mode=case.velocity_mode,
        arc_chunking_mode=case.arc_chunking_mode,
    )


def tokenize_case(case: ArcParityCase, module=None) -> tuple[np.ndarray, np.ndarray]:
    batch = tokenizer_for(case, module).transform({"raw": case.actions.copy()})
    return batch["token"], batch["preserved"]
