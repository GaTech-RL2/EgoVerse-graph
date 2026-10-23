"""Bounds for the ARC replay-tempo multipliers.

Dependency-free on purpose: the decoder (torch, tokenizers) and the browser
dashboard (neither) must agree on one range without importing each other.
"""

import math

# Bounds on either tempo multiplier, wherever it comes from (config or browser).
ARC_SPEED_RANGE = (0.25, 4.0)


def validate_arc_speed(value, name="speed"):
    """A finite real tempo multiplier inside ARC_SPEED_RANGE, as a float."""
    lo, hi = ARC_SPEED_RANGE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ARC {name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"ARC {name} must be within [{lo}, {hi}]")
    return value
