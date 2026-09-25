"""Explicit, opt-in historical execution semantics (not a precision policy)."""


def validate_compatibility_mode(mode: str) -> str:
    if mode not in ("current", "legacy_c12"):
        raise ValueError("compatibility_mode must be 'current' or 'legacy_c12'")
    return mode
