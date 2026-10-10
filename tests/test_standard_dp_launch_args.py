"""The maintained DP helper cannot silently select a retired single-source recipe."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("profile", ["", "chain_manual4919_retimed_v1"])
def test_retired_standard_dp_default_rejected(monkeypatch, profile):
    monkeypatch.delenv("DP_COTRAIN_STANDARD", raising=False)
    monkeypatch.setenv("DP_SINGLE_SOURCE_STANDARD", profile)
    root = Path(__file__).resolve().parents[1]
    recipe = runpy.run_path(str(root / "scripts/train/standard_dp_launch.py"))["recipe"]
    with pytest.raises(ValueError, match="retired"):
        recipe()
