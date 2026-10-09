"""Local validation limits cannot be lowered into misleading storage failures."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("value", ["0", "4194303", "invalid", "999999999999"])
def test_low_invalid_or_unavailable_space_blocks_before_python(value):
    script = Path(__file__).resolve().parents[1] / "scripts/run_local_cpu_validation.sh"
    env = dict(
        os.environ,
        LOCAL_CPU_TEST_MIN_FREE_KIB=value,
        LOCAL_CPU_TEST_PYTHON="/does-not-exist",
    )
    result = subprocess.run(
        ["bash", str(script), "--version"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 64
    assert "Local CPU validation" in result.stderr
    assert "/does-not-exist" not in result.stderr
