"""CPU-only contract checks; no CUDA or dataset initialization."""

import ast
import re
from pathlib import Path

import pytest


def helper():
    path = Path(__file__).resolve().parents[1] / "scripts/ice/ice_gpu_probe.py"
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "nvml_gpu_uuid"
    )
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["nvml_gpu_uuid"]


@pytest.mark.parametrize("prefix", ["", "GPU-"])
def test_actual_cuda_uuid_is_normalized(prefix):
    bare = "12345678-1234-1234-1234-123456789abc"
    assert helper()(prefix + bare) == "GPU-" + bare


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "MIG-1234",
        "GPU-",
        None,
        b"1234",
        "GPU-12345678-1234-1234-1234-123456789abc; x",
    ],
)
def test_physical_indices_or_unknown_uuid_rejected(value):
    with pytest.raises(RuntimeError):
        helper()(value)
