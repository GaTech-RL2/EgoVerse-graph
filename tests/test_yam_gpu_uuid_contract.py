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

@pytest.mark.parametrize("text", ["f8ae3ac1-0e5a-3e23-c619-d06dd44000a2", "invalid"])
def test_torch_cuuuid_wrapper_keeps_strict_uuid_validation(text):
    wrapper = type(
        "_CUuuid", (), {"__module__": "torch._C", "__str__": lambda self: text}
    )
    if text == "invalid":
        with pytest.raises(RuntimeError):
            helper()(wrapper())
    else:
        assert helper()(wrapper()) == "GPU-" + text


def test_unrelated_stringifiable_object_is_rejected():
    wrapper = type(
        "OtherUUID",
        (),
        {"__str__": lambda self: "f8ae3ac1-0e5a-3e23-c619-d06dd44000a2"},
    )
    with pytest.raises(RuntimeError):
        helper()(wrapper())
