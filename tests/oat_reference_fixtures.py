"""One shared, hash-verified upstream fixture; independent of test collection."""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def reference():
    root = os.environ.get("OAT_REFERENCE_ROOT")
    if not root:
        pytest.skip(
            "Set OAT_REFERENCE_ROOT to the pinned OAT checkout for source parity"
        )
    root = Path(root)
    manifest = json.loads(
        (Path(__file__).parents[1] / "egomimic/models/oat/UPSTREAM.json").read_text()
    )
    for relative, expected in manifest["files"].items():
        assert (
            hashlib.sha256((root / "oat" / relative).read_bytes()).hexdigest()
            == expected
        )
    previous = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        yield lambda name: importlib.import_module("oat." + name)
    finally:
        sys.path[:] = previous
