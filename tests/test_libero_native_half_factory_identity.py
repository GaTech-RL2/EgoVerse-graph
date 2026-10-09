"""Historical half-ResNet factory identity; no torch import for static guard."""

import ast
import hashlib
from pathlib import Path


def test_native_half_factory_matches_historical_architecture_and_initialization():
    path = Path(__file__).parents[1] / "egomimic/models/stems/visual_core.py"
    source = path.read_text()
    tree = ast.parse(source)
    branches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "resnet_model == 'resnet18_half'"
    ]
    assert len(branches) == 1, "native historical half-ResNet factory unavailable"
    assert (
        # Pinned from training source7c253e5, including the non-half else arm.
        # Ignore whitespace only; retain every call, initializer and argument.
        hashlib.sha256(
            ast.dump(branches[0], include_attributes=False).encode()
        ).hexdigest()
        == "9521e4594c50d2e98b25fc9c07ef2d0c9aac42c83b57e0d0f55c8128056d8a75"
    )
