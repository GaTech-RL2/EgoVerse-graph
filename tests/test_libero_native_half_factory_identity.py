"""Historical half-ResNet factory identity; no torch import for static guard."""
import ast
import hashlib
from pathlib import Path

def test_native_half_factory_matches_historical_architecture_and_initialization():
    path=Path(__file__).parents[1]/'egomimic/models/stems/visual_core.py'
    tree=ast.parse(path.read_text())
    branches=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="resnet_model == 'resnet18_half'"]
    assert len(branches)==1, 'native historical half-ResNet factory unavailable'
    assert hashlib.sha256(ast.dump(branches[0],include_attributes=False).encode()).hexdigest()=='f8b1abdd95123842e25d0a35e4eb9ec5505068c105f1386968fab7f4a8e6ec07'
