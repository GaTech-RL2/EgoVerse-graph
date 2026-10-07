"""Historical half-ResNet factory identity; no torch import for static guard."""
import ast
import hashlib
from pathlib import Path

def test_native_half_factory_matches_historical_architecture_and_initialization():
    path=Path(__file__).parents[1]/'egomimic/models/stems/visual_core.py'
    source=path.read_text()
    tree=ast.parse(source)
    branches=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="resnet_model == 'resnet18_half'"]
    assert len(branches)==1, 'native historical half-ResNet factory unavailable'
    assert hashlib.sha256(ast.get_source_segment(source,branches[0]).encode()).hexdigest()=='8b5b660b258b4bda63444714d5417ad1c7764ab7a3e436b5797651b861e2d0b8'
