"""Checkpoint-specific ARC execution, frozen at the training revision.

The source snapshots are byte-identical Git blobs. Only imports are redirected
into a private namespace. Helper functions and the evaluator's decode methods
are compiled directly from those snapshots, not reimplemented here.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

TRAINING_COMMIT = "bbb813a0ebb8fe6337bd61827f510771842bffee"
SOURCE_HASHES = {
    "tokenizer.py": "677fb7ad388057ed84311cf6bf4c4696f171d855b021f550cc7d93051cedba9f",
    "open_loop.py": "c380f83820fd7711c698124071e27ee7a435d5d9ddcac8f05fc75ef06c8bb6a2",
    "distance.py": "d73556f46064d6db549d1faa7a6272733698ad0a3b8def615c20ca94642d7b5a",
}
_TOKENIZER = "egomimic.rldb.zarr.arc_length_tokenizer"
_DISTANCE = "egomimic.eval.distance_budget_dtw"


def verify_sources():
    root = Path(__file__).parent
    for filename, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned ARC training source hash mismatch: {filename}")


@lru_cache(maxsize=1)
def frozen_modules():
    verify_sources()
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(__name__ + "._tokenizer", root / "tokenizer.py")
    tokenizer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tokenizer
    spec.loader.exec_module(tokenizer)
    modules = {_TOKENIZER: tokenizer}

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    def execute(name, filename, tree=None):
        module = ModuleType(__name__ + "._" + name)
        module.__file__ = str(root / filename)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=private_import)
        code = compile(tree if tree is not None else (root / filename).read_text(), module.__file__, "exec")
        exec(code, module.__dict__)
        return module

    distance = execute("distance", "distance.py")
    modules[_DISTANCE] = distance
    tree = ast.parse((root / "open_loop.py").read_text())
    body = []
    method_names = {"_clock_seconds", "_is_arc_prediction", "_decode_prediction_with_steps"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("egomimic."):
            if node.module not in modules:
                continue
        if isinstance(node, ast.ClassDef):
            if node.name == "OpenLoopSimEval":
                # Discard metric/video lifecycle, preserving exact decode methods.
                node.bases = []
                node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in method_names]
                assert {item.name for item in node.body} == method_names
                body.append(node)
            continue
        else:
            body.append(node)
    evaluator = execute("open_loop", "open_loop.py", ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])))
    return tokenizer, evaluator


class PinnedArcDecoder:
    """Decode unnormalized native (1,100,20) predictions to executable frames.

    There is deliberately no output padding, fixed frame replan limit, timing
    floor, or global decoder replacement. All-held predictions yield 400 frames.
    """
    shape = (100, 20)
    training_commit = TRAINING_COMMIT

    def __init__(self, mode, training_commit=TRAINING_COMMIT):
        if training_commit != TRAINING_COMMIT or mode not in ("pw", "logsec"):
            raise ValueError("Unsupported pinned training revision or timing mode")
        verify_sources()
        _, source = frozen_modules()
        self.mode = mode
        self.reference = source.OpenLoopSimEval()
        self.reference.action_mode = "arc"
        self.reference.resampled_vector_length = 100
        self.reference.min_distance_unit = 0.42
        self.reference.rotation_distance_unit = float(np.deg2rad(100))
        self.reference.arc_chunking_mode = "multistream"
        self.reference.arc_execution_cap_mode = "waypoints"
        self.reference.execute_fraction = 0.3
        self.reference.control_dt = 1 / 30
        self.reference.velocity_mode = "per_waypoint" if mode == "pw" else "duration"
        self.reference.velocity_layout = "rate_clock" if mode == "pw" else "log_clock6"
        self.reference.gripper_distance_unit = None
        self.reference.gripper_stream_mode = "events"
        self.reference._arc_tokenizer = None

    def __call__(self, native_tokens):
        values = native_tokens.detach().cpu() if torch.is_tensor(native_tokens) else torch.as_tensor(np.asarray(native_tokens))
        if tuple(values.shape) != (1, 100, 20) or not bool(torch.isfinite(values).all()):
            raise ValueError("Pinned ARC decoder requires finite (1,100,20) unnormalized tokens")
        # This is the evaluator's exact conversion. Log seconds are undone ONCE.
        seconds = self.reference._clock_seconds(values)
        if not bool(torch.isfinite(seconds).all()):
            raise ValueError("Nonfinite decoded ARC timing; refusing the prediction")
        prediction = seconds[0].numpy()
        decoded, steps = self.reference._decode_prediction_with_steps(prediction)
        if decoded.shape != (steps, 14) or not np.isfinite(decoded).all():
            raise ValueError("Invalid pinned ARC decoded trajectory")
        return decoded[None]
