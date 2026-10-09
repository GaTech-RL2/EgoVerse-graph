"""Execute maintained lifecycle methods without model/data/runtime imports."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names):
    tree = ast.parse((ROOT / path).read_text())
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


F = functions("egomimic/trainHydra.py", {"_needs_final_validation"})[
    "_needs_final_validation"
]
N = functions(
    "egomimic/eval/libero_action_flow_eval.py",
    {"on_validation_end", "has_completed_validation"},
)


class Native:
    on_validation_end = N["on_validation_end"]
    has_completed_validation = N["has_completed_validation"]

    def __init__(self, group="valid", batches=None, sanity=False):
        self.artifact_root = "unused"
        self.artifact_identity = {}
        self.group = group
        self.artifact_batches = [{"batch_index": 0}] if batches is None else batches
        self.model = types.SimpleNamespace(global_step=2)
        self.trainer = types.SimpleNamespace(sanity_checking=sanity)
        self._completed_validation_step = None


class Lifecycle(unittest.TestCase):
    def publication(self, e, error=None):
        module = types.ModuleType("egomimic.benchmarks.libero.action_flow_artifacts")
        module.canonical_sha = lambda identity: "identity"

        def writer(root, payload):
            if error:
                raise error

        module.write_artifact = writer
        with patch.dict(sys.modules, {module.__name__: module}):
            e.on_validation_end()

    def test_periodic_terminal_completion_skips_only_duplicate(self):
        e = Native()
        self.assertTrue(F(e, 2))
        self.publication(e)
        self.assertFalse(F(e, 2))
        self.assertTrue(F(e, 3))
        self.assertTrue(F(e, 80000))

    def test_missing_generic_and_nonboolean_capability_keep_final(self):
        for e in (
            None,
            object(),
            types.SimpleNamespace(has_completed_validation=lambda step: 1),
        ):
            self.assertTrue(F(e, 2))

    def test_empty_sanity_and_other_group_keep_final(self):
        for e in (Native(batches=[]), Native(sanity=True), Native(group="train_viz")):
            self.publication(e)
            self.assertTrue(F(e, 2))

    def test_new_empty_or_failed_pass_invalidates_prior_completion(self):
        tree = ast.parse(
            (ROOT / "egomimic/eval/libero_action_flow_eval.py").read_text()
        )
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "on_validation_start"
        )
        e = Native()
        e._completed_validation_step = 2
        exec(
            compile(
                ast.Module(body=method.body[:1], type_ignores=[]),
                "native-start",
                "exec",
            ),
            {"self": e},
        )
        e.artifact_batches = []
        self.publication(e)
        self.assertTrue(F(e, 2))

    def test_failed_immutable_publication_does_not_complete(self):
        e = Native()
        with self.assertRaises(FileExistsError):
            self.publication(e, FileExistsError("immutable collision"))
        self.assertTrue(F(e, 2))
        self.assertTrue(e.artifact_batches)

    def test_completion_reset_precedes_replayed_valid_batch(self):
        tree = ast.parse(
            (ROOT / "egomimic/eval/libero_action_flow_eval.py").read_text()
        )
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "on_validation_step"
        )
        # Execute the actual initial statements, stopping before model inference.
        statements = method.body[:2]
        e = Native()
        e._completed_validation_step = 2
        exec(
            compile(
                ast.Module(body=statements, type_ignores=[]), "native-reset", "exec"
            ),
            {"self": e, "batch_idx": 0, "dataloader_idx": 0},
        )
        self.assertTrue(F(e, 2))


if __name__ == "__main__":
    unittest.main()
