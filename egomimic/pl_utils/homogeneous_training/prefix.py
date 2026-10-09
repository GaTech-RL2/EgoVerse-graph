"""Task-local integration of PR198 shared observation batching.

No weighted sampler, loss-reduction change, model replacement, or adoption.
Only the deterministic two-stage observation prefix is grouped. Native Action
Flow source-major execution remains intact after that boundary, preserving the
ordering of noise/dropout draws in the objective paths. Scheduled proof required.
"""
import ast
import hashlib
from collections import OrderedDict
from pathlib import Path


def load_method(path, class_name, method, namespace):
    tree = ast.parse(Path(path).read_text())
    owners = [node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name == class_name]
    assert len(owners) == 1
    methods = [node for node in owners[0].body if isinstance(node, ast.FunctionDef)
               and node.name == method]
    assert len(methods) == 1
    # Execute the exact maintained method AST, not a hand-rewritten equivalent.
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method]


class HomogeneousPolicy:
    def __init__(self, task):
        self.task = Path(task)
        self.calls = 0
        self.grouped_encoder_calls = 0
        self.saved = None

    def install(self, algo):
        from collections.abc import Mapping
        from egomimic.pipeline.algo import PipelineAlgo
        from egomimic.pipeline.core import Pipeline, Stage
        from egomimic.pipeline.stages_sampler import FusedObsEncoder, KeyedFeatureProjection
        assert isinstance(algo, PipelineAlgo) and self.saved is None
        stages = list(algo.pipeline.stages)
        assert len(stages) >= 3
        assert type(stages[0]) is KeyedFeatureProjection
        assert type(stages[1]) is FusedObsEncoder
        assert not stages[1].forward_context, "context-dependent fusion unsupported"
        source = Path(__file__).parent / "prefix_source"
        for name, expected in {
            "core.py": "f15bb44f9f20fa1eed34cb89059a44587f19beeeb58bdc743245de994eff7b4b",
            "sampler.py": "152b674184e0b93580ca0e6db7f58e9471a48a50838df502af032b52b8334ac0",
            "batch_utils.py": "f00aaf14e85b39c72f2a6db0527e3a636739a2053268869350af38e4532c8737",
        }.items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
        namespace = {"Mapping": Mapping}
        batch_namespace = {}
        exec(compile((source / "batch_utils.py").read_text(),
                     str(source / "batch_utils.py"), "exec"), batch_namespace)
        namespace["map_batches"] = batch_namespace["map_batches"]
        stage_method = load_method(source / "core.py", "Stage", "execute_batches", namespace.copy())
        pipeline_method = load_method(source / "core.py", "Pipeline", "execute_batches", namespace.copy())
        fused_method = load_method(source / "sampler.py", "FusedObsEncoder", "execute_batches", namespace.copy())
        assert "execute_batches" not in vars(Stage) and "execute_batches" not in vars(Pipeline)
        assert "execute_batches" not in vars(FusedObsEncoder)
        saved = ("_execute" in vars(algo), vars(algo).get("_execute"))
        prefix, tail = Pipeline(stages[:2]), Pipeline(stages[2:])
        self.saved = (algo, Stage, Pipeline, FusedObsEncoder, saved)
        Stage.execute_batches = stage_method
        Pipeline.execute_batches = pipeline_method
        def fused(stage, batches, *, mode):
            # Counter proves real grouping, not just policy initialization.
            inputs = {source: {key: batch[key] for key in stage.reads}
                      for source, batch in batches.items()}
            groups = list(batch_namespace["compatible_groups"](inputs))
            assert len(batches) == 2 and len(groups) == 1, "no compatible grouping"
            self.grouped_encoder_calls += 1
            return fused_method(stage, batches, mode=mode)
        FusedObsEncoder.execute_batches = fused
        def execute(batch, *, mode):
            import torch
            assert mode == "train", "training-only isolated policy"
            algo._validate_groups(batch)
            assert len(batch) == 2
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
            prepared = prefix.execute_batches(batch, mode=mode)
            assert torch.equal(rng, torch.get_rng_state()), "stochastic prefix incompatible"
            assert all(torch.equal(a, b) for a, b in zip(
                cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [])), "CUDA stochastic prefix incompatible"
            self.calls += 1
            return OrderedDict((source, tail.execute(value, mode=mode))
                               for source, value in prepared.items())
        algo._execute = execute
        return self

    def restore(self):
        if self.saved is None:
            return
        algo, Stage, Pipeline, FusedObsEncoder, saved = self.saved
        if saved[0]:
            algo._execute = saved[1]
        else:
            del algo._execute
        del Stage.execute_batches
        del Pipeline.execute_batches
        del FusedObsEncoder.execute_batches
        self.saved = None
