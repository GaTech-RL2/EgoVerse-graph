"""Test-only maintained homogeneous prefix with native source RNG replay.

The isolated baseline supplies the recipe. This is not a production default:
no guessed noise schedule, extra model forward, crop disable or reseeding.
Every candidate forward validates draw signatures, complete consumption and
final RNG state. Full-model scheduled exactness proof is still required.
"""
from collections import OrderedDict
from contextlib import nullcontext
from homogeneous_cotrain_v1 import HomogeneousPolicy
from homogeneous_rng_replay_v2 import NativeDrawReplay, native_split_crop, rng_state, set_rng, same_rng


class HomogeneousNativeReplay:
    def __init__(self, task):
        self.task = task
        self.draws = NativeDrawReplay()
        self.rows = {}
        self.sources = None
        self.ready = False
        self.calls = 0
        self.grouped_encoder_calls = 0
        self.restore_action = None

    def install_capture(self, algo):
        from egomimic.models.stems.visual_core import VisualCore
        assert self.restore_action is None and not self.ready
        encoder = algo.pipeline.stages[1].encoder
        original_execute = algo._execute
        execute_saved = ("_execute" in vars(algo), vars(algo).get("_execute"))
        packed_saved = ("forward_packed" in vars(encoder), vars(encoder).get("forward_packed"))
        packed = encoder.forward_packed
        crop = VisualCore._crop
        paths = {id(module): name for name, module in encoder.named_modules() if isinstance(module,VisualCore)}
        assert paths, "native visual crop missing"

        def packed_capture(*args, **kwargs):
            source, _ = self.draws.boundary
            with self.draws.at(source,"prefix"):
                return packed(*args, **kwargs)

        def crop_capture(module, images):
            assert id(module) in paths and self.draws.boundary[1] == "prefix"
            source = self.draws.boundary[0]
            key = (source,paths[id(module)])
            assert key not in self.rows, "repeated crop owner requires distinct proof"
            self.rows[key] = len(images)
            return crop(module,images)

        def execute(batch, *, mode):
            if self.ready:
                return original_execute(batch,mode=mode)
            assert mode == "train" and len(batch) == 2
            algo._validate_groups(batch)
            self.sources = tuple(batch)
            start = rng_state()
            encoder.forward_packed = packed_capture
            VisualCore._crop = crop_capture
            try:
                with self.draws.functions("capture"):
                    result = OrderedDict()
                    for source,value in batch.items():
                        with self.draws.at(source,"tail"):
                            result[source] = algo.pipeline.execute(dict(value),mode=mode)
            finally:
                restore_packed()
                VisualCore._crop = crop
            final = rng_state()
            # Recipe completeness is an executable gate against native dropout
            # or another random operator absent from the recorded draw families.
            try:
                set_rng(start)
                replay_final = self.draws.draw_native_order()
                assert same_rng(final,replay_final), "unrecorded native RNG recipe"
            finally:
                set_rng(final)
                self.draws.values.clear()
                self.draws.positions.clear()
            assert all((source,path) in self.rows for source in self.sources for path in paths.values())
            self.ready = True
            return result

        def restore_packed():
            if packed_saved[0]:
                encoder.forward_packed = packed_saved[1]
            elif "forward_packed" in vars(encoder):
                del encoder.forward_packed

        algo._execute = execute
        def restore():
            restore_packed()
            VisualCore._crop = crop
            if execute_saved[0]: algo._execute = execute_saved[1]
            else: del algo._execute
            self.restore_action = None
        self.restore_action = restore
        return self

    def install_candidate(self, algo):
        import torch
        from egomimic.pipeline.core import Pipeline
        from egomimic.models.stems.visual_core import VisualCore
        assert self.ready and self.restore_action is None
        base = HomogeneousPolicy(self.task).install(algo)
        # Counters belong to this hook installation, including cached reactivation.
        self.calls = self.grouped_encoder_calls = 0
        stages = list(algo.pipeline.stages)
        prefix,tail = Pipeline(stages[:2]),Pipeline(stages[2:])
        encoder = stages[1].encoder
        paths = {id(module): name for name,module in encoder.named_modules() if isinstance(module,VisualCore)}
        assert set(paths.values()) == {path for _,path in self.rows}
        crop = VisualCore._crop

        def split_crop(module,images):
            assert id(module) in paths, "unknown visual owner"
            rows = [self.rows[(source,paths[id(module)])] for source in self.sources]
            return native_split_crop(module,images,self.sources,rows,self.draws,crop)

        def execute(batch, *, mode):
            assert mode == "train" and tuple(batch) == self.sources
            algo._validate_groups(batch)
            final = self.draws.draw_native_order()
            try:
                with self.draws.functions("replay"):
                    VisualCore._crop = split_crop
                    try:
                        prepared = prefix.execute_batches(batch,mode=mode)
                    finally:
                        VisualCore._crop = crop
                    assert same_rng(final,rng_state()), "unrecorded grouped prefix RNG"
                    result = OrderedDict()
                    for source,value in prepared.items():
                        with self.draws.at(source,"tail"):
                            result[source] = tail.execute(value,mode=mode)
                self.draws.complete(final)
            finally:
                VisualCore._crop = crop
                self.draws.values.clear()
                self.draws.positions.clear()
            self.calls += 1
            base.calls += 1
            self.grouped_encoder_calls = base.grouped_encoder_calls
            assert self.calls == self.grouped_encoder_calls
            return result

        algo._execute = execute
        def restore():
            VisualCore._crop = crop
            base.restore()
            self.restore_action = None
        self.restore_action = restore
        return self

    def restore(self):
        if self.restore_action:
            self.restore_action()
