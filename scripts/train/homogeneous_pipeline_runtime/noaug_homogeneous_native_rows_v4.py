"""Full native visual graphs inside maintained grouped-prefix RNG replay.

Test candidate only. Must pass actual model boundaries and full exact-loss
verification before any production use. No visual batch64 speed claim.
"""
from homogeneous_cotrain_v2 import HomogeneousNativeReplay
from noaug_visual_native_rows_v4 import native_visual_rows


class HomogeneousNativeRows(HomogeneousNativeReplay):
    def __init__(self, task):
        super().__init__(task)
        self.visual_contexts = []
        self.visual_stats = []
        self.visual_verified_updates = 0

    def install_candidate(self, algo):
        from egomimic.models.stems.visual_core import VisualCore
        super().install_candidate(algo)
        try:
            visual = [m for m in algo.pipeline.stages[1].encoder.modules()
                      if isinstance(m, VisualCore)]
            assert len(visual) == 1, "unproved visual inventory"
            assert self.sources and all(
                self.rows[(s, p)] == 32 for s, p in self.rows)
            for module in visual:
                context = native_visual_rows(module, 32)
                self.visual_contexts.append(context)
                self.visual_stats.append(context.__enter__())
            execute = algo._execute

            def guarded(batch, *, mode):
                before = [(s["grouped_calls"], s["native_chunks"])
                          for s in self.visual_stats]
                result = execute(batch, mode=mode)
                after = [(s["grouped_calls"], s["native_chunks"])
                         for s in self.visual_stats]
                assert all(b - a == 1 and d - c == 2
                           for (a, c), (b, d) in zip(before, after)), (
                               "native visual graph inactive or extra calls")
                self.visual_verified_updates += 1
                return result

            algo._execute = guarded
            return self
        except BaseException:
            self.restore()
            raise

    def restore(self):
        for context in reversed(self.visual_contexts):
            context.__exit__(None, None, None)
        self.visual_contexts.clear()
        self.visual_stats.clear()
        super().restore()
