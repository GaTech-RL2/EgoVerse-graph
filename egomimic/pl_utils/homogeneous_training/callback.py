"""Explicitly authorized numerically non-equivalent homogeneous training.

Maintained PR198 prefix + native crop/RNG replay; DiT-half alone. No source
sampling, weighting, objective or augmentation change. First update captures
the native draw recipe, without another model forward; subsequent updates
must activate grouping. Validation retains native execution.
"""
from pathlib import Path
from lightning import Callback
from .replay import HomogeneousNativeReplay
from .checkpoint_policy import policy


class HomogeneousDiTHalf(Callback):
    def __init__(self, task=None):
        self.task = Path(task) if task else Path(__file__).parent
        self.plan = HomogeneousNativeReplay(self.task)
        self.counts = {}
        self.active = False
        self.updates = 0
        self.capture_updates = 0
        self.grouped_updates = 0
        self.saved = None

    def on_fit_start(self, trainer, module):
        from egomimic.pipeline.algo import PipelineAlgo
        from egomimic.models import unite_dit
        algo = module.model
        assert isinstance(algo, PipelineAlgo) and algo.nets is module.nets
        assert self.saved is None
        stages = tuple(algo.pipeline.stages)
        assert type(stages[0]).__name__ == "KeyedFeatureProjection"
        assert type(stages[1]).__name__ == "FusedObsEncoder"
        assert sum(type(s).__name__ == "SharedSpeedCondition" for s in stages) == 1
        self.algo = algo
        self.saved = (unite_dit.checkpoint, "_execute" in vars(algo), vars(algo).get("_execute"))
        self.native_execute = algo._execute
        unite_dit.checkpoint = policy(unite_dit.checkpoint, "dit-half", "dit", self.counts)
        self.plan.install_capture(algo)
        capture = algo._execute

        def execute(batch, *, mode):
            if mode != "train" or not algo.nets.training:
                return self.native_execute(batch, mode=mode)
            assert len(batch) == 2
            if not self.active:
                result = capture(batch, mode=mode)
                assert self.plan.ready
                self.capture_updates += 1
                self.plan.restore()
                self.plan.install_candidate(algo)
                self.candidate_execute = algo._execute
                algo._execute = execute
                self.active = True
            else:
                before = self.plan.grouped_encoder_calls
                result = self.candidate_execute(batch, mode=mode)
                assert self.plan.grouped_encoder_calls == before + 1
                self.grouped_updates += 1
            self.updates += 1
            return result
        algo._execute = execute

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        self.before = (self.updates, self.grouped_updates, self.counts.get("dit/direct", 0), self.counts.get("dit/checkpoint", 0))

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        import torch
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        assert torch.isfinite(loss).all()
        assert self.updates == self.before[0] + 1
        assert self.capture_updates == 1
        if self.updates > 1:
            assert self.grouped_updates == self.before[1] + 1
        assert self.counts.get("dit/direct", 0) > self.before[2]
        assert self.counts.get("dit/checkpoint", 0) > self.before[3]
        module.log("Train/Execution/HomogeneousGroupedUpdates", float(self.grouped_updates), on_step=True, on_epoch=False)
        module.log("Train/Execution/DiTHalfDirectCalls", float(self.counts["dit/direct"]), on_step=True, on_epoch=False)

    def state_dict(self):
        return {"updates": self.updates, "capture_updates": self.capture_updates,
                "grouped_updates": self.grouped_updates, "policy_counts": dict(self.counts),
                "exact_equivalence": False, "user_accepted_numerical_delta": True}

    def load_state_dict(self, state):
        # Recapture from resumed model's next real native update. Recipe itself
        # is not tensor/model state and must not be guessed from another attempt.
        assert state["user_accepted_numerical_delta"] is True

    def restore(self):
        if self.saved is None:
            return
        from egomimic.models import unite_dit
        self.plan.restore()
        unite_dit.checkpoint = self.saved[0]
        if self.saved[1]: self.algo._execute = self.saved[2]
        elif "_execute" in vars(self.algo): del self.algo._execute
        self.saved = None

    def on_fit_end(self, trainer, module):
        self.restore()

    def on_exception(self, trainer, module, exception):
        self.restore()
