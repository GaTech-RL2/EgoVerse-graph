"""Explicitly authorized numerically non-equivalent homogeneous training.

Maintained PR198 prefix + native crop/RNG replay; DiT-half alone. No source
sampling, weighting, objective or augmentation change. First update captures
the native draw recipe, without another model forward; subsequent updates
must activate grouping. Validation retains native execution.
"""
from pathlib import Path
from lightning import Callback
from homogeneous_cotrain_v2 import HomogeneousNativeReplay
from optimization_parity_v3 import policy


class HomogeneousDiTHalf(Callback):
    def __init__(self, task):
        self.task = Path(task)
        self.plan = HomogeneousNativeReplay(self.task)
        self.counts = {}
        self.active = False
        self.updates = 0
        self.capture_updates = 0
        self.grouped_updates = 0
        self.saved = None
        self.plans = {}
        self.signature = None
        self.last_update_grouped = False

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

        def execute(batch, *, mode):
            if mode != "train" or not algo.nets.training:
                return self.native_execute(batch, mode=mode)
            assert len(batch) == 2
            signature = batch_shape_signature(batch)
            if signature != self.signature:
                self.plan.restore()
                algo._execute = self.native_execute
                if signature not in self.plans:
                    assert len(self.plans) < 64, "unexpectedly many batch layouts"
                    self.plans[signature] = HomogeneousNativeReplay(self.task)
                self.plan = self.plans[signature]
                self.active = self.plan.ready
                if self.active:
                    self.plan.install_candidate(algo)
                    self.candidate_execute = algo._execute
                else:
                    self.plan.install_capture(algo)
                    self.capture_execute = algo._execute
                algo._execute = execute
                self.signature = signature
            self.last_update_grouped = self.active
            if not self.active:
                result = self.capture_execute(batch, mode=mode)
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
        assert self.capture_updates == len(self.plans)
        assert self.capture_updates + self.grouped_updates == self.updates
        if self.last_update_grouped:
            assert self.grouped_updates == self.before[1] + 1
        assert self.counts.get("dit/direct", 0) > self.before[2]
        assert self.counts.get("dit/checkpoint", 0) > self.before[3]
        module.log("Train/Execution/HomogeneousGroupedUpdates", float(self.grouped_updates), on_step=True, on_epoch=False)
        module.log("Train/Execution/HomogeneousShapeCaptures", float(self.capture_updates), on_step=True, on_epoch=False)
        module.log("Train/Execution/DiTHalfDirectCalls", float(self.counts["dit/direct"]), on_step=True, on_epoch=False)

    def state_dict(self):
        return {"updates": self.updates, "capture_updates": self.capture_updates,
                "grouped_updates": self.grouped_updates, "policy_counts": dict(self.counts),
                "batch_shape_cache_version": 1,
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


def batch_shape_signature(batch):
    """Describe native tensor layouts, never sample values or episode identities."""
    import torch
    from collections.abc import Mapping
    def shape(value):
        if torch.is_tensor(value):
            return ("tensor", tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
        if isinstance(value, Mapping):
            return ("mapping", tuple((key, shape(child)) for key, child in value.items()))
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(shape(child) for child in value))
        return ("metadata", type(value).__name__)
    return shape(batch)


class ResumeBatchProbe(Callback):
    """Smoke-only real-model exercise of full, short, grouped-short, reused-full."""
    def __init__(self):
        self.calls = 0
        self.rows = []
    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        import torch
        self.calls += 1
        source = "pushshapes_sim_u_socket"
        if self.calls in (2, 3):
            def trim(value):
                if torch.is_tensor(value) and value.ndim and value.shape[0] == 32:
                    return value[:13]
                if isinstance(value, dict): return {k: trim(v) for k,v in value.items()}
                if isinstance(value, list) and len(value) == 32: return value[:13]
                return value
            batch[source] = trim(batch[source])
        self.rows.append(13 if self.calls in (2,3) else 32)
    def on_fit_end(self, trainer, module):
        assert self.rows == [32,13,13,32], self.rows
