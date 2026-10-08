"""Typed execution callback proposal. AV0 requires its own actual rawproof; scalar objective1 waiver is never inherited."""
import json
from typed_execution_contract_v1 import validate_runtime, validate_source_batches, require_exactness_proof, digest
from homogeneous_dithalf_training import batch_shape_signature
from pathlib import Path
from lightning import Callback
from noaug_homogeneous_native_rows_v4 import HomogeneousNativeRows
from optimization_parity_v3 import policy


class HomogeneousDiTHalf(Callback):
    def __init__(self, task, reference_task, contract_path):
        self.task = Path(task).resolve(strict=True)
        contract_path = Path(contract_path).resolve(strict=True)
        assert contract_path.parent == self.task, "contract must belong to proof task"
        self.architecture_contract = json.loads(contract_path.read_text())
        self.exact_equivalence = None
        self.task = Path(task)
        self.reference_task = Path(reference_task).resolve(strict=True)
        self.plan = HomogeneousNativeRows(self.reference_task)
        self.counts = {}
        self.active = False
        self.updates = 0
        self.capture_updates = 0
        self.grouped_updates = 0
        self.saved = None
        self.regular_signature = None
        self.signature = None
        self.native_variable_updates = 0
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
        validate_runtime(stages, self.architecture_contract)
        self.exact_equivalence = require_exactness_proof(self.task, self.architecture_contract)
        self.algo = algo
        self.saved = (unite_dit.checkpoint, "_execute" in vars(algo), vars(algo).get("_execute"))
        self.native_execute = algo._execute
        unite_dit.checkpoint = policy(unite_dit.checkpoint, "dit-half", "dit", self.counts)
        def execute(batch, *, mode):
            if mode != "train" or not algo.nets.training:
                return self.native_execute(batch, mode=mode)
            rows = validate_variable_source_batches(batch, self.architecture_contract)
            signature = batch_shape_signature(batch)
            regular = rows == self.architecture_contract["source_rows"]
            if regular and self.regular_signature is None:
                self.regular_signature = signature
            if not regular or signature != self.regular_signature:
                # Close native-32 visual contexts before executing shorter rows.
                self.plan.restore()
                algo._execute = execute
                self.signature = None
                self.active = False
                self.last_update_grouped = False
                result = self.native_execute(batch, mode=mode)
                self.native_variable_updates += 1
                self.updates += 1
                return result
            validate_source_batches(batch, self.architecture_contract)
            if signature != self.signature:
                self.plan.restore()
                algo._execute = self.native_execute
                self.active = self.plan.ready
                if self.active:
                    self.plan.install_candidate(algo)
                    self.candidate_execute = algo._execute
                else:
                    self.plan.install_capture(algo)
                    self.capture_execute = algo._execute
                self.signature = signature
                algo._execute = execute
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
                native_before = self.plan.visual_verified_updates
                result = self.candidate_execute(batch, mode=mode)
                assert self.plan.grouped_encoder_calls == before + 1
                assert self.plan.visual_verified_updates == native_before + 1
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
        assert self.capture_updates in (0, 1)
        assert self.capture_updates + self.grouped_updates + self.native_variable_updates == self.updates
        if self.last_update_grouped:
            assert self.grouped_updates == self.before[1] + 1
        assert self.counts.get("dit/direct", 0) > self.before[2]
        assert self.counts.get("dit/checkpoint", 0) > self.before[3]
        module.log("Train/Execution/HomogeneousGroupedUpdates", float(self.grouped_updates), on_step=True, on_epoch=False)
        module.log("Train/Execution/NativeVariableBatchUpdates", float(self.native_variable_updates), on_step=True, on_epoch=False)
        module.log("Train/Execution/DiTHalfDirectCalls", float(self.counts["dit/direct"]), on_step=True, on_epoch=False)

    def state_dict(self):
        return {'updates': self.updates, 'capture_updates': self.capture_updates, 'grouped_updates': self.grouped_updates, 'policy_counts': dict(self.counts), 'native_variable_updates': self.native_variable_updates, 'variable_batch_policy': 'native', 'architecture_contract_sha256': digest(self.architecture_contract), 'exact_equivalence': self.exact_equivalence, 'user_accepted_numerical_delta': self.architecture_contract['numerical_contract'] == 'known-numerical-delta-authorized'}

    def load_state_dict(self, state):
        assert state['architecture_contract_sha256'] == digest(self.architecture_contract)
        if self.architecture_contract['numerical_contract'] == 'raw-required':
            assert state['exact_equivalence'] is True
            assert state['user_accepted_numerical_delta'] is False
        self.updates = 0
        self.capture_updates = 0
        self.grouped_updates = 0

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


def validate_variable_source_batches(batch, contract):
    if set(batch) != set(contract["source_rows"]):
        raise ValueError("Every update must retain both source batches")
    rows = {}
    for source, nominal in contract["source_rows"].items():
        value = batch[source]["embodiment"]
        if not hasattr(value, "shape") or len(value.shape) not in (1, 2):
            raise ValueError("Invalid embodiment batch")
        actual = int(value.shape[0])
        if not 0 < actual <= nominal:
            raise ValueError("Empty or oversized source batch")
        rows[source] = actual
    return rows
