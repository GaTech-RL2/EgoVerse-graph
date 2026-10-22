"""Observe real crash restoration and early updates without saving or stopping.

Actual distributed/CUDA execution is mandatory; CPU tests do not certify the
production path. The scientific RNG restore function runs unchanged. A temporary
observer verifies its resulting state before the first training computation.
"""

import functools
import hashlib
import importlib
import math
import os
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from scripts.train.joint.contract import bound, canonical_uuid, need, sha, write_once


def equal_state(actual, saved):
    if isinstance(saved, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and actual.dtype == saved.dtype
                and actual.shape == saved.shape
                and torch.equal(actual.detach().cpu(), saved.detach().cpu()))
    if isinstance(saved, np.ndarray):
        return (isinstance(actual, np.ndarray) and actual.dtype == saved.dtype
                and np.array_equal(actual, saved))
    if isinstance(saved, dict):
        return (isinstance(actual, dict) and actual.keys() == saved.keys()
                and all(equal_state(actual[key], value) for key, value in saved.items()))
    if isinstance(saved, (list, tuple)):
        return (type(actual) is type(saved) and len(actual) == len(saved)
                and all(equal_state(a, b) for a, b in zip(actual, saved)))
    return type(actual) is type(saved) and actual == saved


def weights_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_restored_state(trainer, module, saved, *, start_step, world_size):
    """Strict loaded-object equality, not just constructor/config compatibility."""
    need(trainer.world_size == world_size == 4 and trainer.precision == "bf16-mixed"
         and trainer.global_step == saved["global_step"] == start_step,
         "Crash restore topology, precision or global step changed")
    need(equal_state(module.state_dict(), saved["state_dict"]), "Weights were not exactly restored")
    need(len(trainer.optimizers) == 1 and isinstance(trainer.optimizers[0], torch.optim.AdamW),
         "Expected one actual restored AdamW optimizer")
    optimizer = trainer.optimizers[0]
    need(equal_state(optimizer.state_dict(), saved["optimizer_states"][0]),
         "AdamW moments or parameter groups were not exactly restored")
    need(saved.get("lr_schedulers", []) == [] and not trainer.lr_scheduler_configs,
         "Constant1e-4 recovery cannot introduce a scheduler")
    need(all(group["lr"] == 1e-4 and group["weight_decay"] == .01
             for group in optimizer.param_groups), "Restored AdamW LR or decay changed")
    need(all(float(value["step"]) == start_step and all(
                 value[name].dtype == torch.float32 and torch.isfinite(value[name]).all()
                 for name in ("exp_avg", "exp_avg_sq"))
             for value in optimizer.state.values()) and bool(optimizer.state),
         "AdamW state is not finiteFP32 at the restored update")
    need(equal_state(trainer.datamodule.state_dict(), saved["PixelProprioDataModule"])
         and trainer.datamodule.consumed_batches == start_step,
         "Exact sampling cursor/context was not restored")
    need(equal_state(module.training_behavior._pending_rng, saved["mot_rng_state"]),
         "Native per-rank RNG restoration was not armed")
    return optimizer


class CrashRestoreProof(Callback):
    """Emit evidence after actual restored updates; do not add saves/validation."""

    def __init__(self, *, checkpoint, inspection_path, inspection_sha256, output,
                 world_size, start_step, proof_updates=2, diagnostic=False,
                 validation_interval=20000, request_sha256=None):
        need(world_size == 4 and type(start_step) is int and start_step > 0
             and start_step % 20000 == 0 and type(proof_updates) is int
             and 2 <= proof_updates < 20000, "Invalid actual crash-restore proof range")
        need(type(diagnostic) is bool and (
                 (not diagnostic and validation_interval == 20000)
                 or (diagnostic and validation_interval == proof_updates == 2)),
             "Only an explicit two-update diagnostic may use diagnostic validation cadence")
        self.checkpoint, self.output = str(checkpoint), Path(output)
        self.inspection_binding = dict(path=str(inspection_path), sha256=inspection_sha256)
        self.world_size, self.start_step, self.proof_updates = world_size, start_step, proof_updates
        self.diagnostic, self.validation_interval = diagnostic, validation_interval
        self.request_sha256 = request_sha256
        self._rng_original = self._rng_observer = self._rng_module = None
        self.precision_hook = None
        self.done = False
        self.observed_saves = []
        self.validation_steps = []

    def on_train_start(self, trainer, pl_module):
        need(torch.distributed.is_available() and torch.distributed.is_initialized()
             and torch.distributed.get_world_size() == self.world_size
             and torch.distributed.get_rank() == trainer.global_rank
             and torch.cuda.is_available(), "Actual four-rank CUDA/DDP is required")
        inspection = bound(self.inspection_binding)
        need(inspection["schema"] == "joint-crash-checkpoint-inspection-v1"
             and inspection["status"] == "PASS"
             and inspection["global_step"] == self.start_step
             and inspection["world_size"] == self.world_size
             and inspection["checkpoint"]["path"] == self.checkpoint
             and sha(self.checkpoint) == inspection["checkpoint"]["sha256"],
             "Crash checkpoint differs from its exact allocated inspection")
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.optimizer = verify_restored_state(
            trainer, pl_module, saved, start_step=self.start_step, world_size=self.world_size)
        provenance = pl_module.hparams.config_tree.run_provenance
        need(provenance.run_id == inspection["run_id"]
             and provenance.source_revision == inspection["source_revision"]
             and pl_module.data_context.fingerprint() == inspection["data_context_sha256"],
             "Restored runtime source/run/data context differs from inspection")
        self.network = pl_module.model.pipeline.stage_by_id("joint_denoiser").model
        need(all(p.requires_grad and p.dtype == torch.float32 for p in self.network.parameters()),
             "Every active parameter must remain trainableFP32")
        self.initial_weights = weights_digest(pl_module.state_dict())
        gathered = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, self.initial_weights)
        need(len(set(gathered)) == 1, "Initial restored weights differ across ranks")
        self.initial_all_rank_weights = gathered
        self.run_id, self.source_revision = inspection["run_id"], inspection["source_revision"]
        self.checkpoint_sha256 = inspection["checkpoint"]["sha256"]
        from scripts.clusters.pace.gpu_admission import cuda_uuid
        self.gpu_uuid = cuda_uuid(torch.cuda.get_device_properties(pl_module.device))
        self.fp32_losses = self.bf16_outputs = self.gradients_checked = 0
        self.optimizer_update_lrs, self.validation_steps = [], []
        self.rng_restore_observed = False
        self._install_rng_observer(pl_module, saved["mot_rng_state"]["ranks"][trainer.global_rank])
        del saved
        self.output.mkdir(parents=True, exist_ok=True)

        def observe(module, args, outputs):
            if not pl_module.training or not torch.is_grad_enabled() or self.done:
                return
            need(isinstance(outputs, dict) and set(outputs) == set(self.network.streams)
                 and all(value.dtype == torch.bfloat16 and torch.isfinite(value).all()
                         for value in outputs.values()),
                 "Actual resumed active-stream forward must be finiteBF16")
            self.bf16_outputs += 1
        self.precision_hook = self.network.register_forward_hook(observe)

    def _install_rng_observer(self, module, expected):
        implementation = importlib.import_module(type(module.training_behavior).__module__)
        need(implementation.__name__ == "egomimic.pl_utils.world_model_behavior",
             "Unexpected scientific RNG restoration implementation")
        original = implementation._restore_rng_state
        need(not getattr(original, "_crash_restore_observer", False), "Duplicate RNG restore observer")
        self._rng_module, self._rng_original = implementation, original

        @functools.wraps(original)
        def observe(state):
            need(equal_state(state, expected), "Native RNG selected another saved rank")
            original(state)
            actual = implementation._capture_rng_state(current_cuda_only=True)
            need(equal_state(actual, expected), "Actual native RNG state was not exactly restored")
            self.rng_restore_observed = True
            self._remove_rng_observer()
        observe._crash_restore_observer = True
        self._rng_observer = observe
        implementation._restore_rng_state = observe

    def _remove_rng_observer(self):
        if self._rng_module is not None:
            need(self._rng_module._restore_rng_state is self._rng_observer,
                 "Scientific RNG restore function changed while observed")
            self._rng_module._restore_rng_state = self._rng_original
            self._rng_module = self._rng_original = self._rng_observer = None

    def on_before_backward(self, trainer, pl_module, loss):
        if self.done:
            return
        need(self.rng_restore_observed and pl_module.training_behavior._pending_rng is None,
             "Actual native per-rank RNG restore was not observed before training")
        need(loss.dtype == torch.float32 and bool(torch.isfinite(loss)),
             "Actual resumed MSE must be finiteFP32")
        self.fp32_losses += 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.done:
            return
        need(optimizer is self.optimizer and all(group["lr"] == 1e-4
             and group["weight_decay"] == .01 for group in optimizer.param_groups),
             "Actual resumed optimizer policy changed")
        parameters = list(self.network.parameters())
        need(all(p.grad is not None and p.grad.dtype == torch.float32
                 and torch.isfinite(p.grad).all() for p in parameters),
             "Every resumed parameter needs a finiteFP32 gradient")
        self.gradients_checked += 1
        self.optimizer_update_lrs.append(1e-4)

    def on_validation_start(self, trainer, pl_module):
        need(not trainer.sanity_checking and trainer.global_step > self.start_step,
             "Crash resume replayed boundary/sanity validation")
        need(trainer.global_step % self.validation_interval == 0,
             "Crash resume changed validation cadence")
        need(trainer.global_step not in self.validation_steps, "Crash resume repeated validation")
        self.validation_steps.append(trainer.global_step)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Observe every native save, including ones before the early proof ends."""
        step = trainer.global_step
        need(type(step) is int and type(checkpoint["global_step"]) is int
             and checkpoint["global_step"] == step
             and step > self.start_step, "Crash resume replayed a boundary checkpoint save")
        expected = step == self.start_step + self.proof_updates if self.diagnostic else step % 20000 == 0
        need(expected, "Crash resume attempted an extra off-cadence checkpoint save")
        need(step not in self.observed_saves, "Crash resume repeated a checkpoint save")
        self.observed_saves.append(step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.done:
            return
        count = trainer.global_step - self.start_step
        need(0 <= count <= self.proof_updates, "Skipped actual crash-restore proof boundary")
        if count != self.proof_updates:
            return
        need(count == self.fp32_losses == self.bf16_outputs == self.gradients_checked
             == len(self.optimizer_update_lrs)
             and trainer.datamodule.consumed_batches == trainer.global_step
             and all(float(value["step"]) == trainer.global_step
                     for value in self.optimizer.state.values())
             and not trainer.lr_scheduler_configs and not self.validation_steps,
             "Missing actual optimizer/precision/sampler proof or repeated validation")
        final = weights_digest(pl_module.state_dict())
        gathered = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, final)
        need(len(set(gathered)) == 1 and final != self.initial_weights,
             "Restored parameters did not update equally on all ranks")
        record = dict(
            schema="joint-crash-restore-rank-v1", status="PASS",
            job_id=os.environ["SLURM_JOB_ID"], rank=trainer.global_rank,
            run_id=self.run_id, source_revision=self.source_revision,
            gpu_uuid=self.gpu_uuid, diagnostic=self.diagnostic,
            request_sha256=self.request_sha256,
            world_size=self.world_size, start_step=self.start_step,
            global_step=trainer.global_step, actual_updates=count,
            inspection_sha256=self.inspection_binding["sha256"],
            checkpoint=self.checkpoint, checkpoint_sha256=self.checkpoint_sha256,
            strict_weights_optimizer_sampler_rng=True,
            actual_native_rank_rng_restored=True, fp32_losses=self.fp32_losses,
            finite_bf16_outputs=self.bf16_outputs, finite_fp32_gradients=self.gradients_checked,
            optimizer_update_lrs=self.optimizer_update_lrs,
            initial_weights_sha256=self.initial_weights, initial_all_rank_weights=self.initial_all_rank_weights,
            weights_sha256=final, all_rank_weights=gathered,
            no_extra_save=True, no_replayed_validation=True,
            save_cadence_guard_installed=True, observed_saves_through_proof=list(self.observed_saves),
        )
        write_once(self.output / f"restore-step{self.start_step:09d}-rank{trainer.global_rank}.json", record)
        self.done = True
        self._remove_hooks()

    def _remove_hooks(self):
        if self.precision_hook is not None:
            self.precision_hook.remove()
            self.precision_hook = None
        self._remove_rng_observer()

    def on_exception(self, trainer, pl_module, exception):
        self._remove_hooks()

    def on_train_end(self, trainer, pl_module):
        self._remove_hooks()
        need(self.done, "Training ended before actual crash restoration was proved")
        if self.diagnostic:
            expected = self.start_step + self.proof_updates
            evaluator = pl_module.evaluator
            metrics = {str(key): float(value) for key, value in trainer.callback_metrics.items()
                       if str(key).startswith("val/libero_gripper/")}
            need(trainer.global_step == expected and self.validation_steps == [expected]
                 and self.observed_saves == [expected]
                 and evaluator.has_completed_validation(expected)
                 and evaluator.global_validation_batches == 100
                 and evaluator._validation_batches == 100 // self.world_size
                 and len(metrics) == 2 * len(self.network.streams)
                 and all(math.isfinite(value) for value in metrics.values()),
                 "Actual recovery diagnostic did not finish all100 scheduled validation anchors")
            write_once(self.output / f"restore-validation-rank{trainer.global_rank}.json", dict(
                schema="joint-crash-restore-validation-rank-v1", status="PASS",
                job_id=os.environ["SLURM_JOB_ID"], rank=trainer.global_rank,
                world_size=self.world_size, start_step=self.start_step,
                global_step=expected, validation_steps=self.validation_steps,
                global_validation_batches=100, local_validation_batches=100 // self.world_size,
                inspection_sha256=self.inspection_binding["sha256"], metrics=metrics,
                observed_saves=self.observed_saves,
                actual_scheduled_validation_after_optimizer=True,
            ))


def validate_rank_proofs(output, *, inspection, inspection_sha256, start_step,
                         world_size=4, expected_job_id=None, diagnostic=False,
                         expected_request_sha256=None, expected_updates=2):
    """Verify actual per-rank artifacts; synthetic tests never create runtime PASS."""
    import json

    rows = []
    directory = Path(output)
    for rank in range(world_size):
        path = directory / f"restore-step{start_step:09d}-rank{rank}.json"
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2_000_000,
             "Missing immutable actual rank restore proof")
        row = json.loads(path.read_text())
        need(row["schema"] == "joint-crash-restore-rank-v1" and row["status"] == "PASS"
             and row["rank"] == rank and row["world_size"] == world_size == 4
             and row["start_step"] == start_step
             and row["global_step"] == start_step + row["actual_updates"]
             and row["actual_updates"] == expected_updates
             and row["inspection_sha256"] == inspection_sha256
             and row["checkpoint"] == inspection["checkpoint"]["path"]
             and row["checkpoint_sha256"] == inspection["checkpoint"]["sha256"]
             and row["run_id"] == inspection["run_id"]
             and row["source_revision"] == inspection["source_revision"]
             and row["diagnostic"] is diagnostic,
             "Actual rank proof differs from the current inspected recovery")
        count = row["actual_updates"]
        need(row["strict_weights_optimizer_sampler_rng"] is True
             and row["actual_native_rank_rng_restored"] is True
             and row["fp32_losses"] == row["finite_bf16_outputs"]
             == row["finite_fp32_gradients"] == count
             and row["optimizer_update_lrs"] == [1e-4] * count
             and row["no_extra_save"] is True and row["no_replayed_validation"] is True
             and row["save_cadence_guard_installed"] is True
             and (row["observed_saves_through_proof"] in ([], [start_step + expected_updates])
                  if diagnostic else row["observed_saves_through_proof"] == []),
             "Actual optimizer/precision/RNG proof is incomplete")
        need(expected_job_id is None or str(row["job_id"]) == str(expected_job_id),
             "Restore proof belongs to another allocation")
        need(expected_request_sha256 is None
             or row["request_sha256"] == expected_request_sha256,
             "Restore proof belongs to another request")
        rows.append(row)
    need(len({str(row["job_id"]) for row in rows}) == 1
         and len({canonical_uuid(row["gpu_uuid"]) for row in rows}) == world_size,
         "Restore ranks do not share one allocation with unique physical GPUs")
    initial = [row["initial_weights_sha256"] for row in rows]
    final = [row["weights_sha256"] for row in rows]
    need(len(set(initial)) == len(set(final)) == 1 and initial[0] != final[0]
         and all(row["initial_all_rank_weights"] == initial
                 and row["all_rank_weights"] == final for row in rows)
         and len({row["actual_updates"] for row in rows}) == 1,
         "Actual restored and updated all-rank weights disagree")
    if diagnostic:
        for rank, row in enumerate(rows):
            value = json.loads((directory / f"restore-validation-rank{rank}.json").read_text())
            need(value["schema"] == "joint-crash-restore-validation-rank-v1"
                 and value["status"] == "PASS" and value["rank"] == rank
                 and value["world_size"] == world_size
                 and str(value["job_id"]) == str(row["job_id"])
                 and value["start_step"] == start_step
                 and value["global_step"] == start_step + expected_updates
                 and value["validation_steps"] == [start_step + expected_updates]
                 and value["global_validation_batches"] == 100
                 and value["local_validation_batches"] == 100 // world_size
                 and value["inspection_sha256"] == inspection_sha256
                 and value["actual_scheduled_validation_after_optimizer"] is True
                 and value["observed_saves"] == [start_step + expected_updates]
                 and value["metrics"] and all(math.isfinite(float(v))
                                              for v in value["metrics"].values()),
                 "Missing actual all-rank scheduled recovery validation")
    return rows


def assert_actual_restore(request, *, expected_job_id=None, expected_request_sha256=None,
                          diagnostic=False):
    """Production postflight entry point used before full completion is written."""
    recovery = request["crash_recovery"]
    inspection = bound(recovery["inspection"])
    need(inspection["schema"] == "joint-crash-checkpoint-inspection-v1"
         and inspection["status"] == "PASS"
         and inspection["run_id"] == request["run_id"]
         and inspection["source_revision"] == request["source_revision"],
         "Wrong crash inspection for actual restoration proof")
    return validate_rank_proofs(
        Path(request["output"]) / "restore-proof", inspection=inspection,
        inspection_sha256=recovery["inspection"]["sha256"],
        start_step=inspection["global_step"], world_size=request["world_size"],
        expected_job_id=(os.environ.get("SLURM_JOB_ID")
                         if expected_job_id is None else expected_job_id),
        diagnostic=diagnostic, expected_request_sha256=expected_request_sha256)
