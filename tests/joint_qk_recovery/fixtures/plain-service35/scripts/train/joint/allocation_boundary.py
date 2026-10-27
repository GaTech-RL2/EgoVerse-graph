"""End an allocation at a scheduled validation boundary without changing its budget.

The existing Lightning full-state loader owns continuation. This callback only
requests a graceful stop and publishes an additional recovery checkpoint after
validation, so restoring it also restores the post-validation RNG state. It
never submits jobs, signals supervisors, removes checkpoints or edits max_steps.
"""

from pathlib import Path

from lightning import Callback

from scripts.train.joint.contract import need, sha, write_once


def check_interval(start_step, stop_step, target_step, cadence):
    values = (start_step, stop_step, target_step, cadence)
    need(all(type(v) is int for v in values), "Integer allocation steps required")
    need(
        cadence > 0 and 0 <= start_step < stop_step < target_step
        and start_step % cadence == stop_step % cadence == target_step % cadence == 0,
        "Allocation boundary must preserve the scheduled cadence and final budget",
    )


class ScheduledAllocationBoundary(Callback):
    def __init__(self, *, start_step, stop_step, target_step, cadence, output):
        check_interval(start_step, stop_step, target_step, cadence)
        self.start_step, self.stop_step = start_step, stop_step
        self.target_step, self.cadence = target_step, cadence
        self.output = Path(output)
        need(self.output.is_absolute(), "Absolute allocation evidence path required")
        self.receipt = self.output / f"allocation-step{stop_step:09d}.json"

    def on_train_start(self, trainer, pl_module):
        need(trainer.max_steps == self.target_step, "Scientific budget changed")
        need(trainer.global_step == self.start_step, "Wrong restored optimizer step")
        need(
            trainer.accumulate_grad_batches == 1
            and trainer.val_check_interval == self.cadence
            and trainer.check_val_every_n_epoch is None
            and not trainer.sanity_checking,
            "Allocation must retain the configured optimizer/validation cadence",
        )
        checkpoint = trainer.checkpoint_callback
        need(
            checkpoint is not None
            and checkpoint._every_n_train_steps == self.cadence
            and checkpoint.save_top_k == -1
            and checkpoint.save_last == "link",
            "Immutable scheduled-checkpoint retention is required",
        )
        need(not self.receipt.exists(), "Allocation boundary already completed")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        need(trainer.global_step <= self.stop_step, "Allocation overshot its boundary")
        if trainer.global_step == self.stop_step:
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        need(trainer.global_step == self.stop_step, "Allocation ended early")
        need(
            pl_module.evaluator.has_completed_validation(self.stop_step),
            "Boundary validation must complete before recovery publication",
        )
        need(
            trainer.datamodule.consumed_batches == self.stop_step,
            "Sampler cursor differs from completed optimizer updates",
        )
        self.output.mkdir(parents=True, exist_ok=True)
        recovery = self.output / (
            f"allocation-recovery-epoch{trainer.current_epoch:04d}"
            f"-step{self.stop_step:09d}.ckpt"
        )
        need(not recovery.exists(), "Recovery checkpoint must never be overwritten")
        # All ranks participate in the native distributed RNG checkpoint hook.
        trainer.save_checkpoint(recovery)
        trainer.strategy.barrier("allocation-recovery-published")
        if trainer.is_global_zero:
            write_once(self.receipt, dict(
                schema="joint-allocation-boundary-v1", status="PASS",
                start_step=self.start_step, global_step=self.stop_step,
                target_step=self.target_step, cadence=self.cadence,
                world_size=trainer.world_size, validation_completed=True,
                consumed_batches=trainer.datamodule.consumed_batches,
                checkpoint=dict(path=str(recovery), sha256=sha(recovery),
                                size_bytes=recovery.stat().st_size),
                checkpoint_saved_after_validation=True,
            ))
        trainer.strategy.barrier("allocation-receipt-published")
