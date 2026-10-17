"""Optimizer/checkpoint policy for native OAT/ARC benchmark graphs."""

import json
from pathlib import Path

import torch
from lightning import Callback

from egomimic.pl_utils.training_behavior import TrainingBehavior
from egomimic.utils.ema_callback import EMACallback


class EarlyRecoveryCheckpoint(Callback):
    """Keep an additional immutable full-state recovery save before normal cadence.

    Extracted from the verified native LIBERO DurableCheckpointGuard; scheduled
    cadence saves remain owned by Lightning ModelCheckpoint. No signal is sent.
    """

    def __init__(self, output, first_step=1000):
        self.output = Path(output)
        self.first_step = int(first_step)
        if not self.output.is_absolute() or self.first_step <= 0:
            raise ValueError(
                "Recovery save requires an absolute output and positive step"
            )
        self.done = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del pl_module, outputs, batch, batch_idx
        step = int(trainer.global_step)
        if self.done or step != self.first_step:
            return
        if int(trainer.world_size) != 1:
            raise ValueError(
                "Early recovery publication currently requires world size one"
            )
        path = (
            self.output
            / "checkpoints"
            / (f"recovery-epoch-{int(trainer.current_epoch):04d}-step-{step:09d}.ckpt")
        )
        if path.exists():
            raise FileExistsError("Refusing to overwrite an early recovery checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(path), weights_only=False)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if (
            payload.get("global_step") != step
            or payload.get("ema_num_updates") != step
            or not payload.get("optimizer_states")
            or not payload.get("lr_schedulers")
            or not payload.get("data_context")
        ):
            raise ValueError(
                "Early recovery checkpoint lacks exact full training state"
            )

        def finite(value):
            if torch.is_tensor(value):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(item) for item in value)
            if isinstance(value, float):
                return bool(torch.isfinite(torch.tensor(value)))
            return True

        if not finite(payload):
            raise ValueError("Nonfinite early recovery training state")
        from egomimic.utils.slurm_requeue import _atomic_json_no_overwrite, _sha256

        _atomic_json_no_overwrite(
            path.with_suffix(".ckpt.recovery.json"),
            {
                "status": "PASS_DURABLE_STATE",
                "global_step": step,
                "ema_num_updates": step,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "full_state": True,
            },
        )
        self.done = True


class OATBatchBudgetCallback(Callback):
    """Match the released four-rank, drop-last optimizer budget on any device count."""

    def __init__(
        self, global_batch_size=1024, report_path=None, expected_source_batches=None
    ):
        self.global_batch_size = int(global_batch_size)
        self.report_path = report_path
        self.expected_source_batches = (
            None
            if expected_source_batches is None
            else {
                str(name): int(size) for name, size in expected_source_batches.items()
            }
        )
        if self.expected_source_batches is not None and (
            not self.expected_source_batches
            or any(size <= 0 for size in self.expected_source_batches.values())
        ):
            raise ValueError("Every source batch must be positive")
        self.budget = None

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        datasets = trainer.datamodule.train_datasets
        if self.expected_source_batches is not None:
            self._configure_joint_budget(trainer, datasets)
            return
        if len(datasets) != 1:
            raise ValueError("OAT budget requires exactly one replay training dataset")
        name, dataset = next(iter(datasets.items()))
        params = trainer.datamodule.train_dataloader_params[name]
        microbatch = int(params["batch_size"])
        accumulation = int(trainer.accumulate_grad_batches)
        effective = microbatch * int(trainer.world_size) * accumulation
        if effective != self.global_batch_size or not params.get("drop_last", False):
            raise ValueError("OAT requires its configured global batch and drop_last")
        if trainer.limit_train_batches != 1.0:
            raise ValueError(
                "Disable the OAT batch budget callback for limited smoke tests"
            )
        steps = len(dataset) // effective
        if not steps:
            raise ValueError("Training replay is smaller than one global batch")
        # Accelerate's four-rank BatchSamplerShard drops an incomplete GLOBAL
        # batch. Lightning otherwise takes a final partial accumulation step.
        trainer.limit_train_batches = steps * accumulation
        epoch_limit = int(trainer.max_epochs)
        step_limit = int(getattr(trainer, "max_steps", -1))
        limits = [value for value in (steps * epoch_limit, step_limit) if value > 0]
        if not limits:
            raise ValueError("OAT requires a finite positive epoch or step budget")
        total_steps = min(limits)
        self.budget = {
            "train_examples": len(dataset),
            "microbatch_size": microbatch,
            "world_size": int(trainer.world_size),
            "gradient_accumulation": accumulation,
            "global_batch_size": effective,
            "optimizer_steps_per_epoch": steps,
            "microbatches_per_epoch": steps * accumulation,
            "epochs": epoch_limit,
            "max_steps": step_limit,
            "total_optimizer_steps": total_steps,
            "dropped_examples_per_epoch": len(dataset) % effective,
        }
        if trainer.is_global_zero:
            encoded = json.dumps(self.budget, indent=2) + "\n"
            print("OAT_TRAINING_BUDGET", encoded, flush=True)
            if self.report_path:
                path = Path(self.report_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(encoded)

    def _configure_joint_budget(self, trainer, datasets):
        """Preserve fixed per-source participation in max-size cycling loaders."""
        if set(datasets) != set(self.expected_source_batches):
            raise ValueError("Joint budget source identities differ from datasets")
        if trainer.datamodule.train_loader_mode != "max_size_cycle":
            raise ValueError("Fixed joint batches require max_size_cycle")
        if trainer.limit_train_batches != 1.0:
            raise ValueError(
                "Disable the batch budget callback for limited smoke tests"
            )
        world = int(trainer.world_size)
        accumulation = int(trainer.accumulate_grad_batches)
        sources = {}
        for name, size in self.expected_source_batches.items():
            params = trainer.datamodule.train_dataloader_params[name]
            if int(params["batch_size"]) != size or not params.get("drop_last", False):
                raise ValueError(
                    f"Joint source {name} must preserve batch size and drop_last"
                )
            effective = size * world * accumulation
            # Lightning's distributed sampler pads to equal rank lengths.
            rank_examples = (len(datasets[name]) + world - 1) // world
            microbatches = rank_examples // size
            if microbatches < accumulation:
                raise ValueError(
                    f"Joint source {name} is smaller than one optimizer batch"
                )
            sources[name] = {
                "train_examples": len(datasets[name]),
                "microbatch_size": size,
                "global_batch_size": effective,
                "available_microbatches": microbatches,
            }
        effective = sum(s["global_batch_size"] for s in sources.values())
        if effective != self.global_batch_size:
            raise ValueError("Joint source batches differ from configured global batch")
        microbatches = max(s["available_microbatches"] for s in sources.values())
        steps = microbatches // accumulation
        trainer.limit_train_batches = steps * accumulation
        epoch_limit = int(trainer.max_epochs)
        step_limit = int(getattr(trainer, "max_steps", -1))
        limits = [value for value in (steps * epoch_limit, step_limit) if value > 0]
        if not limits:
            raise ValueError("Joint budget requires a finite epoch or step budget")
        self.budget = {
            "sources": sources,
            "loader_mode": "max_size_cycle",
            "world_size": world,
            "gradient_accumulation": accumulation,
            "global_batch_size": effective,
            "optimizer_steps_per_epoch": steps,
            "microbatches_per_epoch": steps * accumulation,
            "epochs": epoch_limit,
            "max_steps": step_limit,
            "total_optimizer_steps": min(limits),
        }
        if trainer.is_global_zero:
            encoded = json.dumps(self.budget, indent=2) + "\n"
            print("OAT_TRAINING_BUDGET", encoded, flush=True)
            if self.report_path:
                path = Path(self.report_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(encoded)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        del trainer, pl_module
        checkpoint["training_budget"] = self.budget


class OATTrainingBehavior(TrainingBehavior):
    def __init__(
        self, learning_rate=5e-5, obs_enc_lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)
    ):
        super().__init__()
        self.learning_rate, self.obs_enc_lr = learning_rate, obs_enc_lr
        self.weight_decay, self.betas = weight_decay, tuple(betas)

    def configure_optimizers(self):
        from egomimic.models.oat.checkpoint import validate_input_representation
        from egomimic.pipeline.stages_oat import OATPolicyStage, OATTokenizerStage

        validate_input_representation(self.context.model.pipeline.stages)

        for stage in self.context.model.pipeline.stages:
            if isinstance(stage, OATPolicyStage):
                return stage.policy.get_optimizer(
                    self.learning_rate,
                    self.obs_enc_lr,
                    self.weight_decay,
                    self.betas,
                )
            if isinstance(stage, OATTokenizerStage):
                return stage.tokenizer.get_optimizer(
                    self.learning_rate, self.weight_decay, self.betas
                )
        # Continuous ARC head: same observation learning rate and AdamW groups.
        groups = {}
        for name, parameter in self.context.nets.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = self.obs_enc_lr if ".encoder." in name else self.learning_rate
            decay = self.weight_decay if parameter.ndim >= 2 else 0.0
            groups.setdefault((lr, decay), []).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": params, "lr": lr, "weight_decay": decay}
                for (lr, decay), params in groups.items()
            ],
            betas=self.betas,
        )

    def on_save_checkpoint(self, checkpoint):
        from egomimic.models.oat.checkpoint import validate_input_representation

        # Lightning skips callback save hooks for weights-only checkpoints.
        # Preserve EMA *weights* and budget provenance without optimizer state.
        # Full-state saves already ran these hooks; never run them twice (in
        # particular while validation has temporarily swapped EMA parameters).
        if "optimizer_states" not in checkpoint:
            trainer = self.context.trainer
            for callback in trainer.callbacks:
                if isinstance(callback, (OATBatchBudgetCallback, OATEMACallback)):
                    callback.on_save_checkpoint(trainer, self.context, checkpoint)
        checkpoint["oat_input_representation"] = validate_input_representation(
            self.context.model.pipeline.stages
        )
        stages = [
            stage
            for stage in self.context.model.pipeline.stages
            if hasattr(stage, "normalizer_state")
        ]
        if not stages:
            raise RuntimeError(
                "Benchmark checkpoint requires bound dataset normalization"
            )
        checkpoint["normalizer_state"] = stages[0].normalizer_state
        checkpoint["benchmark_data_context"] = stages[0].data_context
        from egomimic.pipeline.stages_oat import OATPolicyStage

        for stage in self.context.model.pipeline.stages:
            if isinstance(stage, OATPolicyStage):
                checkpoint["oat_tokenizer_config"] = (
                    stage.policy.action_tokenizer._native_config
                )

    def on_load_checkpoint(self, checkpoint):
        from egomimic.models.oat.checkpoint import validate_input_representation

        validate_input_representation(self.context.model.pipeline.stages, checkpoint)
        reference = checkpoint.get("normalizer_state", {}).get("benchmark_context")
        for stage in self.context.model.pipeline.stages:
            if hasattr(stage, "normalizer_state"):
                if stage.normalizer_state.get("benchmark_context") != reference:
                    raise ValueError(
                        "Cannot load with a different benchmark dataset/split/observations"
                    )


class OATDiffusionTrainingBehavior(OATTrainingBehavior):
    """Keep the released DP optimizer groups, including its condition encoder."""

    def configure_optimizers(self):
        from egomimic.models.oat.diffusion import GraphDiffusionTransformer
        from egomimic.pipeline.stages_diffusion import DiffusionDenoiserStage
        from egomimic.pipeline.stages_oat import OATObservationStage

        stages = self.context.model.pipeline.stages
        observations = [s for s in stages if isinstance(s, OATObservationStage)]
        denoisers = [s for s in stages if isinstance(s, DiffusionDenoiserStage)]
        if len(observations) != 1 or len(denoisers) != 1:
            raise ValueError(
                "Released DP requires one observation encoder and denoiser"
            )
        model = denoisers[0].policy.model
        if not isinstance(model, GraphDiffusionTransformer):
            raise TypeError("Released DP optimizer requires its diffusion Transformer")
        groups = model.get_optim_groups(
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        groups.append(
            {
                "params": observations[0].encoder.parameters(),
                "lr": self.obs_enc_lr,
                "weight_decay": self.weight_decay,
            }
        )
        return torch.optim.AdamW(groups, betas=self.betas)


class OATEMACallback(EMACallback):
    """Use the upstream zero-based EMA update counter with shared checkpoint I/O."""

    def __init__(self, final_checkpoint_path=None, **kwargs):
        super().__init__(**kwargs)
        self.final_checkpoint_path = final_checkpoint_path

    def _schedule(self, update):
        return super()._schedule(update - 1)

    def on_train_end(self, trainer, pl_module):
        # The released budget (5001 epochs) is not divisible by the ten-epoch
        # checkpoint cadence. Persist the actual final optimizer/EMA state.
        if self.final_checkpoint_path is not None:
            trainer.save_checkpoint(
                self.final_checkpoint_path.format(
                    epoch=trainer.current_epoch, step=trainer.global_step
                )
            )
