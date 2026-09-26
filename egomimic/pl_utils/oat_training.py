"""Optimizer/checkpoint policy for native OAT/ARC benchmark graphs."""

import json
from pathlib import Path

import torch
from lightning import Callback

from egomimic.pl_utils.training_behavior import TrainingBehavior
from egomimic.utils.ema_callback import EMACallback


class OATBatchBudgetCallback(Callback):
    """Match the released four-rank, drop-last optimizer budget on any device count."""

    def __init__(self, global_batch_size=1024, report_path=None):
        self.global_batch_size = int(global_batch_size)
        self.report_path = report_path
        self.budget = None

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        datasets = trainer.datamodule.train_datasets
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
        self.budget = {
            "train_examples": len(dataset),
            "microbatch_size": microbatch,
            "world_size": int(trainer.world_size),
            "gradient_accumulation": accumulation,
            "global_batch_size": effective,
            "optimizer_steps_per_epoch": steps,
            "microbatches_per_epoch": steps * accumulation,
            "epochs": int(trainer.max_epochs),
            "total_optimizer_steps": steps * int(trainer.max_epochs),
            "dropped_examples_per_epoch": len(dataset) % effective,
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
        from egomimic.pipeline.stages_oat import OATPolicyStage, OATTokenizerStage
        from egomimic.models.oat.checkpoint import validate_input_representation

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
            trainer.save_checkpoint(self.final_checkpoint_path)
