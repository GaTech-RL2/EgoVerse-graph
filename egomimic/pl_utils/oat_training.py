"""Optimizer/checkpoint policy for native OAT/ARC benchmark graphs."""

import torch

from egomimic.pl_utils.training_behavior import TrainingBehavior
from egomimic.utils.ema_callback import EMACallback


class OATTrainingBehavior(TrainingBehavior):
    def __init__(
        self, learning_rate=5e-5, obs_enc_lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)
    ):
        super().__init__()
        self.learning_rate, self.obs_enc_lr = learning_rate, obs_enc_lr
        self.weight_decay, self.betas = weight_decay, tuple(betas)

    def configure_optimizers(self):
        from egomimic.pipeline.stages_oat import OATPolicyStage, OATTokenizerStage

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
        reference = checkpoint.get("normalizer_state", {}).get("benchmark_context")
        for stage in self.context.model.pipeline.stages:
            if hasattr(stage, "normalizer_state"):
                if stage.normalizer_state.get("benchmark_context") != reference:
                    raise ValueError(
                        "Cannot load with a different benchmark dataset/split/observations"
                    )


class OATEMACallback(EMACallback):
    """Use the upstream zero-based EMA update counter with shared checkpoint I/O."""

    def _schedule(self, update):
        return super()._schedule(update - 1)
