"""Original Action Flow losses with the released OAT DP optimizer recipe."""

import torch

from egomimic.pipeline.stages_oat import OATObservationStage
from egomimic.pl_utils.training_behavior_action_flow import ActionFlowTrainingBehavior


class OATMatchedActionFlowTrainingBehavior(ActionFlowTrainingBehavior):
    """Preserve the Action Flow training path; partition only optimizer groups."""

    def __init__(self, policy_lr=5e-5, obs_enc_lr=1e-5, betas=(0.9, 0.95), **kwargs):
        super().__init__(**kwargs)
        self.policy_lr = float(policy_lr)
        self.obs_enc_lr = float(obs_enc_lr)
        self.betas = tuple(betas)

    def configure_optimizers(self):
        stages = self.context.model.pipeline.stages
        observations = [s for s in stages if isinstance(s, OATObservationStage)]
        if len(observations) != 1:
            raise ValueError("Matched comparison requires one released observation encoder")
        observation_parameters = list(observations[0].encoder.parameters())
        observation_ids = {id(p) for p in observation_parameters}
        all_parameters = list(self.context.nets.parameters())
        policy_parameters = [p for p in all_parameters if id(p) not in observation_ids]
        if not observation_parameters or not policy_parameters:
            raise ValueError("Both released learning-rate groups must be nonempty")
        grouped_ids = [id(p) for p in observation_parameters + policy_parameters]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {
            id(p) for p in all_parameters
        }:
            raise ValueError("Optimizer groups must cover every parameter exactly once")
        # The upstream 'constant' schedule ignores its warmup field.
        return {
            "optimizer": torch.optim.AdamW(
                [
                    {"params": policy_parameters, "lr": self.policy_lr},
                    {"params": observation_parameters, "lr": self.obs_enc_lr},
                ],
                betas=self.betas,
                weight_decay=0.0,
            )
        }
