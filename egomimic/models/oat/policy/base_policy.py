from typing import Dict, List, Optional, Tuple, Union

import torch

from egomimic.models.oat.model.common.module_attr_mixin import ModuleAttrMixin
from egomimic.models.oat.model.common.normalizer import LinearNormalizer


class BasePolicy(ModuleAttrMixin):
    n_obs_steps: int
    n_action_steps: int

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        from egomimic.benchmarks.libero.rollout import load_policy

        return load_policy(checkpoint, **kwargs)

    def get_optimizer(self, *args, **kwargs):
        return torch.optim.AdamW(self.parameters(), *args, **kwargs)

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(
        self, normalizer: Union[LinearNormalizer, List[LinearNormalizer]]
    ):
        raise NotImplementedError()

    def get_observation_encoder(self):
        raise NotImplementedError()

    def get_observation_modalities(self) -> List[str]:
        raise NotImplementedError()

    def get_observation_ports(self) -> List[str]:
        raise NotImplementedError()

    def get_policy_name(self) -> str:
        raise NotImplementedError()

    def create_dummy_observation(
        self,
        batch_size: int,
        horizon: int,
        obs_key_shapes: Dict[str, Tuple[int]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        for obs_port, obs_shape in obs_key_shapes.items():
            obs_dict[obs_port] = torch.randn(
                size=(batch_size, horizon, *obs_shape),
            ).to(device)
        return obs_dict
