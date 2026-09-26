from typing import Dict, List, Union

from egomimic.models.oat.model.common.module_attr_mixin import ModuleAttrMixin
from egomimic.models.oat.model.common.normalizer import LinearNormalizer


class BaseObservationEncoder(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    def forward(self, obs_dict: Union[Dict, List[Dict]]) -> Dict:
        raise NotImplementedError

    def modalities(self) -> List[str]:
        raise NotImplementedError

    def output_feature_dim(self) -> int:
        raise NotImplementedError

    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError
