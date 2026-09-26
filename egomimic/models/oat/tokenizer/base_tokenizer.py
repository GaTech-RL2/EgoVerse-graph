import torch

from egomimic.models.oat.model.common.module_attr_mixin import ModuleAttrMixin


class BaseTokenizer(ModuleAttrMixin):
    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        from egomimic.models.oat.factory import load_tokenizer

        return load_tokenizer(checkpoint, **kwargs)

    def get_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError

    def set_normalizer(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def autoencode(self, *args, **kwargs):
        raise NotImplementedError

    def tokenize(self, *args, **kwargs):
        raise NotImplementedError

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
