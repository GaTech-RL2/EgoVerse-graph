"""OAT's FAST adapter with an embedded, pinned processor instead of remote code."""

import os
from copy import deepcopy

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from egomimic.models.oat.factory import identity_normalizer
from egomimic.models.oat.tokenizer.base_tokenizer import BaseTokenizer
from egomimic.models.oat.tokenizer.fast.processing_action_tokenizer import (
    UniversalActionProcessor,
)

PROCESSOR_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FASTTok(BaseTokenizer):
    def __init__(self, processor_config, training_data_context=None):
        super().__init__()
        config = deepcopy(dict(processor_config))
        if config.pop("revision") != PROCESSOR_REVISION:
            raise ValueError("FAST processor revision differs from the pinned source")
        bpe = Tokenizer.from_str(config.pop("bpe_json"))
        if (
            config["scale"] <= 0
            or config["time_horizon"] < 1
            or config["action_dim"] < 1
            or not 0 < bpe.get_vocab_size() <= config["vocab_size"]
        ):
            raise ValueError("Invalid fitted FAST processor")
        self.fast_tok = UniversalActionProcessor(
            PreTrainedTokenizerFast(
                tokenizer_object=bpe, clean_up_tokenization_spaces=False
            ),
            **config,
        )
        self.vocab_size = self.fast_tok.vocab_size
        # MultiDataset applies the real affine once, as for native OAT.
        self.normalizer = identity_normalizer(["action"])
        self._native_config = {
            "processor_config": deepcopy(dict(processor_config)),
            "training_data_context": deepcopy(training_data_context),
        }
        self._training_data_context = deepcopy(training_data_context)

    @classmethod
    def from_processor(cls, processor, training_data_context=None):
        return cls(
            {
                "revision": PROCESSOR_REVISION,
                "bpe_json": processor.bpe_tokenizer.backend_tokenizer.to_str(),
                **{
                    key: getattr(processor, key)
                    for key in (
                        "scale",
                        "vocab_size",
                        "min_token",
                        "time_horizon",
                        "action_dim",
                    )
                },
            },
            training_data_context=training_data_context,
        )

    def tokenize(self, samples):
        nsamples = self.normalizer["action"].normalize(samples)
        return self.fast_tok(nsamples.detach().float().cpu().numpy())

    def detokenize(self, tokens, horizon=None, dim=None):
        nsamples = torch.from_numpy(
            self.fast_tok.decode(tokens, time_horizon=horizon, action_dim=dim)
        ).to(dtype=torch.float32, device=self.device)
        return self.normalizer["action"].unnormalize(nsamples)
