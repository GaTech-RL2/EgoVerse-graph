"""Qwen pooling helpers: no HuggingFace download."""

import torch

from egomimic.models.stems.text_encoders import qwen_last_token_pool


def test_qwen_last_token_pool_left_padding():
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4, 1)
    # Both rows end on a real token -> left-padded recipe uses [:, -1].
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.long)
    pooled = qwen_last_token_pool(hidden, mask)
    assert pooled.squeeze(-1).tolist() == [3.0, 7.0]


def test_qwen_last_token_pool_right_padding():
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4, 1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
    pooled = qwen_last_token_pool(hidden, mask)
    assert pooled.squeeze(-1).tolist() == [1.0, 6.0]
