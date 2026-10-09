import pytest
import torch

from egomimic.models.oat.tokenizer.oat.quantizer.fsq import FSQ
from scripts.audit_components import offline_static_buffer_construction


@pytest.mark.parametrize("levels", [(8, 5, 5, 5), (3, 4)])
def test_static_lookup_buffers_match_real_constructor_and_rng(levels):
    original = FSQ.__init__
    baseline = FSQ(list(levels))
    before = torch.get_rng_state().clone()
    with offline_static_buffer_construction():
        candidate = FSQ(list(levels))
        with torch.device("meta"):
            meta = FSQ(list(levels))
    assert FSQ.__init__ is original
    assert torch.equal(before, torch.get_rng_state())
    assert candidate.codebook_size == meta.codebook_size == baseline.codebook_size
    assert not tuple(candidate.parameters())
    for name, value in baseline.named_buffers():
        actual = dict(candidate.named_buffers())[name]
        metadata = dict(meta.named_buffers())[name]
        assert torch.equal(value, actual)
        assert metadata.device.type == "meta"
        assert metadata.shape == value.shape and metadata.dtype == value.dtype


def test_scope_restores_constructor_on_failure():
    original = FSQ.__init__
    linspace = torch.linspace
    with pytest.raises(RuntimeError, match="fixture failure"):
        with offline_static_buffer_construction():
            raise RuntimeError("fixture failure")
    assert FSQ.__init__ is original
    assert torch.linspace is linspace


def test_full_tokenizer_layout_and_scalar_schedule_scope():
    from egomimic.models.oat.factory import make_tokenizer

    baseline = make_tokenizer()
    before = torch.get_rng_state().clone()
    with offline_static_buffer_construction(), torch.device("meta"):
        meta = make_tokenizer()
        assert torch.linspace(0, 1, 3).device.type == "meta"
    assert torch.equal(before, torch.get_rng_state())
    assert baseline.state_dict().keys() == meta.state_dict().keys()
    for name, value in baseline.state_dict().items():
        candidate = meta.state_dict()[name]
        assert candidate.shape == value.shape and candidate.dtype == value.dtype
