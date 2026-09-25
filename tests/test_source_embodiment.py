"""Weighted-mixture train sources may carry a non-embodiment key (PR #141)."""

import pytest

from egomimic.trainHydra import _source_embodiment


class _Leaf:
    def __init__(self, embodiment):
        self.embodiment = embodiment


def test_embodiment_key_is_used_directly():
    assert _source_embodiment("yam_bimanual", _Leaf("eva_bimanual")) == "yam_bimanual"


def test_extra_source_takes_its_leaf_embodiment():
    assert _source_embodiment("abc_yam_bimanual", _Leaf("yam_bimanual")) == "yam_bimanual"


def test_ambiguous_source_is_rejected():
    with pytest.raises(ValueError):
        _source_embodiment("abc_yam_bimanual", _Leaf(None))
