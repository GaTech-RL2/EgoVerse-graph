"""Pinned experiment pools survive installation outside the author's checkout."""

from importlib.resources import files

import pytest

from egomimic.rldb.filters import DatasetFilter


def test_packaged_allowlist_membership_and_memo_identity():
    name = "hydra_configs/data/abc_arc/rl2_towels394_episodes.txt"
    allowed = files("egomimic").joinpath(name).read_text().split()
    assert len(set(allowed)) == 394
    selection = DatasetFilter(episode_allowlist_file="package://egomimic/" + name)
    assert all(selection.matches({"episode_hash": value}) for value in allowed)
    assert not selection.matches({"episode_hash": "not-in-pool"})
    assert not selection.matches({"episode_hash": allowed[0], "is_deleted": True})
    assert (
        selection.episode_hashes == frozenset()
    )  # Filter before split, not a per-split pin.
    assert selection.cache_key() != DatasetFilter().cache_key()


def test_missing_or_empty_allowlist_cannot_broaden_selection(tmp_path):
    path = tmp_path / "episodes.txt"
    with pytest.raises(FileNotFoundError):
        DatasetFilter(episode_allowlist_file=str(path))
    path.write_text("")
    with pytest.raises(ValueError, match="Empty episode allowlist"):
        DatasetFilter(episode_allowlist_file=str(path))
