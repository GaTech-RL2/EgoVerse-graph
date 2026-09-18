import hashlib
import json

import pytest

from egomimic.rldb.zarr.manifest_resolver import load_split
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def write_manifest(tmp_path, rows, status="READY"):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"status": status, "held_out_embodiments": ["heldout"], "episodes": rows}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_embodiment_siblings_cannot_cross_splits(tmp_path):
    rows = [{"embodiment": "a", "episode_id": "one", "split": "train", "source_family": "source/one"},
            {"embodiment": "b", "episode_id": "copy", "split": "valid", "source_family": "source/one"}]
    with pytest.raises(ValueError, match="crosses splits"):
        load_split(*write_manifest(tmp_path, rows), "a", "train")


def test_heldout_and_pilot_data_are_rejected(tmp_path):
    rows = [{"embodiment": "a", "episode_id": "one", "split": "train", "source_family": "one"}]
    with pytest.raises(ValueError, match="Held-out"):
        load_split(*write_manifest(tmp_path, rows), "heldout", "train")
    with pytest.raises(ValueError, match="not promoted"):
        load_split(*write_manifest(tmp_path, rows, "PILOT"), "a", "train")


def test_pinned_split_preserved_and_changed_bytes_rejected(tmp_path):
    rows = [{"embodiment": "a", "episode_id": "one", "split": "valid", "source_family": "one"}]
    path, sha = write_manifest(tmp_path, rows)
    assert load_split(path, sha, "a", "valid") == rows
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_split(path, sha, "a", "valid")


def test_articulated_checkpoint_ids_and_small_circle_alias_are_preserved():
    # IDs from the previously published articulated normalizer contract.
    names = ["u_socket", "chain_gripper", "gripper", "suction", "umi",
             "triangle", "scoop", "flipper", "spring"]
    assert [get_embodiment_id("pushshapes_sim_" + name) for name in names] == list(range(19, 28))
    assert get_embodiment_id("pushshapes_sim_small_circle") == 17
    assert get_embodiment_id("pushshapes_sim_circle_small") == 17
