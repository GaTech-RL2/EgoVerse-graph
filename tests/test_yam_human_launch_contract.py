"""Regression tests for the frozen two-domain YAM+human launch boundary."""

import hashlib
import json

import pytest

from scripts.train.verify_yam_human_action_flow_launch import DOMAINS, split_identities


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest():
    result = {
        "split_seed": 42,
        "valid_ratio": 0.01,
        "train_frame_windows_total": 376264,
        "proportional_train_window_probabilities": {
            "yam": 249565 / 376264,
            "human": 126699 / 376264,
        },
    }
    for domain, (
        _,
        train_count,
        valid_count,
        train_windows,
        valid_windows,
        prefix,
    ) in DOMAINS.items():
        rows = []
        summary = {
            "episodes": train_count + valid_count,
            "train_episodes": train_count,
            "valid_episodes": valid_count,
            "train_frame_windows": train_windows,
            "valid_frame_windows": valid_windows,
        }
        for group, count, windows in (
            ("train", train_count, train_windows),
            ("valid", valid_count, valid_windows),
        ):
            names = [f"{domain}-{group}-{index:03d}" for index in range(count)]
            summary[f"{group}_ids_sha256"] = hashlib.sha256(
                "".join(f"{name}\n" for name in names).encode()
            ).hexdigest()
            for index, name in enumerate(names):
                rows.append(
                    {
                        "split": group,
                        "episode_hash": name,
                        "zarr_processed_path": f"{prefix}{name}.zarr",
                        "num_frames": windows - count + 1 if index == 0 else 1,
                    }
                )
        result[domain] = {"summary": summary, "episodes": rows}
    return result


def test_exact_proportional_split_accepts_all_episodes(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps(_manifest()))
    ids = split_identities(path, _sha(path))
    assert set(ids) == {"yam_train", "yam_valid", "human_train", "human_valid"}


@pytest.mark.parametrize(
    "mutation", ["duplicate_path", "balanced_sampling", "wrong_holdout"]
)
def test_split_rejects_science_contract_drift(tmp_path, mutation):
    split = _manifest()
    if mutation == "duplicate_path":
        split["human"]["episodes"][1]["zarr_processed_path"] = split["human"][
            "episodes"
        ][0]["zarr_processed_path"]
    elif mutation == "balanced_sampling":
        split["proportional_train_window_probabilities"] = {"yam": 0.5, "human": 0.5}
    else:
        split["human"]["episodes"][0]["split"] = "valid"
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError):
        split_identities(path, _sha(path))
