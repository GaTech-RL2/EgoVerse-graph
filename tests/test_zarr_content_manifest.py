import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

from egomimic.rldb.zarr.content_manifest import (
    build_content_manifest,
    canonical_manifest_bytes,
)

ROOT = Path(__file__).parents[1]
GENERATOR = ROOT / "scripts" / "data" / "materialize_zarr_content_manifest.py"
VALIDATOR = ROOT / "scripts" / "ice" / "validate_planar_dataset.py"


def _tiny_episodes(root: Path) -> dict[str, Path]:
    root.mkdir()
    episodes = {}
    for index in range(3):
        episode_id = f"episode_{index:03d}"
        path = root / f"{episode_id}.zarr"
        group = zarr.open_group(str(path), mode="w")
        group.create_array(
            "actions",
            data=np.asarray([[index, index + 1]], dtype=np.int64),
            chunks=(1, 2),
        )
        group.attrs["episode"] = index
        episodes[episode_id] = path
    return episodes


def _split_manifest(module, names, source_root):
    train, valid = module.split_names(names)
    source_paths = {name: source_root / name for name in names}
    generator = ROOT / "scripts" / "data" / "materialize_local_episode_split.py"
    return {
        "schema_version": 1,
        "status": "PASS",
        "split_algorithm": module.SPLIT_ALGORITHM,
        "split_seed": 42,
        "valid_ratio": 0.01,
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "cross_domain_train_valid_resolved_path_overlap_count": 0,
        "domains": {
            module.DOMAIN: {
                "folder_path": str(source_root),
                "total_count": len(names),
                "inventory_names_sha256": module.names_sha256(names),
                "train_count": len(train),
                "train_names_sha256": module.names_sha256(train),
                "train_resolved_paths_sha256": module.paths_sha256(
                    source_paths[name] for name in train
                ),
                "train_ids": sorted(train),
                "valid_count": len(valid),
                "valid_names_sha256": module.names_sha256(valid),
                "valid_resolved_paths_sha256": module.paths_sha256(
                    source_paths[name] for name in valid
                ),
                "valid_ids": sorted(valid),
                "id_overlap_count": 0,
                "resolved_path_overlap_count": 0,
                "union_count": len(names),
                "union_matches_inventory": True,
            }
        },
    }


def test_manifest_is_path_independent_and_byte_sensitive(tmp_path):
    first = _tiny_episodes(tmp_path / "source")
    second_root = tmp_path / "mirror"
    second_root.mkdir()
    second = {}
    for episode_id, source in first.items():
        destination = second_root / source.name
        import shutil

        shutil.copytree(source, destination)
        second[episode_id] = destination

    first_manifest = build_content_manifest(first)
    second_manifest = build_content_manifest(second)
    assert first_manifest == second_manifest
    assert canonical_manifest_bytes(first_manifest) == canonical_manifest_bytes(
        second_manifest
    )

    chunk = next(
        path
        for path in sorted(second["episode_001"].rglob("*"))
        if path.is_file() and path.name not in {"zarr.json", ".zgroup", ".zattrs"}
    )
    chunk.write_bytes(chunk.read_bytes() + b"tamper")
    tampered = build_content_manifest(second)
    assert tampered["aggregate_sha256"] != first_manifest["aggregate_sha256"]
    rows = {row["episode_id"]: row for row in tampered["episodes"]}
    expected = {row["episode_id"]: row for row in first_manifest["episodes"]}
    assert rows["episode_001"]["sha256"] != expected["episode_001"]["sha256"]
    assert rows["episode_000"] == expected["episode_000"]


def test_generator_and_validator_reject_byte_tamper(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_planar_dataset", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    episodes = _tiny_episodes(tmp_path / "dataset")
    content_path = tmp_path / "content.json"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output",
            str(content_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    content_sha = hashlib.sha256(content_path.read_bytes()).hexdigest()

    split = _split_manifest(module, set(episodes), Path("/canonical/source"))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split, sort_keys=True, separators=(",", ":")) + "\n")
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()

    def validate(output: Path, *, check: bool):
        return subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--repo",
                str(ROOT),
                "--dataset-root",
                str(tmp_path / "dataset"),
                "--manifest",
                str(split_path),
                "--expected-manifest-sha256",
                split_sha,
                "--content-manifest",
                str(content_path),
                "--expected-content-manifest-sha256",
                content_sha,
                "--output",
                str(output),
            ],
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    validate(tmp_path / "pass.json", check=True)
    payload = json.loads((tmp_path / "pass.json").read_text())
    assert payload["content_identity"]["aggregate_sha256"] == json.loads(
        content_path.read_text()
    )["aggregate_sha256"]

    chunk = next(
        path
        for path in sorted(episodes["episode_002"].rglob("*"))
        if path.is_file() and path.name not in {"zarr.json", ".zgroup", ".zattrs"}
    )
    chunk.write_bytes(chunk.read_bytes() + b"tamper")
    failed = validate(tmp_path / "fail.json", check=False)
    assert failed.returncode != 0
    assert "Zarr content manifest mismatch" in failed.stderr
    assert not (tmp_path / "fail.json").exists()


def test_validator_requires_both_content_manifest_flags(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--repo",
            str(ROOT),
            "--dataset-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "split.json"),
            "--expected-manifest-sha256",
            "a" * 64,
            "--content-manifest",
            str(tmp_path / "content.json"),
            "--output",
            str(tmp_path / "result.json"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "must be supplied together" in completed.stderr
