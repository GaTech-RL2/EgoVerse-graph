#!/usr/bin/env python3
"""Validate two physical Planar corpora against one immutable co-training split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from validate_planar_dataset import (
    SPLIT_ALGORITHM,
    SPLIT_SEED,
    VALID_RATIO,
    _atomic_json,
    _digest,
    sha256_file,
    validate_content_identity,
    validate_physical_inventory,
)


DOMAINS = ("pushshapes_sim_chain_gripper", "pushshapes_sim_u_socket")


def _domain_args(parser: argparse.ArgumentParser, domain: str) -> None:
    slug = "chain" if domain.endswith("chain_gripper") else "usocket"
    parser.add_argument(f"--{slug}-dataset-root", required=True, type=Path)
    parser.add_argument(f"--{slug}-content-manifest", required=True, type=Path)
    parser.add_argument(f"--expected-{slug}-content-manifest-sha256", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    for domain in DOMAINS:
        _domain_args(parser, domain)
    return parser.parse_args()


def _load_json(path: Path, label: str) -> Mapping:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} root must be an object")
    return payload


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    expected_manifest_sha = _digest(
        args.expected_manifest_sha256, "expected manifest SHA-256"
    )
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError("co-training split manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path, "split manifest")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise RuntimeError("co-training split manifest must be a schema-1 PASS")
    if manifest.get("split_seed") != SPLIT_SEED:
        raise RuntimeError("co-training split seed must be 42")
    if manifest.get("valid_ratio") != VALID_RATIO:
        raise RuntimeError("co-training validation ratio must be 1%")
    if manifest.get("split_algorithm") != SPLIT_ALGORITHM:
        raise RuntimeError("co-training split algorithm mismatch")
    if manifest.get("cross_domain_train_valid_resolved_path_overlap_count") != 0:
        raise RuntimeError("co-training manifest records cross-domain overlap")
    domains = manifest.get("domains")
    if not isinstance(domains, Mapping) or set(domains) != set(DOMAINS):
        raise RuntimeError(f"co-training split must contain exactly {DOMAINS!r}")

    generator = repo / "scripts" / "data" / "materialize_local_episode_split.py"
    generator_sha = sha256_file(generator)
    if generator_sha != _digest(manifest.get("generator_sha256"), "generator_sha256"):
        raise RuntimeError("split generator differs from repository source")

    sys.path.insert(0, str(repo))
    from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver

    reports = {}
    all_resolved_paths = {}
    for domain in DOMAINS:
        slug = "chain" if domain.endswith("chain_gripper") else "usocket"
        root = getattr(args, f"{slug}_dataset_root").expanduser().resolve(strict=True)
        content_path = (
            getattr(args, f"{slug}_content_manifest").expanduser().resolve(strict=True)
        )
        expected_content_sha = _digest(
            getattr(args, f"expected_{slug}_content_manifest_sha256"),
            f"expected {slug} content manifest SHA-256",
        )
        actual_content_sha = sha256_file(content_path)
        if actual_content_sha != expected_content_sha:
            raise RuntimeError(f"{slug} content manifest SHA-256 mismatch")
        entries = LocalEpisodeResolver._get_local_filtered_paths(root)
        single_manifest = dict(manifest)
        single_manifest["domains"] = {domain: domains[domain]}
        report = validate_physical_inventory(single_manifest, entries, root)
        content = _load_json(content_path, f"{slug} content manifest")
        content_identity = {
            "manifest": str(content_path),
            "manifest_sha256": actual_content_sha,
            **validate_content_identity(content, entries),
        }
        reports[domain] = {
            "physical_inventory": report,
            "content_identity": content_identity,
        }
        all_resolved_paths[domain] = {Path(path).resolve(strict=True) for path, _ in entries}

    if all_resolved_paths[DOMAINS[0]] & all_resolved_paths[DOMAINS[1]]:
        raise RuntimeError("co-training physical corpora share resolved episode paths")

    payload = {
        "schema_version": 1,
        "status": "COTRAIN_DATASETS_VALIDATED",
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "split_seed": SPLIT_SEED,
        "valid_ratio": VALID_RATIO,
        "split_algorithm": SPLIT_ALGORITHM,
        "generator_sha256": generator_sha,
        "cross_domain_resolved_path_overlap_count": 0,
        "domains": reports,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
