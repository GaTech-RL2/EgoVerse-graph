#!/usr/bin/env python3
"""Materialize a path-independent content manifest for resolved Zarr episodes."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite content manifest: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(repo))

    from egomimic.rldb.zarr.content_manifest import (  # noqa: PLC0415
        build_content_manifest,
        canonical_manifest_bytes,
    )
    episodes = {}
    for path in sorted(dataset_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            zarr.open_group(str(path), mode="r")
        except Exception:
            continue
        episode_id = path.name[:-5] if path.name.endswith(".zarr") else path.name
        if episode_id in episodes:
            raise RuntimeError(f"duplicate suffixless episode ID: {episode_id}")
        episodes[episode_id] = path
    manifest = build_content_manifest(episodes)
    encoded = canonical_manifest_bytes(manifest)
    output = args.output.expanduser().resolve()
    atomic_create(output, encoded)
    print(
        "[zarr-content-manifest] "
        f"episodes={manifest['episode_count']} files={manifest['file_count']} "
        f"bytes={manifest['total_bytes']} aggregate={manifest['aggregate_sha256']} "
        f"manifest_sha256={hashlib.sha256(encoded).hexdigest()} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
