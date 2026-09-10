"""Path-independent content identities for directory-backed Zarr episodes.

The digest covers every regular file below an episode using a framed stream of
its episode-relative POSIX name, byte size, and contents.  Dataset locations,
mtimes, owners, and traversal order are deliberately excluded so an exact
mirror has the same identity on every cluster.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATUS = "ZARR_CONTENT_MANIFEST"
ALGORITHM = {
    "aggregate": "sha256-framed-episode-records-v1",
    "episode": "sha256-framed-relative-name-size-bytes-v1",
    "file_order": "episode-relative-posix-path-utf8-bytewise",
    "path_scope": "episode-relative-only",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EPISODE_DOMAIN = b"egomimic-zarr-episode-content-v1\x00"
_AGGREGATE_DOMAIN = b"egomimic-zarr-dataset-content-v1\x00"
_CHUNK_SIZE = 16 * 1024 * 1024


def _frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big", signed=False))
    digest.update(value)


def _uint64(digest: Any, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"expected a nonnegative integer, got {value!r}")
    digest.update(value.to_bytes(8, "big", signed=False))


def _regular_files(root: Path) -> list[tuple[bytes, Path]]:
    """Return bytewise-sorted relative file names without following symlinks."""

    files: list[tuple[bytes, Path]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"content manifest refuses directory symlink: {candidate}")
        for name in file_names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"content manifest refuses file symlink: {candidate}")
            if not candidate.is_file():
                raise RuntimeError(f"content manifest found a non-regular file: {candidate}")
            relative = candidate.relative_to(root).as_posix().encode("utf-8")
            files.append((relative, candidate))
    files.sort(key=lambda item: item[0])
    if len({relative for relative, _ in files}) != len(files):
        raise RuntimeError(f"duplicate relative file name below {root}")
    return files


def hash_episode(path: Path) -> dict[str, Any]:
    """Hash one directory-backed Zarr episode without including its root path."""

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"episode is not a directory: {root}")
    digest = hashlib.sha256()
    digest.update(_EPISODE_DOMAIN)
    total_bytes = 0
    files = _regular_files(root)
    for relative, file_path in files:
        before = file_path.stat()
        size = int(before.st_size)
        _frame(digest, relative)
        _uint64(digest, size)
        consumed = 0
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
                consumed += len(chunk)
        after = file_path.stat()
        stable_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if consumed != size or stable_before != stable_after:
            raise RuntimeError(f"episode file changed while hashing: {file_path}")
        total_bytes += size
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _aggregate_episode_records(episodes: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(_AGGREGATE_DOMAIN)
    for episode in episodes:
        _frame(digest, episode["episode_id"].encode("utf-8"))
        _uint64(digest, episode["file_count"])
        _uint64(digest, episode["total_bytes"])
        digest.update(bytes.fromhex(episode["sha256"]))
    return digest.hexdigest()


def build_content_manifest(episodes: Mapping[str, Path]) -> dict[str, Any]:
    """Build a deterministic content manifest for an episode-id/path mapping."""

    if not isinstance(episodes, Mapping) or not episodes:
        raise ValueError("content manifest needs at least one episode")
    rows = []
    seen_paths: set[Path] = set()
    for raw_episode_id in sorted(episodes, key=lambda value: str(value).encode("utf-8")):
        episode_id = str(raw_episode_id)
        if not episode_id or "/" in episode_id or "\\" in episode_id:
            raise ValueError(f"invalid episode id: {episode_id!r}")
        path = Path(episodes[raw_episode_id]).expanduser().resolve(strict=True)
        if path in seen_paths:
            raise ValueError(f"multiple episode IDs resolve to {path}")
        seen_paths.add(path)
        rows.append({"episode_id": episode_id, **hash_episode(path)})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "algorithm": ALGORITHM,
        "episode_count": len(rows),
        "file_count": sum(row["file_count"] for row in rows),
        "total_bytes": sum(row["total_bytes"] for row in rows),
        "aggregate_sha256": _aggregate_episode_records(rows),
        "episodes": rows,
    }
    validate_content_manifest(payload)
    return payload


def validate_content_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a manifest's schema and recompute its aggregate identity."""

    if not isinstance(payload, Mapping):
        raise RuntimeError("content manifest root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != STATUS:
        raise RuntimeError("content manifest has an unsupported schema or status")
    if payload.get("algorithm") != ALGORITHM:
        raise RuntimeError("content manifest hash algorithm differs")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise RuntimeError("content manifest episodes must be a non-empty list")
    rows: list[dict[str, Any]] = []
    for raw in raw_episodes:
        if not isinstance(raw, Mapping) or set(raw) != {
            "episode_id",
            "file_count",
            "total_bytes",
            "sha256",
        }:
            raise RuntimeError("content manifest episode record has non-canonical keys")
        episode_id = raw["episode_id"]
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or "/" in episode_id
            or "\\" in episode_id
        ):
            raise RuntimeError("content manifest has an invalid episode id")
        file_count = raw["file_count"]
        total_bytes = raw["total_bytes"]
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 0
            or isinstance(total_bytes, bool)
            or not isinstance(total_bytes, int)
            or total_bytes < 0
        ):
            raise RuntimeError("content manifest has invalid episode counts")
        sha256 = raw["sha256"]
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise RuntimeError("content manifest has an invalid episode digest")
        rows.append(dict(raw))
    expected_order = sorted(rows, key=lambda row: row["episode_id"].encode("utf-8"))
    if rows != expected_order or len({row["episode_id"] for row in rows}) != len(rows):
        raise RuntimeError("content manifest episode records must be sorted and unique")
    expected_scalars = {
        "episode_count": len(rows),
        "file_count": sum(row["file_count"] for row in rows),
        "total_bytes": sum(row["total_bytes"] for row in rows),
        "aggregate_sha256": _aggregate_episode_records(rows),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"content manifest {key} mismatch")
    if set(payload) != {
        "schema_version",
        "status",
        "algorithm",
        "episode_count",
        "file_count",
        "total_bytes",
        "aggregate_sha256",
        "episodes",
    }:
        raise RuntimeError("content manifest has non-canonical top-level keys")
    return dict(expected_scalars)


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode a validated manifest deterministically for an external SHA-256."""

    validate_content_manifest(payload)
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
