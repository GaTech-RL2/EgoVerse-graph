"""Restore the pinned companion tests/docs without replacing checkout code."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def restore(root: Path) -> list[str]:
    root = root.resolve()
    revision = (root / ".github/validation-ref").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Validation must be pinned to a full commit SHA")
    available = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", revision + "^{commit}"],
        capture_output=True,
    )
    if available.returncode:
        git(root, "fetch", "--no-tags", "--depth=1", "origin", revision)
    manifest = json.loads(
        git(root, "show", revision + ":.github/validation-paths.json")
    )
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Validation manifest must map file paths to SHA-256 hashes")
    tracked = set(git(root, "ls-files", "-z").decode().split("\0"))
    for name, digest in manifest.items():
        path = PurePosixPath(name)
        if (
            not name
            or str(path) != name
            or path.is_absolute()
            or any(part in {"..", ".git"} for part in path.parts)
            or "\\" in name
            or name.startswith("-")
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"Invalid validation manifest entry: {name}")
        if name in tracked:
            raise ValueError(f"Validation would replace tracked checkout file: {name}")
        destination = root / name
        if any(parent.is_symlink() for parent in destination.parents if parent != root):
            raise ValueError(f"Validation path traverses a symlink: {name}")
        if destination.is_symlink():
            raise ValueError(f"Validation destination is a symlink: {name}")
        if destination.exists() and (
            not destination.is_file()
            or hashlib.sha256(destination.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"Validation would replace a local file: {name}")

    # Validate the complete archive before writing anything. Never extract a
    # companion source tree: the manifest contains only non-runtime artifacts.
    archive = git(root, "archive", "--format=tar", revision, "--", *manifest)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle:
            if member.isdir():
                continue
            if member.name not in manifest or not member.isfile():
                raise ValueError(f"Unexpected validation archive entry: {member.name}")
            content = bundle.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != manifest[member.name]:
                raise ValueError(f"Validation hash mismatch: {member.name}")
            files[member.name] = (content, member.mode)
    if files.keys() != manifest.keys():
        raise ValueError("Validation archive does not contain the complete manifest")
    for name, (content, mode) in files.items():
        destination = root / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(content)
            destination.chmod(mode & 0o777)
    print(f"Restored {len(files)} validation files from {revision}")
    return list(files)


if __name__ == "__main__":
    restore(Path(__file__).resolve().parents[2])
