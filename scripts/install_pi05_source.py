"""Install the pinned OpenPI model source without its simulator dependency bundle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REVISION = "981483dca0fd9acba698fea00aa6e52d56a66c58"
SOURCE = f"git+https://github.com/GaTech-RL2/openpi.git@{REVISION}"
REQUIRED = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.53.2",
    "flax": "0.10.2",
    "jax": "0.5.3",
    "orbax-checkpoint": "0.11.13",
}


def apply_transformers_patch(root, files, fetch, backup_root):
    """Verify all bytes first, preserve originals, then replace file inodes.

    In-place writes can mutate uv's hardlinked cache and other environments.
    Atomic replacement deliberately breaks that link only inside this venv.
    """
    root, backup_root = Path(root), Path(backup_root)
    prepared = []
    for relative, expected in files.items():
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Patch paths must stay inside the Transformers package")
        destination = root / relative
        if destination.is_symlink():
            raise ValueError(f"Refusing to patch a symlink: {destination}")
        data = fetch(relative.as_posix())
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Pinned Transformers patch hash mismatch: {relative}")
        prepared.append((relative, destination, data, expected))
    receipt = []
    for relative, destination, data, expected in prepared:
        before = destination.read_bytes() if destination.exists() else None
        before_hash = hashlib.sha256(before).hexdigest() if before is not None else None
        backup = None
        if before is not None and before_hash != expected:
            backup = backup_root / before_hash / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists() and backup.read_bytes() != before:
                raise ValueError(f"Original-file backup changed: {backup}")
            if not backup.exists():
                backup.write_bytes(before)
        if before_hash != expected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as handle:
                handle.write(data)
                temporary = Path(handle.name)
            temporary.chmod(0o644)
            os.replace(temporary, destination)
        receipt.append(
            {
                "path": str(relative),
                "sha256": expected,
                "previous_sha256": before_hash,
                "backup": str(backup) if backup else None,
            }
        )
    return receipt


def install_transformers_patch():
    manifest = json.loads(
        Path(__file__).with_name("pi05_transformers_manifest.json").read_text()
    )
    if manifest["source_revision"] != REVISION or manifest[
        "transformers_version"
    ] != importlib.metadata.version("transformers"):
        raise ValueError(
            "Transformers patch manifest does not match the pinned runtime"
        )
    root = importlib.metadata.distribution("transformers").locate_file("transformers")

    def fetch(relative):
        url = f"https://raw.githubusercontent.com/GaTech-RL2/openpi/{REVISION}/src/openpi/models_pytorch/transformers_replace/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()

    return apply_transformers_patch(
        root, manifest["files"], fetch, Path(sys.prefix) / "pi05-transformers-originals"
    )


def main():
    versions = {name: importlib.metadata.version(name) for name in REQUIRED}
    if versions != REQUIRED:
        raise RuntimeError(
            f"Install the locked PI extra before OpenPI source: {versions}"
        )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            f"openpi @ {SOURCE}",
        ],
        check=True,
    )
    distribution = importlib.metadata.distribution("openpi")
    provenance = json.loads(distribution.read_text("direct_url.json"))
    if provenance.get("vcs_info", {}).get("commit_id") != REVISION:
        raise RuntimeError(
            "Installed OpenPI does not match the required immutable revision"
        )
    if {name: importlib.metadata.version(name) for name in REQUIRED} != versions:
        raise RuntimeError("Source installation changed the locked model runtime")
    payload = {
        "source": provenance,
        "runtime_versions": versions,
        "policy": "pinned-model-source-without-upstream-dependency-bundle",
        "transformers_patch": install_transformers_patch(),
    }
    path = Path(sys.prefix) / "pi05-source-receipt.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
