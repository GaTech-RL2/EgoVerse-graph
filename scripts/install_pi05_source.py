"""Install the pinned OpenPI model source without its simulator dependency bundle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
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
