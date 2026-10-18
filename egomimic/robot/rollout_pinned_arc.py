"""YAM rollout entry point with hash-bound historical ARC model routing.

Uses the existing rollout loop, calibration, safety checks and dashboard.
Only policy loading gains a provenance guard. Importing this module opens no
hardware. Running it without --check-config has the normal robot semantics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from egomimic.robot import rollout
from egomimic.robot.pinned_arc_bbb813a0 import (
    TRAINING_COMMIT,
    PinnedArcDecoder,
    verify_sources,
)

_original_load_policy = rollout.load_policy
_original_validate_config = rollout.validate_rollout_config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(config):
    if config.get("kind") != "graph":
        return None
    checkpoint = Path(config["checkpoint"]).resolve(strict=True)
    binding = checkpoint.with_suffix(".pinned-arc.json")
    if not binding.exists():
        from egomimic.pipeline.inference_config import find_inference_config

        sidecar = config.get("inference_config") or find_inference_config(checkpoint)
        if sidecar and "pinned_arc_bbb813a0" in Path(sidecar).read_text():
            raise ValueError("Pinned ARC checkpoint binding is missing")
        return None
    manifest = json.loads(binding.read_text())
    if manifest["training_commit"] != TRAINING_COMMIT:
        raise ValueError("Unexpected pinned ARC training revision")
    for key in ("checkpoint", "training_config", "normalizer_path", "inference_config"):
        item = manifest["files"][key]
        expected = checkpoint.parent / item["name"]
        actual = Path(config.get(key, expected)).resolve(strict=True)
        if (
            actual != expected.resolve(strict=True)
            or file_sha256(actual) != item["sha256"]
        ):
            raise ValueError(
                f"Pinned ARC {key} path or SHA-256 does not match checkpoint binding"
            )
    package = Path(__file__).with_name("pinned_arc_bbb813a0")
    for name, expected in manifest["adapter_files"].items():
        if file_sha256(package / name) != expected:
            raise ValueError(f"Pinned ARC adapter changed: {name}")
    verify_sources()
    return manifest


def load_policy(config):
    binding = validate_bundle(config)
    policy = _original_load_policy(config)
    if binding is not None:
        decoder = policy.adapter.decoder
        if not isinstance(decoder, PinnedArcDecoder) or decoder.mode != binding["mode"]:
            raise ValueError("Checkpoint was routed to the wrong historical decoder")
        if policy.replan_every is not None or policy.output_shape is not None:
            raise ValueError(
                "Pinned ARC must execute its variable-length first-stream prefix without additional truncation or padding"
            )
    return policy


def validate_rollout_config(config):
    _original_validate_config(config)
    # The dashboard may select either pinned model later in this process.
    if float(config.get("frequency", 0)) != 30.0:
        raise ValueError(
            "Pinned ARC rollout requires the training control frequency of 30 Hz"
        )


def main():
    rollout.load_policy = load_policy
    rollout.validate_rollout_config = validate_rollout_config
    rollout.main()


if __name__ == "__main__":
    main()
