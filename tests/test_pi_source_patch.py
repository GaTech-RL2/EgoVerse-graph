"""Pinned OpenPI patching must never modify another environment's hardlinks."""

import hashlib
import os

import pytest

from scripts.install_pi05_source import apply_transformers_patch


def test_verified_patch_preserves_cache_hardlink_and_original(tmp_path):
    cache = tmp_path / "cached.py"
    cache.write_bytes(b"original code")
    root = tmp_path / "transformers"
    root.mkdir()
    target = root / "model.py"
    os.link(cache, target)
    content = b"pinned patched code"
    manifest = {"model.py": hashlib.sha256(content).hexdigest()}
    receipt = apply_transformers_patch(
        root, manifest, lambda path: content, tmp_path / "backups"
    )
    assert target.read_bytes() == content
    assert cache.read_bytes() == b"original code"
    assert target.stat().st_ino != cache.stat().st_ino
    from pathlib import Path

    assert Path(receipt[0]["backup"]).read_bytes() == b"original code"
    assert (
        apply_transformers_patch(
            root, manifest, lambda path: content, tmp_path / "backups"
        )[0]["sha256"]
        == receipt[0]["sha256"]
    )


def test_every_hash_is_preflighted_before_patch_mutation(tmp_path):
    root = tmp_path / "transformers"
    root.mkdir()
    (root / "a.py").write_bytes(b"original")
    expected = hashlib.sha256(b"valid").hexdigest()
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_transformers_patch(
            root,
            {"a.py": expected, "b.py": "wrong"},
            lambda path: b"valid",
            tmp_path / "backups",
        )
    assert (root / "a.py").read_bytes() == b"original"
    assert not (root / "b.py").exists()
