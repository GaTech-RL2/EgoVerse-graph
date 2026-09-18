"""Resolve an explicit, content-pinned dataset split without random resplitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .content_manifest import hash_episode
from .zarr_dataset_multi import (
    LocalEpisodeResolverWithEmbodimentOverride,
    episode_names_sha256,
)


def load_split(manifest_path, manifest_sha256, embodiment, split):
    path = Path(manifest_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("Dataset manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    if manifest.get("status") != "READY":
        raise ValueError("Dataset is not promoted for training")
    if embodiment in manifest.get("held_out_embodiments", []):
        raise ValueError(f"Held-out embodiment cannot enter a training/validation loader: {embodiment}")
    if split not in ("train", "valid"):
        raise ValueError(f"Unknown split {split}")
    families, identities = {}, set()
    for row in manifest["episodes"]:
        if row["split"] not in ("train", "valid"):
            raise ValueError("Invalid episode split")
        identity = (row["embodiment"], row["episode_id"])
        if identity in identities:
            raise ValueError(f"Duplicate episode identity: {identity}")
        identities.add(identity)
        family = row["source_family"]
        if families.setdefault(family, row["split"]) != row["split"]:
            raise ValueError(f"Source family crosses splits: {family}")
        if row["embodiment"] in manifest.get("held_out_embodiments", []):
            raise ValueError("Held-out data found in the training manifest")
    rows = [row for row in manifest["episodes"] if row["embodiment"] == embodiment and row["split"] == split]
    if not rows:
        raise ValueError(f"Empty pinned split: {embodiment}/{split}")
    return rows


class ManifestEpisodeResolver(LocalEpisodeResolverWithEmbodimentOverride):
    """Read a staged split, verify its bytes, and preserve the manifest assignment.

    Use with MultiDataset mode=total: the manifest already chose this split.
    Staging directories are <root>/<embodiment>/<split>/*.zarr.
    """
    def __init__(self, root, manifest_path, manifest_sha256, embodiment, split,
                 key_map=None, transform_list=None):
        self.rows = load_split(manifest_path, manifest_sha256, embodiment, split)
        super().__init__(
            folder_path=Path(root) / embodiment / split,
            embodiment_override=embodiment,
            expected_episode_count=len(self.rows),
            expected_episode_names_sha256=episode_names_sha256(row["episode_id"] for row in self.rows),
            key_map=key_map, transform_list=transform_list,
        )

    def resolve(self, **kwargs):
        for row in self.rows:
            content = hash_episode(self.folder_path / (row["episode_id"] + ".zarr"))
            if content["sha256"] != row["sha256"]:
                raise ValueError(f"Episode content mismatch: {row['episode_id']}")
        return super().resolve(**kwargs)
