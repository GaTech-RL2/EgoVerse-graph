from pathlib import Path

import numpy as np
import zarr

from egomimic.rldb.zarr.content_manifest import (
    build_content_manifest,
    canonical_manifest_bytes,
)

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
