"""Immutable decoded replay arrays shared through read-only file mappings.

The source is a local, immutable Zarr replay. A process lock and atomic publish
allow DDP ranks to prepare the same cache concurrently. Workers reopen mappings
after spawning; decoded image data is never pickled or copied per worker.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr


def _source_signature(source):
    """Detect local replay replacement/edits, including changes to chunk files."""
    digest = hashlib.sha256(str(source.resolve()).encode())
    for directory, dirs, files in os.walk(source):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            stat = path.stat()
            digest.update(
                json.dumps(
                    [
                        str(path.relative_to(source)),
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    ],
                    separators=(",", ":"),
                ).encode()
            )
    return digest.hexdigest()


class DecodedReplayArrays:
    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.manifest = manifest
        self._arrays = None

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    def open(self):
        if self._arrays is None:
            arrays = {}
            for key, info in self.manifest["arrays"].items():
                path = self.directory / info["file"]
                if path.stat().st_size != info["file_bytes"]:
                    raise ValueError(f"Truncated decoded replay array: {path}")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if (
                    list(array.shape) != info["shape"]
                    or array.dtype.str != info["dtype"]
                ):
                    raise ValueError(f"Decoded replay schema differs: {path}")
                arrays[key] = array
            self._arrays = arrays
        return self._arrays


def prepare_decoded_replay(source, keys, *, cache_root=None, block_bytes=32 << 20):
    """Decode once without changing dtype, shape, values or the source replay.

    Files stay on local storage; the OS shares their resident pages across all
    ranks/workers. Only a bounded chunk block is materialized during preparation.
    Incomplete builds are never published, and changed sources get new caches.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("Decoded replay requires a local Zarr directory")
    if block_bytes < 1:
        raise ValueError("block_bytes must be positive")
    keys = sorted(set(keys))
    if not keys:
        raise ValueError("Decoded replay requires at least one array")
    root = (
        Path(cache_root) if cache_root else source.with_name(source.name + ".decoded")
    )
    root = root.resolve()
    if root == source or source in root.parents:
        raise ValueError("Decoded cache must be outside the source replay")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        signature = _source_signature(source)
        identity = {
            "version": 1,
            "source": str(source),
            "signature": signature,
            "keys": keys,
        }
        token = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        destination = root / token
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text())
            if manifest["identity"] != identity or sorted(manifest["arrays"]) != keys:
                raise ValueError("Decoded replay manifest differs from source")
            result = DecodedReplayArrays(destination, manifest)
            result.open()  # Reject incomplete arrays before starting workers.
            return result

        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=root))
        try:
            data = zarr.open_group(str(source), mode="r")["data"]
            manifest = {"identity": identity, "arrays": {}}
            for key in keys:
                array = data[key]
                if array.dtype.hasobject or not array.shape or array.shape[0] == 0:
                    raise ValueError(f"Unsupported decoded replay array: {key}")
                filename = hashlib.sha256(key.encode()).hexdigest() + ".npy"
                path = temporary / filename
                decoded = np.lib.format.open_memmap(
                    path, mode="w+", dtype=array.dtype, shape=array.shape
                )
                frame_bytes = array.dtype.itemsize * math.prod(array.shape[1:])
                chunk_frames = array.chunks[0]
                block = (
                    max(1, block_bytes // max(1, frame_bytes * chunk_frames))
                    * chunk_frames
                )
                digest = hashlib.sha256()
                for start in range(0, array.shape[0], block):
                    values = np.ascontiguousarray(array[start : start + block])
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite decoded replay array: {key}")
                    decoded[start : start + len(values)] = values
                    digest.update(memoryview(values).cast("B"))
                decoded.flush()
                del decoded
                manifest["arrays"][key] = {
                    "file": filename,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "file_bytes": path.stat().st_size,
                    "decoded_sha256": digest.hexdigest(),
                }
            if _source_signature(source) != signature:
                raise ValueError("Source replay changed during cache preparation")
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return DecodedReplayArrays(destination, manifest)
