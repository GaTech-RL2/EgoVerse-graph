"""Lossless indexed RGB frame caches with bounded random-access decoding.

JPEG is decoded once during preparation. Each canonical RGB frame is then
stored with lossless Blosc/LZ4 compression; array offsets are memory mapped.
This avoids expanding long simulation corpora to terabytes of raw RGB files.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from numcodecs import Blosc


@lru_cache(maxsize=192)
def _packed(path):
    path = Path(path)
    offsets = np.load(str(path) + ".offsets.npy", mmap_mode="r", allow_pickle=False)
    payload = np.memmap(str(path) + ".blosc", dtype=np.uint8, mode="r")
    return offsets, payload


def read_frames(path, indices, shape):
    offsets, payload = _packed(str(path))
    codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.NOSHUFFLE)
    frames = []
    for index in np.asarray(indices, dtype=np.int64).reshape(-1):
        if index < 0 or index + 1 >= len(offsets):
            raise IndexError(index)
        raw = codec.decode(payload[int(offsets[index]):int(offsets[index + 1])])
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape))
        frames.append(frame)
    return np.stack(frames)


def write_frames(path, frames):
    """Write an iterable of uint8 RGB images, validating every encoded frame."""
    path = Path(path)
    codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.NOSHUFFLE)
    offsets = [0]
    shape = None
    with Path(str(path) + ".blosc").open("xb") as stream:
        for image in frames:
            image = np.ascontiguousarray(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Expected uint8 HWC RGB")
            if shape is not None and tuple(image.shape) != shape:
                raise ValueError("Frame geometry changed within episode")
            shape = tuple(image.shape)
            encoded = codec.encode(image)
            if codec.decode(encoded) != image.tobytes():
                raise ValueError("Lossless RGB cache round trip failed")
            stream.write(encoded)
            offsets.append(offsets[-1] + len(encoded))
    if shape is None:
        raise ValueError("Cannot cache an empty image sequence")
    np.save(str(path) + ".offsets.npy", np.asarray(offsets, dtype=np.uint64), allow_pickle=False)
    return {"format": "blosc_rgb_v1", "shape": list(shape), "frames": len(offsets) - 1,
            "payload_bytes": offsets[-1], "uncompressed_bytes": (len(offsets) - 1) * int(np.prod(shape))}
