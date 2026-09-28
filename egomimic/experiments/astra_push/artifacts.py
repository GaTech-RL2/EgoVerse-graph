"""Immutable experiment records, hashes, and named random streams."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def named_seed(name, seed=17):
    return int.from_bytes(
        hashlib.sha256(f"astrapush-v1:{seed}:{name}".encode()).digest()[:4], "big"
    )


def publish_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    )
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return hashlib.sha256(data).hexdigest()
