"""Stage pinned, read-only training inputs on OSMO before requesting GPU tasks."""

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid manifest-relative path: {value!r}")
    return Path(*path.parts)


def copy_stream(stream, destination, *, expected_bytes, expected_sha256=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("xb") as output:
        shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
    if temporary.stat().st_size != expected_bytes:
        raise ValueError(f"Input size mismatch: {destination}")
    actual = digest(temporary)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"Input SHA-256 mismatch: {destination}")
    # No replacement of an existing input, even if another task races this one.
    os.link(temporary, destination)
    temporary.unlink()
    return {"bytes": expected_bytes, "sha256": actual}


def prepare(data_manifest, weight_manifest, tokenizer_manifest, tokenizer_root, output):
    import boto3
    from botocore.config import Config

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(max_pool_connections=8, retries={"max_attempts": 8}),
    )
    data = json.loads(Path(data_manifest).read_text())
    downloads = []
    for episode in data["episodes"]:
        for item in episode["objects"]:
            if not item["key"].startswith(episode["prefix"]):
                raise ValueError("Data object is outside its pinned episode")
            target = relative_path(
                f"data/{episode['source']}/{episode['episode_hash']}.zarr"
            )
            target /= relative_path(item["key"][len(episode["prefix"]) :])
            downloads.append((item, target))

    def fetch(record):
        item, relative = record
        response = client.get_object(
            Bucket=data["bucket"], Key=item["key"], IfMatch=item["etag"]
        )
        with response["Body"] as stream:
            receipt = copy_stream(
                stream, output / relative, expected_bytes=item["bytes"]
            )
        return {
            "path": str(relative),
            "key": item["key"],
            "etag": item["etag"],
            **receipt,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        data_receipts = list(pool.map(fetch, downloads))
    weights = json.loads(Path(weight_manifest).read_text())
    weight_receipts = []
    for spec in weights["files"]:
        print("Downloading pinned weight input", spec["path"], flush=True)
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "egoverse-integration-input-audit"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            receipt = copy_stream(
                response,
                output / relative_path(spec["path"]),
                expected_bytes=spec["bytes"],
                expected_sha256=spec["sha256"],
            )
        weight_receipts.append({**spec, **receipt})
    tokens = json.loads(Path(tokenizer_manifest).read_text())
    for spec in tokens["files"]:
        path = Path(tokenizer_root) / relative_path(spec["file"])
        deadline = time.monotonic() + 300
        # Explicit rsync to stage-inputs sends only these files; no tokens/credentials.
        while (
            not path.is_file() or path.stat().st_size != spec["bytes"]
        ) and time.monotonic() < deadline:
            time.sleep(2)
        with path.open("rb") as stream:
            copy_stream(
                stream,
                output / "tokenizer" / relative_path(spec["file"]),
                expected_bytes=spec["bytes"],
                expected_sha256=spec["sha256"],
            )
    for source, name in [
        (data_manifest, "data-manifest.json"),
        (weight_manifest, "weight-manifest.json"),
        (tokenizer_manifest, "tokenizer-manifest.json"),
    ]:
        with Path(source).open("rb") as stream:
            copy_stream(
                stream,
                output / name,
                expected_bytes=Path(source).stat().st_size,
                expected_sha256=digest(source),
            )
    receipt = {
        "status": "verified",
        "data_manifest_sha256": digest(data_manifest),
        "weight_manifest_sha256": digest(weight_manifest),
        "tokenizer_manifest_sha256": digest(tokenizer_manifest),
        "objects": data_receipts,
        "weights": weight_receipts,
        "tokenizer": tokens,
    }
    with (output / "input-receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    print(
        "INPUTS_VERIFIED",
        len(data_receipts),
        "objects",
        len(weight_receipts),
        "weight files",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-manifest",
        "weight-manifest",
        "tokenizer-manifest",
        "tokenizer-root",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    arguments = vars(parser.parse_args())
    prepare(**arguments)
