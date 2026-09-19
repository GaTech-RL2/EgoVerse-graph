"""Stage a frozen episode manifest without changing any source bytes.

Archives are checksum verified, extracted once, then exposed through read-only
symlink views. Existing files and links are never replaced.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tarfile
import zipfile

import boto3

from egomimic.rldb.zarr.content_manifest import hash_episode


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative path within the dataset: {value}")
    return path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if digest(args.manifest) != args.sha256:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "READY":
        raise ValueError("Only promoted manifests may be staged")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    c = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
                     aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                     aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                     region_name="auto")
    archives = {}
    for row in manifest["episodes"]:
        archive = row["archive"]
        previous = archives.setdefault(archive["sha256"], archive)
        if previous != archive:
            raise ValueError("Inconsistent archive identity in manifest")

    def extract(item):
        sha, archive = item
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise ValueError("Invalid archive checksum")
        directory = root / "_objects" / sha
        receipt = directory / ".extracted.json"
        if receipt.exists():
            if json.loads(receipt.read_text()) != archive:
                raise ValueError(f"Cached archive identity changed: {sha}")
            return
        if directory.exists():
            raise FileExistsError(f"Unverified partial directory preserved: {directory}")
        directory.mkdir(parents=True)
        bucket, key = archive["uri"].removeprefix("s3://").split("/", 1)
        download = directory / ".archive.download"
        c.download_file(bucket, key, str(download))
        if digest(download) != sha:
            raise ValueError(f"Archive checksum mismatch: {key}")
        if "bytes" in archive and download.stat().st_size != archive["bytes"]:
            raise ValueError(f"Archive size mismatch: {key}")
        if zipfile.is_zipfile(download):
            with zipfile.ZipFile(download) as z:
                for name in z.namelist():
                    safe_relative(name)
                z.extractall(directory)
        else:
            with tarfile.open(download) as tar:
                for member in tar.getmembers():
                    safe_relative(member.name)
                    if member.issym() or member.islnk():
                        raise ValueError("Dataset archives must contain regular files")
                tar.extractall(directory, filter="data")
        receipt.write_text(json.dumps(archive, sort_keys=True) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(extract, archives.items()))

    def stage(row):
        source = root / "_objects" / row["archive"]["sha256"] / safe_relative(row["archive_path"])
        if hash_episode(source)["sha256"] != row["sha256"]:
            raise ValueError(f"Episode checksum mismatch: {row['episode_id']}")
        views = [str(safe_relative(row["embodiment"]) / safe_relative(row["split"]))]
        views += row.get("views", [])
        for view in views:
            destination = root / safe_relative(view) / (str(safe_relative(row["episode_id"])) + ".zarr")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
            destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(stage, manifest["episodes"]))
    receipt = root / ("STAGED-" + args.sha256 + ".json")
    value = {"status": "PASS", "manifest_sha256": args.sha256,
             "episodes": len(manifest["episodes"]), "archives": len(archives)}
    receipt.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
