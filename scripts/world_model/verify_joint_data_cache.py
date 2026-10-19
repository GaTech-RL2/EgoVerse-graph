"""Allocated-host admission of the immutable complete joint-training data cache.

No ML imports. The producer hashes every artifact; this admission binds that
receipt and native completion, rehashes the manifest, and checks all file sizes.
The DataModule still performs its full content/provenance validation afterward.
"""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_cache(request, node, cache_parent=Path("/tmp/pixel-joint-complete-data")):
    binding = request["data_cache"]
    receipt_path, native_path = (
        Path(binding["receipt"]),
        Path(binding["native_receipt"]),
    )
    require(
        sha(receipt_path) == binding["receipt_sha256"], "Cache receipt hash mismatch"
    )
    require(
        sha(native_path) == binding["native_receipt_sha256"],
        "Native completion hash mismatch",
    )
    receipt = json.loads(receipt_path.read_text())
    job = str(binding["job_id"])
    native = native_path.read_text().split()
    fields = dict(part.split("=", 1) for part in native if "=" in part)
    require(
        fields.get("JobId") == job
        and fields.get("JobName") == "joint-complete-data-v1"
        and fields.get("JobState") == "COMPLETED"
        and fields.get("ExitCode") == "0:0"
        and fields.get("UserId", "").startswith("paphiwetsa3(")
        and fields.get("NodeList") == node,
        "Cache producer lacks exact successful native completion",
    )
    cache = Path(binding["root"])
    require(
        job.isdigit()
        and cache == cache_parent / job
        and cache.resolve() == cache
        and cache.is_dir()
        and node == binding["node"] == receipt["node"],
        "Wrong physical cache location or allocation node",
    )
    logical = str(Path(request["data_manifest"]).parent)
    require(
        receipt["schema"] == "joint-complete-data-cache-v1"
        and receipt["status"] == "PASS"
        and str(receipt["job_id"]) == job
        and receipt["cache_root"] == str(cache)
        and receipt["logical_data_root"] == logical
        and receipt["manifest_sha256"] == request["data_manifest_sha256"]
        and receipt["files"] == 1502
        and receipt["rgb_files"] == 500
        and receipt["splits"] == {"train": 450, "val": 50},
        "Complete joint cache contract mismatch",
    )
    require(
        sha(cache / "manifest.json") == request["data_manifest_sha256"],
        "Cached manifest changed",
    )
    manifest = json.loads((cache / "manifest.json").read_text())
    expected = {"manifest.json": (request["data_manifest_sha256"], False)}

    def add(name, digest, rgb=False):
        path = Path(name)
        require(
            not path.is_absolute() and ".." not in path.parts, "Unbounded data artifact"
        )
        require(name not in expected, "Duplicate data artifact")
        expected[name] = (digest, rgb)

    add(manifest["normalization"]["path"], manifest["normalization"]["sha256"])
    episodes = manifest["episodes"]
    require(
        len(episodes) == 500
        and {k: sum(e["split"] == k for e in episodes) for k in ("train", "val")}
        == receipt["splits"],
        "Cached manifest split changed",
    )
    for episode in episodes:
        add(episode["path"], episode["sha256"])
        add(episode["rgb_path"], episode["rgb_sha256"], True)
        add(
            episode["action_provenance"]["path"], episode["action_provenance"]["sha256"]
        )
    records = receipt["artifacts"]
    require(
        len(records) == 1502 and len(expected) == 1502, "Incomplete artifact inventory"
    )
    seen, total, rgb_count = set(), 0, 0
    for record in records:
        name = record["path"]
        require(
            name in expected and name not in seen,
            "Unexpected or duplicate cache artifact",
        )
        seen.add(name)
        require(
            (record["sha256"], record["rgb_hardlink"]) == expected[name],
            "Artifact digest/role mismatch",
        )
        target = cache / name
        require(
            target.resolve(strict=True) == target
            and stat.S_ISREG(target.lstat().st_mode),
            "Cache artifact must be a regular non-symlink file",
        )
        require(
            type(record["bytes"]) is int
            and record["bytes"] > 0
            and target.stat().st_size == record["bytes"],
            "Missing, partial or resized cache artifact",
        )
        total += record["bytes"]
        rgb_count += record["rgb_hardlink"]
    require(
        seen == set(expected)
        and total == receipt["total_bytes"] > 0
        and rgb_count == 500,
        "Cache inventory totals mismatch",
    )
    return dict(
        status="DATA_CACHE_ADMISSION_PASS",
        files=1502,
        rgb_files=500,
        total_bytes=total,
        producer_job=job,
        node=node,
        root=str(cache),
        receipt_sha256=binding["receipt_sha256"],
        native_receipt_sha256=binding["native_receipt_sha256"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("proof", type=Path)
    args = parser.parse_args()
    require(
        os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Fresh allocation required"
    )
    proof = verify_cache(
        json.loads(args.request.read_text()), os.environ["SLURMD_NODENAME"]
    )
    with args.proof.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
