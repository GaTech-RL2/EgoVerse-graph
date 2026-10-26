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
    if binding.get("kind") == "prepared-track-v1":
        return verify_track_cache(request, node)
    if binding.get("kind") == "prepared-pixel-v1":
        return verify_pixel_cache(request, node)
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


def track_inventory(root, manifest_sha):
    """Exact prepared XYZ contract, sharing the existing immutable cache owner."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    identity = manifest.get("track_identity", {})
    require(manifest.get("schema") == "track-proprio-prepared-v1"
            and identity == dict(grid_shape=[28, 40], reference_camera="agentview",
                                selection="regular_reference_view_grid", point_identity="persistent_surface_point",
                                frame="fixed_reference", units="meters", interpolated_endpoints=False)
            and manifest["normalization"]["scope"] == "train_only", "Unbound track geometry/normalization")
    episodes = manifest["episodes"]
    require(len(episodes) == 500 and {k: sum(ep["split"] == k for ep in episodes) for k in ("train", "val")}
            == dict(train=450, val=50), "Track split changed")
    physical = [(ep["native_source"]["source_demo_sha256"], ep["native_source"]["episode"]) for ep in episodes]
    require(len(set(physical)) == len(physical), "Duplicate physical track episodes")
    expected = {"manifest.json": manifest_sha,
                manifest["normalization"]["path"]: manifest["normalization"]["sha256"]}
    for ep in episodes:
        require(not any(key in ep for key in ("rgb_path", "rgb_sha256")), "RGB cache in XYZ corpus")
        for name, digest in ((ep["path"], ep["sha256"]),
                             (ep["action_provenance"]["path"], ep["action_provenance"]["sha256"])):
            require(name not in expected, "Duplicate track artifact")
            expected[name] = digest
    expected["provenance/source-manifest.json"] = manifest["source_manifest_sha256"]
    locations = root / "provenance/native-locations.json"
    canonical = json.dumps(json.loads(locations.read_text()), sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(hashlib.sha256(canonical.encode()).hexdigest() == manifest["native_locations_sha256"],
            "Native track locations changed")
    expected["provenance/native-locations.json"] = sha(locations)
    require(len(expected) == 1004, "Incomplete prepared track inventory")
    for name in expected:
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unbounded track artifact")
    return expected


def verify_track_producer(binding):
    path = Path(binding["native_receipt"])
    require(sha(path) == binding["native_receipt_sha256"], "Track producer accounting changed")
    fields = dict(p.split("=", 1) for p in path.read_text().split() if "=" in p)
    require(fields.get("JobId") == str(binding["job_id"])
            and fields.get("JobName") == binding["producer_name"]
            and fields.get("UserId") == f"{binding['user']}({binding['uid']})"
            and fields.get("NodeList") == binding["producer_node"]
            and fields.get("JobState") == "COMPLETED" and fields.get("ExitCode") == "0:0",
            "Track producer lacks exact native completion")


def verify_track_cache(request, node):
    binding = request["data_cache"]
    verify_track_producer(binding)
    require(sha(binding["receipt"]) == binding["receipt_sha256"], "Track staging receipt changed")
    receipt = json.loads(Path(binding["receipt"]).read_text())
    root = Path(binding["root"])
    require(root.is_absolute() and root.resolve(strict=True) == root and root.stat().st_uid == binding["uid"]
            and receipt["schema"] == "prepared-track-cache-v1" and receipt["status"] == "PASS"
            and receipt["cache_root"] == str(root)
            and receipt["manifest_sha256"] == request["data_manifest_sha256"]
            and receipt["producer_native_sha256"] == binding["native_receipt_sha256"],
            "Track cache placement/receipt changed")
    require(sha(root / "manifest.json") == request["data_manifest_sha256"], "Track manifest changed")
    expected = track_inventory(root, request["data_manifest_sha256"])
    seen, total = set(), 0
    for item in receipt["artifacts"]:
        name = item["path"]
        require(name in expected and name not in seen and item["sha256"] == expected[name], "Track inventory changed")
        seen.add(name)
        path = root / name
        require(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode)
                and path.stat().st_size == item["bytes"] > 0, "Missing/partial track artifact")
        total += item["bytes"]
    require(seen == set(expected) and receipt["total_bytes"] == total, "Track inventory incomplete")
    return dict(status="DATA_CACHE_ADMISSION_PASS", spatial_stream="track", files=len(seen),
                rgb_files=0, total_bytes=total, producer_job=str(binding["job_id"]), node=node,
                root=str(root), receipt_sha256=binding["receipt_sha256"],
                native_receipt_sha256=binding["native_receipt_sha256"])


def stage_track_cache(request, source, node):
    return stage_prepared_cache(request, source, node, spatial="track")


def pixel_inventory(root, manifest_sha, request=None):
    """Deduplicate combined RGB/state HDF5s while rejecting conflicting roles."""
    root = Path(root)
    require(sha(root / "manifest.json") == manifest_sha, "Pixel manifest changed")
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["schema"] == "pixel-proprio-prepared-v1", "Wrong pixel corpus schema")
    expected = {"manifest.json": manifest_sha}
    def add(name, digest):
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Unbounded pixel artifact")
        require(name not in expected or expected[name] == digest, "Conflicting shared pixel artifact")
        expected[name] = digest
    add(manifest["normalization"]["path"], manifest["normalization"]["sha256"])
    for episode in manifest["episodes"]:
        add(episode["path"], episode["sha256"])
        add(episode["rgb_path"], episode["rgb_sha256"])
        native_subtask = request and request.get("kind") == "native-subtask-world-model-v1"
        provenance = episode.get("action_provenance")
        if provenance:
            add(provenance["path"], provenance["sha256"])
        else:
            require(native_subtask and episode["embodiment"] not in request["interfaces"]["action_dims"],
                    "Missing command provenance for an action-bearing embodiment")
        if native_subtask:
            native = episode.get("native_source", {})
            if native.get("provenance_path"):
                add(native["provenance_path"], native["provenance_sha256"])
    if request and request.get("kind") == "native-subtask-world-model-v1":
        logical = Path(request["data_manifest"]).parent
        for binding in request["subtask_contract"].values():
            relative = Path(binding["path"]).relative_to(logical)
            add(str(relative), binding["sha256"])
            require(sha(root / relative) == binding["sha256"], "Changed native subtask sidecar")
    return expected


def verify_pixel_cache(request, node):
    binding = request["data_cache"]
    verify_track_producer(binding)  # same exact native terminal/owner/resource binding
    require(sha(binding["receipt"]) == binding["receipt_sha256"], "Pixel staging receipt changed")
    receipt = json.loads(Path(binding["receipt"]).read_text())
    root = Path(binding["root"])
    require(root.is_absolute() and root.resolve(strict=True) == root and root.stat().st_uid == binding["uid"]
            and node == binding["node"] == receipt["staging_node"]
            and receipt["schema"] == "prepared-pixel-cache-v1" and receipt["status"] == "PASS"
            and str(receipt["staging_job_id"]) == str(binding["job_id"])
            and receipt["cache_root"] == str(root)
            and receipt["manifest_sha256"] == request["data_manifest_sha256"], "Pixel cache placement changed")
    expected = pixel_inventory(root, request["data_manifest_sha256"], request)
    seen, total = set(), 0
    for item in receipt["artifacts"]:
        name = item["path"]
        require(name in expected and name not in seen and item["sha256"] == expected[name], "Pixel inventory changed")
        seen.add(name)
        path = root / name
        require(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode)
                and path.stat().st_size == item["bytes"] > 0, "Missing/partial pixel artifact")
        total += item["bytes"]
    require(seen == set(expected) and total == receipt["total_bytes"], "Incomplete pixel inventory")
    return dict(status="DATA_CACHE_ADMISSION_PASS", files=len(seen), total_bytes=total,
                producer_job=str(binding["job_id"]), node=node, root=str(root),
                receipt_sha256=binding["receipt_sha256"], native_receipt_sha256=binding["native_receipt_sha256"])


def stage_pixel_cache(request, source, node):
    return stage_prepared_cache(request, source, node, spatial="pixel")


def stage_prepared_cache(request, source, node, *, spatial):
    """One immutable copy/hash implementation for prepared track or pixel corpora."""
    import shutil
    import tempfile
    require(spatial in ("pixel", "track"), "Unknown prepared corpus")
    binding = request["data_cache"]
    require(binding.get("kind") == "prepared-" + spatial + "-v1"
            and os.getuid() == binding["uid"], "Wrong prepared staging owner")
    if spatial == "track":
        verify_track_producer(binding)
    else:
        require(node == binding["node"], "Wrong pixel staging node")
    source, destination = Path(source), Path(binding["root"])
    require(source.resolve(strict=True) == source and not destination.exists()
            and destination.parent.resolve(strict=True) == destination.parent, "Preserve existing/partial corpus")
    require(sha(source / "manifest.json") == request["data_manifest_sha256"], "Preparation manifest changed")
    inventory = pixel_inventory if spatial == "pixel" else track_inventory
    expected = (inventory(source, request["data_manifest_sha256"], request) if spatial == "pixel"
                else inventory(source, request["data_manifest_sha256"]))
    needed = sum((source / name).stat().st_size for name in expected)
    require(shutil.disk_usage(destination.parent).free > needed + request["reserved_checkpoint_bytes"],
            "Insufficient prepared data plus checkpoint reserve")
    with tempfile.TemporaryFile(dir=destination.parent) as f:
        f.write(b"prepared-cache-route-check"); f.flush(); os.fsync(f.fileno()); f.seek(0)
        require(f.read() == b"prepared-cache-route-check", "Prepared write/read/fsync failed")
    destination.mkdir()
    records = []
    for name in sorted(expected, key=lambda name: name == "manifest.json"):
        src, dst = source / name, destination / name
        require(src.resolve(strict=True) == src and src.is_file(), "Unsafe source artifact")
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        with src.open("rb") as inp, dst.open("xb") as out:
            for block in iter(lambda: inp.read(8 << 20), b""):
                out.write(block); digest.update(block); count += len(block)
            out.flush(); os.fsync(out.fileno())
        require(digest.hexdigest() == expected[name] and dst.stat().st_size == count > 0,
                "Prepared transfer bytes/hash mismatch")
        records.append(dict(path=name, sha256=digest.hexdigest(), bytes=count))
    result = dict(schema="prepared-" + spatial + "-cache-v1", status="PASS", staging_job_id=os.environ["SLURM_JOB_ID"],
                  staging_node=node, cache_root=str(destination), manifest_sha256=request["data_manifest_sha256"],
                  total_bytes=sum(r["bytes"] for r in records), artifacts=records)
    if spatial == "track":
        result["producer_native_sha256"] = binding["native_receipt_sha256"]
    with Path(binding["receipt"]).open("x") as out:
        json.dump(result, out, indent=2); out.write("\n"); out.flush(); os.fsync(out.fileno())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("proof", type=Path)
    parser.add_argument("--stage-from", type=Path)
    args = parser.parse_args()
    require(
        os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Fresh allocation required"
    )
    request, node = json.loads(args.request.read_text()), os.environ["SLURMD_NODENAME"]
    stage = stage_pixel_cache if request["data_cache"].get("kind") == "prepared-pixel-v1" else stage_track_cache
    proof = (stage(request, args.stage_from, node) if args.stage_from else verify_cache(request, node))
    with args.proof.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
