"""Upload Yam HDF5 demos.

Normal interactive upload (renames demos and creates RLDB metadata)::

    python -m egomimic.scripts.data_upload.yam_uploader

Frozen, no-clobber backup (preserves filenames and creates no metadata)::

    python -m egomimic.scripts.data_upload.yam_uploader \
      --backup-dir demos/yam_gello \
      --backup-manifest /tmp/egoverse-yam-backup/source.tsv \
      --backup-prefix staging/yam/manual-backup \
      --manifest-sha256 EXPECTED_SHA256
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

MIB = 1024 * 1024


@dataclass(frozen=True)
class BackupItem:
    name: str
    size: int


@dataclass(frozen=True)
class FinalizeItem:
    source_name: str
    source_key: str
    destination_key: str
    metadata_key: str
    timestamp_ms: int
    size: int
    operator: str
    metadata: dict


class UploadProgress:
    def __init__(self, total_bytes):
        self.total_bytes = total_bytes
        self.transferred = 0
        self.started = time.monotonic()
        self.last_report = 0.0
        self.lock = threading.Lock()

    def __call__(self, byte_count):
        with self.lock:
            self.transferred += byte_count
            now = time.monotonic()
            if now - self.last_report < 10:
                return
            self.last_report = now
            elapsed = max(now - self.started, 0.001)
            gib = self.transferred / (1024**3)
            total_gib = self.total_bytes / (1024**3)
            mib_s = self.transferred / elapsed / MIB
            percent = 100 * self.transferred / self.total_bytes
            print(
                f"Progress: {gib:.2f}/{total_gib:.2f} GiB "
                f"({percent:.1f}%) at {mib_s:.1f} MiB/s",
                flush=True,
            )


def collect_files(local_dir):
    return sorted(
        path
        for path in Path(local_dir).iterdir()
        if path.is_file() and path.suffix == ".hdf5"
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_backup_manifest(manifest_path):
    items = []
    seen = set()
    for line_number, raw_line in enumerate(manifest_path.read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"Malformed manifest line {line_number}: {raw_line!r}")
        name = fields[0]
        if Path(name).name != name or not name.endswith(".hdf5"):
            raise ValueError(
                f"Unsafe manifest filename on line {line_number}: {name!r}"
            )
        if name in seen:
            raise ValueError(f"Duplicate manifest filename: {name}")
        size = int(fields[1])
        if size <= 0:
            raise ValueError(f"Invalid size for {name}: {size}")
        seen.add(name)
        items.append(BackupItem(name=name, size=size))
    if not items:
        raise ValueError("Backup manifest is empty")
    return items


def _head_size(s3, bucket, key):
    try:
        return s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return None
        raise


def _head_object(s3, bucket, key):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return None
        raise


def _canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exact_plan(path, plan):
    payload = _canonical_json(plan)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing finalize plan differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build_finalize_plan(args, items):
    source_dir = args.backup_dir.expanduser().resolve()
    operators = list(dict.fromkeys(args.operator))
    if not operators:
        raise ValueError("Provide at least one --operator value")

    assignments = [operators[index % len(operators)] for index in range(len(items))]
    random.Random(args.operator_seed).shuffle(assignments)

    source_prefix = args.backup_prefix.strip("/")
    destination_prefix = args.canonical_prefix.strip("/")
    planned = []
    timestamps = set()
    for item, operator in zip(items, assignments, strict=True):
        path = source_dir / item.name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item.size:
            raise ValueError(f"Frozen source size changed: {path}")
        timestamp_ms = int(path.stat().st_mtime * 1000)
        if timestamp_ms in timestamps:
            raise ValueError(f"Duplicate canonical timestamp: {timestamp_ms}")
        timestamps.add(timestamp_ms)
        metadata = {
            "embodiment": args.embodiment,
            "episode_hash": timestamp_ms,
            "lab": args.lab,
            "objects": json.loads(args.objects_json),
            "operator": operator,
            "rig_name": args.rig_name,
            "scene": args.scene,
            "task": args.task,
            "task_description": args.task_description,
        }
        planned.append(
            FinalizeItem(
                source_name=item.name,
                source_key=f"{source_prefix}/{item.name}",
                destination_key=f"{destination_prefix}/{timestamp_ms}.hdf5",
                metadata_key=f"{destination_prefix}/{timestamp_ms}_metadata.json",
                timestamp_ms=timestamp_ms,
                size=item.size,
                operator=operator,
                metadata=metadata,
            )
        )

    counts = {operator: assignments.count(operator) for operator in operators}
    plan = {
        "schema_version": 1,
        "bucket": args.bucket,
        "manifest_sha256": args.manifest_sha256.lower(),
        "operator_seed": args.operator_seed,
        "operator_counts": counts,
        "source_prefix": source_prefix,
        "destination_prefix": destination_prefix,
        "items": [item.__dict__ for item in planned],
    }
    return planned, plan


def _owned_destination(head, item, manifest_sha256):
    if head is None or head["ContentLength"] != item.size:
        return False
    metadata = head.get("Metadata", {})
    return (
        metadata.get("source-key") == item.source_key
        and metadata.get("source-manifest-sha256") == manifest_sha256
        and metadata.get("source-size") == str(item.size)
    )


def _multipart_copy_object(
    s3,
    *,
    bucket,
    source_key,
    destination_key,
    source_size,
    source_etag,
    metadata,
    part_size,
):
    """Copy a large R2 object without one long-running CopyObject request."""
    upload = s3.create_multipart_upload(
        Bucket=bucket,
        Key=destination_key,
        ContentType="application/x-hdf5",
        Metadata=metadata,
    )
    upload_id = upload["UploadId"]
    completed_parts = []
    try:
        for part_number, start in enumerate(range(0, source_size, part_size), 1):
            end = min(start + part_size, source_size) - 1
            result = s3.upload_part_copy(
                Bucket=bucket,
                Key=destination_key,
                PartNumber=part_number,
                UploadId=upload_id,
                CopySource={"Bucket": bucket, "Key": source_key},
                CopySourceIfMatch=source_etag,
                CopySourceRange=f"bytes={start}-{end}",
            )
            completed_parts.append(
                {
                    "ETag": result["CopyPartResult"]["ETag"],
                    "PartNumber": part_number,
                }
            )
        s3.complete_multipart_upload(
            Bucket=bucket,
            Key=destination_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": completed_parts},
        )
    except BaseException:
        s3.abort_multipart_upload(
            Bucket=bucket,
            Key=destination_key,
            UploadId=upload_id,
        )
        raise


def run_finalize_snapshot(args):
    manifest_path = args.backup_manifest.expanduser().resolve()
    actual_manifest_sha256 = _sha256(manifest_path)
    if actual_manifest_sha256 != args.manifest_sha256.lower():
        raise ValueError(
            "Manifest SHA-256 mismatch: "
            f"expected {args.manifest_sha256.lower()}, got {actual_manifest_sha256}"
        )
    items = read_backup_manifest(manifest_path)
    if args.limit is not None:
        items = items[: args.limit]
    planned, plan = build_finalize_plan(args, items)
    plan_path = args.plan_output.expanduser().resolve()
    plan_sha256 = _write_exact_plan(plan_path, plan)
    print(
        f"Finalize plan: {plan_path} ({len(planned)} episodes, SHA-256 {plan_sha256})",
        flush=True,
    )
    print(f"Operator counts: {plan['operator_counts']}", flush=True)
    if args.plan_only:
        return

    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client, load_env

    load_env(required=True)
    endpoint = os.environ.get("R2_ENDPOINT_URL", "").rstrip("/")
    expected_endpoint = args.expected_endpoint.rstrip("/")
    if endpoint != expected_endpoint:
        raise ValueError(
            f"Refusing unexpected R2 endpoint: expected {expected_endpoint!r}, "
            f"got {endpoint!r}"
        )
    s3 = get_boto3_s3_client()

    conflicts = []
    pending = []
    for item in planned:
        source_head = _head_object(s3, args.bucket, item.source_key)
        if source_head is None:
            conflicts.append(f"missing staging object: {item.source_key}")
            continue
        if source_head["ContentLength"] != item.size:
            conflicts.append(
                f"staging size mismatch: {item.source_key} "
                f"({source_head['ContentLength']} != {item.size})"
            )
            continue

        destination_head = _head_object(s3, args.bucket, item.destination_key)
        metadata_head = _head_object(s3, args.bucket, item.metadata_key)
        destination_ok = _owned_destination(
            destination_head, item, args.manifest_sha256.lower()
        )
        metadata_ok = False
        if metadata_head is not None:
            body = s3.get_object(Bucket=args.bucket, Key=item.metadata_key)[
                "Body"
            ].read()
            metadata_ok = body == _canonical_json(item.metadata)

        if destination_head is not None and not destination_ok:
            conflicts.append(f"unowned destination exists: {item.destination_key}")
        if metadata_head is not None and not metadata_ok:
            conflicts.append(f"different metadata exists: {item.metadata_key}")
        if destination_head is None or metadata_head is None:
            pending.append(item)

    if conflicts:
        raise ValueError("Finalize preflight failed:\n- " + "\n- ".join(conflicts))

    print(
        f"Finalize preflight passed: {len(planned) - len(pending)} complete, "
        f"{len(pending)} pending",
        flush=True,
    )
    if not args.apply:
        print("Dry run only; pass --apply to create canonical objects.", flush=True)
        return

    def finalize(item):
        destination_head = _head_object(s3, args.bucket, item.destination_key)
        if destination_head is None:
            source_head = _head_object(s3, args.bucket, item.source_key)
            _multipart_copy_object(
                s3,
                bucket=args.bucket,
                source_key=item.source_key,
                destination_key=item.destination_key,
                source_size=item.size,
                source_etag=source_head["ETag"],
                metadata={
                    "source-key": item.source_key,
                    "source-manifest-sha256": args.manifest_sha256.lower(),
                    "source-size": str(item.size),
                },
                part_size=args.copy_part_size_mib * MIB,
            )
        metadata_head = _head_object(s3, args.bucket, item.metadata_key)
        if metadata_head is None:
            s3.put_object(
                Bucket=args.bucket,
                Key=item.metadata_key,
                Body=_canonical_json(item.metadata),
                ContentType="application/json",
            )
        return item

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(finalize, item) for item in pending]
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            print(
                f"Finalized {index}/{len(pending)}: {item.source_name} -> "
                f"{item.timestamp_ms}",
                flush=True,
            )

    for item in planned:
        destination_head = _head_object(s3, args.bucket, item.destination_key)
        if not _owned_destination(destination_head, item, args.manifest_sha256.lower()):
            raise ValueError(f"Final verification failed: {item.destination_key}")
        body = s3.get_object(Bucket=args.bucket, Key=item.metadata_key)["Body"].read()
        if body != _canonical_json(item.metadata):
            raise ValueError(f"Final verification failed: {item.metadata_key}")
    print(
        f"Canonical YAM upload complete: {len(planned)} episodes under "
        f"s3://{args.bucket}/{args.canonical_prefix.strip('/')}/",
        flush=True,
    )


def run_snapshot_backup(args):
    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client, load_env

    source_dir = args.backup_dir.expanduser().resolve()
    manifest_path = args.backup_manifest.expanduser().resolve()
    actual_manifest_sha256 = _sha256(manifest_path)
    if actual_manifest_sha256 != args.manifest_sha256.lower():
        raise ValueError(
            "Manifest SHA-256 mismatch: "
            f"expected {args.manifest_sha256.lower()}, got {actual_manifest_sha256}"
        )

    items = read_backup_manifest(manifest_path)
    for item in items:
        path = source_dir / item.name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != item.size:
            raise ValueError(
                f"Frozen source changed for {item.name}: "
                f"manifest={item.size}, actual={actual_size}"
            )

    load_env(required=True)
    endpoint = os.environ.get("R2_ENDPOINT_URL", "").rstrip("/")
    expected_endpoint = args.expected_endpoint.rstrip("/")
    if endpoint != expected_endpoint:
        raise ValueError(
            f"Refusing unexpected R2 endpoint: expected {expected_endpoint!r}, "
            f"got {endpoint!r}"
        )

    bucket = args.bucket
    prefix = args.backup_prefix.strip("/")
    s3 = get_boto3_s3_client()
    pending = []
    skipped_bytes = 0
    for item in items:
        key = f"{prefix}/{item.name}"
        remote_size = _head_size(s3, bucket, key)
        if remote_size is None:
            pending.append(item)
        elif remote_size == item.size:
            skipped_bytes += item.size
            print(f"Already verified: s3://{bucket}/{key}", flush=True)
        else:
            raise ValueError(
                f"No-clobber conflict at s3://{bucket}/{key}: "
                f"local={item.size}, remote={remote_size}"
            )

    total_bytes = sum(item.size for item in pending)
    print(
        f"Snapshot: {len(items)} files; {len(pending)} pending; "
        f"{skipped_bytes / (1024**3):.2f} GiB already verified",
        flush=True,
    )
    if not pending:
        print("Backup is already complete.", flush=True)
        return

    progress = UploadProgress(total_bytes)
    transfer_config = TransferConfig(
        max_concurrency=args.parts,
        multipart_threshold=args.part_size_mib * MIB,
        multipart_chunksize=args.part_size_mib * MIB,
        use_threads=True,
    )

    def upload(item):
        path = source_dir / item.name
        key = f"{prefix}/{item.name}"
        # The preflight above enforces no-clobber before any worker starts.
        s3.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/x-hdf5"},
            Config=transfer_config,
            Callback=progress,
        )
        remote_size = _head_size(s3, bucket, key)
        if remote_size != item.size:
            raise ValueError(
                f"Post-upload size mismatch for {item.name}: "
                f"local={item.size}, remote={remote_size}"
            )
        return item

    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(upload, item): item for item in pending}
        for future in as_completed(futures):
            item = future.result()
            completed_bytes += item.size
            print(
                f"Verified: {item.name} ({completed_bytes / (1024**3):.2f}/"
                f"{total_bytes / (1024**3):.2f} GiB committed)",
                flush=True,
            )

    print(
        f"Backup complete: {len(items)} files at s3://{bucket}/{prefix}/",
        flush=True,
    )


def yam_uploader():
    from egomimic.scripts.data_upload.abstract_upload import Uploader

    return Uploader(
        embodiment="yam",
        datatype=".hdf5",
        collect_files=collect_files,
        defaults={"rig_name": "yam"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        type=Path,
        metavar="DIRECTORY",
        help="List input demos without credentials or uploading",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Direct-upload a frozen snapshot from this directory",
    )
    parser.add_argument(
        "--backup-manifest",
        type=Path,
        help="Tab-separated filename/size manifest for --backup-dir",
    )
    parser.add_argument(
        "--backup-prefix",
        help="Destination prefix for snapshot backup (filenames are preserved)",
    )
    parser.add_argument("--manifest-sha256", help="Expected manifest SHA-256")
    parser.add_argument(
        "--finalize-snapshot",
        action="store_true",
        help="Finalize a staged snapshot into canonical HDF5 + metadata objects",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Create/validate the deterministic finalize plan without S3 access",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply canonical server-side copies after finalize preflight",
    )
    parser.add_argument("--limit", type=int, help="Limit finalize items")
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("/tmp/yam-finalize-plan.json"),
    )
    parser.add_argument("--canonical-prefix", default="raw_v2/yam")
    parser.add_argument("--operator", action="append", default=[])
    parser.add_argument("--operator-seed", type=int, default=20260914)
    parser.add_argument("--lab")
    parser.add_argument("--task")
    parser.add_argument("--task-description")
    parser.add_argument("--scene")
    parser.add_argument("--objects-json", default="null")
    parser.add_argument("--rig-name")
    parser.add_argument("--embodiment", default="yam_bimanual")
    parser.add_argument(
        "--expected-endpoint",
        default="https://1beb594fb475d71c4420f7b693524e19.r2.cloudflarestorage.com",
        help="Exact allowed R2 endpoint",
    )
    parser.add_argument("--bucket", default="rldb")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parts", type=int, default=4)
    parser.add_argument("--part-size-mib", type=int, default=64)
    parser.add_argument(
        "--copy-part-size-mib",
        type=int,
        default=256,
        help="Server-side multipart-copy part size used by --finalize-snapshot",
    )
    args = parser.parse_args()
    if args.list is not None:
        for path in collect_files(args.list):
            print(path)
        return
    backup_args = (
        args.backup_dir,
        args.backup_manifest,
        args.backup_prefix,
        args.manifest_sha256,
    )
    if args.finalize_snapshot:
        if not all(value is not None for value in backup_args):
            parser.error(
                "--backup-dir, --backup-manifest, --backup-prefix, and "
                "--manifest-sha256 are required with --finalize-snapshot"
            )
        metadata_args = (
            args.lab,
            args.task,
            args.task_description,
            args.scene,
            args.rig_name,
        )
        if not args.operator or not all(metadata_args):
            parser.error(
                "--operator, --lab, --task, --task-description, "
                "--scene, and --rig-name are required with --finalize-snapshot"
            )
        try:
            json.loads(args.objects_json)
        except json.JSONDecodeError as error:
            parser.error(f"--objects-json must be valid JSON: {error}")
        run_finalize_snapshot(args)
        return
    if any(value is not None for value in backup_args):
        if not all(value is not None for value in backup_args):
            parser.error(
                "--backup-dir, --backup-manifest, --backup-prefix, and "
                "--manifest-sha256 are required together"
            )
        if args.workers < 1 or args.parts < 1 or args.part_size_mib < 5:
            parser.error("workers/parts must be >= 1 and part-size-mib must be >= 5")
        run_snapshot_backup(args)
        return
    asyncio.run(yam_uploader().run())


if __name__ == "__main__":
    main()
