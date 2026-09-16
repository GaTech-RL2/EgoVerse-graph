"""Upload one immutable YAM manifest through no-overwrite presigned PUT URLs.

The plan contains credentials in the form of temporary URLs. Keep it owner-only,
never print it, and invoke this module with a bounded worker count, for example::

    python -m egomimic.scripts.data_upload.yam_presigned_uploader \
        --plan /secure/yam-upload-plan.json \
        --receipt /secure/yam-upload-receipt.jsonl \
        --lock /secure/yam-upload.lock \
        --workers 3
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import h5py


def now() -> str:
    return datetime.now(UTC).isoformat()


def emit(
    receipt: Path, payload: dict[str, object], receipt_lock: threading.Lock
) -> None:
    payload["timestamp"] = now()
    with receipt_lock, receipt.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_state(root: Path, item: dict[str, object]) -> tuple[Path, dict[str, int]]:
    name = item.get("name")
    if not isinstance(name, str) or Path(name).name != name:
        raise RuntimeError(f"unsafe source name: {name!r}")
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"source file is absent or unsafe: {path}")
    stat = path.stat()
    state = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    expected = {"size": item.get("size"), "mtime_ns": item.get("mtime_ns")}
    if state != expected:
        raise RuntimeError(f"source changed since plan creation: {path}")
    with h5py.File(path, "r") as episode:
        if not bool(episode.attrs.get("complete", False)):
            raise RuntimeError(f"collector did not mark the demo complete: {path}")
    return path, state


def _put_connection(
    url: object, timeout_s: float
) -> tuple[http.client.HTTPSConnection, str]:
    if not isinstance(url, str):
        raise TypeError("plan entry does not contain a string PUT URL")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise RuntimeError("plan entry has an invalid PUT URL")
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    return http.client.HTTPSConnection(
        parts.hostname, port=parts.port, timeout=timeout_s
    ), target


def _put_stream(url: object, stream, size: int, timeout_s: float) -> int:
    connection, target = _put_connection(url, timeout_s)
    try:
        connection.putrequest("PUT", target, skip_accept_encoding=True)
        connection.putheader("Content-Length", str(size))
        connection.putheader("If-None-Match", "*")
        connection.endheaders()
        while chunk := stream.read(8 * 1024 * 1024):
            connection.send(chunk)
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def upload_raw(item: dict[str, object], path: Path) -> int:
    with path.open("rb") as handle:
        return _put_stream(item.get("raw_put_url"), handle, path.stat().st_size, 7200)


def upload_metadata(item: dict[str, object]) -> int:
    metadata = item.get("metadata")
    if not isinstance(metadata, str):
        raise TypeError("plan entry does not contain JSON metadata text")
    payload = metadata.encode()
    from io import BytesIO

    return _put_stream(
        item.get("metadata_put_url"), BytesIO(payload), len(payload), 300
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent episode uploads; each worker handles one HDF5 plus metadata.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    if args.plan.stat().st_mode & 0o077:
        raise SystemExit("refusing plan that is readable outside its owner")
    if not args.receipt.parent.is_dir():
        raise SystemExit(f"receipt parent does not exist: {args.receipt.parent}")
    if args.receipt.exists():
        raise SystemExit(f"refusing to append to an existing receipt: {args.receipt}")

    plan = json.loads(args.plan.read_text())
    root = Path(plan["source_root"])
    items = plan.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit("plan must contain a non-empty items list")

    try:
        lock_fd = os.open(args.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise SystemExit(
            f"refusing duplicate execution; lock exists: {args.lock}"
        ) from error
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
        lock.write(now() + "\n")
        lock.flush()
        os.fsync(lock.fileno())

    receipt_lock = threading.Lock()

    def upload_item(item: dict[str, object]) -> tuple[str, bool]:
        name = str(item.get("name"))
        episode_id = item.get("episode_id")
        try:
            path, before = source_state(root, item)
            raw_status = upload_raw(item, path)
            _, after_raw = source_state(root, item)
            raw_ok = raw_status in {200, 201}
            emit(
                args.receipt,
                {
                    "name": name,
                    "episode_id": episode_id,
                    "object": "hdf5",
                    "status": raw_status,
                    "ok": raw_ok,
                    "source_before": before,
                    "source_after": after_raw,
                },
                receipt_lock,
            )
            if not raw_ok:
                return name, False
            metadata_status = upload_metadata(item)
            metadata_ok = metadata_status in {200, 201}
            emit(
                args.receipt,
                {
                    "name": name,
                    "episode_id": episode_id,
                    "object": "metadata",
                    "status": metadata_status,
                    "ok": metadata_ok,
                },
                receipt_lock,
            )
            return name, metadata_ok
        except Exception as error:  # noqa: BLE001 - receipt every worker failure.
            emit(
                args.receipt,
                {
                    "name": name,
                    "episode_id": episode_id,
                    "object": "exception",
                    "ok": False,
                    "error_type": type(error).__name__,
                },
                receipt_lock,
            )
            return name, False

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(upload_item, item) for item in items]
        for completed, future in enumerate(as_completed(futures), 1):
            name, ok = future.result()
            failures += not ok
            print(f"COMPLETE={completed}/{len(items)} name={name} ok={ok}", flush=True)

    print(f"ITEMS={len(items)}", flush=True)
    print(f"FAILURES={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
