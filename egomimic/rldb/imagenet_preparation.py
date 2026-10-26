"""Allocated-CPU raw ImageNet inventory and complete native RGB decode.

The raw producer never loads Qwen features, constructs a model, or writes image
caches. Its result certifies bytes/decode only. Native allocation/host completion
and subsequent class-language admission remain separate external gates.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import re
import stat
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import PIL

from egomimic.rldb import imagenet_imagefolder as adapter


RAW_SCHEMA = "imagenet-raw-image-inventory-v1"
RAW_STATUS = "PASS_IMAGE_BYTES_AND_DECODE_ONLY"
ADAPTER_SHA256 = "b43103a338b1cb732c1db6d3c07516699c162a9517f2212cc648948d9d93d737"
SIDECARS = {"tar_extract_script.sh~": 0, "tar_extract_script.sh": 108}
_WNID = re.compile(r"n[0-9]{8}")
_SHA = re.compile(r"[0-9a-f]{64}")
_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode())


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_once(path, value):
    path = Path(path)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return dict(path=str(path), sha256=_sha(path.read_bytes()), bytes=path.stat().st_size)


def decoder_binding():
    """Bind the exact sealed adapter module, decoder version and preprocessing."""
    source = Path(adapter.__file__)
    digest = _sha(source.read_bytes())
    if digest != ADAPTER_SHA256:
        raise ValueError("Native decoder adapter differs from the sealed image adapter")
    return dict(module="egomimic.rldb.imagenet_imagefolder", module_sha256=digest,
                function="_decode_rgb160", library="Pillow", version=PIL.__version__,
                preprocessing=copy.deepcopy(adapter.PREPROCESSING))


def _physical_directory(value, name):
    path = Path(value)
    if not path.is_absolute() or str(path) != str(path.resolve(strict=True)):
        raise ValueError(name + " must be an explicit physical absolute path; aliases are rejected")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(name + " must be a physical directory")
    return path


def _signature(path):
    item = path.stat(follow_symlinks=False)
    return [item.st_dev, item.st_ino, item.st_mode, item.st_size,
            item.st_mtime_ns, item.st_ctime_ns]


def _read_regular(path):
    """No-follow source read, rejecting a changed file during the read."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Source member is not a regular file: " + str(path))
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (any(getattr(before, key) != getattr(after, key) for key in fields)
                or len(raw) != before.st_size
                or any(getattr(after, key) != getattr(current, key)
                       for key in fields)):
            raise ValueError("Source member changed while reading: " + str(path))
        return raw
    finally:
        os.close(fd)


def _sidecars(value):
    if not isinstance(value, list):
        raise ValueError("Explicit train sidecar exclusions with actual byte sizes and SHA256 are required")
    result = {}
    for row in value:
        if (not isinstance(row, dict) or row.get("name") not in SIDECARS
                or type(row.get("bytes")) is not int
                or row["bytes"] != SIDECARS[row["name"]]
                or row["name"] in result):
            raise ValueError("Only exact explicitly hashed known train sidecars may be excluded")
        hashed = (set(row) == {"name", "bytes", "sha256"}
                  and _SHA.fullmatch(str(row.get("sha256"))) is not None)
        unreadable = (set(row) == {"name", "bytes", "status", "uid", "mode", "mtime_seconds"}
                      and row["name"] == "tar_extract_script.sh" and row["bytes"] == 108
                      and row["status"] == "UNREADABLE_EXCLUDED"
                      and type(row["uid"]) is int and row["uid"] >= 0
                      and isinstance(row["mode"], str) and re.fullmatch(r"[0-7]{4}", row["mode"]) is not None
                      and type(row["mtime_seconds"]) is int and row["mtime_seconds"] >= 0)
        if not (hashed or unreadable):
            raise ValueError("Sidecar requires an actual hash or the exact known108B unreadable recipe")
        result[row["name"]] = row
    return result


def _roots(train_root, val_root, exclusions, expected_classes):
    train = _physical_directory(train_root, "train_root")
    val = _physical_directory(val_root, "val_root")
    if train.name != "train" or val.name != "val" or train.parent != val.parent:
        raise ValueError("Explicit physical train/val siblings are required")
    result, directories, excluded = {}, [], []
    for split, root in (("train", train), ("val", val)):
        directories.append((root, _signature(root)))
        classes = {}
        seen_excluded = set()
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.name in exclusions and split == "train":
                row = exclusions[path.name]
                if path.is_symlink():
                    raise ValueError("A sidecar exclusion cannot admit a symlink")
                before = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("A sidecar exclusion requires an actual regular file")
                try:
                    raw = _read_regular(path)
                except PermissionError as error:
                    if row.get("status") != "UNREADABLE_EXCLUDED" or error.errno != errno.EACCES:
                        raise
                    current = path.stat(follow_symlinks=False)
                    keys = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid")
                    if (any(getattr(current, key) != getattr(before, key) for key in keys)
                            or before.st_size != row["bytes"] or before.st_uid != row["uid"]
                            or format(stat.S_IMODE(before.st_mode), "04o") != row["mode"]
                            or before.st_mtime_ns // 1000000000 != row["mtime_seconds"]):
                        raise ValueError("Actual unreadable sidecar metadata differs from reviewed exclusion") from error
                    excluded.append(dict(split="train", **row,
                        read_attempt=dict(operation="open_read_O_NOFOLLOW", outcome="PermissionError", errno=errno.EACCES)))
                else:
                    if row.get("status") == "UNREADABLE_EXCLUDED":
                        raise ValueError("An unreadable sidecar recipe cannot exclude a readable file")
                    if len(raw) != row["bytes"] or _sha(raw) != row["sha256"]:
                        raise ValueError("Actual train sidecar differs from its explicit exclusion")
                    excluded.append(dict(split="train", **row))
                seen_excluded.add(path.name)
            elif _WNID.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
                classes[path.name] = path
            else:
                raise ValueError("Unknown root sidecar, linked or non-WNID member: " + str(path))
        if split == "train" and seen_excluded != set(exclusions):
            raise ValueError("An explicitly excluded train sidecar is absent")
        if len(classes) != expected_classes:
            raise ValueError("Actual ImageNet class count differs from the explicit contract")
        result[split] = classes
    if set(result["train"]) != set(result["val"]):
        raise ValueError("Physical TRAIN/VAL WNID sets differ")
    return {"train": train, "val": val}, result, directories, excluded


def _tasks(classes, directories):
    for split in ("train", "val"):
        for wnid, directory in sorted(classes[split].items()):
            directories.append((directory, _signature(directory)))
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
            if not entries:
                raise ValueError("An ImageNet class is empty: " + str(directory))
            for path in entries:
                if (path.is_symlink() or not path.is_file()
                        or path.suffix.lower() not in _EXTENSIONS):
                    raise ValueError("Unexpected or linked image class member: " + str(path))
                yield split, wnid, path


def _decode_task(task):
    split, wnid, path = task
    raw = _read_regular(path)
    if not raw:
        raise ValueError("Empty source image: " + str(path))
    # This is the sealed adapter's exact decode, not an approximate probe.
    rgb = adapter._decode_rgb160(raw)
    if rgb.shape != (160, 160, 3) or str(rgb.dtype) != "uint8":
        raise ValueError("Native RGB decoder violated its exact shape/dtype contract")
    return split, dict(path=split + "/" + wnid + "/" + path.name,
                       wnid=wnid, bytes=len(raw), sha256=_sha(raw),
                       rgb160_sha256=_sha(rgb.tobytes(order="C")))


def _ordered_bounded_map(tasks, *, workers, max_inflight):
    """Bound submitted futures, retaining canonical input order."""
    iterator, pending = iter(tasks), deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="imagenet-decode") as pool:
        try:
            for _ in range(max_inflight):
                item = next(iterator, None)
                if item is None:
                    break
                pending.append(pool.submit(_decode_task, item))
            while pending:
                yield pending.popleft().result()
                item = next(iterator, None)
                if item is not None:
                    pending.append(pool.submit(_decode_task, item))
        finally:
            for future in pending:
                future.cancel()


def verify_images(train_root, val_root, output_path, *, expected_classes=1000,
                  expected_train_images=1281167, expected_val_images=50000,
                  workers=4, max_inflight=16, train_sidecar_exclusions=None):
    """Exhaust all source images once; never retry/reuse an existing intent."""
    for name, value in (("expected_classes", expected_classes), ("expected_train_images", expected_train_images),
                        ("expected_val_images", expected_val_images), ("workers", workers), ("max_inflight", max_inflight)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if workers > 4 or max_inflight > 16 or max_inflight < workers:
        raise ValueError("Decode concurrency must be bounded at workers<=4 and workers<=inflight<=16")
    output = Path(output_path).absolute()
    if not output.parent.is_dir() or any(Path(str(output) + suffix).exists()
            for suffix in ("", ".intent.json", ".progress.jsonl", ".failure.json", ".payload")):
        raise ValueError("Fresh immutable output namespace and existing parent directory are required")
    expected = {"train": expected_train_images, "val": expected_val_images}
    intent = dict(schema="imagenet-raw-image-inventory-intent-v1", started_at_utc=_now(),
                  train_root=str(train_root), val_root=str(val_root), expected_classes=expected_classes,
                  expected_counts=expected, workers=workers, max_inflight=max_inflight,
                  train_sidecar_exclusions=train_sidecar_exclusions, data_admitted=False,
                  producer_pid=os.getpid(), producer_job_id=os.environ.get("SLURM_JOB_ID"))
    _write_once(str(output) + ".intent.json", intent)
    journal = None
    rows = {"train": [], "val": []}
    try:
        journal = Path(str(output) + ".progress.jsonl").open("x")
        def progress(event):
            entry = {"at_utc": _now()}
            entry.update(event)
            journal.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            journal.flush(); os.fsync(journal.fileno())
        progress(dict(event="STARTED", expected_counts=expected))
        decoder = decoder_binding()
        exclusions = _sidecars(train_sidecar_exclusions)
        roots, classes, directories, excluded = _roots(train_root, val_root, exclusions, expected_classes)
        total_bytes = {"train": 0, "val": 0}
        for split, row in _ordered_bounded_map(_tasks(classes, directories), workers=workers, max_inflight=max_inflight):
            rows[split].append(row); total_bytes[split] += row["bytes"]
            if len(rows[split]) > expected[split]:
                raise ValueError("Actual source count exceeds explicit " + split + " count")
            if (len(rows["train"]) + len(rows["val"])) % 256 == 0:
                progress(dict(event="PROGRESS", decoded_counts={s: len(v) for s, v in rows.items()},
                              raw_bytes=total_bytes, last_path=row["path"]))
        if {s: len(v) for s, v in rows.items()} != expected:
            raise ValueError("Complete actual source counts differ from the explicit ImageNet contract")
        if any(_signature(path) != signature for path, signature in directories):
            raise ValueError("A source directory changed while the exhaustive inventory was running")
        splits = {split: dict(count=len(part), raw_bytes=total_bytes[split],
                             records_sha256=_digest(part), images=part) for split, part in rows.items()}
        content = dict(dataset="ILSVRC2012_ImageNet1k", class_count=expected_classes,
                       wnids=sorted(classes["train"]), preprocessing=copy.deepcopy(adapter.PREPROCESSING), splits=splits)
        progress(dict(event="ALL_IMAGES_DECODED", decoded_counts=expected,
                      content_identity_sha256=_digest(content)))
        report = dict(schema=RAW_SCHEMA, status=RAW_STATUS, data_admitted=False,
                      started_at_utc=intent["started_at_utc"], completed_at_utc=_now(),
                      decoder=decoder, portable_content=content, content_identity_sha256=_digest(content),
                      physical_sources={s: str(p) for s, p in roots.items()},
                      excluded_sidecars=excluded, concurrency=dict(workers=workers, max_inflight=max_inflight),
                      producer=dict(job_id=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(),
                                    native_completion="EXTERNAL_REQUIRED", host_completion="EXTERNAL_REQUIRED"))
        payload = Path(str(output) + ".payload")
        _write_once(payload, report)
        os.link(payload, output)  # Atomic no-clobber publication; retain the payload hardlink.
        _fsync_directory(output.parent)
        progress(dict(event="PUBLISHED", report_sha256=_sha(output.read_bytes()), status=RAW_STATUS))
        return dict(path=str(output), sha256=_sha(output.read_bytes()), status=RAW_STATUS,
                    counts=expected, data_admitted=False)
    except BaseException as error:
        failure = dict(schema="imagenet-raw-image-inventory-failure-v1", status="FAILED_NO_DATA_ADMISSION",
                       at_utc=_now(), error_type=type(error).__name__, error=str(error),
                       decoded_counts={s: len(v) for s, v in rows.items()}, data_admitted=False)
        if journal is not None:
            progress(dict(event="FAILED", **failure))
        _write_once(str(output) + ".failure.json", failure)
        raise
    finally:
        if journal is not None:
            journal.close()


def validate_raw_inventory(report_path, report_sha256, *, expected_classes=1000,
                           expected_train_images=1281167, expected_val_images=50000):
    """Validate immutable raw report only; never re-open/decode source images."""
    path = Path(report_path)
    if Path(str(path) + ".failure.json").exists():
        raise ValueError("A failure receipt prevents accepting this raw inventory namespace")
    report = adapter._read_json(path, report_sha256, maximum=1024 * 1024 * 1024)
    if (set(report) != {"schema", "status", "data_admitted", "started_at_utc", "completed_at_utc",
                        "decoder", "portable_content", "content_identity_sha256", "physical_sources",
                        "excluded_sidecars", "concurrency", "producer"}
            or report.get("schema") != RAW_SCHEMA or report.get("status") != RAW_STATUS
            or report.get("data_admitted") is not False
            or report.get("decoder") != decoder_binding()):
        raise ValueError("Actual raw byte/decode report or exact decoder identity mismatch")
    roots = report.get("physical_sources")
    if (not isinstance(roots, dict) or set(roots) != {"train", "val"}
            or any(not isinstance(value, str) or not Path(value).is_absolute()
                   or str(Path(value)) != value or any(part in (".", "..") for part in Path(value).parts)
                   for value in roots.values())
            or Path(roots["train"]).name != "train" or Path(roots["val"]).name != "val"
            or Path(roots["train"]).parent != Path(roots["val"]).parent):
        raise ValueError("Raw source provenance requires canonical absolute sibling train/val paths")
    exclusions = report.get("excluded_sidecars")
    if (not isinstance(exclusions, list)
            or any(not isinstance(row, dict) or row.get("split") != "train" for row in exclusions)):
        raise ValueError("Raw excluded-sidecar provenance is invalid")
    recipes = []
    for row in exclusions:
        if row.get("status") == "UNREADABLE_EXCLUDED":
            if (set(row) != {"split", "name", "bytes", "status", "uid", "mode", "mtime_seconds", "read_attempt"}
                    or row["read_attempt"] != dict(operation="open_read_O_NOFOLLOW", outcome="PermissionError", errno=errno.EACCES)):
                raise ValueError("Unreadable exclusion lacks its actual permission-denied read attempt")
            recipes.append({k: v for k, v in row.items() if k not in ("split", "read_attempt")})
        elif set(row) == {"split", "name", "bytes", "sha256"}:
            recipes.append({k: row[k] for k in ("name", "bytes", "sha256")})
        else:
            raise ValueError("Excluded sidecar provenance has an unknown schema")
    _sidecars(recipes)
    concurrency = report.get("concurrency")
    if (not isinstance(concurrency, dict) or set(concurrency) != {"workers", "max_inflight"}
            or type(concurrency["workers"]) is not int or type(concurrency["max_inflight"]) is not int
            or not 1 <= concurrency["workers"] <= 4
            or not concurrency["workers"] <= concurrency["max_inflight"] <= 16):
        raise ValueError("Raw producer concurrency was not explicitly bounded")
    producer = report.get("producer")
    if (not isinstance(producer, dict) or set(producer) != {"job_id", "pid", "native_completion", "host_completion"}
            or type(producer["pid"]) is not int or producer["pid"] < 1
            or producer["job_id"] is not None and (not isinstance(producer["job_id"], str)
                or re.fullmatch(r"[1-9][0-9]*", producer["job_id"]) is None)
            or producer["native_completion"] != "EXTERNAL_REQUIRED"
            or producer["host_completion"] != "EXTERNAL_REQUIRED"):
        raise ValueError("Raw report cannot substitute its own native/host completion")
    try:
        started = datetime.fromisoformat(report["started_at_utc"])
        completed = datetime.fromisoformat(report["completed_at_utc"])
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise ValueError("Invalid raw producer time interval")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid raw producer time interval") from error
    content = report.get("portable_content")
    if (not isinstance(content, dict) or set(content) != {"dataset", "class_count", "wnids", "preprocessing", "splits"}
            or content["dataset"] != "ILSVRC2012_ImageNet1k" or content["class_count"] != expected_classes
            or type(content["class_count"]) is not int or content["preprocessing"] != adapter.PREPROCESSING
            or report.get("content_identity_sha256") != _digest(content)
            or set(content["splits"]) != {"train", "val"}):
        raise ValueError("Raw content identity/count/preprocessing mismatch")
    wnids = content["wnids"]
    if (not isinstance(wnids, list) or len(wnids) != expected_classes or wnids != sorted(set(wnids))
            or any(_WNID.fullmatch(str(x)) is None for x in wnids)):
        raise ValueError("Invalid exact raw WNID set")
    expected = {"train": expected_train_images, "val": expected_val_images}
    seen = set()
    for split, part in content["splits"].items():
        rows = part.get("images")
        if (not isinstance(rows, list) or len(rows) != expected[split]
                or type(part.get("count")) is not int or part["count"] != len(rows)
                or part.get("records_sha256") != _digest(rows)):
            raise ValueError("Incomplete raw split/decode inventory")
        paths, classes = [], set()
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"path", "wnid", "bytes", "sha256", "rgb160_sha256"}
                    or row["wnid"] not in wnids or type(row["bytes"]) is not int or row["bytes"] < 1
                    or any(_SHA.fullmatch(str(row[k])) is None for k in ("sha256", "rgb160_sha256"))):
                raise ValueError("Invalid decoded source-file record")
            adapter._relative_image_path(row["path"], split, row["wnid"])
            if row["path"] in seen:
                raise ValueError("Duplicate raw source-file record")
            seen.add(row["path"]); paths.append(row["path"]); classes.add(row["wnid"])
        if paths != sorted(paths) or classes != set(wnids) or part.get("raw_bytes") != sum(r["bytes"] for r in rows):
            raise ValueError("Raw canonical ordering/class set/byte totals differ")
    return report


def join_raw_inventory(report_path, report_sha256, class_language_path, class_language_sha256,
                       feature_cache_path, feature_cache_sha256, output_path, *,
                       expected_classes=1000, expected_train_images=1281167, expected_val_images=50000):
    """Join genuine raw report with exact class/Qwen bindings, without image IO."""
    report = validate_raw_inventory(report_path, report_sha256, expected_classes=expected_classes,
                    expected_train_images=expected_train_images, expected_val_images=expected_val_images)
    classes, _ = adapter.load_class_language(class_language_path, class_language_sha256,
                    feature_cache_path, feature_cache_sha256, expected_classes=expected_classes)
    content = report["portable_content"]
    if set(content["wnids"]) != set(classes["classes"]):
        raise ValueError("Actual raw WNIDs differ from actual class-name/Qwen binding")
    splits = {}
    for split, part in content["splits"].items():
        rows = [{key: row[key] for key in ("path", "wnid", "bytes", "sha256")} for row in part["images"]]
        splits[split] = dict(count=len(rows), records_sha256=adapter.state_fingerprint(rows), images=rows)
    manifest = dict(schema=adapter.DATA_SCHEMA, dataset=content["dataset"],
                    format="ImageFolder/train_and_val_WNID_directories", class_count=expected_classes,
                    class_language_sha256=class_language_sha256, feature_cache_sha256=feature_cache_sha256,
                    preprocessing=copy.deepcopy(adapter.PREPROCESSING), splits=splits)
    adapter._validate_manifest(manifest, classes, class_language_sha256, feature_cache_sha256, expected_classes)
    binding = _write_once(output_path, manifest)
    return dict(schema="imagenet-raw-inventory-class-language-join-v1", status="JOINED_REQUIRES_EXTERNAL_ADMISSION",
                raw_report=dict(path=str(report_path), sha256=report_sha256),
                raw_content_identity_sha256=report["content_identity_sha256"],
                class_language=dict(path=str(class_language_path), sha256=class_language_sha256),
                feature_cache=dict(path=str(feature_cache_path), sha256=feature_cache_sha256),
                manifest=binding, images_rehashed_or_decoded=0, data_admitted=False)
