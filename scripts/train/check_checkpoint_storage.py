#!/usr/bin/env python3
"""Fail closed unless a target filesystem can retain every planned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PASS_STATUS = "CHECKPOINT_STORAGE_VALIDATED"
FAIL_STATUS = "CHECKPOINT_STORAGE_FAILED"


class StorageError(RuntimeError):
    """Raised when filesystem evidence cannot prove the storage contract."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _optional_checkpoint_size(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive byte count")
    return parsed


def _reserve(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("safety reserve must be positive")
    return parsed


def storage_evidence(
    target: Path,
    *,
    planned_checkpoint_count: int,
    measured_checkpoint_bytes: int | None,
    expected_checkpoint_bytes: int | None,
    safety_reserve_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure a filesystem and calculate a conservative retention budget."""

    resolved = target.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise StorageError(f"filesystem target is not a directory: {resolved}")
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise StorageError(f"filesystem target is not writable/searchable: {resolved}")
    if planned_checkpoint_count <= 0:
        raise StorageError("planned checkpoint count must be positive")
    if measured_checkpoint_bytes is None and expected_checkpoint_bytes is None:
        raise StorageError(
            "at least one of measured or expected checkpoint bytes is required"
        )
    checkpoint_sizes = [
        size
        for size in (measured_checkpoint_bytes, expected_checkpoint_bytes)
        if size is not None
    ]
    if any(size <= 0 for size in checkpoint_sizes):
        raise StorageError("checkpoint byte counts must be positive")
    if safety_reserve_bytes <= 0:
        raise StorageError("safety reserve must be positive")

    stats = os.stat(resolved)
    filesystem = os.statvfs(resolved)
    fragment_size = int(filesystem.f_frsize)
    if fragment_size <= 0:
        raise StorageError("filesystem reported a non-positive fragment size")
    available_bytes = int(filesystem.f_bavail) * fragment_size
    free_bytes = int(filesystem.f_bfree) * fragment_size
    capacity_bytes = int(filesystem.f_blocks) * fragment_size
    if not 0 <= available_bytes <= free_bytes <= capacity_bytes:
        raise StorageError("filesystem reported inconsistent capacity counters")
    read_only_flag = int(getattr(os, "ST_RDONLY", 1))
    is_read_only = bool(int(filesystem.f_flag) & read_only_flag)

    bytes_per_checkpoint = max(checkpoint_sizes)
    checkpoint_budget_bytes = planned_checkpoint_count * bytes_per_checkpoint
    required_available_bytes = checkpoint_budget_bytes + safety_reserve_bytes
    headroom_after_checkpoints_bytes = available_bytes - checkpoint_budget_bytes
    report = {
        "target": str(resolved),
        "filesystem_device": int(stats.st_dev),
        "filesystem_fragment_size_bytes": fragment_size,
        "filesystem_capacity_bytes": capacity_bytes,
        "filesystem_free_bytes": free_bytes,
        "filesystem_available_bytes": available_bytes,
        "filesystem_read_only": is_read_only,
        "planned_checkpoint_count": planned_checkpoint_count,
        "measured_checkpoint_bytes": measured_checkpoint_bytes,
        "expected_checkpoint_bytes": expected_checkpoint_bytes,
        "bytes_per_checkpoint_budget": bytes_per_checkpoint,
        "checkpoint_budget_bytes": checkpoint_budget_bytes,
        "safety_reserve_bytes": safety_reserve_bytes,
        "required_available_bytes": required_available_bytes,
        "headroom_after_checkpoints_bytes": headroom_after_checkpoints_bytes,
    }
    failures = []
    if is_read_only:
        failures.append(
            {"field": "filesystem_read_only", "expected": False, "observed": True}
        )
    if available_bytes < required_available_bytes:
        failures.append(
            {
                "field": "filesystem_available_bytes",
                "expected_at_least": required_available_bytes,
                "observed": available_bytes,
                "shortfall_bytes": required_available_bytes - available_bytes,
            }
        )
    return report, failures


def distributed_budget(plan: dict[str, Any], target: Path, planned_count: int, measured_bytes: int, reserve: int, local_quota: dict[str, Any], now: float) -> dict[str, Any]:
    """Require measured pair budgets, current quota, and verified mirror evidence."""
    def require(ok, message):
        if not ok: raise StorageError(message)
    def positive(value): return isinstance(value,int) and not isinstance(value,bool) and value>0
    require(plan.get('schema_version')==1 and plan.get('status')=='DISTRIBUTED_RETENTION_READY','Unverified distributed retention plan')
    rows=plan.get('runs');require(isinstance(rows,list) and rows,'Missing retained runs')
    targets=[];local_budget=0;archive_budget=0;archive_files=0
    require(positive(reserve),'Invalid storage reserve')
    for row in rows:
        require(Path(row['local_root']).is_absolute(),'Local output must be absolute')
        root=Path(row['local_root']).resolve()
        require(len(root.parts)>=4,'Local output too broad')
        require(root not in targets,'Duplicate output root');targets.append(root)
        size=row.get('measured_checkpoint_bytes');scheduled=row.get('scheduled_count');extra=row.get('additional_checkpoint_count');retain=row.get('retain_local')
        require(positive(size) and positive(scheduled) and positive(extra) and positive(retain) and retain>=2,'Missing measured size/cadence/recovery budget')
        require(row.get('checkpoint_measurement_status')=='STRICT_FULL_STATE_RELOAD_PASS','Checkpoint measurement lacks strict reload')
        require(row.get('preserve_all_cadence') is True,'Every cadence checkpoint must be archived')
        local_budget+=(retain+1)*size
        archive_budget+=(scheduled+extra+1)*size
        archive_files+=scheduled+extra+1
    target=target.resolve();require(target in targets,'Target absent from exact retention plan')
    row=rows[targets.index(target)]
    require(row['scheduled_count']==planned_count and row['measured_checkpoint_bytes']==measured_bytes,'Target count or measured size mismatch')
    archive=plan.get('archive_quota',{})
    require(archive.get('status')=='UID_QUOTA_VALIDATED','Archive UID quota unverified')
    require(positive(archive.get('uid')) and archive.get('uid')==plan.get('archive_uid') and archive.get('username')==plan.get('archive_username') and archive.get('username'),'Archive account identity mismatch')
    require(isinstance(archive.get('checked_at_unix'),(int,float)) and 0<=now-archive['checked_at_unix']<=600,'Archive quota evidence stale/future')
    require(positive(archive.get('available_bytes')) and positive(archive.get('available_inodes')),'Missing archive block/inode headroom')
    require(archive.get('authority') in ('native_lfs_quota','native_zfs_user_and_group_quota','native_nfs4_owner_quota'),'Unsupported quota authority')
    require(Path(archive['target']).is_absolute(),'Archive quota root must be absolute')
    mount=Path(archive['target']).resolve()
    require(mount.is_absolute() and len(mount.parts)>=4,'Unspecific archive root')
    for row in rows:
        remote=Path(row['remote_root'])
        require(remote.is_absolute() and remote!=mount and mount in remote.parents,'Archive target escapes verified quota root')
    mirror=plan.get('mirror_cycle',{})
    require(mirror.get('status')=='CHECKPOINT_MIRROR_CYCLE_VERIFIED' and mirror.get('cycle_errors')==0,'Missing zero-error checkpoint mirror cycle')
    require(mirror.get('remote_sha256_verified') is True,'Mirror lacks remote SHA proof')
    require(isinstance(mirror.get('checked_at_unix'),(int,float)) and 0<=now-mirror['checked_at_unix']<=600,'Mirror cycle stale/future')
    require(set(mirror.get('parent_run_ids',[]))==set(row['id'] for row in rows),'Mirror cycle parent identities mismatch')
    require(local_quota.get('available_bytes',-1)>=local_budget+reserve,'Local account quota cannot stage pair recovery+inflight')
    require(local_quota.get('available_inodes',-1)>=(sum(row['retain_local']+1 for row in rows)+32),'Local inode quota insufficient')
    require(archive['available_bytes']>=archive_budget+reserve,'Archive quota cannot preserve all checkpoints+inflight')
    require(archive['available_inodes']>=archive_files+32,'Archive inode quota insufficient')
    return {'layout':'distributed_verified_retention','pair_local_checkpoint_budget_bytes':local_budget,'pair_archive_checkpoint_budget_bytes':archive_budget,'pair_archive_checkpoint_budget_files':archive_files,'local_quota':local_quota,'archive_quota':archive,'mirror_cycle':mirror,'planned_checkpoint_count':planned_count,'measured_checkpoint_bytes':measured_bytes,'safety_reserve_bytes':reserve}


def distributed_evidence(args):
    import importlib.util
    path=Path(os.environ['ICE_DISTRIBUTED_RETENTION_PLAN']).resolve(strict=True)
    expected=os.environ['ICE_DISTRIBUTED_RETENTION_PLAN_SHA256']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise StorageError('Retention plan SHA mismatch')
    plan=json.loads(path.read_text())
    for field in ('archive_quota','mirror_cycle'):
        bound=plan[field+'_receipt'];receipt=Path(bound['path']).resolve(strict=True)
        if hashlib.sha256(receipt.read_bytes()).hexdigest()!=bound['sha256']:raise StorageError('Bound '+field+' receipt SHA mismatch')
        if json.loads(receipt.read_text())!=plan[field]:raise StorageError('Bound '+field+' receipt content mismatch')
    mirror=Path(__file__).parents[1]/'ice'/'ice_checkpoint_mirror.py'
    spec=importlib.util.spec_from_file_location('storage_mirror_quota',mirror);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    def existing_parent(raw):
        value=Path(raw).resolve()
        while not value.is_dir():
            if value==value.parent:raise StorageError('No existing storage parent')
            value=value.parent
        return value
    local_mount=Path(plan['local_quota_mount']).resolve(strict=True)
    for row in plan['runs']:
        if os.stat(existing_parent(row['local_root'])).st_dev!=os.stat(local_mount).st_dev:raise StorageError('Pair output on another quota filesystem')
    target_parent=existing_parent(args.target)
    quota=module.lustre_quota(local_mount,target_parent,30)
    if args.measured_checkpoint_bytes is None:raise StorageError('Distributed mode requires actual strict smoke size')
    if args.expected_checkpoint_bytes is not None and args.expected_checkpoint_bytes>args.measured_checkpoint_bytes:raise StorageError('Measured checkpoint does not cover expected size')
    report=distributed_budget(plan,args.target,args.planned_checkpoint_count,args.measured_checkpoint_bytes,args.safety_reserve_bytes,quota,time.time())
    fs_report,failures=storage_evidence(target_parent,planned_checkpoint_count=sum(row['retain_local']+1 for row in plan['runs']),measured_checkpoint_bytes=max(row['measured_checkpoint_bytes'] for row in plan['runs']),expected_checkpoint_bytes=None,safety_reserve_bytes=args.safety_reserve_bytes)
    return {**report,'local_filesystem':fs_report,'retention_plan_sha256':expected},failures


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite storage evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument(
        "--planned-checkpoint-count", required=True, type=_positive_int
    )
    parser.add_argument(
        "--measured-checkpoint-bytes", type=_optional_checkpoint_size
    )
    parser.add_argument(
        "--expected-checkpoint-bytes", type=_optional_checkpoint_size
    )
    parser.add_argument("--safety-reserve-bytes", required=True, type=_reserve)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.measured_checkpoint_bytes is None
        and args.expected_checkpoint_bytes is None
    ):
        parser.error(
            "one or both of --measured-checkpoint-bytes and "
            "--expected-checkpoint-bytes is required"
        )
    return args


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": FAIL_STATUS,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "failures": [],
    }
    exit_code = 1
    try:
        if os.environ.get('ICE_DISTRIBUTED_RETENTION_PLAN'):
            report, failures = distributed_evidence(args)
        else:
            report, failures = storage_evidence(
            args.target,
            planned_checkpoint_count=args.planned_checkpoint_count,
            measured_checkpoint_bytes=args.measured_checkpoint_bytes,
            expected_checkpoint_bytes=args.expected_checkpoint_bytes,
            safety_reserve_bytes=args.safety_reserve_bytes,
        )
        evidence.update({"storage": report, "failures": failures})
        if not failures:
            evidence["status"] = PASS_STATUS
            exit_code = 0
    except (OSError, StorageError, KeyError, ValueError) as exc:
        evidence["failures"] = [{"field": "storage", "error": str(exc)}]
    _atomic_json(output, evidence)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
