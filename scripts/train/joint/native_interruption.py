"""Read-only native interruption evidence for the maintained Elmo service.

This module never cancels, submits, requeues, or imports ML libraries. A capture
is evidence, not restart authorization. Terminal history must be captured while
Slurm still has it: an absent record is deliberately not accepted as success.
"""
from datetime import datetime, timezone
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess

from scripts.train.joint.contract import bound, need, sha, write_once


TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "CANCELLED", "OUT_OF_MEMORY"}
INFRASTRUCTURE = {"TIMEOUT", "NODE_FAIL", "PREEMPTED"}
NATIVE_KEYS = ("JobId", "JobName", "UserId", "Account", "QOS", "Partition", "ReqNodeList",
               "NumCPUs", "MinMemoryNode", "TimeLimit", "Requeue", "Restarts", "Command")


def fields(raw):
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)", raw)
    need(len(pairs) == len(dict(pairs)), "Duplicate or multiple native job records")
    return dict(pairs)


def process_identity(expected, *, cpu=False):
    owner = re.fullmatch(r"[^()]+\((\d+)\)", expected["UserId"])
    need(owner is not None and int(owner[1]) == os.getuid()
         and expected["JobId"] == os.environ.get("SLURM_JOB_ID")
         and expected["NodeList"] == os.environ.get("SLURMD_NODENAME")
         and os.environ.get("SLURM_RESTART_COUNT", "0") == "0",
         "Actual process UID/node/fresh allocation differs from the bound native producer")
    if cpu:
        need(not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS"),
             "CPU producer unexpectedly owns GPUs")


def destination_probe(output, job, required_bytes):
    """Preserve a fresh exclusive fsync/readback probe beside the real output."""
    import shutil

    output = Path(output)
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
         and not output.exists() and type(required_bytes) is int and required_bytes > 0
         and str(job).isdecimal(), "Exact fresh production destination/storage reserve required")
    free = shutil.disk_usage(output.parent).free
    need(free > required_bytes, "Insufficient actual production destination free space")
    path = output.parent / (".native-recovery-fsync-" + str(job) + ".probe")
    payload = (str(output) + "\n" + str(job) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    need(path.read_bytes() == payload, "Actual production destination readback failed")
    fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(output=str(output), probe_path=str(path), probe_sha256=sha(path),
                free_bytes=free, reserved_bytes=required_bytes, actual_write_fsync_readback=True)


def native_identity(raw, expected, state):
    """Bind exact owner, node, resources, walltime and fresh-allocation identity."""
    f = fields(raw)
    need(set(expected) == set(NATIVE_KEYS) | {"NodeList", "TresPerNode"},
         "Complete exact parent native resource binding required")
    need(all(f.get(k, "") == str(v) for k, v in expected.items()),
         "Actual native identity/resources differ from the bound parent")
    need(expected["Requeue"] == expected["Restarts"] == "0"
         and f.get("JobState") == state and f.get("NumNodes") == "1"
         and f.get("NumTasks") == "1", "Wrong fresh single-node native allocation")
    need(f.get("ReqTRES") and f.get("AllocTRES"), "Actual requested and allocated TRES missing")
    for key in ("ReqTRES", "AllocTRES"):
        parts = f[key].split(",")
        need("cpu=" + expected["NumCPUs"] in parts and "node=1" in parts,
             "Native CPU/node TRES differ")
        mem = expected["MinMemoryNode"]
        allowed = {"mem=" + mem}
        if mem.endswith("G"):
            allowed.add("mem=" + str(int(mem[:-1]) * 1024) + "M")
        need(bool(set(parts) & allowed), "Native allocated memory differs")
        gpu = [p for p in parts if "gpu" in p.lower()]
        if expected["TresPerNode"] == "gres/gpu:h100:4":
            need(set(gpu) <= {"gres/gpu=4", "gres/gpu:h100=4"},
                 "Conflicting optional GPU TRES")
        else:
            need(expected["TresPerNode"] == "" and not gpu,
                 "CPU producer unexpectedly owns GPUs")
    if expected["TresPerNode"] == "gres/gpu:h100:4":
        need(f.get("JOB_GRES") == "gpu:h100:4", "Actual four-H100 JOB_GRES missing")
        allocated = f.get("GRES", "")
        if state == "RUNNING" or allocated:
            match = re.fullmatch(r"gpu:h100:4\(IDX:([0-9,-]+)\)", allocated)
            need(match is not None, "Actual four-H100 allocation indices missing")
            indices = []
            for part in match[1].split(","):
                endpoints = part.split("-")
                need(len(endpoints) <= 2, "Invalid GPU index range")
                first, last = int(endpoints[0]), int(endpoints[-1])
                need(0 <= first <= last < 8, "Invalid H100 index range")
                indices.extend(range(first, last + 1))
            need(len(indices) == len(set(indices)) == 4, "GPU indices are not four distinct devices")
    else:
        need(f.get("JOB_GRES", "(null)") == "(null)" and not f.get("GRES"),
             "CPU producer has nonempty native GPU allocation")
    if state in TERMINAL:
        end = f.get("EndTime", "")
        need(end not in {"", "Unknown", "None", "N/A"}
             and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", end)
             and re.fullmatch(r"\d+:\d+", f.get("ExitCode", "")),
             "Authoritative terminal EndTime/ExitCode missing")
    return f


def classify_failure(terminal, worker_tail):
    """Only mechanical infrastructure evidence admits automatic recovery.

    Cancellation, OOM and arbitrary FAILED remain blocked for diagnosis. A
    future operational-stop extension must bind its actual guarded cancellation
    and prior crash; it must not reinterpret a human cancellation here.
    """
    state = terminal["JobState"]
    scientific = re.compile(r"(?:nan loss|nonfinite|non-finite|FloatingPointError|"
                            r"(?:loss|mse)[^\n]{0,80}?(?:[:=(]\s*|\s+)[+-]?(?:nan|inf(?:inity)?)(?![a-z])|"
                            r"CUDA error: device-side assert|AssertionError|ValueError|"
                            r"out of memory|OutOfMemoryError)", re.I)
    need(not scientific.search(worker_tail),
         "Scientific, deterministic, or capacity failure requires diagnosis")
    if state in INFRASTRUCTURE:
        return state
    if state == "FAILED":
        patterns = (r"(?:ProcessGroupNCCL|NCCL)[^\n]*(?:timed out|timeout|watchdog)",
                    r"Watchdog caught collective operation timeout")
        need(terminal["ExitCode"] != "0:0" and any(re.search(p, worker_tail, re.I) for p in patterns),
             "FAILED without preserved infrastructure worker-crash evidence")
        return "VERIFIED_NCCL_WORKER_CRASH"
    raise ValueError("Terminal state is not automatically recoverable: " + state)


def check_steps(raw, job):
    """squeue --steps is live: any row for the failed parent blocks release."""
    rows = [line.split("|", 1)[0].strip() for line in raw.splitlines() if line.strip()]
    need(not any(row == str(job) or row.startswith(str(job) + ".") for row in rows),
         "Failed parent still has active steps")
    return True


def check_queue(raw, names, allowed_job=None):
    """Complete exact owned queue, using explicit scientific names from all attempts."""
    need(names and len(names) == len(set(names)), "Complete unique scientific writer names required")
    for line in raw.splitlines():
        if not line.strip():
            continue
        columns = line.split("|")
        need(len(columns) == 3, "Unparseable native job queue")
        job, name, state = (v.strip() for v in columns)
        if name in names:
            need(job == allowed_job, "Other active/pending scientific writer: " + job)
            need(state == "RUNNING", "Current admitted allocation is not RUNNING")
    return True


def read_tail(path):
    """Bound bytes and lines without loading unbounded worker logs."""
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "Actual bounded worker log missing")
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        raw = stream.read(65536)
    return b"\n".join(raw.splitlines()[-64:]).decode("utf8", errors="replace")


def parent_csv_paths(parent):
    from scripts.train.joint.production_history import start_step

    relative = Path("csv") / f"step{start_step(parent):09d}" / "metrics.csv"
    return {"local": str(Path(parent["local_logs"]) / relative),
            "mirror": str(Path(parent["output"]) / "log-mirror" / relative)}


def newest_loss(snapshots):
    """Select newest numeric-step loss row before any finiteness filtering."""
    candidates = []
    for name, snapshot in snapshots.items():
        columns = next(csv.reader([snapshot["header"]]))
        need(len(columns) == len(set(columns)) and "step" in columns, "Ambiguous loss CSV header")
        for position, row in enumerate(csv.DictReader(io.StringIO(snapshot["tail"]), fieldnames=columns)):
            if list(row.values()) == columns:
                continue
            validation = {key: float(value) for key, value in row.items() if isinstance(key, str)
                          and key.startswith("val/") and value not in (None, "")}
            need(all(math.isfinite(value) for value in validation.values()),
                 "Observed validation is nonfinite; scientific diagnosis required")
            metrics = {key: value for key, value in row.items() if isinstance(key, str)
                       and key.lower().startswith("train/") and ("loss" in key.lower() or "mse" in key.lower())
                       and value not in (None, "")}
            if not metrics:
                continue
            need(None not in row and all(value is not None for value in row.values()), "Truncated loss row")
            step = float(row["step"])
            need(math.isfinite(step) and step.is_integer() and step >= 0, "Invalid numeric loss step")
            # NaN and Inf deliberately stay eligible here; newest is selected first.
            numeric = {key: float(value) for key, value in metrics.items()}
            candidates.append(dict(source=name, step=int(step), position=position, metrics=numeric))
    need(candidates, "No actual numeric loss row; scientific state unresolved (absence is not zero-update proof)")
    newest = max(row["step"] for row in candidates)
    selected = {}
    for row in candidates:
        if row["step"] == newest:
            selected[row["source"]] = row
    rows = list(selected.values())
    required = {"train/loss", "train/joint_image_mse", "train/joint_proprio_mse",
                "train/joint_camera_mse", "train/source_0_loss_action"}
    need(all(required <= {key.lower() for key in row["metrics"]} for row in rows),
         "Newest loss row lacks total/image/proprio/camera/action-support metrics")
    need(all(all(math.isfinite(value) for value in row["metrics"].values()) for row in rows),
         "Newest numeric loss row is nonfinite; scientific diagnosis required")
    need(all(row["metrics"] == rows[0]["metrics"] for row in rows), "Local/mirror newest loss rows disagree")
    # Lightning CSV retains the previous value of other-embodiment metrics.
    # Presence alone cannot identify the current batch: require exact agreement
    # of all three current generic metrics with one embodiment's metrics.
    metrics = {key.lower(): value for key, value in rows[0]["metrics"].items()}
    active = [embodiment for embodiment in ("eva_bimanual", "human_bimanual")
              if all(metrics.get(f"train/joint_{stream}_mse/{embodiment}") == metrics[f"train/joint_{stream}_mse"]
                     for stream in ("image", "proprio", "camera"))]
    need(len(active) == 1, "Newest row active-embodiment/stream support is unknown or ambiguous")
    if active[0] == "human_bimanual":
        need(metrics["train/source_0_loss_action"] == 0, "Human batch unexpectedly has action loss")
    else:
        need("train/joint_action_mse" in metrics and "train/joint_action_mse/eva_bimanual" in metrics
             and metrics["train/joint_action_mse"] == metrics["train/joint_action_mse/eva_bimanual"],
             "Robot batch lacks the actual supervised action metric")
    return dict(step=newest, metrics=rows[0]["metrics"], finite=True, selected_rows=rows,
                active_embodiment=active[0], supervised_action=active[0] == "eva_bimanual",
                selection="highest_numeric_step_then_last_loss_row_per_source_before_finiteness")


def loss_snapshot(parent):
    snapshots, sources = {}, {}
    for name, value in parent_csv_paths(parent).items():
        path = Path(value)
        if not path.exists():
            sources[name] = dict(path=value, exists=False)
            continue
        need(path.is_file() and not path.is_symlink(), "Actual loss CSV must be a regular file")
        before = path.stat()
        with path.open("rb") as stream:
            header = stream.readline(65536)
            need(header.endswith(b"\n"), "Loss CSV header exceeds bound/is incomplete")
            offset = max(0, before.st_size - 1048576)
            stream.seek(offset)
            raw = stream.read(1048576)
        after = path.stat()
        need((before.st_size, before.st_mtime_ns, before.st_ino) ==
             (after.st_size, after.st_mtime_ns, after.st_ino), "Loss CSV changed during bounded observation")
        if offset:
            raw = raw.partition(b"\n")[2]  # never parse a partial leading row
        lines = raw.splitlines()
        need(len(lines) >= 64 or offset == 0, "Loss CSV tail exceeds bounded 1MiB window")
        snapshots[name] = dict(header=header.decode("utf8"), tail=b"\n".join(lines[-64:]).decode("utf8"))
        sources[name] = dict(path=value, exists=True, size_bytes=before.st_size,
                             mtime_ns=before.st_mtime_ns, inode=before.st_ino, device=before.st_dev)
    return sources, snapshots


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=25, check=False)
    need(result.returncode == 0, "Native query failed; history unavailable: " + result.stderr[:1000])
    return result.stdout


def capture(request_path):
    """Capture actual parent evidence. Caller must use an owned CPU capture job."""
    q = json.loads(Path(request_path).read_text())
    need(q["schema"] == "native-elmo-interruption-capture-v1", "Wrong capture request")
    parent = bound(q["parent_request"])
    need(parent["kind"] == "native-subtask-world-model-v1" and parent["phase"] == "full",
         "Only exact native Elmo production failure may enter this path")
    job, expected = q["parent_job_id"], q["parent_native"]
    need(str(job).isdecimal() and expected["JobId"] == str(job)
         and expected["JobName"] == parent["job_name"], "Parent job/name/request binding differs")
    need(os.environ.get("SLURM_JOB_ID", "").isdecimal()
         and not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS")
         and os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "Fresh allocated CPU capture required")
    process_identity(q["producer_native"], cpu=True)
    producer = command(["/usr/local/bin/scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o", "-dd"])
    native_identity(producer, q["producer_native"], "RUNNING")
    need(q["producer_native"]["JobId"] == os.environ["SLURM_JOB_ID"]
         and q["producer_native"]["NodeList"] == os.environ["SLURMD_NODENAME"],
         "CPU producer is not this actual job/node")
    accepted = bound(q["parent_running_admission"])
    need(accepted["job_id"] == str(job)
         and accepted["request_sha256"] == q["parent_request"]["sha256"],
         "Actual launch claim does not identify parent request")
    native_identity(accepted["native"], expected, "RUNNING")
    raw = command(["/usr/local/bin/scontrol", "show", "job", str(job), "-o", "-dd"])
    terminal = fields(raw)
    need(terminal.get("JobState") in TERMINAL, "Parent is not terminal")
    native_identity(raw, expected, terminal["JobState"])
    steps = command(["/usr/local/bin/squeue", "--steps", "--noheader", "--jobs", str(job), "--format=%i|%T"])
    check_steps(steps, job)
    tails = {key: read_tail(value) for key, value in q["worker_logs"].items()}
    need(tails, "Preserved bounded stdout/stderr evidence required")
    sources, snapshots = loss_snapshot(parent)
    output = Path(q["output"])
    output.mkdir()  # an existing or partial capture is immutable, never retried
    csv_evidence = {"loss-" + name + "-" + part + ".txt": text for name, snap in snapshots.items()
                    for part, text in snap.items()}
    for name, text in {"parent-native.txt": raw, "parent-live-steps.txt": steps,
                       "producer-running-native.txt": producer, **csv_evidence,
                       **{k + ".tail64.txt": v for k, v in tails.items()}}.items():
        need(Path(name).name == name, "Capture filename escaped evidence directory")
        with (output / name).open("x") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    try:
        loss = newest_loss(snapshots)
        reason = classify_failure(terminal, "\n".join(tails.values()))
    except (ValueError, KeyError) as error:
        write_once(output / "interruption-blocked.json", dict(
            schema="native-elmo-interruption-blocked-v1", status="SCIENTIFIC_OR_INFRASTRUCTURE_DIAGNOSIS_REQUIRED",
            request_sha256=sha(request_path), parent_request=q["parent_request"], parent_job_id=str(job),
            loss_sources=sources, error=str(error), automatically_restart_authorized=False,
            evidence={p.name: dict(path=str(p), sha256=sha(p)) for p in output.iterdir()}))
        raise
    write_once(output / "interruption-capture.json", dict(
        schema="native-elmo-interruption-evidence-v1", status="CAPTURED_REQUIRES_NATIVE_PRODUCER_COMPLETION",
        request_sha256=sha(request_path), parent_request=q["parent_request"], parent_job_id=str(job),
        run_id=parent["run_id"], terminal_state=terminal["JobState"], terminal_exit=terminal["ExitCode"],
        terminal_end=terminal["EndTime"], reason=reason, producer_job_id=os.environ["SLURM_JOB_ID"],
        captured_at=datetime.now(timezone.utc).isoformat(),
        newest_numeric_loss=loss, loss_sources=sources,
        evidence={p.name: dict(path=str(p), sha256=sha(p)) for p in output.iterdir()},
        parent_running_admission=q["parent_running_admission"], parent_native=expected,
        automatically_restart_authorized=False))


if __name__ == "__main__":
    import sys
    capture(sys.argv[1])
