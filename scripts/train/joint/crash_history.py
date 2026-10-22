"""Read retained crash-attempt history without rewriting any historical bytes.

This module does no checkpoint loading and never changes a file. A parent whose
workers continued beyond the restored checkpoint contributes events only through
that checkpoint. Its later rows remain preserved as lost, unsaved work.
"""

import csv
import math
from pathlib import Path
import re

from scripts.train.joint.contract import bound, need


CADENCE = 20000


def _start_and_parent(request):
    crash, continuation = request.get("crash_recovery"), request.get("continuation")
    need(not (crash is not None and continuation is not None),
         "A request cannot be both crash recovery and scheduled continuation")
    if crash is not None:
        need("checkpoint_inspection" not in crash, "Ambiguous crash inspection binding")
        parent = bound(crash["parent_request"])
        receipt = bound(crash["inspection"])
        need(receipt["schema"] == "joint-crash-checkpoint-inspection-v1"
             and receipt["status"] == "PASS"
             and receipt["parent_request_sha256"] == crash["parent_request"]["sha256"]
             and receipt["run_id"] == request["run_id"]
             and receipt["source_revision"] == request["source_revision"],
             "Unbound crash history inspection")
        start = receipt["global_step"]
    elif continuation is not None:
        parent = bound(continuation["parent_request"])
        receipt = bound(continuation["completion"])
        need(receipt["status"] == "PASS" and receipt["run_id"] == request["run_id"]
             and receipt["request_sha256"] == continuation["parent_request"]["sha256"],
             "Unbound scheduled history completion")
        start = receipt["global_step"]
    else:
        return 0, None
    need(type(start) is int and 0 < start < request["target_steps"]
         and start % CADENCE == 0, "Invalid retained history boundary")
    need(parent["run_id"] == request["run_id"]
         and parent["source_revision"] == request["source_revision"]
         and parent["target_steps"] == request["target_steps"]
         and parent["world_size"] == request["world_size"]
         and parent["active_streams"] == request["active_streams"],
         "Historical attempt belongs to another scientific identity")
    return start, parent


def history_segments(request, target=None):
    """Return chronological immutable CSV/output segments with inclusive cutoffs.

    A segment owns optimizer milestones ``start_step < step <= through_step``.
    CSV rows at zero-based ``step-1`` map to their actual optimizer milestone.
    Empty segments are retained so a zero-progress failed attempt cannot hide an
    unexpected checkpoint or duplicated boundary validation.
    """
    target = request["target_steps"] if target is None else target
    need(type(target) is int and 0 < target <= request["target_steps"]
         and target % CADENCE == 0, "Invalid history target")
    rows, seen_outputs, seen_logs = [], set(), set()
    current, through = request, target
    for _ in range(32):
        output, log = Path(current["output"]), Path(current["local_logs"])
        need(output.is_absolute() and log.is_absolute()
             and ".." not in output.parts and ".." not in log.parts,
             "Unsafe retained attempt path")
        resolved_output, resolved_log = output.resolve(), log.resolve()
        need(all(resolved_output != previous and resolved_output not in previous.parents
                 and previous not in resolved_output.parents for previous in seen_outputs)
             and all(resolved_log != previous and resolved_log not in previous.parents
                     and previous not in resolved_log.parents for previous in seen_logs),
             "Repeated history output/log root, nested attempt or request cycle")
        seen_outputs.add(resolved_output); seen_logs.add(resolved_log)
        start, parent = _start_and_parent(current)
        need(start <= through, "History cutoff precedes its restored state")
        relative = Path("csv") / f"step{start:09d}" / "metrics.csv"
        direct, mirrored = log / relative, output / "log-mirror" / relative
        path = direct if direct.is_file() else mirrored
        rows.append(dict(request=current, start_step=start, through_step=through,
                         csv_path=path, output=output))
        if parent is None:
            return rows[::-1]
        current, through = parent, start
    raise ValueError("Retained crash history is cyclic or exceeds32 attempts")


def collect_milestones(request, target=None):
    """Collect exactly one immutable checkpoint per owned optimizer milestone.

    An unexpected newer parent checkpoint is not silently ignored. Its existence
    requires the highest-checkpoint selection/inspection to be reconciled first.
    ``last.ckpt`` is an alias, never a second retained milestone.
    """
    result = {}
    for segment in history_segments(request, target):
        directory = segment["output"] / "checkpoints"
        if not directory.is_dir():
            need(segment["start_step"] == segment["through_step"] and not directory.exists(),
                 "Missing retained checkpoint directory")
            # A failed resumed process can end before its first new20k save.
            # Such a zero-owned segment contributes no immutable milestone.
            continue
        for path in directory.glob("*.ckpt"):
            if path.name == "last.ckpt":
                need(path.is_symlink(), "Last checkpoint must be a symlink")
                continue
            match = re.search(r"([0-9]{9})\.ckpt$", path.name)
            need(match is not None and "epoch" in path.name and "step" in path.name
                 and path.is_file() and not path.is_symlink(),
                 "Unexpected or mutable retained checkpoint")
            step = int(match[1])
            need(step % CADENCE == 0 and segment["start_step"] < step
                 <= segment["through_step"],
                 "Unreconciled newer checkpoint or repeated resume-boundary save")
            need(step not in result, "Duplicate immutable checkpoint across attempts")
            result[step] = path
    expected = set(range(CADENCE, (request["target_steps"] if target is None else target)+1,
                         CADENCE))
    need(set(result) == expected, "Missing retained checkpoint history")
    return result


def _true_validation_step(raw):
    number = float(raw)
    need(math.isfinite(number) and number.is_integer() and number >= 0,
         "Invalid validation CSV step")
    number = int(number)
    if number > 0 and number % CADENCE == 0:
        return number
    if (number + 1) % CADENCE == 0:
        return number + 1
    raise ValueError("Validation does not occur at the20k optimizer cadence")


def collect_validation(request, target=None, streams=None):
    """Merge full finite events and reject actual repeated boundary validation."""
    streams = request["active_streams"] if streams is None else streams
    metric = {"image": "image_normalized_mse", "proprio": "proprio_normalized_mse",
              "action": "action_command_mse"}
    expected_keys = {f"val/libero_gripper/{kind}/{metric[name]}"
                     for name in streams for kind in ("forecast", "completion")}
    result = {}
    for segment in history_segments(request, target):
        local, raw_steps = {}, {}
        if not segment["csv_path"].is_file():
            need(segment["start_step"] == segment["through_step"]
                 and not segment["csv_path"].exists(), "Missing retained validation CSV")
            # No scheduled optimizer milestone is owned by this failed attempt.
            # Existing files, if any, are still scanned below for replayed events.
            continue
        with segment["csv_path"].open() as stream:
            for row in csv.DictReader(stream):
                values = {key: float(value) for key, value in row.items()
                          if key.startswith("val/libero_gripper/") and value}
                if not values:
                    continue
                step = _true_validation_step(row["step"])
                if step > segment["through_step"]:
                    # Preserve failed-attempt data beyond the actual restored
                    # checkpoint without incorporating it into completed history.
                    continue
                need(step > segment["start_step"],
                     "Repeated validation at or before a resumed boundary")
                need(all(math.isfinite(value) for value in values.values())
                     and set(values) <= expected_keys, "Nonfinite or unexpected validation metric")
                if step in raw_steps:
                    need(raw_steps[step] == row["step"], "Ambiguous validation step convention")
                raw_steps[step] = row["step"]
                event = local.setdefault(step, {})
                need(not (event.keys() & values.keys()), "Repeated validation metric/event")
                event.update(values)
        need(not (result.keys() & local.keys()), "Repeated validation across crash attempts")
        need(all(set(event) == expected_keys for event in local.values()),
             "Incomplete scheduled validation event")
        result.update(local)
    expected = set(range(CADENCE, (request["target_steps"] if target is None else target)+1,
                         CADENCE))
    need(set(result) == expected, "Missing finite scheduled validation history")
    return result
