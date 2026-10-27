"""Retain immutable production checkpoints and validation across crash attempts.

Every old file is preserved. The selected production checkpoint defines the
committed prefix of a failed attempt; later unsaved work is audit history only.
"""
import csv
import math
from pathlib import Path
import re

from scripts.train.joint.contract import bound, need, sha
from scripts.train.joint.checkpoint_inspection import attempt_chain, scheduled_filename


def has_interruption(request):
    if request.get("kind") != "native-subtask-world-model-v1":
        return False
    return any(row.get("interruption") is not None for row in attempt_chain(request))


def start_step(request):
    if request.get("interruption"):
        return request["interruption"]["checkpoint"]["global_step"]
    if request.get("continuation"):
        return bound(request["continuation"]["completion"])["global_step"]
    return 0


def segments(request, stop):
    """Chronological attempt ownership intervals: (start, end]."""
    attempts = attempt_chain(request)[::-1]
    result = []
    for index, attempt in enumerate(attempts):
        start = start_step(attempt)
        end = start_step(attempts[index + 1]) if index + 1 < len(attempts) else stop
        need(type(start) is type(end) is int and 0 <= start <= end <= request["target_steps"]
             and start % 20000 == end % 20000 == 0, "Invalid committed attempt history")
        relative = Path("csv") / f"step{start:09d}" / "metrics.csv"
        local, mirror = Path(attempt["local_logs"]) / relative, Path(attempt["output"]) / "log-mirror" / relative
        need(local != mirror, "Local/mirror log namespace collision")
        result.append(dict(request=attempt, start=start, end=end, csv=local if local.is_file() else mirror))
    need(len({str(row["csv"]) for row in result}) == len(result), "Repeated production log namespace")
    return result


def retained_milestones(request, stop):
    milestones, excluded = {}, []
    for row in segments(request, stop):
        directory = Path(row["request"]["output"]) / "checkpoints"
        need(directory.is_dir() and not directory.is_symlink(), "Retained attempt checkpoint directory missing")
        for path in directory.glob("*.ckpt"):
            if path.name == "last.ckpt":
                continue
            need(path.is_file() and not path.is_symlink(), "Invalid retained checkpoint name/type")
            step = scheduled_filename(path.name)
            need(step > 0 and step % 20000 == 0 and step <= request["target_steps"],
                 "Unexpected unscheduled production checkpoint")
            if not row["start"] < step <= row["end"]:
                excluded.append(dict(path=str(path), global_step=step, retained=True))
                continue
            if step in milestones:
                need(milestones[step].resolve() == path.resolve(), "Duplicate committed production milestone")
            milestones[step] = path
    need(set(milestones) == set(range(20000, stop + 1, 20000)), "Committed production history lost a scheduled milestone")
    return milestones, dict(schema="native-elmo-retained-attempt-history-v1", retained_steps=sorted(milestones),
                             files=[dict(global_step=step, path=str(path), sha256=sha(path))
                                    for step, path in sorted(milestones.items())],
                             excluded_from_committed_prefix_but_preserved=excluded)


def scheduled_validation(request, stop, metric_names):
    """Merge disjoint committed prefixes, rejecting duplicate/partial/nonfinite events."""
    validation, evidence = {}, []
    for item in segments(request, stop):
        selected, discarded = {}, set()
        path = item["csv"]
        need(path.is_file() and not path.is_symlink(), "Actual retained validation CSV missing")
        with path.open() as stream:
            for row in csv.DictReader(stream):
                metrics = {k: float(v) for k, v in row.items() if k.startswith("val/") and v}
                if not metrics:
                    continue
                need(all(math.isfinite(value) for value in metrics.values()),
                     "Observed validation is nonfinite, including uncommitted history")
                raw = float(row["step"])
                need(math.isfinite(raw) and raw.is_integer(), "Invalid validation step")
                step = int(raw)
                step = step + 1 if step % 20000 == 19999 else step
                need(step > 0 and step % 20000 == 0 and step <= request["target_steps"],
                     "Unexpected non-scheduled production validation")
                need(step != item["start"], "Resumed attempt repeated checkpoint-boundary validation")
                if not item["start"] < step <= item["end"]:
                    discarded.add(step)
                    continue
                need(set(metrics) <= set(metric_names) and all(math.isfinite(v) for v in metrics.values()),
                     "Unexpected/nonfinite committed validation metric")
                target = selected.setdefault(step, {})
                need(not target.keys() & metrics.keys(), "Repeated validation metric/event within attempt")
                target.update(metrics)
        need(not selected.keys() & validation.keys(), "Repeated validation across committed attempts")
        need(all(set(v) == set(metric_names) for v in selected.values()), "Incomplete scheduled validation event")
        validation.update(selected)
        evidence.append(dict(path=str(path), sha256=sha(path), start=item["start"], end=item["end"],
                             included_steps=sorted(selected), uncommitted_steps_preserved=sorted(discarded)))
    need(set(validation) == set(range(20000, stop + 1, 20000)), "Missing committed scheduled validation event")
    return validation, evidence
