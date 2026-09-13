#!/usr/bin/env python3
"""Read bounded recent W&B telemetry without conflating history and optimizer steps.

This is an observation tool, not a smoke gate or proof that a scheduler job
completed. It never initializes, resumes, or writes a W&B run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import sys
import time


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _config_value(config, key):
    # Exact top-level names take precedence; otherwise use dotted nested paths.
    if key in config:
        return config[key]
    value = config
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(key)
        value = value[part]
    return value


def collect_health(
    api,
    run_path,
    *,
    metrics=(),
    expected_config=None,
    window=1000,
    min_optimizer_step=None,
    max_age_seconds=None,
    now=None,
):
    """Inspect the latest bounded row window and keep sparse metrics separate."""
    if len(run_path.split("/")) != 3 or not all(run_path.split("/")):
        raise ValueError("run path must be entity/project/run-id")
    if not 1 <= window <= 5000:
        raise ValueError("window must be between 1 and 5000 history rows")
    run = api.run(run_path)
    actual_path = "/".join(run.path)
    errors, warnings, missing = [], [], []
    if actual_path != run_path:
        errors.append(f"run identity mismatch: requested {run_path}, returned {actual_path}")
    config_checks = {}
    for key, expected in (expected_config or {}).items():
        try:
            actual = _config_value(run.config, key)
        except KeyError:
            config_checks[key] = {"expected": expected, "present": False, "matches": False}
            errors.append(f"required config identity missing: {key}")
            continue
        matches = actual == expected
        config_checks[key] = {"expected": expected, "actual": actual, "matches": matches}
        if not matches:
            errors.append(f"config identity mismatch: {key}")

    last_row = int(run.lastHistoryStep)
    start = max(0, last_row - window + 1)
    latest, rows_seen, optimizer_step, timestamp = {}, 0, None, None
    previous_step = None
    regressions = []
    if last_row >= 0:
        # Supplying keys asks W&B for rows containing ALL those keys. Training,
        # validation and schedule metrics often occupy different history rows.
        rows = run.scan_history(
            keys=None, min_step=start, max_step=last_row + 1,
            page_size=min(window, 1000),
        )
        for row in rows:
            rows_seen += 1
            if rows_seen > window:
                raise RuntimeError("W&B returned more rows than the bounded window")
            step = row.get("trainer/global_step")
            if _finite_number(step):
                if previous_step is not None and step < previous_step:
                    regressions.append({"from": previous_step, "to": step, "wandb_step": row.get("_step")})
                previous_step = optimizer_step = step
            if _finite_number(row.get("_timestamp")):
                timestamp = row["_timestamp"]
            for key in metrics:
                if key in row:
                    value = row[key]
                    latest[key] = {
                        "value": value if _finite_number(value) else repr(value),
                        "finite": _finite_number(value),
                        "source": "recent_history",
                        "wandb_step": row.get("_step"),
                        "optimizer_step": step if _finite_number(step) else None,
                        "timestamp": row.get("_timestamp"),
                    }

    for key in metrics:
        if key not in latest:
            value = run.summary.get(key)
            if value is not None:
                latest[key] = {
                    "value": value if _finite_number(value) else repr(value),
                    "finite": _finite_number(value),
                    "source": "summary_only",
                    "wandb_step": None, "optimizer_step": None, "timestamp": None,
                }
                warnings.append(f"{key}: summary value has no proven recent optimizer step")
            else:
                missing.append(key)
        if key in latest and not latest[key]["finite"]:
            errors.append(f"latest observed metric is not finite: {key}")

    if optimizer_step is None:
        missing.append("trainer/global_step")
    elif min_optimizer_step is not None and optimizer_step < min_optimizer_step:
        errors.append(f"optimizer step {optimizer_step} is below required {min_optimizer_step}")
    if regressions:
        warnings.append("optimizer steps regress inside the recent window; inspect checkpoint resume history")
    age = None if timestamp is None else (time.time() if now is None else now) - timestamp
    if max_age_seconds is not None:
        if age is None:
            missing.append("_timestamp")
        elif age > max_age_seconds:
            warnings.append(f"latest history is {age:.1f}s old, above {max_age_seconds}s")
    if run.state in {"failed", "crashed"}:
        errors.append(f"W&B run state is {run.state}")
    status = "ERROR" if errors else "INCOMPLETE" if missing else "WARN" if warnings else "PASS"
    return {
        "schema_version": 1, "status": status,
        "run_path": actual_path, "url": run.url, "run_state": run.state,
        "optimizer_step": optimizer_step, "wandb_history_step": last_row,
        "history_window": {"min_step": start, "max_step_exclusive": last_row + 1, "rows_read": rows_seen},
        "latest_history_age_seconds": age,
        "metrics": latest, "config_identity": config_checks,
        "optimizer_step_regressions": regressions,
        "missing": missing, "warnings": warnings, "errors": errors,
        "note": "W&B _step counts history rows; trainer/global_step counts optimizer updates. PASS is telemetry health, not training completion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="entity/project/run-id")
    parser.add_argument("--metric", action="append", default=[], help="required numeric metric; repeat for sparse metrics")
    parser.add_argument("--expect-config", action="append", default=[], metavar="KEY=JSON", help="exact top-level or dotted config identity; strings may be unquoted")
    parser.add_argument("--window", type=int, default=1000, help="recent W&B history rows, at most 5000")
    parser.add_argument("--min-optimizer-step", type=int)
    parser.add_argument("--max-age-seconds", type=float)
    args = parser.parse_args()
    expected = {}
    for entry in args.expect_config:
        key, separator, value = entry.partition("=")
        if not separator or not key:
            parser.error("--expect-config requires KEY=JSON")
        try:
            expected[key] = json.loads(value)
        except json.JSONDecodeError:
            expected[key] = value
    try:
        import wandb

        result = collect_health(
            wandb.Api(timeout=30), args.run, metrics=args.metric,
            expected_config=expected, window=args.window,
            min_optimizer_step=args.min_optimizer_step,
            max_age_seconds=args.max_age_seconds,
        )
    except Exception as exc:
        result = {"status": "ERROR", "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if result["status"] == "ERROR" else 2 if result["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    sys.exit(main())
