#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egomimic.rldb.zarr import arc_length_tokenizer as arc_module


def _source_actions(rows: int = 600) -> np.ndarray:
    t = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    progress = t.copy()
    progress[:8] = 0.0
    progress[8:] = np.linspace(0.0, 1.0, rows - 8, dtype=np.float64)
    actions = np.zeros((rows, 14), dtype=np.float64)
    actions[:, 0] = 0.7 * progress
    actions[:, 1] = 0.08 * np.sin(np.pi * progress)
    actions[:, 2] = 0.03 * progress**2
    actions[:, 7] = -0.5 * progress
    actions[:, 8] = 0.08 * np.cos(np.pi * progress)
    actions[:, 9] = -0.02 * progress**2
    actions[:, 3] = 1.2 * progress
    actions[:, 4] = -0.3 * progress
    actions[:, 5] = np.linspace(np.pi - 0.08, -np.pi + 0.12, rows)
    actions[:, 10] = -0.84 * progress
    actions[:, 11] = 0.24 * progress
    actions[:, 12] = np.linspace(-np.pi + 0.05, np.pi - 0.09, rows)
    actions[:, 6] = 0.5 + 0.4 * np.sin(2.0 * np.pi * t)
    actions[:, 13] = 0.45 + 0.35 * np.cos(1.5 * np.pi * t)
    return actions


def _scalar_slerp(ypr: np.ndarray, indices: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            arc_module._interp_ypr_at_s(
                ypr[index : index + 2],
                np.array([0.0, 1.0], dtype=np.float64),
                float(blend),
            )
            for index, blend in zip(indices, alpha, strict=True)
        ]
    )


@contextmanager
def _rotation_path(function):
    original = arc_module._slerp_segments_ypr
    arc_module._slerp_segments_ypr = function
    try:
        yield
    finally:
        arc_module._slerp_segments_ypr = original


def _tokenizer(mode: str):
    return arc_module.TokenizeBimanualArcLengthCartesian(
        action_key="raw",
        output_action_key="token",
        min_distance_unit=0.40,
        rotation_distance_unit=0.42,
        resampled_vector_length=100,
        velocity_mode="per_waypoint",
        translation_horizon_mode=mode,
    )


def _median_ms(tokenizer, raw: np.ndarray, repeat: int) -> float:
    for _ in range(3):
        tokenizer.transform({"raw": raw.copy()})
    samples = []
    for _ in range(repeat):
        started = time.perf_counter_ns()
        tokenizer.transform({"raw": raw.copy()})
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    raw = _source_actions()
    report = {}
    for mode in ("joint", "race"):
        tokenizer = _tokenizer(mode)
        with _rotation_path(_scalar_slerp):
            reference = tokenizer.transform({"raw": raw.copy()})["token"].copy()
            reference_ms = _median_ms(tokenizer, raw, args.repeat)
        optimized = tokenizer.transform({"raw": raw.copy()})["token"].copy()
        if not np.array_equal(reference, optimized):
            raise RuntimeError(f"{mode} optimized token differs from scalar reference")
        optimized_ms = _median_ms(tokenizer, raw, args.repeat)
        report[mode] = {
            "reference_ms": reference_ms,
            "optimized_ms": optimized_ms,
            "ratio": optimized_ms / reference_ms,
            "exact_token_parity": True,
        }
    report["gate_passed"] = all(row["ratio"] <= 0.70 for row in report.values())
    print(json.dumps(report, indent=None if args.json else 2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
