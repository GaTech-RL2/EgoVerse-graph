#!/usr/bin/env python3
"""Prove the ARC tokenizer speedup is exact, and measure how much it bought.

The reference is not a monkeypatched copy of the current tokenizer -- that only
re-times one helper against itself. It is the tokenizer file as it stood at
``--reference-rev``, loaded as a standalone module (the file depends on nothing
but numpy and scipy), so "before" really is the code the runs were training on.

Two gates, both of which must pass:
  exact parity   every case in the frozen corpus must produce byte-identical
                 tokens and preserved rows under ``np.array_equal``; a tolerance
                 comparison would hide a changed token contract.
  throughput     the optimized median must be at most --max-ratio of the
                 reference median, for every chunking mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tests.arc_token_parity_fixtures import (  # noqa: E402
    ARC_CASES,
    tokenize_case,
    tokenizer_for,
)

TOKENIZER_PATH = "egomimic/rldb/zarr/arc_length_tokenizer.py"
# The commit the organize_stationary group launched on, i.e. the code whose
# tokens must not change.
DEFAULT_REFERENCE_REV = "257c4f80f834272d29467341e055d854cdf702ab"
# The modes whose throughput is gated, each at the production case shape.
BENCHMARK_CASES = {
    "joint_distance": "joint_distance_hybrid_m100_per_waypoint",
    "race": "race_hybrid_m100_per_waypoint",
    "multistream": "multistream_hybrid_m100_per_waypoint",
}


def _load_reference(rev: str):
    """Import the tokenizer as of ``rev`` under its own module name."""
    blob = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:{TOKENIZER_PATH}"],
        check=True,
        capture_output=True,
    ).stdout
    directory = tempfile.mkdtemp(prefix="arc_ref_")
    target = Path(directory) / "arc_length_tokenizer_reference.py"
    target.write_bytes(blob)
    spec = importlib.util.spec_from_file_location(target.stem, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[target.stem] = module
    spec.loader.exec_module(module)
    return module


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _median_ms(tokenizer, raw: np.ndarray, repeat: int) -> float:
    for _ in range(3):  # warm the scipy/numpy paths before timing
        tokenizer.transform({"raw": raw.copy()})
    samples = []
    for _ in range(repeat):
        started = time.perf_counter_ns()
        tokenizer.transform({"raw": raw.copy()})
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-rev", default=DEFAULT_REFERENCE_REV)
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument("--max-ratio", type=float, default=0.70)
    parser.add_argument("--output")
    # The frozen-reference test's GOLDEN table has to come from the reference
    # implementation, not the optimized one, or it would only ever confirm that
    # the optimized code agrees with itself.
    parser.add_argument("--emit-golden")
    args = parser.parse_args()

    reference_module = _load_reference(args.reference_rev)

    mismatches = []
    for case in ARC_CASES:
        expected_token, expected_preserved = tokenize_case(case, reference_module)
        actual_token, actual_preserved = tokenize_case(case)
        if not (
            np.array_equal(expected_token, actual_token)
            and np.array_equal(expected_preserved, actual_preserved)
            and expected_token.shape == actual_token.shape
        ):
            mismatches.append(case.name)

    if args.emit_golden:
        lines = ["GOLDEN = {"]
        for case in ARC_CASES:
            token, preserved = tokenize_case(case, reference_module)
            lines.append(f'    "{case.name}": (')
            lines.append(f"        {tuple(token.shape)},")
            lines.append(f'        "{_sha256(token)}",')
            lines.append(f'        "{_sha256(preserved)}",')
            lines.append("    ),")
        lines.append("}")
        Path(args.emit_golden).write_text("\n".join(lines) + "\n")

    by_name = {case.name: case for case in ARC_CASES}
    throughput = {}
    for mode, case_name in BENCHMARK_CASES.items():
        case = by_name[case_name]
        reference_ms = _median_ms(
            tokenizer_for(case, reference_module), case.actions, args.repeat
        )
        optimized_ms = _median_ms(tokenizer_for(case), case.actions, args.repeat)
        throughput[mode] = {
            "case": case_name,
            "reference_ms": reference_ms,
            "optimized_ms": optimized_ms,
            "ratio": optimized_ms / reference_ms,
            "speedup": reference_ms / optimized_ms,
        }

    report = {
        "reference_rev": args.reference_rev,
        "cases_compared": len(ARC_CASES),
        "exact_parity": not mismatches,
        "parity_mismatches": mismatches,
        "max_ratio": args.max_ratio,
        "throughput": throughput,
    }
    report["throughput_gate_passed"] = all(
        row["ratio"] <= args.max_ratio for row in throughput.values()
    )
    report["gate_passed"] = report["exact_parity"] and report["throughput_gate_passed"]
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
