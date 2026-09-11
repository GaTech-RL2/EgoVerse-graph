#!/usr/bin/env python3
"""Validate and aggregate the paired synthetic-distribution benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = (
    "validation_generation_energy_distance",
    "validation_generation_symmetric_nn_mse",
)


def _finite_number(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value}")
    return number


def aggregate(manifest_path: Path) -> tuple[list[dict], list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    runs = manifest["runs"]
    if len(runs) != 72:
        raise ValueError(f"expected 72 runs, found {len(runs)}")

    rows = []
    by_pair: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for run in runs:
        config = json.loads(Path(run["config"]).read_text())
        if config["source_commit"] != manifest["source_commit"]:
            raise ValueError(f"source mismatch for {run['run_id']}")
        output = Path(config["output_dir"])
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        row = {
            "run_id": run["run_id"],
            "distribution": run["distribution"],
            "dimension_regime": run["dimension_regime"],
            "seed": int(run["seed"]),
            "method": run["method"],
            "parameters": int(summary["trainable_parameters"]),
        }
        for metric in METRICS:
            row[metric] = _finite_number(summary[metric], f"{run['run_id']}:{metric}")
        rows.append(row)
        key = (row["distribution"], row["dimension_regime"], row["seed"])
        if row["method"] in by_pair[key]:
            raise ValueError(f"duplicate method in pair {key}: {row['method']}")
        by_pair[key][row["method"]] = row

    paired = []
    for key, methods in sorted(by_pair.items()):
        if set(methods) != {"direct", "action_flow"}:
            raise ValueError(f"incomplete pair {key}: {sorted(methods)}")
        direct, action = methods["direct"], methods["action_flow"]
        paired.append(
            {
                "distribution": key[0],
                "dimension_regime": key[1],
                "seed": key[2],
                **{
                    f"action_minus_direct_{metric}": action[metric] - direct[metric]
                    for metric in METRICS
                },
            }
        )
    if len(paired) != 36:
        raise ValueError(f"expected 36 paired rows, found {len(paired)}")
    return rows, paired


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, paired = aggregate(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / "RUN_RESULTS.csv", rows)
    _write_csv(args.output_dir / "PAIRED_RESULTS.csv", paired)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in paired:
        groups[(row["distribution"], row["dimension_regime"])].append(row)
    group_summary = []
    for (distribution, dimension), group in sorted(groups.items()):
        item = {
            "distribution": distribution,
            "dimension_regime": dimension,
            "seeds": len(group),
        }
        for metric in METRICS:
            key = f"action_minus_direct_{metric}"
            values = [row[key] for row in group]
            item[f"mean_{key}"] = sum(values) / len(values)
        group_summary.append(item)
    payload = {
        "manifest": str(args.manifest),
        "run_count": len(rows),
        "pair_count": len(paired),
        "interpretation": "negative action-minus-direct values favor Action Flow",
        "groups": group_summary,
    }
    (args.output_dir / "AGGREGATE_RESULTS.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps({"runs": len(rows), "pairs": len(paired)}))


if __name__ == "__main__":
    main()
