#!/usr/bin/env python3
"""Aggregate the complete A--F action-flow comparison across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
    "validation_generation_symmetric_nn_mse",
    "validation_generation_energy_distance",
    "validation_torus_surface_rmse",
    "validation_angular_histogram_l1",
    "validation_angular_support_recall",
    "validation_reconstruction_mse",
    "validation_path_consistency_mse",
    "validation_scale_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = []
    for variant in manifest["variants"]:
        summaries = []
        for seed in manifest["seeds"]:
            matching = []
            for path_text in manifest["configs"]:
                path = Path(path_text)
                config = json.loads(path.read_text())
                if config.get("variant") == variant and config["seed"] == seed:
                    matching.append((path, config))
            if len(matching) != 1:
                raise ValueError(f"missing unique config for {variant} seed {seed}")
            _, config = matching[0]
            summary_path = Path(config["output_dir"]) / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            summaries.append(json.loads(summary_path.read_text()))
        aggregate = {"variant": variant, "seeds": manifest["seeds"]}
        for metric in METRICS:
            values = [summary[metric] for summary in summaries if metric in summary]
            if values:
                aggregate[metric] = {
                    "values": values,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
        rows.append(aggregate)
    result = {
        "schema_version": 1,
        "primary_metric": manifest["primary_metric"],
        "comparison_contract": manifest["comparison_contract"],
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Action-flow affine comparison",
        "",
        "| Variant | Symmetric NN MSE (mean ± sample std) | Seeds |",
        "|---|---:|---|",
    ]
    for row in rows:
        metric = row[manifest["primary_metric"]]
        lines.append(
            f"| {row['variant']} | {metric['mean']:.9g} ± {metric['std']:.9g} "
            f"| {', '.join(map(str, row['seeds']))} |"
        )
    args.output_markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
