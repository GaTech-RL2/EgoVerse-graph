"""Emit an executable argument-vector plan for the complete five-suite study."""

import argparse
import json
import sys
from pathlib import Path

from egomimic.benchmarks.libero.catalog import LIBERO_COMMIT, OAT_COMMIT, TASKS


def campaign_plan(data_root, output_root, python=sys.executable):
    data_root, output_root = Path(data_root), Path(output_root)
    jobs = []
    for suite in TASKS:
        dataset = str(data_root / f"{suite}.zarr")
        outputs = {
            name: output_root / "training" / suite / name
            for name in ("tokenizer", "oat", "arc")
        }
        tokenizer_checkpoint = str(outputs["tokenizer"] / "checkpoints/last.ckpt")
        for name, experiment in (
            ("tokenizer", "libero_oattok"),
            ("oat", "libero_oatpolicy"),
            ("arc", "libero_arc_policy"),
        ):
            args = [
                python,
                "-m",
                "egomimic.trainHydra",
                f"+experiment=oat/{experiment}",
                f"benchmark.suite={suite}",
                f"benchmark.dataset={dataset}",
                f"hydra.run.dir={outputs[name]}",
            ]
            if name == "oat":
                args.append(f"benchmark.tokenizer_checkpoint={tokenizer_checkpoint}")
            jobs.append(
                {
                    "id": f"{suite}/{name}",
                    "requires": [f"{suite}/tokenizer"] if name == "oat" else [],
                    "argv": args,
                }
            )
        jobs.append(
            {
                "id": f"{suite}/reconstruction",
                "requires": [f"{suite}/tokenizer"],
                "argv": [
                    python,
                    "-m",
                    "egomimic.benchmarks.libero.cli",
                    "reconstruct",
                    "--suite",
                    suite,
                    "--dataset",
                    dataset,
                    "--checkpoint",
                    tokenizer_checkpoint,
                    "--output",
                    str(output_root / f"{suite}_reconstruction.json"),
                ],
            }
        )
        for method in ("arc", "oat"):
            jobs.append(
                {
                    "id": f"{suite}/{method}_rollout",
                    "requires": [f"{suite}/{method}"],
                    "argv": [
                        python,
                        "-m",
                        "egomimic.benchmarks.libero.cli",
                        "rollout",
                        "--checkpoint",
                        str(outputs[method] / "checkpoints/last.ckpt"),
                        "--output",
                        str(output_root / method / suite),
                    ],
                }
            )
    jobs.append(
        {
            "id": "complete_comparison",
            "requires": [
                f"{suite}/{method}_rollout"
                for suite in TASKS
                for method in ("arc", "oat")
            ],
            "argv": [
                python,
                "-m",
                "egomimic.benchmarks.libero.cli",
                "compare",
                "--arc-root",
                str(output_root / "arc"),
                "--oat-root",
                str(output_root / "oat"),
                "--output",
                str(output_root / "comparison.json"),
            ],
        }
    )
    return {
        "oat_commit": OAT_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "training_seed": 42,
        "checkpoint_selection": "final EMA; preselected before simulator evaluation",
        "jobs": jobs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.output).open("x") as handle:
        json.dump(campaign_plan(args.data_root, args.output_root), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
