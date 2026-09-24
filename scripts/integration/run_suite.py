"""Run every declared case, preserving failures and releasing model memory between phases."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def run(matrix, suite, inputs, output):
    cases = yaml.safe_load(matrix.read_text())["cases"]
    results = []
    for name, spec in cases.items():
        if spec["suite"] != suite:
            continue
        destination = output / name
        destination.mkdir(parents=True, exist_ok=False)
        status = {"case": name, "status": "failed"}
        for phase in ("train", "verify"):
            command = [
                sys.executable,
                "-m",
                "scripts.integration.run_gate",
                "--matrix",
                str(matrix),
                "--case",
                name,
                "--inputs",
                str(inputs),
                "--output",
                str(destination),
                "--phase",
                phase,
            ]
            print("START", name, phase, flush=True)
            with (destination / f"{phase}.log").open("x") as log:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                code = process.wait()
            status[phase] = code
            if code:
                break
        else:
            status["status"] = "passed"
        results.append(status)
        (output / "suite-progress.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print("CASE_RESULT", json.dumps(status), flush=True)
    if not results or any(row["status"] != "passed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(**vars(args))
