#!/usr/bin/env python
"""Assemble the arm x embodiment rollout table from cells banked in R2.

The launcher copies each cell's log to
s3://rldb/staged/articulated_rollout/<job>/cells/<arm>__<embodiment>.log the
moment that cell finishes, so results survive the preemptions that have killed
every rollout job so far, and a partly-finished job still contributes.

Reads R2 rather than `osmo workflow logs`: a log fetch on these jobs takes over
eight minutes and the earlier log-scraping collector simply timed out and
reported nothing while the data sat in R2. Credentials come from
~/.egoverse_env (R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT_URL).

Usage:
    python tools/collect_artic_results.py --jobs arq-1 arq-2 arq-3 arq-4 arq-5 \
        --csv results/artic_rollout.csv
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys

HELD_OUT = {"umi", "scoop"}
EMB_ORDER = ["u_socket", "gripper", "chain_gripper", "suction", "triangle",
             "flipper", "spring", "umi", "scoop"]
ARM_ORDER = ["arc_dur_D80_M56", "arc_stk_D80_M56", "arc_dur_D80_M16",
             "arc_stk_D80_M16", "dp_paper"]
LABEL = {"arc_dur_D80_M56": "dur_M56", "arc_stk_D80_M56": "stk_M56",
         "arc_dur_D80_M16": "dur_M16", "arc_stk_D80_M16": "stk_M16",
         "dp_paper": "DP"}


def r2_env() -> dict:
    env = dict(os.environ)
    path = os.path.expanduser("~/.egoverse_env")
    if not os.path.exists(path):
        sys.exit(f"no {path}; R2 credentials unavailable")
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    for src, dst in (("R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
                     ("R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY")):
        if src not in env:
            sys.exit(f"{src} missing from ~/.egoverse_env")
        env[dst] = env[src]
    env["AWS_DEFAULT_REGION"] = "auto"
    return env


def pull(jobs: list[str], dest: str, env: dict) -> None:
    os.makedirs(dest, exist_ok=True)
    for job in jobs:
        subprocess.run(
            ["aws", "s3", "cp",
             f"s3://rldb/staged/articulated_rollout/{job}/cells/",
             os.path.join(dest, job), "--recursive",
             "--endpoint-url", env["R2_ENDPOINT_URL"]],
            env=env, capture_output=True, text=True, timeout=600)


def load(dest: str) -> dict:
    rows: dict = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(dest, "*", "*.log")):
        name = os.path.basename(path)[:-4]
        if "__" not in name:
            continue
        arm, emb = name.split("__", 1)
        arm = arm.replace("artic_cotrain7_", "").replace("_R26deg", "")
        text = open(path, errors="ignore").read()
        for m in re.finditer(r"SUMMARY (\{.*\})", text):
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            # Skip the one-episode smoke rows; only the real pass counts.
            if d.get("episodes", 0) >= 2:
                rows[arm][emb] = d
    return rows


def table(rows: dict, csv_path: str | None) -> None:
    arms = [a for a in ARM_ORDER if a in rows] or sorted(rows)
    embs = [e for e in EMB_ORDER if any(e in rows[a] for a in arms)]
    have = sum(len(rows[a]) for a in arms)
    print(f"{have} of {len(arms) * 9} cells scored\n")

    def grid(title: str, field: str, fmt: str) -> None:
        print(title)
        print(f"{'embodiment':16s}" + "".join(f"{LABEL.get(a, a):>9s}" for a in arms)
              + "   role")
        for e in embs:
            cells = "".join(
                format(rows[a][e][field], fmt) if e in rows[a] else f"{'--':>9s}"
                for a in arms)
            print(f"{e:16s}{cells}   "
                  f"{'HELD-OUT' if e in HELD_OUT else 'in-domain'}")
        print()

    grid("PEAK COVERAGE", "peak_coverage_mean", "9.3f")
    grid("SR@0.80", "SR@0.80", "9.3f")

    # Aggregate only over embodiments every arm has, so a mean is never
    # inflated by an arm that happens to hold an easy cell nobody else has.
    common = [e for e in embs if all(e in rows[a] for a in arms)]
    print(f"MEANS over the {len(common)} embodiment(s) common to all arms: {common}")
    for name, group in (("in-domain", [e for e in common if e not in HELD_OUT]),
                        ("held-out", [e for e in common if e in HELD_OUT])):
        if not group:
            continue
        cells = "".join(
            f"{sum(rows[a][e]['peak_coverage_mean'] for e in group) / len(group):9.3f}"
            for a in arms)
        print(f"{name:16s}{cells}")

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w") as fh:
            fh.write("embodiment,role,arm,peak_coverage_mean,peak_coverage_median,"
                     "SR@0.80,SR@0.95,episodes,budget\n")
            for e in embs:
                role = "held-out" if e in HELD_OUT else "in-domain"
                for a in arms:
                    d = rows[a].get(e)
                    if d:
                        fh.write(f"{e},{role},{LABEL.get(a, a)},"
                                 f"{d['peak_coverage_mean']:.4f},"
                                 f"{d['peak_coverage_median']:.4f},{d['SR@0.80']},"
                                 f"{d['SR@0.95']},{d['episodes']},{d['budget']}\n")
        print(f"\nwrote {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--dest", default="/tmp/artic_cells")
    ap.add_argument("--csv")
    ap.add_argument("--no-pull", action="store_true")
    a = ap.parse_args()
    env = r2_env()
    if not a.no_pull:
        pull(a.jobs, a.dest, env)
    table(load(a.dest), a.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
