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


EXPECTED_EPISODES = 40


def load(dest: str, expected: int = EXPECTED_EPISODES) -> dict:
    """Merge seed-window chunks back into one row per arm x embodiment.

    A heavy embodiment cannot finish 40 episodes inside the window this pool
    leaves between preemptions, so the launcher scores it in chunks of
    CHUNK_EPISODES with a shifted evaluator.seed_base. Every episode is
    independent -- the evaluator does env.reset(seed=seed_base+i) and nothing
    carries across -- so seeds 0..39 scored in four chunks are the SAME forty
    episodes as one pass, and merging is exact rather than an approximation.

    Keyed on the seed_base inside each SUMMARY, not on the filename, so a
    renamed or hand-copied log still lands in the right window.
    """
    # (arm, emb) -> seed -> coverage, plus one representative summary
    seeds: dict = collections.defaultdict(dict)
    meta: dict = {}
    for path in glob.glob(os.path.join(dest, "*", "*.log")):
        name = os.path.basename(path)[:-4]
        if "__" not in name:
            continue
        arm, emb = name.split("__", 1)
        emb = emb.split("__s")[0]          # drop the __s<base>n<count> suffix
        arm = arm.replace("artic_cotrain7_", "").replace("_R26deg", "")
        text = open(path, errors="ignore").read()
        for m in re.finditer(r"SUMMARY (\{.*\})", text):
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            cov = d.get("ep_coverages")
            if not cov:
                continue
            base = int(d.get("seed_base", 0))
            # A 1-episode run at seed_base 0 is the smoke probe, not a chunk.
            if len(cov) == 1 and base == 0 and expected > 1:
                continue
            for i, v in enumerate(cov):
                seeds[(arm, emb)][base + i] = float(v)
            meta[(arm, emb)] = d

    rows: dict = collections.defaultdict(dict)
    for (arm, emb), by_seed in seeds.items():
        got = sorted(by_seed)
        arr = [by_seed[k] for k in got]
        n = len(arr)
        d = dict(meta[(arm, emb)])
        d["episodes"] = n
        d["seeds_present"] = got
        d["complete"] = (got == list(range(expected)))
        d["peak_coverage_mean"] = sum(arr) / n
        d["SR@0.80"] = sum(1 for v in arr if v >= 0.80) / n
        d["SR@0.95"] = sum(1 for v in arr if v >= 0.95) / n
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
        def one(a, e):
            if e not in rows[a]:
                return f"{'--':>9s}"
            d = rows[a][e]
            # A partial cell is still only part of the 40 seeds; marking it
            # keeps a 10-seed row from being read as a finished result.
            txt = format(d[field], fmt.replace("9", "8"))
            return txt + (" " if d.get("complete", True) else "*")

        for e in embs:
            cells = "".join(one(a, e) for a in arms)
            print(f"{e:16s}{cells}   "
                  f"{'HELD-OUT' if e in HELD_OUT else 'in-domain'}")
        print()

    grid("PEAK COVERAGE", "peak_coverage_mean", "9.3f")
    grid("SR@0.80", "SR@0.80", "9.3f")
    partial = [(LABEL.get(a, a), e, len(rows[a][e].get("seeds_present", [])))
               for a in arms for e in rows[a] if not rows[a][e].get("complete", True)]
    if partial:
        print("* PARTIAL, fewer than "
              f"{EXPECTED_EPISODES} seeds: "
              + ", ".join(f"{a}/{e} ({n})" for a, e, n in sorted(partial)))
        print()

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
