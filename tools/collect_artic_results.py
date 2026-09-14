#!/usr/bin/env python
"""Collect [sim] SUMMARY lines from the per-arm rollout jobs into one table.

Each job scores one arm across all nine embodiments, so the interesting view --
arm x embodiment, in-domain against held-out -- has to be assembled from
several logs. Fetches tails with `osmo workflow logs -n`, which returns the end
of the log server-side; streaming whole logs truncates around 40 KB.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HELD_OUT = {"umi", "scoop"}
ORDER = ["u_socket", "gripper", "chain_gripper", "suction", "triangle",
         "flipper", "spring", "umi", "scoop"]


def fetch(job: str, lines: int) -> str:
    try:
        r = subprocess.run(
            ["osmo", "workflow", "logs", job, "-t", "rollout", "-n", str(lines)],
            capture_output=True, text=True, timeout=900)
        return r.stdout
    except subprocess.TimeoutExpired:
        return ""


def parse(text: str) -> dict:
    """embodiment -> the LAST summary for it (the 40-episode pass, not the smoke)."""
    out = {}
    for m in re.finditer(r"SUMMARY (\{.*?\})\s*$", text, re.M):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if d.get("episodes", 0) < 2:        # skip the 1-episode smoke rows
            continue
        out[d["embodiment"].replace("pushshapes_sim_", "")] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="job:label pairs, e.g. articroll-a1-1:dur_M56")
    ap.add_argument("--lines", type=int, default=4000)
    ap.add_argument("--csv")
    a = ap.parse_args()

    table, labels = {}, []
    for spec in a.jobs:
        job, _, label = spec.partition(":")
        label = label or job
        labels.append(label)
        rows = parse(fetch(job, a.lines))
        table[label] = rows
        print(f"  {label:12s} {len(rows)} embodiment(s) scored", file=sys.stderr)

    embs = [e for e in ORDER if any(e in table[l] for l in labels)]
    w = max((len(l) for l in labels), default=8) + 2
    print("\nPEAK COVERAGE  (mean over episodes)")
    print(f"{'embodiment':16s}" + "".join(f"{l:>{w}s}" for l in labels) + "   role")
    for e in embs:
        cells = "".join(
            f"{table[l][e]['peak_coverage_mean']:{w}.3f}" if e in table[l]
            else f"{'--':>{w}s}" for l in labels)
        print(f"{e:16s}{cells}   {'HELD-OUT' if e in HELD_OUT else 'in-domain'}")

    print("\nSR@0.80")
    print(f"{'embodiment':16s}" + "".join(f"{l:>{w}s}" for l in labels))
    for e in embs:
        cells = "".join(
            f"{table[l][e]['SR@0.80']:{w}.3f}" if e in table[l]
            else f"{'--':>{w}s}" for l in labels)
        print(f"{e:16s}{cells}")

    ind = [e for e in embs if e not in HELD_OUT]
    held = [e for e in embs if e in HELD_OUT]
    print("\nMEANS")
    print(f"{'group':16s}" + "".join(f"{l:>{w}s}" for l in labels))
    for name, group in (("in-domain", ind), ("held-out", held)):
        cells = ""
        for l in labels:
            vals = [table[l][e]["peak_coverage_mean"] for e in group if e in table[l]]
            cells += f"{sum(vals)/len(vals):{w}.3f}" if vals else f"{'--':>{w}s}"
        print(f"{name:16s}{cells}")

    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w") as fh:
            fh.write("embodiment,role,arm,peak_coverage_mean,SR@0.80,SR@0.95,episodes,budget\n")
            for e in embs:
                role = "held-out" if e in HELD_OUT else "in-domain"
                for l in labels:
                    d = table[l].get(e)
                    if d:
                        fh.write(f"{e},{role},{l},{d['peak_coverage_mean']:.4f},"
                                 f"{d['SR@0.80']},{d['SR@0.95']},{d['episodes']},"
                                 f"{d['budget']}\n")
        print(f"\nwrote {a.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
