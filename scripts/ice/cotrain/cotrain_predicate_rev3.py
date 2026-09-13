#!/usr/bin/env python3
"""Pinned predicate for the UNITE cotrain loop, protocol revision 3 (2026-09-10).

Reads protocol rollout results (rollout_ct3.sh / monitor_ct3.py JSONs) from
~/scratch/rollouts/CT2-rev3 and asks: is there ONE UNITE-cotrain checkpoint whose mean
peak coverage on the canonical level-0 seeds 0-39 beats (a) the best Paper-DP BC checkpoint
and (b) the best Paper-DP cotrain checkpoint, on BOTH embodiments?  Bars use the best DP
checkpoint per embodiment (selection-biased upward, i.e. conservative for UNITE).

Inference setting for the UNITE family (declared 2026-09-10, recorded here): CFG 1.0
(`cfg1`); the checkpoint-embedded CFG 4 (`cfgemb`) is reported alongside.  Protocol state
filtered on every row: horizon revision 3 budget file, full horizon, EMA, replan 8, chunk
start 0.  Prints units_remaining = sum over embodiments of max(0, bar - best) for the
checkpoint with the smallest total shortfall; exit 0 iff a checkpoint clears both bars on
both embodiments (complete=True).  Also prints the BC fallback state (UNITE BC vs DP BC).
"""
import glob, json, os, re, sys
from collections import defaultdict

OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/scratch/rollouts/CT2-rev3")
VARIANT = os.environ.get("UNITE_VARIANT", "cfg1")   # declared family inference setting
UNITE_COTRAIN = ("ctA", "ctAc", "ctA768", "ctB", "ctAema", "ctAann", "ctApin", "ctApinAnn", "ctAwd", "s3ctA", "s3ctA768", "s3ctB", "s3ctApin", "s3ctApinAnn")
UNITE_BC = {"usocket": ("uniteus", "uniteusema", "uniteusann"), "chain": ("unitech", "s3unitech")}
DP_BC = {"usocket": ("dpus",), "chain": ("dpch", "s3dpch")}
DP_CT = ("dpct", "s3dpct")


def load():
    peaks = defaultdict(dict)  # (row, step, emb, variant) -> {seed: peak}, level 0 only
    for path in glob.glob(os.path.join(OUT, "*.json")):
        m = re.match(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L0(?:-s(\d+))?-(cfgemb|cfg1)\.json$", os.path.basename(path))
        if not m:
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        pr = d.get("protocol") or {}
        if pr.get("horizon_revision") != 3 or not pr.get("full_horizon") or not pr.get("use_ema") \
                or pr.get("replan_every") != 8 or pr.get("chunk_start") != 0:
            continue
        key = (m.group(1), int(m.group(2)) * 1000, m.group(3), m.group(5))
        for e in d["episodes"]:
            peaks[key].setdefault(int(e["seed"]), float(e["peak"]))
    return peaks


def canon_mean(seeds):
    vals = [v for s, v in seeds.items() if s < 40]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def best_bar(peaks, rows, emb, variant="cfgemb"):
    cands = []
    for (row, step, e, var), seeds in peaks.items():
        if row in rows and e == emb and var == variant:
            mean, n = canon_mean(seeds)
            if n >= 40:
                cands.append((mean, row, step))
    return max(cands) if cands else None


def main():
    peaks = load()
    report = {"variant": VARIANT, "bars": {}, "closest": None, "units_remaining": None, "complete": False, "bc_fallback": {}}
    bars = {}
    for emb in ("usocket", "chain"):
        bc = best_bar(peaks, DP_BC[emb], emb); ct = best_bar(peaks, DP_CT, emb)
        report["bars"][emb] = {"dp_bc": bc, "dp_cotrain": ct}
        if bc and ct:
            bars[emb] = max(bc[0], ct[0])
        elif bc:
            bars[emb] = bc[0]  # cotrain bar not evaluable yet: provisional
    best = None
    for (row, step, emb, var), seeds in peaks.items():
        if row not in UNITE_COTRAIN or var != VARIANT or emb != "usocket":
            continue
        u_mean, u_n = canon_mean(seeds)
        c = peaks.get((row, step, "chain", var))
        if c is None or u_n < 40:
            continue
        c_mean, c_n = canon_mean(c)
        if c_n < 40 or len(bars) < 2:
            continue
        short = max(0.0, bars["usocket"] - u_mean) + max(0.0, bars["chain"] - c_mean)
        if best is None or short < best[0]:
            best = (short, row, step, u_mean, c_mean)
    if best:
        report["closest"] = {"row": best[1], "step": best[2], "usocket": round(best[3], 4), "chain": round(best[4], 4)}
        report["units_remaining"] = round(best[0], 4)
        report["complete"] = best[0] == 0.0 and all(k in report["bars"] and report["bars"][k]["dp_cotrain"] for k in ("usocket", "chain"))
    for emb in ("usocket", "chain"):
        ub = best_bar(peaks, UNITE_BC[emb], emb, VARIANT); db = report["bars"][emb]["dp_bc"]
        report["bc_fallback"][emb] = {"unite_bc": ub, "dp_bc": db, "shortfall": (round(max(0.0, db[0] - ub[0]), 4) if ub and db else None)}
    print(json.dumps(report, indent=1))
    print(f"units_remaining={report['units_remaining']} complete={report['complete']}")
    sys.exit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
