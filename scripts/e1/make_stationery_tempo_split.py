#!/usr/bin/env python3
"""Split manifest for the rl2 sort-stationery tempo-bucket runs (2026-09-25).

Input: the per-episode bucket table from the Step 0 tempo rerun (tau tertiles over the pooled 424 scored
rl2 episodes: organize_stationary = Elmo 198 + Aidan 29, organize_stationary_updated = aniketh 197;
cuts 0.2288 / 0.2854 m/s). Output: one JSON manifest with explicit episode lists.

  val_<b>     N_VAL episodes per bucket, drawn at random (seed) -- held out of EVERY run
  val         the union of the three (in-training validation for every run)
  train_<b>   the rest of bucket b, trimmed to the smallest bucket's size so all buckets train on the same count
  train_mixed the same count, spread evenly over the three train_<b> pools (a subset of them)

usage: make_stationery_tempo_split.py --buckets buckets_tertiles_424.csv --out manifest.json [--n-val 8 --seed 42]
"""

import argparse
import csv
import json

import numpy as np

B = ("slow", "medium", "fast")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.buckets)))
    info = {r["episode_hash"]: r for r in rows}
    rng = np.random.default_rng(args.seed)
    by = {b: sorted(r["episode_hash"] for r in rows if r["bucket"] == b) for b in B}
    sets = {}
    for b in B:
        perm = list(rng.permutation(by[b]))
        sets[f"val_{b}"] = sorted(perm[: args.n_val])
        sets[f"train_{b}"] = perm[args.n_val:]
    n_train = min(len(sets[f"train_{b}"]) for b in B)
    for b in B:
        sets[f"train_{b}"] = sorted(sets[f"train_{b}"][:n_train])  # already shuffled: trimming drops random episodes
    per = [n_train // 3 + (1 if k < n_train % 3 else 0) for k in range(3)]  # 44/44/45 style
    mixed = []
    for b, k in zip(B, sorted(per)):
        mixed += list(rng.choice(sets[f"train_{b}"], size=k, replace=False))
    sets["train_mixed"] = sorted(mixed)
    sets["val"] = sorted(sum((sets[f"val_{b}"] for b in B), []))
    allv = set(sets["val"])
    for k, v in sets.items():
        if k.startswith("train"):
            assert not allv & set(v), k
    out = {"version": "v1", "created": "2026-09-25", "source": args.buckets, "seed": args.seed, "n_val_per_bucket": args.n_val,
           "pool": {"episodes": len(rows), "tertile_cuts_m_per_s": [0.2288, 0.2854],
                    "tasks": sorted({r["task"] for r in rows}), "operators": sorted({r["operator"] for r in rows})},
           "sets": {}}
    for k, v in sets.items():
        ops, tasks = {}, {}
        for h in v:
            ops[info[h]["operator"]] = ops.get(info[h]["operator"], 0) + 1
        out["sets"][k] = {"n": len(v), "hours": round(sum(float(info[h]["duration_s"]) for h in v) / 3600, 3),
                          "tau_median": round(float(np.median([float(info[h]["tau_ep"]) for h in v])), 4),
                          "buckets": {b: sum(1 for h in v if info[h]["bucket"] == b) for b in B},
                          "operators": ops, "episodes": v}
    json.dump(out, open(args.out, "w"), indent=1)
    for k, v in out["sets"].items():
        print(f"{k:12s} n={v['n']:4d} h={v['hours']:.3f} tau_med={v['tau_median']:.3f} {v['buckets']} {v['operators']}")


if __name__ == "__main__":
    main()
