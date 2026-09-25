#!/usr/bin/env python3
"""Stationery MID-TEMPO HOLDOUT splits, v2 (2026-09-15): v1 plus an in-distribution test set.

Pool: ABC "sort the stationery into containers" + rl2 "organize_stationary" (the YAM
station uploads of 2026-09-14/15), one Step 0 run over both (tau_ep per episode).

Tertiles are cut on tau_ep over the whole pool. The MIDDLE tertile is removed from
training entirely -- no training episode has a tau inside it -- so test_mid is an
interpolation probe between the slow and fast modes the model did see.

  train     = slow tertile + fast tertile, minus val, minus test_in
  val       = N_VAL episodes, half from each training tertile (checkpoint selection)
  test_mid  = N_TEST episodes of the middle tertile (reported, never selected on)
  test_in   = N_TEST_IN episodes, half from each training tertile, disjoint from val
              (reported, never selected on) -- the in-distribution reference, so the
              interpolation penalty is test_mid - test_in rather than test_mid - val,
              which would be read against the set checkpoints were selected on
  mid_all   = every middle-tertile episode (record only)

v1 compatibility: val and test_mid are drawn first, in v1's order and from the same
seeded generator, so they are identical to v1; test_in is drawn afterwards.

Draws are seeded and stratified by source (abc / rl2) in proportion to each tertile's
own composition. Eval episodes need at least MIN_EVAL_FRAMES frames.

usage: build_stationery_midtempo_v2.py --step0 <dir with episodes.csv, summary.json>
         --sources pool_sources.json --out <splits dir>
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def stratified(pool, n, rng):
    by = defaultdict(list)
    for e in sorted(pool, key=lambda e: e["episode"]):
        by[e["source"]].append(e)
    total = len(pool)
    quota = {s: n * len(v) / total for s, v in by.items()}
    take = {s: int(np.floor(q)) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - take[s], reverse=True)[: n - sum(take.values())]:
        take[s] += 1
    out = []
    for s in sorted(by):
        idx = rng.permutation(len(by[s]))[: take[s]]
        out += [by[s][i] for i in idx]
    return out


def pack(lst):
    t = np.array([e["tau"] for e in lst])
    src = defaultdict(int)
    for e in lst:
        src[e["source"]] += 1
    return {
        "n": len(lst),
        "hours": round(sum(e["duration_s"] for e in lst) / 3600, 3),
        "tau_median": round(float(np.median(t)), 4),
        "tau_range": [round(float(t.min()), 4), round(float(t.max()), 4)],
        "tau_cv": round(float(t.std() / t.mean()), 4),
        "sources": dict(src),
        "episodes": sorted(e["episode"] for e in lst),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step0", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-test-in", type=int, default=40)
    ap.add_argument("--min-eval-frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source_of = json.load(open(args.sources))["source_of"]
    rows = list(csv.DictReader(open(os.path.join(args.step0, "episodes.csv"))))
    eps = []
    for r in rows:
        try:
            tau = float(r["tau_ep"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(tau):
            continue
        eps.append({"episode": r["episode"], "tau": tau, "duration_s": float(r["duration_s"]),
                    "n_frames": int(float(r["n_frames"])), "source": source_of[r["episode"]]})
    unanalysed = sorted(set(source_of) - {e["episode"] for e in eps})
    taus = np.array([e["tau"] for e in eps])
    q1, q2 = (float(np.quantile(taus, 1 / 3)), float(np.quantile(taus, 2 / 3)))
    slow = [e for e in eps if e["tau"] < q1]
    mid = [e for e in eps if q1 <= e["tau"] <= q2]
    fast = [e for e in eps if e["tau"] > q2]

    rng = np.random.default_rng(args.seed)
    ok = lambda lst: [e for e in lst if e["n_frames"] >= args.min_eval_frames]  # noqa: E731
    # v1 order: val, then test_mid.
    val = stratified(ok(slow), args.n_val // 2, rng) + stratified(ok(fast), args.n_val - args.n_val // 2, rng)
    val_ids = {e["episode"] for e in val}
    test_mid = stratified(ok(mid), args.n_test, rng)
    # v2: test_in from the training tertiles, excluding val.
    not_val = lambda lst: [e for e in ok(lst) if e["episode"] not in val_ids]  # noqa: E731
    test_in = stratified(not_val(slow), args.n_test_in // 2, rng) + stratified(
        not_val(fast), args.n_test_in - args.n_test_in // 2, rng
    )
    test_in_ids = {e["episode"] for e in test_in}
    train = [e for e in slow + fast if e["episode"] not in val_ids | test_in_ids]

    train_ids = {e["episode"] for e in train}
    train_taus = np.array([e["tau"] for e in train])
    assert not ((train_taus >= q1) & (train_taus <= q2)).any(), "a training episode sits in the held-out band"
    test_mid_ids = {e["episode"] for e in test_mid}
    for a, b, name in ((val_ids, test_mid_ids, "val/test_mid"), (val_ids, test_in_ids, "val/test_in"),
                       (train_ids, val_ids, "train/val"), (train_ids, test_in_ids, "train/test_in"),
                       (train_ids, test_mid_ids, "train/test_mid"), (test_in_ids, test_mid_ids, "test_in/test_mid")):
        assert not (a & b), f"overlap {name}"

    summary = {}
    sp = os.path.join(args.step0, "summary.json")
    if os.path.exists(sp):
        summary = json.load(open(sp))
    by_source = {}
    for s in sorted({e["source"] for e in eps}):
        t = np.array([e["tau"] for e in eps if e["source"] == s])
        by_source[s] = {"n": int(len(t)), "tau_median": round(float(np.median(t)), 4),
                        "tau_q10_q90": [round(float(np.quantile(t, 0.1)), 4), round(float(np.quantile(t, 0.9)), 4)],
                        "in_slow_mid_fast": [sum(e["source"] == s for e in part) for part in (slow, mid, fast)]}

    man = {
        "design": "stationery mid-tempo holdout v2: train = slow + fast tau_ep tertiles minus val and test_in, "
                  "middle tertile held out; test_in = in-distribution reference",
        "version": "v2",
        "tasks": {"abc": "sort the stationery into containers", "rl2": "organize_stationary"},
        "embodiment": "yam_bimanual",
        "seed": args.seed,
        "pool": {"analysed": len(eps), "not_analysed_by_step0": unanalysed,
                 "hours": round(sum(e["duration_s"] for e in eps) / 3600, 2),
                 "tertile_cuts": [round(q1, 4), round(q2, 4)], "by_source": by_source},
        "step0_summary": summary,
        "sets": {"train": pack(train), "val": pack(val), "test_mid": pack(test_mid), "test_in": pack(test_in),
                 "slow_tertile": pack(slow), "mid_all": pack(mid), "fast_tertile": pack(fast)},
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(man, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    for k in ("train", "val", "test_mid", "test_in", "mid_all"):
        open(os.path.join(args.out, f"{k}.txt"), "w").write("\n".join(man["sets"][k]["episodes"]) + "\n")
    brief = {k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in man["sets"].items()}
    print(json.dumps({"pool": man["pool"], "sets": brief}, indent=1))


if __name__ == "__main__":
    main()
