#!/usr/bin/env python
"""Give a held-out embodiment the pooled statistics of the training set.

normalize() looks stats up by embodiment id and, finding none, returns the
tensor UNCHANGED -- so a held-out embodiment would be fed raw pixels and raw
pose into a network trained on normalized inputs, and would still produce a
plausible-looking coverage number. sim_rollout_planar_eval asserts against that,
which is how this surfaced.

A held-out embodiment has no statistics by definition. Using one training
embodiment's stats makes the result depend on which donor is picked, so this
pools: every field is the element-wise mean across the training embodiments.
The choice is a real methodological knob, so it is recorded in the output file
under "pooled_from" and printed at run time.

Also prints the spread of quantile_1/quantile_99 across the donors. If the
embodiments barely differ, pooling is uncontroversial; if they differ a lot,
that is something the reader should know before trusting the held-out row.

Usage:
    python tools/pool_norm_stats.py norm_stats.json --out augmented.json \
        --sources 19 20 21 22 24 26 27 --targets 23 25
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stats", help="norm_stats.json from the training run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", required=True,
                    help="embodiment ids that WERE trained on")
    ap.add_argument("--targets", nargs="+", required=True,
                    help="held-out embodiment ids to synthesise stats for")
    a = ap.parse_args()

    payload = json.load(open(a.stats))
    stats = payload["stats"]
    missing = [s for s in a.sources if s not in stats]
    if missing:
        print(f"FATAL: source ids absent from {a.stats}: {missing}", file=sys.stderr)
        print(f"       present: {sorted(stats)}", file=sys.stderr)
        return 1

    # Keys must exist in every donor; a key present in only some would be
    # pooled over a different population than the rest.
    keysets = [set(stats[s]) for s in a.sources]
    common = set.intersection(*keysets)
    dropped = set.union(*keysets) - common
    if dropped:
        print(f"  NOTE: keys not present in every donor, skipped: {sorted(dropped)}")

    print(f"  pooling {len(common)} key(s) over {len(a.sources)} donors")
    for key in sorted(common):
        fields = set.intersection(*(set(stats[s][key]) for s in a.sources))
        if not {"quantile_1", "quantile_99"} <= fields:
            continue
        q1 = np.array([np.asarray(stats[s][key]["quantile_1"], dtype=np.float64)
                       for s in a.sources])
        q99 = np.array([np.asarray(stats[s][key]["quantile_99"], dtype=np.float64)
                        for s in a.sources])
        # Measure disagreement against the SPAN normalization maps to [-1, 1],
        # not against the mean: a channel centred near zero (any angle) makes a
        # mean-relative percentage meaningless.
        span = np.abs(q99.mean(axis=0) - q1.mean(axis=0)) + 1e-9
        for field, arr in (("quantile_1", q1), ("quantile_99", q99)):
            spread = (arr.max(axis=0) - arr.min(axis=0)) / span
            print(f"    {key}.{field}: donors disagree by up to "
                  f"{float(spread.max())*100:.1f}% of the normalized span")

    for target in a.targets:
        if target in stats:
            print(f"  {target} already present -- leaving it alone")
            continue
        entry = {}
        for key in sorted(common):
            fields = set.intersection(*(set(stats[s][key]) for s in a.sources))
            entry[key] = {
                f: np.mean([np.asarray(stats[s][key][f], dtype=np.float64)
                            for s in a.sources], axis=0).tolist()
                for f in sorted(fields)
            }
        stats[target] = entry
        print(f"  wrote pooled stats for held-out embodiment id {target}")

    payload.setdefault("pooled_from", {})
    for target in a.targets:
        payload["pooled_from"][target] = {
            "sources": list(a.sources),
            "method": "elementwise mean of each stat field across sources",
        }
    json.dump(payload, open(a.out, "w"))
    print(f"  wrote {a.out} with ids {sorted(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
