#!/usr/bin/env python3
"""Tiny manifest for the stationery mid-tempo smoke, drawn from the real v2 manifest.

train    = the 2 shortest ABC + 2 shortest rl2 episodes of the real train split
val / test_mid / test_in = the 2 shortest episodes of each real set
All at least MIN_FRAMES long so a 100-frame chunk and an open-loop walk both exist.

usage: make_smoke_manifest.py --manifest <splits_v2/manifest.json> --step0 <step0 dir>
         --sources <pool_sources.json> --out <smoke dir>
"""

import argparse
import copy
import csv
import json
import os

MIN_FRAMES = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--step0", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    source_of = json.load(open(args.sources))["source_of"]
    frames = {r["episode"]: int(float(r["n_frames"]))
              for r in csv.DictReader(open(os.path.join(args.step0, "episodes.csv")))}

    def shortest(eps, n, source=None):
        pool = [e for e in eps if frames.get(e, 0) >= MIN_FRAMES and (source is None or source_of[e] == source)]
        return sorted(pool, key=lambda e: (frames[e], e))[:n]

    sets = man["sets"]
    picks = {
        "train": shortest(sets["train"]["episodes"], 2, "abc") + shortest(sets["train"]["episodes"], 2, "rl2"),
        "val": shortest(sets["val"]["episodes"], 2),
        "test_mid": shortest(sets["test_mid"]["episodes"], 2),
        "test_in": shortest(sets["test_in"]["episodes"], 2),
    }
    smoke = copy.deepcopy(man)
    smoke["version"] = f"{man.get('version', 'v1')}-smoke"
    smoke["design"] = "SMOKE subset of " + man["design"]
    for k, eps in picks.items():
        smoke["sets"][k] = {
            "n": len(eps),
            "hours": round(sum(frames[e] for e in eps) / 30 / 3600, 3),
            "sources": {s: sum(source_of[e] == s for e in eps) for s in sorted({source_of[e] for e in eps})},
            "episodes": eps,
            "frames": {e: frames[e] for e in eps},
        }
    os.makedirs(args.out, exist_ok=True)
    json.dump(smoke, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in smoke["sets"].items()
                      if k in picks}, indent=1))


if __name__ == "__main__":
    main()
