# 2026-09-27: add two pools to the stationery tempo manifest, keeping the shared 24-episode held-out set out of training:
#   train_slowpace = the 229 organize_stationary eps (Elmo 200 + Aidan 29, instructed slow pace) minus held-out
#   train_all      = every usable rl2 stationery ep (organize_stationary + organize_stationary_updated) minus held-out
import csv
import json
import os

M = "scripts/e1/stationery_tempo_manifest.json"
man = json.load(open(M))
rows = [
    r
    for r in csv.DictReader(open(os.path.expanduser("~/rl2_stat_rows_2026-09-27.csv")))
    if float(r["num_frames"] or -1) > 0 and r["zarr_processed_path"]
]
val = set(man["sets"]["val"]["episodes"])
ops = {r["episode_hash"]: r["operator"] for r in rows}
dur = {r["episode_hash"]: float(r["num_frames"]) / 30 for r in rows}
buck = {}
for k in (
    "train_slow",
    "train_medium",
    "train_fast",
    "val_slow",
    "val_medium",
    "val_fast",
):
    for h in man["sets"][k]["episodes"]:
        buck[h] = k.split("_")[1]


def entry(eps):
    o = {}
    for h in eps:
        o[ops[h]] = o.get(ops[h], 0) + 1
    b = {x: sum(1 for h in eps if buck.get(h) == x) for x in ("slow", "medium", "fast")}
    b["unbucketed"] = sum(1 for h in eps if h not in buck)
    return {
        "n": len(eps),
        "hours": round(sum(dur[h] for h in eps) / 3600, 3),
        "tau_median": None,
        "buckets": b,
        "operators": o,
        "episodes": sorted(eps),
    }


slow = [r["episode_hash"] for r in rows if r["task"] == "organize_stationary"]
alls = [r["episode_hash"] for r in rows]
assert len(slow) == 229 and len(alls) == 427, (len(slow), len(alls))
man["sets"]["train_slowpace"] = entry([h for h in slow if h not in val])
man["sets"]["train_all"] = entry([h for h in alls if h not in val])
man["version"] = "v1+pools-2026-09-27"
json.dump(man, open(M, "w"), indent=1)
for k in ("train_slowpace", "train_all"):
    e = man["sets"][k]
    print(k, e["n"], e["hours"], e["operators"], e["buckets"])
