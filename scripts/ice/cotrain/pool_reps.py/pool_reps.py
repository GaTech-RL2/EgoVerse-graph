#!/usr/bin/env python3
"""Pool protocol replicates: the monitor's rep1 file (<row>-<step>k-<emb>-L0-s<blk>-<var>.json) plus -rep2/-rep3.
Prints per (row, step, emb): mean of per-replicate means (each replicate = 80 seeds pooled, or 40 canonical), SD across replicates, and the grand mean over all episodes."""
import glob, json, os, re, sys, statistics as st
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/scratch/rollouts/CT2-rev3")
pat = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L0-s(\d+)-(cfgemb|cfg1)(?:-rep(\d))?\.json$")
data = {}  # (row, step, emb, var) -> {rep: {seed: peak}}
for p in glob.glob(os.path.join(D, "*.json")):
    m = pat.match(os.path.basename(p))
    if not m: continue
    row, stepk, emb, blk, var, rep = m.groups(); rep = int(rep or 1)
    try: d = json.load(open(p))
    except Exception: continue
    if (d.get("protocol") or {}).get("horizon_revision") != 3: continue
    bucket = data.setdefault((row, int(stepk), emb, var), {}).setdefault(rep, {})
    for e in d["episodes"]: bucket[int(e["seed"])] = float(e["peak"])
want = {("dpct", 240, "usocket", "cfgemb"), ("dpct", 240, "chain", "cfgemb"), ("dpus", 180, "usocket", "cfgemb"), ("dpch", 240, "chain", "cfgemb"), ("ctA", 120, "usocket", "cfg1"), ("ctA", 120, "chain", "cfg1")}
print(f"{'row':6} {'step':>5} {'emb':8} {'reps':>4}  {'pooled80 per rep':>34}  {'mean±SD':>13}  {'canon40 per rep':>30}  {'mean±SD':>13}  {'grand':>6}")
for key in sorted(want):
    reps = data.get(key, {})
    full = {r: s for r, s in reps.items() if len(s) >= 80}
    if not full: print(key, "no complete replicate yet"); continue
    p80 = [st.mean(s.values()) for r, s in sorted(full.items())]
    c40 = [st.mean(v for k, v in s.items() if k < 40) for r, s in sorted(full.items())]
    allv = [v for s in full.values() for v in s.values()]
    sd = lambda x: st.stdev(x) if len(x) > 1 else float("nan")
    print(f"{key[0]:6} {key[1]:>4}k {key[2]:8} {len(full):>4}  {str([round(x,3) for x in p80]):>34}  {st.mean(p80):.3f}±{sd(p80):.3f}  {str([round(x,3) for x in c40]):>30}  {st.mean(c40):.3f}±{sd(c40):.3f}  {st.mean(allv):.3f}")
