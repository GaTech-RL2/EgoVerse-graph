#!/usr/bin/env python3
"""Summarise the -final protocol evaluation: level 0 (seeds 0-39, replicates) and OEC-56 per checkpoint x embodiment x CFG."""
import glob, json, os, re, statistics as st, math, sys
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/scratch/rollouts/CT2-rev3")
L0 = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L0-s0-(cfg[0-9.]+|cfgemb)-final-rep(\d)\.json$")
OEC = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-OEC\d+-\d+-(cfg[0-9.]+|cfgemb)-final-L(\d+)\.json$")
l0, oec = {}, {}
for p in glob.glob(os.path.join(D, "*.json")):
    b = os.path.basename(p); m = L0.match(b); mo = OEC.match(b)
    if not (m or mo): continue
    try: d = json.load(open(p))
    except Exception: continue
    if (d.get("protocol") or {}).get("horizon_revision") != 3: continue
    if m:
        row, stepk, emb, cfg, rep = m.groups()
        l0.setdefault((row, int(stepk), emb, cfg), {})[int(rep)] = {int(e["seed"]): float(e["peak"]) for e in d["episodes"]}
    else:
        row, stepk, emb, cfg, lvl = mo.groups()
        oec.setdefault((row, int(stepk), emb, cfg), {})[int(lvl)] = [float(e["peak"]) for e in d["episodes"]]
def cfgkey(c): return -1.0 if c == "cfgemb" else float(c[3:])
print("== LEVEL 0 (seeds 0-39): per-replicate means | mean ± SE over 40 per-seed means | SR@0.80")
for k in sorted(l0, key=lambda k: (k[0], k[1], k[2], cfgkey(k[3]))):
    reps = [r for r in l0[k].values() if len(r) >= 40]
    if not reps: continue
    means = [st.mean(r.values()) for r in reps]
    ps = [st.mean(r[s] for r in reps) for s in range(40)]
    se = st.stdev(ps) / math.sqrt(40)
    sr = sum(1 for r in reps for v in r.values() if v >= 0.80) / sum(len(r) for r in reps)
    print(f"  {k[0]:7} {k[1]}k {k[2]:8} {k[3]:7} reps={len(reps)} {[round(x,3) for x in means]} -> {st.mean(means):.3f} ± {se:.3f}  SR80 {sr:.2f}")
print("== OEC-56 (30 levels x 5 seeds): mean peak | levels scored | never-moved | SR@0.80")
for k in sorted(oec, key=lambda k: (k[0], k[1], k[2], cfgkey(k[3]))):
    lv = oec[k]; allv = [v for vs in lv.values() for v in vs]
    nm = sum(1 for v in allv if v < 0.02); sr = sum(1 for v in allv if v >= 0.80)
    print(f"  {k[0]:7} {k[1]}k {k[2]:8} {k[3]:7} mean {st.mean(allv):.3f}  levels {len(lv):2d}/30  never-moved {nm}/{len(allv)}  SR80 {sr}/{len(allv)}")
