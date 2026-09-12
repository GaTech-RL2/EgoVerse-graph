"""Replicate-mean comparison (8 x 80 seeds per cell) with paired SE against DP cotrain."""
import glob, json, os, re, statistics as st, math, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.expanduser(sys.argv[1]); out = sys.argv[2]
pat = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L0-s(\d+)-(cfgemb|cfg1)(?:-rep(\d))?\.json$")
data = {}
for p in glob.glob(os.path.join(D, "*.json")):
    m = pat.match(os.path.basename(p))
    if not m: continue
    row, stepk, emb, blk, var, rep = m.groups(); rep = int(rep or 1); d = json.load(open(p))
    if (d.get("protocol") or {}).get("horizon_revision") != 3: continue
    b = data.setdefault((row, int(stepk), emb, var), {}).setdefault(rep, {}); b.update({int(e["seed"]): float(e["peak"]) for e in d["episodes"]})
def reps(k): return [s for s in data.get(k, {}).values() if len(s) >= 80]
def per_seed(k):
    r = reps(k); return {s: st.mean(x[s] for x in r) for s in range(80)}
def mean_se(k):
    ps = per_seed(k); v = list(ps.values()); return st.mean(v), st.stdev(v) / math.sqrt(80)
cells = {
 "Sweep 2 (U-Socket 3000 + chain 3000)": {
   "usocket": [("DP BC", ("dpus", 180, "usocket", "cfgemb")), ("DP cotrain", ("dpct", 240, "usocket", "cfgemb")), ("UNITE cotrain\n120k", ("ctA", 120, "usocket", "cfg1")), ("UNITE cotrain\n+EMA 240k", ("ctAema", 240, "usocket", "cfg1"))],
   "chain":   [("DP BC", ("dpch", 240, "chain", "cfgemb")), ("DP cotrain", ("dpct", 240, "chain", "cfgemb")), ("UNITE cotrain\n120k", ("ctA", 120, "chain", "cfg1")), ("UNITE cotrain\n+EMA 240k", ("ctAema", 240, "chain", "cfg1"))]},
 "Sweep 3 (+ 1,919 chain obstacle episodes)": {
   "usocket": [("DP cotrain", ("s3dpct", 180, "usocket", "cfgemb")), ("UNITE cotrain\n180k", ("s3ctA", 180, "usocket", "cfg1")), ("UNITE cotrain\n210k", ("s3ctA", 210, "usocket", "cfg1")), ("UNITE cotrain\n240k", ("s3ctA", 240, "usocket", "cfg1"))],
   "chain":   [("DP cotrain", ("s3dpct", 240, "chain", "cfgemb")), ("UNITE cotrain\n180k", ("s3ctA", 180, "chain", "cfg1")), ("UNITE cotrain\n210k", ("s3ctA", 210, "chain", "cfg1")), ("UNITE cotrain\n240k", ("s3ctA", 240, "chain", "cfg1"))]}}
color = {"DP BC": "#8a8a8a", "DP cotrain": "#111111"}
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
for i, (sweep, embs) in enumerate(cells.items()):
    for j, emb in enumerate(("usocket", "chain")):
        ax = axes[i, j]; labels, means, ses, cols = [], [], [], []
        for label, key in embs[emb]:
            if not reps(key): continue
            m, se = mean_se(key); labels.append(f"{label}"); means.append(m); ses.append(se)
            cols.append(color.get(label, "#0e7a86"))
        x = range(len(means))
        ax.bar(x, means, yerr=ses, color=cols, capsize=4, width=0.6)
        for xi, m, se in zip(x, means, ses): ax.text(xi, m + se + 0.008, f"{m:.3f}", ha="center", fontsize=9)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0.45, 0.78); ax.set_ylabel("mean peak coverage" if j == 0 else ""); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{sweep} · {'U-Socket (budget 318)' if emb=='usocket' else 'Chain gripper (budget 688)'}", fontsize=10, loc="left")
fig.suptitle("Protocol level 0, replicate means: 8 replicates × 80 seeds per cell; error bars = SE over the 80 per-seed means (UNITE at CFG 1.0)", fontsize=10.5)
fig.text(0.01, 0.005, "8 replicates × 80 seeds per cell. Paired by seed vs DP cotrain — sweep 2: UNITE 120k U-Socket −0.027±0.021, chain +0.024±0.022; vs DP BC −0.006±0.020, +0.067±0.023 (t 3.0).\nSweep 3: UNITE 240k U-Socket −0.025±0.028, chain +0.013±0.024; UNITE 180k chain +0.038±0.022 (t 1.8), U-Socket −0.089±0.030 (t −3.0).", fontsize=7.5, color="#444")
fig.tight_layout(rect=(0, 0.045, 1, 0.96)); fig.savefig(out, dpi=170); print("wrote", out)
