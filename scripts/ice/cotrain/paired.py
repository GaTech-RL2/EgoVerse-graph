import glob, json, os, re, statistics as st, math
D = os.path.expanduser("~/scratch/rollouts/CT2-rev3")
pat = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L0-s(\d+)-(cfgemb|cfg1)(?:-rep(\d))?\.json$")
data = {}
for p in glob.glob(os.path.join(D, "*.json")):
    m = pat.match(os.path.basename(p))
    if not m: continue
    row, stepk, emb, blk, var, rep = m.groups(); rep = int(rep or 1)
    d = json.load(open(p))
    if (d.get("protocol") or {}).get("horizon_revision") != 3: continue
    b = data.setdefault((row, int(stepk), emb, var), {}).setdefault(rep, {})
    for e in d["episodes"]: b[int(e["seed"])] = float(e["peak"])
def per_seed(key):
    reps = [s for s in data[key].values() if len(s) >= 80]
    return {seed: st.mean(s[seed] for s in reps) for seed in range(80)}, len(reps)
def paired(a, b):
    A, na = per_seed(a); B, nb = per_seed(b)
    d = [A[s] - B[s] for s in range(80)]
    m = st.mean(d); se = st.stdev(d) / math.sqrt(len(d)); t = m / se
    wins = sum(1 for x in d if x > 0); losses = sum(1 for x in d if x < 0)
    return m, se, t, wins, losses, na, nb
import sys
cands = [("ctA", 120, "UNITE cotrain 120k")]
for extra in sys.argv[1:]:  # e.g. ctAema:240
    r, stp = extra.split(":"); cands.append((r, int(stp), f"{r} {stp}k"))
for emb, dpbc in (("usocket", ("dpus", 180)), ("chain", ("dpch", 240))):
    dc = ("dpct", 240, emb, "cfgemb"); db = (dpbc[0], dpbc[1], emb, "cfgemb")
    for row, stp, label in cands:
        u = (row, stp, emb, "cfg1")
        if u not in data or not [s for s in data[u].values() if len(s) >= 80]: print(f"{emb:8} {label}: no complete replicate"); continue
        for name, other in (("DP cotrain 240k", dc), ("DP BC", db)):
            m, se, t, w, l, na, nb = paired(u, other)
            print(f"{emb:8} {label:18} vs {name:16}: mean diff {m:+.3f} ± {se:.3f} (t={t:+.2f}, seeds won {w}/lost {l}, reps {na}/{nb})")
    m, se, t, w, l, na, nb = paired(dc, db)
    print(f"{emb:8} DP cotrain 240k  vs DP BC          : mean diff {m:+.3f} ± {se:.3f} (t={t:+.2f}, seeds won {w}/lost {l})")
