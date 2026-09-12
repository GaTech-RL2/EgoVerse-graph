import glob, json, os, re, statistics as st, math
D = os.path.expanduser("~/scratch/rollouts/CT2-rev3")
L0 = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L0-s0-(cfg[0-9.]+|cfgemb)-final-rep(\d)\.json$")
l0 = {}
for p in glob.glob(os.path.join(D, "*.json")):
    m = L0.match(os.path.basename(p))
    if not m: continue
    row, stepk, emb, cfg, rep = m.groups(); d = json.load(open(p))
    l0.setdefault((row, int(stepk), emb, cfg), {})[int(rep)] = {int(e["seed"]): float(e["peak"]) for e in d["episodes"]}
def ps(k):
    reps = [r for r in l0[k].values() if len(r) >= 40]; return {s: st.mean(r[s] for r in reps) for s in range(40)}
def paired(a, b):
    A, B = ps(a), ps(b); d = [A[s] - B[s] for s in range(40)]; m = st.mean(d); se = st.stdev(d) / math.sqrt(40); return m, se, m / se, sum(x > 0 for x in d)
print("== UNITE cotrain (CFG 1.0) minus DP cotrain, same 40 seeds, 3 reps each side")
for sweep, u, dp in (("sweep 2", ("ctA", 120), ("dpct", 240)), ("sweep 3", ("s3ctA", 180), ("s3dpct", 240))):
    for emb in ("usocket", "chain"):
        m, se, t, w = paired((u[0], u[1], emb, "cfg1.0"), (dp[0], dp[1], emb, "cfgemb"))
        print(f"  {sweep} {emb:8}: {m:+.3f} ± {se:.3f} (t={t:+.2f}, seeds won {w}/40)")
print("== CFG x minus CFG 1.0, same checkpoint")
for u in (("ctA", 120), ("s3ctA", 180)):
    for emb in ("usocket", "chain"):
        for cfg in ("cfg1.5", "cfg2.0", "cfg3.0", "cfg4.0", "cfg6.0"):
            m, se, t, w = paired((u[0], u[1], emb, cfg), (u[0], u[1], emb, "cfg1.0"))
            print(f"  {u[0]:6} {emb:8} {cfg}: {m:+.3f} ± {se:.3f} (t={t:+.2f})")
