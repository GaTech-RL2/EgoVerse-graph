"""Eight-replicate campaign tables restated on the canonical seeds 0-39 (and pooled 0-79 alongside)."""
import glob, json, os, re, statistics as st, math
D = os.path.expanduser("~/scratch/rollouts/CT2-rev3")
pat = re.compile(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L0-s(\d+)-(cfgemb|cfg1)(?:-rep(\d))?\.json$")
data = {}
for p in glob.glob(os.path.join(D, "*.json")):
    m = pat.match(os.path.basename(p))
    if not m: continue
    row, stepk, emb, blk, var, rep = m.groups(); rep = int(rep or 1); d = json.load(open(p))
    if (d.get("protocol") or {}).get("horizon_revision") != 3: continue
    b = data.setdefault((row, int(stepk), emb, var), {}).setdefault(rep, {}); b.update({int(e["seed"]): float(e["peak"]) for e in d["episodes"]})
def reps(k, n): return [s for s in data.get(k, {}).values() if all(x in s for x in range(n))]
def stats(k, n):
    r = reps(k, n); ps = [st.mean(x[s] for x in r) for s in range(n)]
    return st.mean(ps), st.stdev(ps) / math.sqrt(n), len(r)
def paired(a, b, n):
    A, B = reps(a, n), reps(b, n)
    pa = {s: st.mean(x[s] for x in A) for s in range(n)}; pb = {s: st.mean(x[s] for x in B) for s in range(n)}
    d = [pa[s] - pb[s] for s in range(n)]; m = st.mean(d); se = st.stdev(d) / math.sqrt(n); return m, se, m / se
cells = {"usocket": [("DP BC", ("dpus", 180, "usocket", "cfgemb")), ("DP cotrain", ("dpct", 240, "usocket", "cfgemb")), ("UNITE cotrain 120k", ("ctA", 120, "usocket", "cfg1")), ("UNITE cotrain+EMA 240k", ("ctAema", 240, "usocket", "cfg1")),
                     ("s3 DP cotrain 180k", ("s3dpct", 180, "usocket", "cfgemb")), ("s3 UNITE cotrain 180k", ("s3ctA", 180, "usocket", "cfg1")), ("s3 UNITE cotrain 240k", ("s3ctA", 240, "usocket", "cfg1"))],
         "chain":   [("DP BC", ("dpch", 240, "chain", "cfgemb")), ("DP cotrain", ("dpct", 240, "chain", "cfgemb")), ("UNITE cotrain 120k", ("ctA", 120, "chain", "cfg1")), ("UNITE cotrain+EMA 240k", ("ctAema", 240, "chain", "cfg1")),
                     ("s3 DP cotrain 240k", ("s3dpct", 240, "chain", "cfgemb")), ("s3 UNITE cotrain 180k", ("s3ctA", 180, "chain", "cfg1")), ("s3 UNITE cotrain 240k", ("s3ctA", 240, "chain", "cfg1"))]}
for emb in ("usocket", "chain"):
    print(f"== {emb}: canonical 40 (mean ± SE, reps) | pooled 80")
    for label, k in cells[emb]:
        m40, se40, n40 = stats(k, 40); m80, se80, n80 = stats(k, 80)
        print(f"  {label:24} {m40:.3f} ± {se40:.3f} ({n40} reps) | {m80:.3f} ± {se80:.3f} ({n80} reps)")
    pairs = [("UNITE cotrain 120k - DP BC", ("ctA", 120), ("dpus", 180) if emb == "usocket" else ("dpch", 240)), ("UNITE cotrain 120k - DP cotrain", ("ctA", 120), ("dpct", 240)),
             ("UNITE+EMA 240k - DP cotrain", ("ctAema", 240), ("dpct", 240)), ("DP cotrain - DP BC", ("dpct", 240), ("dpus", 180) if emb == "usocket" else ("dpch", 240)),
             ("s3 UNITE 180k - s3 DP cotrain", ("s3ctA", 180), ("s3dpct", 180 if emb == "usocket" else 240)), ("s3 UNITE 240k - s3 DP cotrain", ("s3ctA", 240), ("s3dpct", 180 if emb == "usocket" else 240))]
    for label, a, b in pairs:
        va = "cfg1" if a[0].startswith(("ctA", "s3ctA")) else "cfgemb"; vb = "cfg1" if b[0].startswith(("ctA", "s3ctA")) else "cfgemb"
        m, se, t = paired((a[0], a[1], emb, va), (b[0], b[1], emb, vb), 40); m8, se8, t8 = paired((a[0], a[1], emb, va), (b[0], b[1], emb, vb), 80)
        print(f"  paired {label:32} 40: {m:+.3f} ± {se:.3f} (t {t:+.2f}) | 80: {m8:+.3f} ± {se8:.3f} (t {t8:+.2f})")
