import glob, json, os, re, time, subprocess, csv, collections
CEDAR = "/storage/cedar/cedar0/cedarp-dxu345-0/agao81/runs"
rows = json.load(open(os.path.expanduser("~/scratch/autoresearch/orchestrator-260909-1520/monitor_ct3_rows_s3.json")))
# in-code rows of monitor_ct3.py that are not in the json
extra = {"ctAc": dict(sweep="unite-cotrain-2", prefix="ctA-compiled-h384-240k", embs=["usocket","chain"], unite=True, oec={}),
         "ctB": dict(sweep="unite-cotrain-2", prefix="ctB-h384-240k", embs=["usocket","chain"], unite=True, oec={}),
         "unitech": dict(sweep="unite-cotrain-2", prefix="unite-chain-points6-h384-240k", embs=["chain"], unite=True, oec={}),
         "dpch": dict(sweep="unite-cotrain-2", prefix="dp_paper-chain-points6-240k", embs=["chain"], unite=False, oec={})}
for k, v in extra.items(): rows.setdefault(k, v)
# rates from logs
rate = {}
for f in glob.glob(os.path.expanduser("~/scratch/logs/ct2-*.log")):
    tag = re.sub(r"-\d+\.log$", "", os.path.basename(f))[4:]
    txt = open(f, "rb").read()[-200000:].decode("utf8", "ignore")
    m = re.findall(r"([0-9.]+)it/s", txt)
    if m: rate[tag] = float(m[-1])
# scored units from summary
scored = collections.defaultdict(set)  # row -> {(step, emb, level, variant)}
n_by = {}
for r in csv.DictReader((l for l in open(os.path.expanduser("~/scratch/rollouts/CT2-rev3/summary.tsv")) if l.strip() and not l.startswith("#")), delimiter="\t"):
    if int(r["n"]) >= 80 and r["level"] == "0":
        scored[r["row"]].add((int(r["step"]), r["emb"], r["variant"]))
    if r["level"] != "0":
        scored[r["row"]].add((int(r["step"]), r["emb"], "oec", int(r["level"])))
now = time.time()
# job state per run dir: the runner writes runner-state; use squeue names for liveness
live = {}
for line in subprocess.run("squeue -u $USER -h -o '%j %T %M'", shell=True, capture_output=True, text=True).stdout.splitlines():
    parts = line.split()
    if len(parts) >= 3: live[parts[0]] = (parts[1], parts[2])
def elapsed_s(t):
    d, _, hms = t.rpartition("-"); h, m, sec = ([0, 0] + [int(x) for x in hms.split(":")])[-3:]
    return (int(d) if d else 0) * 86400 + h * 3600 + m * 60 + sec
print(f"{'row':10} {'latest ckpt':>11} {'est step':>8} {'it/s':>5} {'train ETA':>9} {'job':>9}  {'L0 scored through':>17}  {'OEC done (120/180/240k)':>22}")
for row, spec in sorted(rows.items()):
    d = sorted(glob.glob(os.path.join(CEDAR, spec["sweep"], spec["prefix"] + "-*")))
    if not d: print(f"{row:10} (no run dir)"); continue
    d = d[-1]
    cks = [(int(re.search(r"step=(\d+)", c).group(1)), os.path.getmtime(c)) for c in glob.glob(os.path.join(d, "checkpoints", "epoch-*step=*.ckpt"))]
    last, mt = max(cks) if cks else (0, now)
    tag = spec["prefix"]; r = rate.get(tag)
    done = last >= 240000
    est = 240000 if done else (int(last + (now - mt) * r) if r else None)
    jobname = "ct2-" + spec["prefix"]
    alive = [k for k in live if k.startswith(jobname)]
    state = "done" if done else (live[alive[0]][0] if alive else "DEAD")
    if state == "DEAD":
        est = last
    elif not done and alive and r:
        # a resumed/requeued segment cannot have advanced more than its own elapsed time
        est = int(last + min(now - mt, elapsed_s(live[alive[0]][1])) * r)
    eta = "done" if done else ("DEAD" if state == "DEAD" else (f"{(240000 - min(est,240000)) / r / 3600:4.1f} h" if r and est is not None else "?"))
    variant = "cfg1" if spec["unite"] else "cfgemb"
    embs = spec["embs"]
    full = [s for s in range(30000, 240001, 30000) if all((s, e, variant) in scored[row] for e in embs)]
    thru = max(full) if full else 0
    oec = ""
    if spec.get("oec"):
        parts = []
        for s in (120000, 180000, 240000):
            lv = sum(1 for e in embs for l in range(1, 31) if (s, e, "oec", l) in scored[row])
            parts.append(f"{lv}/{30*len(embs)}")
        oec = " ".join(parts)
    print(f"{row:10} {last:>11} {str(est):>8} {str(r):>5} {eta:>9} {state:>9}  {thru:>17}  {oec:>22}")
