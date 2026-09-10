#!/usr/bin/env python3
"""Protocol (sim_v2 rev-3) rollout monitor for the cotrain sweeps (2026-09-10).

Every pass: for each row, for each landed checkpoint, submit the missing protocol
rollouts through rollout_ct3.sh and rewrite the summary. Level 0 uses seeds 0-39
(canonical) and 40-79 (extension); obstacle levels use the OEC-56 bank (5 seeds each).
UNITE rows are rolled out at the embedded CFG (canonical) and at CFG 1.0 (labeled
NON-PROTOCOL). State: <OUT>/monitor-state.json. Never launches training.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

CEDAR = "/storage/cedar/cedar0/cedarp-dxu345-0/agao81/runs"
OUT = os.path.expanduser(os.environ.get("CT3_OUT", "~/scratch/rollouts/CT2-rev3"))
HELPER = os.path.expanduser("~/scratch/autoresearch/orchestrator-260909-1520/rollout_ct3.sh")
STATE = os.path.join(OUT, "monitor-state.json")
STEPS = tuple(range(30000, 240001, 30000))
MAX_INFLIGHT = int(os.environ.get("CT3_MAX_INFLIGHT", "24"))
OEC_LEVELS = tuple(range(1, 31))

# row -> dict(sweep_dir, prefix, embodiments, budget_set, unite, oec_levels_by_emb)
ROWS = {
    # sweep 2 (chain trained on chain_gripper_3000_v2 only)
    "ctA":     dict(sweep="unite-cotrain-2", prefix="ctA-h384-240k",                 embs=("usocket", "chain"), bset="s2", unite=True,  oec={}),
    "ctAc":    dict(sweep="unite-cotrain-2", prefix="ctA-compiled-h384-240k",        embs=("usocket", "chain"), bset="s2", unite=True,  oec={}),
    "ctB":     dict(sweep="unite-cotrain-2", prefix="ctB-h384-240k",                 embs=("usocket", "chain"), bset="s2", unite=True,  oec={}),
    "dpct":    dict(sweep="unite-cotrain-2", prefix="dp_paper-cotrain-points6-240k", embs=("usocket", "chain"), bset="s2", unite=False, oec={}),
    "dpus":    dict(sweep="unite-cotrain-2", prefix="dp_paper-usocket-240k",         embs=("usocket",),         bset="s2", unite=False, oec={}),
    "dpch":    dict(sweep="unite-cotrain-2", prefix="dp_paper-chain-points6-240k",   embs=("chain",),           bset="s2", unite=False, oec={}),
    "uniteus": dict(sweep="unite-cotrain-2", prefix="unite-usocket-h384-240k",       embs=("usocket",),         bset="s2", unite=True,  oec={}),
    "unitech": dict(sweep="unite-cotrain-2", prefix="unite-chain-points6-h384-240k", embs=("chain",),           bset="s2", unite=True,  oec={}),
}
# sweep 3 rows are appended by monitor_ct3_rows_s3.json if present (same schema, JSON)
_extra = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_ct3_rows_s3.json")
if os.path.exists(_extra):
    for k, v in json.load(open(_extra)).items():
        v["embs"] = tuple(v["embs"]); v["oec"] = {e: tuple(l) for e, l in v.get("oec", {}).items()}
        ROWS[k] = v


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def queued_names():
    return set(sh("squeue -u $USER -h -o %j").split())


def find_run(row):
    dirs = sorted(glob.glob(os.path.join(CEDAR, ROWS[row]["sweep"], ROWS[row]["prefix"] + "-*")))
    return dirs[-1] if dirs else None


def checkpoints(run_dir):
    out = {}
    for path in glob.glob(os.path.join(run_dir, "checkpoints", "*.ckpt")):
        m = re.search(r"step=(\d+)\.ckpt$", path)
        if m and int(m.group(1)) in STEPS and time.time() - os.path.getmtime(path) > 180:
            out[int(m.group(1))] = os.path.basename(path)
    return out


def jobs_for(row):
    """Yield (tag, helper-args) for every rollout this row needs per checkpoint."""
    cfg = ROWS[row]
    variants = [("cfgemb", [])] if not cfg["unite"] else [("cfgemb", []), ("cfg1", ["--cfg", "1.0"])]
    for emb in cfg["embs"]:
        for block in (0, 40):
            for vname, vargs in variants:
                yield f"{row}-STEP-{emb}-L0-s{block}-{vname}", ["0", "--seed-block", str(block), "--budget-set", cfg["bset"], *vargs]
        for chunk in cfg["oec"].get(emb, ()):  # "a-b" level ranges -> one GPU job, results <tag>-L<level>.json
            for vname, vargs in variants:
                yield f"{row}-STEP-{emb}-OEC{chunk}-{vname}", [str(chunk), "--budget-set", cfg["bset"], *vargs]


def submit_missing(state):
    queued = queued_names()
    inflight = sum(1 for n in queued if n.startswith("ro3-"))
    submitted = []
    for row in ROWS:
        run_dir = find_run(row)
        if not run_dir:
            continue
        for step, ckpt in sorted(checkpoints(run_dir).items()):
            for tag_t, hargs in jobs_for(row):
                tag = tag_t.replace("STEP", f"{step // 1000}k")
                emb = tag.split("-")[2]
                level = hargs[0]
                if "-" in level:  # OEC chunk: only at the configured steps, done when every level file exists
                    if step not in ROWS[row].get("oec_steps", (120000, 180000, 240000)):
                        continue
                    a, b = (int(v) for v in level.split("-"))
                    if all(os.path.exists(os.path.join(OUT, f"{tag}-L{l}.json")) for l in range(a, b + 1)):
                        continue
                elif os.path.exists(os.path.join(OUT, f"{tag}.json")):
                    continue
                if f"ro3-{tag}" in queued:
                    continue
                if state.get("submitted", {}).get(tag, 0) >= 2:
                    continue
                if inflight >= MAX_INFLIGHT:
                    return submitted
                cmd = f'OUTDIR="{OUT}" {HELPER} "{run_dir}" "{ckpt}" {emb} {level} {tag} ' + " ".join(hargs[1:])
                out = sh(cmd).strip()
                job = out.splitlines()[-1] if out else ""
                state.setdefault("submitted", {})[tag] = state.get("submitted", {}).get(tag, 0) + 1
                state.setdefault("jobs", {})[tag] = job
                submitted.append((tag, job))
                inflight += 1
    return submitted


def load_results():
    peaks = defaultdict(dict)  # (row, step, emb, level, variant) -> {seed: peak}
    meta = {}
    for path in glob.glob(os.path.join(OUT, "*.json")):
        name = os.path.basename(path)
        m = re.match(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-L(\d+)(?:-s(\d+))?-(cfgemb|cfg1)\.json$", name)
        if not m:
            m2 = re.match(r"([A-Za-z0-9]+)-(\d+)k-(usocket|chain)-OEC\d+-\d+-(cfgemb|cfg1)-L(\d+)\.json$", name)
            if not m2:
                continue
            row, stepk, emb, variant, level = m2.groups()
        else:
            row, stepk, emb, level, _blk, variant = m.groups()
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if "episodes" not in d:
            continue
        key = (row, int(stepk) * 1000, emb, int(level), variant)
        for e in d["episodes"]:
            peaks[key].setdefault(int(e["seed"]), float(e["peak"]))
        meta[key] = {"budget": (d.get("protocol") or {}).get("max_steps"), "label": (d.get("protocol") or {}).get("label")}
    return peaks, meta


def summarize(peaks, meta):
    lines = ["row\tstep\temb\tlevel\tvariant\tn\tmean_peak\tsr80\tsr95\tnever_moved\tbudget\tseeds0-39_mean"]
    stats = {}
    for key, seeds in sorted(peaks.items()):
        vals = list(seeds.values()); n = len(vals)
        canon = [v for s, v in seeds.items() if s < 40] if key[3] == 0 else vals
        mean = sum(vals) / n
        stats[key] = (mean, n, sum(canon) / len(canon) if canon else None, len(canon))
        lines.append(f"{key[0]}\t{key[1]}\t{key[2]}\t{key[3]}\t{key[4]}\t{n}\t{mean:.3f}\t{sum(v >= 0.8 for v in vals)}/{n}\t{sum(v >= 0.95 for v in vals)}/{n}\t{sum(v < 0.1 for v in vals)}\t{meta[key]['budget']}\t{(sum(canon)/len(canon)) if canon else float('nan'):.3f}")
    # bars: DP BC best checkpoint on canonical seeds (level 0, n>=40)
    bars = {}
    for emb, bc in (("usocket", "dpus"), ("chain", "dpch")):
        cands = [(v[2], k[1]) for k, v in stats.items() if k[0] == bc and k[2] == emb and k[3] == 0 and v[3] >= 40]
        if cands:
            bars[emb] = max(cands)
    verdict = ["", "# canonical DP BC bars (level 0, seeds 0-39, best checkpoint): " + json.dumps(bars)]
    best = None
    for row in ("ctA", "ctAc", "ctB"):
        for step in STEPS:
            for variant in ("cfgemb", "cfg1"):
                u = stats.get((row, step, "usocket", 0, variant)); c = stats.get((row, step, "chain", 0, variant))
                if not u or not c or u[3] < 40 or c[3] < 40 or len(bars) < 2:
                    continue
                short = max(0.0, bars["usocket"][0] - u[2]) + max(0.0, bars["chain"][0] - c[2])
                if best is None or short < best[0]:
                    best = (short, row, step, variant, u[2], c[2])
    if best:
        verdict.append(f"# closest cotrain checkpoint (canonical seeds): {best[1]} @ {best[2]} [{best[3]}] usocket={best[4]:.3f} chain={best[5]:.3f} shortfall={best[0]:.3f}" + ("  -> PREDICATE MET" if best[0] == 0 else ""))
    else:
        verdict.append("# predicate: not evaluable yet")
    text = "\n".join(lines + verdict) + "\n"
    with open(os.path.join(OUT, "summary.tsv"), "w") as f:
        f.write(text)
    return text


def one_pass():
    os.makedirs(OUT, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    submitted = submit_missing(state)
    json.dump(state, open(STATE, "w"), indent=1)
    text = summarize(*load_results())
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] submitted {len(submitted)}: {submitted}")
    print(text)
    sys.stdout.flush()


if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True:
            try:
                one_pass()
            except Exception as exc:
                print("monitor error:", repr(exc)); sys.stdout.flush()
            time.sleep(600)
    else:
        one_pass()
