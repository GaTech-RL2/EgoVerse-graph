#!/usr/bin/env python3
"""Harvest the stationery tempo-bucket open-loop evals (scripts/e1/stattempo_eval.sbatch).

Selection per run: the checkpoint with the lowest paired_mse on the full 24-episode held-out set, computed as the
executed-step-weighted mean over val_slow / val_medium / val_fast (== the evaluator's micro average on their union).
Reported: every (run, step, bucket) paired_mse, plus the selected and final checkpoint per run.

usage: harvest_stattempo.py [--root /storage/project/r-dxu345-0/agao81/runs/stat_tempo_eval]
"""
import argparse, glob, json, os, re
from collections import defaultdict

B = ("slow", "medium", "fast")
M = ("paired_mse", "xyz_mse", "ypr_mse", "grip_mse", "mse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/storage/project/r-dxu345-0/agao81/runs/stat_tempo_eval")
    a = ap.parse_args()
    ev = defaultdict(dict)  # (run, step) -> bucket -> metrics
    for p in glob.glob(os.path.join(a.root, "*", "step*_val_*", "open_loop_sim.json")):
        run = p.split(os.sep)[-3]
        m = re.fullmatch(r"step(\d+)_val_(slow|medium|fast)", p.split(os.sep)[-2])
        if not m: continue
        d = json.load(open(p))
        ev[(run, int(m.group(1)))][m.group(2)] = {**{k: float(d["micro"][k]) for k in M}, "n": int(d["executed_control_steps"]), "eps": int(d["episodes"]), "cov": float(d["coverage"])}
    rows, sel = [], {}
    for (run, step), bk in sorted(ev.items()):
        if len(bk) < 3: continue
        n = sum(bk[b]["n"] for b in B)
        allv = {k: sum(bk[b][k] * bk[b]["n"] for b in B) / n for k in M}
        rows.append(dict(run=run, step=step, **{f"{b}_paired": bk[b]["paired_mse"] for b in B}, all_paired=allv["paired_mse"],
                         all_xyz=allv["xyz_mse"], all_grip=allv["grip_mse"], min_cov=min(bk[b]["cov"] for b in B)))
    by = defaultdict(list)
    for r in rows: by[r["run"]].append(r)
    out = {}
    for run, rs in sorted(by.items()):
        best = min(rs, key=lambda r: r["all_paired"]); fin = max(rs, key=lambda r: r["step"])
        out[run] = {"selected": best, "final": fin, "steps": [r["step"] for r in rs]}
    json.dump({"rows": rows, "selection": out}, open(os.path.join(a.root, "harvest.json"), "w"), indent=1)
    with open(os.path.join(a.root, "harvest.tsv"), "w") as f:
        f.write("run\tstep\tslow\tmedium\tfast\tall24\tmin_cov\n")
        for r in rows: f.write(f"{r['run']}\t{r['step']}\t{r['slow_paired']:.5f}\t{r['medium_paired']:.5f}\t{r['fast_paired']:.5f}\t{r['all_paired']:.5f}\t{r['min_cov']:.3f}\n")
    print("| run | selected step | slow | medium | fast | all 24 | final (240k) all 24 |")
    print("|---|---|---|---|---|---|---|")
    for run, s in out.items():
        b, f = s["selected"], s["final"]
        print(f"| {run} | {b['step']//1000}k | {b['slow_paired']:.5f} | {b['medium_paired']:.5f} | {b['fast_paired']:.5f} | {b['all_paired']:.5f} | {f['all_paired']:.5f} |")


if __name__ == "__main__":
    main()
