#!/usr/bin/env python3
"""Harvest the stationery mid-tempo open-loop evals and select each row's checkpoint.

Reads <runs>/<row>_s42/eval/step<N>_<set>/{open_loop_sim.json, ckpt_path.txt} written by
scripts/e1/stationery_midtempo_eval.sbatch. Numbers come straight from the evaluator's JSON
(Aniketh's open_loop_sim, top-level ``micro`` = executed-step-weighted means over episodes).

Selection: per row, the checkpoint with the lowest val ``paired_mse`` (xyz + gripper, the
lab's paired columns). test_mid and test_in are never used to select. Reported per row:
the selected and the final (highest-step) checkpoint on val / test_mid / test_in, plus the
interpolation penalty test_mid - test_in.

usage: harvest_stationery_midtempo.py --runs <runs/stationery_midtempo> [--rows time arcdur]
writes <runs>/harvest.tsv, <runs>/selection.json and prints a markdown table.
"""

import argparse
import json
import os
import re

METRICS = ("paired_mse", "mse", "xyz_mse", "ypr_mse", "grip_mse")
SETS = ("val", "test_mid", "test_in")


def load_row(run_dir):
    evals = {}
    eval_root = os.path.join(run_dir, "eval")
    if not os.path.isdir(eval_root):
        return evals
    for name in sorted(os.listdir(eval_root)):
        m = re.fullmatch(r"step(\d+)_(val|test_mid|test_in)", name)
        path = os.path.join(eval_root, name, "open_loop_sim.json")
        if not m or not os.path.isfile(path):
            continue
        res = json.load(open(path))
        ckpt_file = os.path.join(eval_root, name, "ckpt_path.txt")
        evals[(int(m.group(1)), m.group(2))] = {
            **{k: float(res["micro"][k]) for k in METRICS},
            "coverage": float(res["coverage"]),
            "episodes": int(res["episodes"]),
            "executed_control_steps": int(res["executed_control_steps"]),
            "execute_control_steps": int(res.get("execute_control_steps", -1)),
            "ckpt": open(ckpt_file).read().strip() if os.path.isfile(ckpt_file) else None,
            "json": path,
        }
    return evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--rows", nargs="+", default=["time", "arcdur"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows_out, selection, tsv = {}, {}, ["row\tstep\tset\t" + "\t".join(METRICS) + "\tcoverage\tepisodes\tckpt"]
    for row in args.rows:
        run_dir = os.path.join(args.runs, f"{row}_s{args.seed}")
        evals = load_row(run_dir)
        for (step, es), v in sorted(evals.items()):
            tsv.append(f"{row}\t{step}\t{es}\t" + "\t".join(f"{v[k]:.6f}" for k in METRICS)
                       + f"\t{v['coverage']:.4f}\t{v['episodes']}\t{v['ckpt']}")
        val_steps = sorted(s for (s, es) in evals if es == "val")
        if not val_steps:
            print(f"{row}: no val evals yet")
            continue
        best = min(val_steps, key=lambda s: (evals[(s, "val")]["paired_mse"], s))
        final = max(val_steps)
        pick = {}
        for label, step in (("selected", best), ("final", final)):
            reads = {es: evals.get((step, es)) for es in SETS}
            pen = None
            if reads["test_mid"] and reads["test_in"]:
                pen = reads["test_mid"]["paired_mse"] - reads["test_in"]["paired_mse"]
            pick[label] = {"step": step, "ckpt": evals[(step, "val")]["ckpt"], "reads": reads,
                           "interp_penalty_paired_mse": pen}
        selection[row] = {"rule": "min val open_loop_sim micro paired_mse", "val_curve": {
            s: evals[(s, "val")]["paired_mse"] for s in val_steps}, **pick}
        rows_out[row] = pick

    with open(os.path.join(args.runs, "harvest.tsv"), "w") as f:
        f.write("\n".join(tsv) + "\n")
    with open(os.path.join(args.runs, "selection.json"), "w") as f:
        json.dump(selection, f, indent=1)

    fmt = lambda r, k: "–" if r is None else f"{r[k]:.5f}"  # noqa: E731
    print("| row | ckpt | step | val paired | test_in paired | test_mid paired | test_mid − test_in | test_mid mse / xyz / ypr / grip | coverage |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row, pick in rows_out.items():
        for label in ("selected", "final"):
            p = pick[label]
            r = p["reads"]
            tm = r["test_mid"]
            comps = "–" if tm is None else " / ".join(f"{tm[k]:.5f}" for k in ("mse", "xyz_mse", "ypr_mse", "grip_mse"))
            pen = "–" if p["interp_penalty_paired_mse"] is None else f"{p['interp_penalty_paired_mse']:+.5f}"
            cov = "–" if tm is None else f"{tm['coverage']:.3f}"
            print(f"| {row} | {label} | {p['step']} | {fmt(r['val'], 'paired_mse')} | {fmt(r['test_in'], 'paired_mse')} "
                  f"| {fmt(tm, 'paired_mse')} | {pen} | {comps} | {cov} |")


if __name__ == "__main__":
    main()
