"""UNITE vs DP under the sim_v2 eval protocol (rev-3 budgets): closed-loop curves by checkpoint.

Input: peaks_export.json  {"row|step|emb|level|variant": {seed: peak}}.
Level-0 panels pool seeds 0-79 (error bars = standard error). OEC-56 panels pool the
150 rollouts (30 levels x 5 seeds) of a checkpoint. UNITE rows at the declared CFG 1.0.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open(sys.argv[1]))
out = sys.argv[2]
stamp = sys.argv[3] if len(sys.argv) > 3 else ""

# row -> (label, color, linestyle, linewidth, family)
ROWS = {
    "dpus":     ("DP BC (U-Socket only)",              "#6b6b6b", "-",  1.6),
    "dpch":     ("DP BC (chain only)",                 "#6b6b6b", "-",  1.6),
    "dpct":     ("DP cotrain",                          "#111111", "-",  2.2),
    "uniteus":  ("UNITE BC (U-Socket only)",           "#7fbfc8", "-",  1.6),
    "unitech":  ("UNITE BC (chain only)",              "#7fbfc8", "-",  1.6),
    "ctA":      ("UNITE cotrain A (h384)",             "#0e7a86", "-",  2.6),
    "ctAc":     ("UNITE cotrain A, compiled replica",  "#0e7a86", ":",  1.2),
    "ctB":      ("UNITE cotrain B (split tokenizer)",  "#0e7a86", "--", 1.6),
    "ctA768":   ("UNITE cotrain A, full width (h768)", "#1b4f8a", "-",  1.8),
    "ctAema":   ("UNITE cotrain A, EMA 0.9999",        "#1b4f8a", "--", 1.2),
    "ctAann":   ("UNITE cotrain A, late anneal",       "#1b4f8a", ":",  1.2),
    "s3dpct":   ("DP cotrain + chain obstacle scenes",     "#b8681c", "-",  2.2),
    "s3dpch":   ("DP BC chain + obstacle scenes",          "#d9a066", "-",  1.6),
    "s3ctA":    ("UNITE cotrain A + chain obstacle scenes","#c0392b", "-",  2.6),
    "s3ctA768": ("UNITE cotrain A h768 + obstacle scenes", "#c0392b", "--", 1.6),
    "s3unitech":("UNITE BC chain + obstacle scenes",       "#e8998d", "-",  1.6),
}
STEPS = [30000 * i for i in range(1, 9)]


def series(row, emb, level0=True):
    xs, ys, es = [], [], []
    for step in STEPS:
        if level0:
            key = f"{row}|{step}|{emb}|0|" + ("cfgemb" if row.startswith(("dp", "s3dp")) else "cfg1")
            seeds = data.get(key)
            if not seeds or len(seeds) < 40:
                continue
            vals = np.array(list(seeds.values()))
        else:
            vals = []
            var = "cfgemb" if row.startswith(("dp", "s3dp")) else "cfg1"
            for lvl in range(1, 31):
                seeds = data.get(f"{row}|{step}|{emb}|{lvl}|{var}")
                if seeds:
                    vals += list(seeds.values())
            if len(vals) < 100:
                continue
            vals = np.array(vals)
        xs.append(step / 1000); ys.append(vals.mean()); es.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    return xs, ys, es


SIMPLE = {  # panel -> [(row, label, color, lw)]
    ("usocket", True):  [("dpus", "DP BC", "#8a8a8a", 2.0), ("dpct", "DP cotrain", "#111111", 2.4), ("uniteus", "UNITE BC", "#7fbfc8", 2.0), ("ctA", "UNITE cotrain", "#0e7a86", 2.8)],
    ("chain",   True):  [("dpch", "DP BC", "#8a8a8a", 2.0), ("dpct", "DP cotrain", "#111111", 2.4), ("unitech", "UNITE BC", "#7fbfc8", 2.0), ("ctA", "UNITE cotrain", "#0e7a86", 2.8)],
    ("usocket", False): [("dpus", "DP BC", "#8a8a8a", 2.0), ("s3dpct", "DP cotrain (+ chain obstacle data)", "#111111", 2.4), ("uniteus", "UNITE BC", "#7fbfc8", 2.0), ("s3ctA", "UNITE cotrain (+ chain obstacle data)", "#0e7a86", 2.8)],
    ("chain",   False): [("s3dpch", "DP BC (+ obstacle data)", "#8a8a8a", 2.0), ("s3dpct", "DP cotrain (+ chain obstacle data)", "#111111", 2.4), ("s3unitech", "UNITE BC (+ obstacle data)", "#7fbfc8", 2.0), ("s3ctA", "UNITE cotrain (+ chain obstacle data)", "#0e7a86", 2.8)],
}
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
panels = [
    (axes[0, 0], "usocket", True,  "U-Socket · in-domain (level 0, seeds 0–79, budget 318 steps)"),
    (axes[0, 1], "chain",   True,  "Chain gripper · in-domain (level 0, seeds 0–79, budget 688 steps)"),
    (axes[1, 0], "usocket", False, "U-Socket · scene generalisation (OEC-56: 30 obstacle levels × 5 seeds)"),
    (axes[1, 1], "chain",   False, "Chain gripper · obstacle levels (OEC-56, trained on them)"),
]
for ax, emb, level0, title in panels:
    for r, label, color, lw in SIMPLE[(emb, level0)]:
        xs, ys, es = series(r, emb, level0)
        if not xs:
            ax.plot([], [], color=color, lw=lw, label=label + " (not scored yet)")
            continue
        ax.errorbar(xs, ys, yerr=es, color=color, lw=lw, marker="o", ms=4, capsize=2.5, elinewidth=0.9, label=label)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_ylim(0, 0.85 if level0 else 0.6)
    ax.set_xticks([x / 1000 for x in STEPS]); ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, loc="lower right" if level0 else "upper left", frameon=False)
for ax in axes[1]:
    ax.set_xlabel("optimizer steps (thousands); global batch 64")
for ax in axes[:, 0]:
    ax.set_ylabel("mean peak coverage (IoU)")
fig.suptitle("UNITE vs Paper-DP under the sim_v2 eval protocol (horizon rev 3, full horizon, EMA, replan 8; UNITE at CFG 1.0)" + (f"  ·  {stamp}" if stamp else ""), fontsize=11.5)
fig.text(0.01, 0.005, "Error bars: standard error over episodes. UNITE cotrain = topology A, hidden 384 (the best UNITE variant so far). Bottom row: rows trained with the chain gripper's 1,919 obstacle episodes.\n"
         "Each point is one protocol reading of 80 episodes (one policy sample per replan); replicate-mean comparisons of the selected checkpoints are in the companion figure.", fontsize=7.5, color="#555")
fig.tight_layout(rect=(0, 0.035, 1, 0.97))
fig.savefig(out, dpi=170)
print("wrote", out)
