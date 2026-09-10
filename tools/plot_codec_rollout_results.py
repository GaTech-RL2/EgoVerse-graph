#!/usr/bin/env python
"""Plot the 260M paper-DP three-way codec comparison.

Reads the per-episode PEAK coverage arrays that were extracted from the
`[sim] SUMMARY` lines of osmo workflows `usevalpaper4-1` and `usevaldurM-1`.

These are NON-PROTOCOL repo-local rollout numbers (see osmo/usocket_codec_rollout.yaml).
They must never be pooled with canonical results, so every figure says so.

Usage:
    python tools/plot_codec_rollout_results.py [--outdir DIR]
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

# Display name -> csv column. Order is the plot order.
PAPER = [
    ("duration ARC\n[x,y,dt_t,cos,sin,dt_r]", "arc_dur_paper"),
    ("stacked ARC\n[x,y,v,cos,sin,w]", "arc_stk_paper"),
    ("Diffusion Policy\n(no codec)", "dp_paper"),
]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

BANNER = ("NON-PROTOCOL repo-local rollout — not comparable with canonical results.  "
          "40 level-0 episodes, seeds 0-39, replan_every=8, budget=318 frames.")


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows])
            for k in rows[0] if k != "seed"}


def bootstrap_ci(x, stat, n_boot=10000, seed=0):
    """Percentile bootstrap CI. SR is a proportion, so a normal CI is wrong near 0/1."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = np.array([stat(x[i]) for i in idx])
    return np.percentile(draws, [2.5, 97.5])


def panel_metrics(ax, data):
    """Grouped bars: mean coverage, SR@0.80, SR@0.95, with bootstrap CIs."""
    metrics = [
        ("mean PEAK\ncoverage", lambda a: a.mean()),
        ("SR@0.80", lambda a: (a >= 0.80).mean()),
        ("SR@0.95", lambda a: (a >= 0.95).mean()),
    ]
    width = 0.26
    xs = np.arange(len(metrics))
    for j, (label, col) in enumerate(PAPER):
        a = data[col]
        vals, los, his = [], [], []
        for _, fn in metrics:
            v = fn(a)
            lo, hi = bootstrap_ci(a, fn)
            vals.append(v)
            los.append(max(v - lo, 0))
            his.append(max(hi - v, 0))
        pos = xs + (j - 1) * width
        ax.bar(pos, vals, width, label=label, color=COLORS[j],
               edgecolor="black", linewidth=0.6)
        ax.errorbar(pos, vals, yerr=[los, his], fmt="none",
                    ecolor="black", elinewidth=1.0, capsize=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("value")
    ax.set_title("All three representations agree within noise\n"
                 "(bars = point estimate, whiskers = 95% bootstrap CI)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)


def panel_sorted(ax, data):
    """Sorted per-episode coverage. Overlapping curves == a null result, visually."""
    for j, (label, col) in enumerate(PAPER):
        a = np.sort(data[col])[::-1]
        ax.plot(np.arange(1, len(a) + 1), a, marker="o", ms=3, lw=1.4,
                color=COLORS[j], label=label.replace("\n", " "))
    ax.axhline(0.95, ls="--", lw=1.0, color="gray")
    ax.axhline(0.80, ls=":", lw=1.0, color="gray")
    ax.text(40, 0.955, "SR@0.95", ha="right", va="bottom", fontsize=7, color="gray")
    ax.text(40, 0.805, "SR@0.80", ha="right", va="bottom", fontsize=7, color="gray")
    ax.set_xlabel("episode rank (best to worst)")
    ax.set_ylabel("PEAK coverage")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Per-episode coverage, sorted\n"
                 "curves overlap: no representation dominates", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)


def panel_outcomes(ax, data):
    """Where the episodes actually land. The 0.0 bucket is total failure."""
    buckets = [
        ("total failure\n(cov = 0)", lambda a: (a == 0).mean(), "#C44E52"),
        ("partial\n(0 < cov < 0.80)", lambda a: ((a > 0) & (a < 0.80)).mean(), "#CCB974"),
        ("near miss\n(0.80-0.95)", lambda a: ((a >= 0.80) & (a < 0.95)).mean(), "#8172B3"),
        ("success\n(cov >= 0.95)", lambda a: (a >= 0.95).mean(), "#55A868"),
    ]
    labels = [l.replace("\n", " ") for l, _ in PAPER]
    bottom = np.zeros(len(PAPER))
    for name, fn, color in buckets:
        vals = np.array([fn(data[col]) for _, col in PAPER])
        ax.barh(labels, vals, left=bottom, color=color, edgecolor="black",
                linewidth=0.6, label=name)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 0.06:
                ax.text(b + v / 2, i, f"{int(round(v * 40))}", ha="center",
                        va="center", fontsize=8)
        bottom += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of 40 episodes  (labels = episode counts)")
    ax.set_title("Outcome breakdown\nfailures dominate every arm alike", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.invert_yaxis()


def panel_paired(ax, data):
    """Paired deltas on the SAME seeds -- the test that actually has power here."""
    pairs = [("arc_dur_paper", "arc_stk_paper"), ("arc_dur_paper", "dp_paper"),
             ("arc_stk_paper", "dp_paper")]
    short = {"arc_dur_paper": "duration", "arc_stk_paper": "stacked", "dp_paper": "DP"}
    labels, deltas, errs, ps = [], [], [], []
    for a_key, b_key in pairs:
        d = data[a_key] - data[b_key]
        sem = d.std(ddof=1) / np.sqrt(len(d))
        t, p = stats.ttest_rel(data[a_key], data[b_key])
        labels.append(f"{short[a_key]} - {short[b_key]}")
        deltas.append(d.mean())
        errs.append(1.96 * sem)
        ps.append(p)

    y = np.arange(len(labels))
    ax.errorbar(deltas, y, xerr=errs, fmt="o", color="#4C72B0", capsize=4, ms=6)
    ax.axvline(0, color="black", lw=1.2)
    for i, (dv, p) in enumerate(zip(deltas, ps)):
        ax.text(dv, i + 0.16, f"$\\Delta$={dv:+.3f}, p={p:.2f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylim(-0.5, len(labels) - 0.15)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("difference in mean PEAK coverage (paired, same 40 seeds)")
    ax.set_xlim(-0.25, 0.25)
    ax.set_title("Every interval straddles zero\n"
                 "no pair is distinguishable (all p > 0.78)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(RESULTS, "figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = load(os.path.join(RESULTS, "usevalpaper4_ep_coverages.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_metrics(axes[0, 0], data)
    panel_sorted(axes[0, 1], data)
    panel_paired(axes[1, 0], data)
    panel_outcomes(axes[1, 1], data)

    fig.suptitle("U-Socket action codec at 260M (paper-DP recipe, 240k steps)\n"
                 "identical training contract; only the action representation differs",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.005, BANNER, ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0, 0.025, 1, 0.955])
    out = os.path.join(args.outdir, "codec_260m_overview.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)

    # Scale comparison: is 6x the parameters buying anything?
    durm = load(os.path.join(RESULTS, "usevaldurM_ep_coverages.csv"))
    fig2, ax = plt.subplots(figsize=(9, 5.2))
    entries = [
        ("duration ARC\n260M", data["arc_dur_paper"], "#4C72B0"),
        ("stacked ARC\n260M", data["arc_stk_paper"], "#DD8452"),
        ("DP\n260M", data["dp_paper"], "#55A868"),
        ("duration M36\n40M", durm["arc_duration_M36"], "#B0B0B0"),
        ("duration M16\n40M", durm["arc_duration_M16"], "#D0D0D0"),
    ]
    xs = np.arange(len(entries))
    means = [e[1].mean() for e in entries]
    cis = [bootstrap_ci(e[1], lambda a: a.mean()) for e in entries]
    los = [max(m - c[0], 0) for m, c in zip(means, cis)]
    his = [max(c[1] - m, 0) for m, c in zip(means, cis)]
    ax.bar(xs, means, 0.62, color=[e[2] for e in entries],
           edgecolor="black", linewidth=0.6)
    ax.errorbar(xs, means, yerr=[los, his], fmt="none", ecolor="black",
                elinewidth=1.0, capsize=4)
    for x, m in zip(xs, means):
        ax.text(x, m + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)
    # The two 40M bars above are the WORST 40M cells (duration M16/M36), which
    # would flatter the big models. The best 40M cells are drawn as reference
    # lines instead: their per-episode arrays were not retained, so they get a
    # point estimate and no interval. See HANDOVER.md section 3e.
    for val, name, style in ((0.557, "DP standard 40M", "--"),
                             (0.519, "stacked ARC M16 40M", "-.")):
        ax.axhline(val, ls=style, lw=1.3, color="#C44E52")
        # y in data coords, x in axes fraction, so the label can't run off the edge
        ax.text(0.99, val + 0.006, f"{name} = {val:.3f}", ha="right", va="bottom",
                fontsize=8, color="#C44E52", transform=ax.get_yaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels([e[0] for e in entries], fontsize=8.5)
    ax.set_ylabel("mean PEAK coverage")
    ax.set_ylim(0, 0.85)
    ax.set_title("260M paper recipe vs 40M standard recipe\n"
                 "every 260M CI contains the best 40M result: 6x the parameters\n"
                 "is not clearly buying coverage", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig2.text(0.5, 0.005, BANNER + "\nRed reference lines are point estimates only "
              "-- per-episode arrays for those runs were not retained, so no CI.",
              ha="center", fontsize=7, color="#555555")
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    out2 = os.path.join(args.outdir, "codec_260m_vs_40m.png")
    fig2.savefig(out2, dpi=160)
    print("wrote", out2)


if __name__ == "__main__":
    main()
