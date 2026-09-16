#!/usr/bin/env python
"""Group the articulated rollout table by CONTACT MECHANISM and chart each group.

The mechanism taxonomy is taken from the simulator's own agent docstrings in
Tsimulation/sim_v2/pushshapes/agents.py, not inferred from the embodiment names:

  USocketAgent     "U-shaped socket that can LATCH onto the object and rotate it"
  GripperAgent     "Parallel-jaw gripper: 4-DOF (x, y, angle, jaw)"
  ChainGripperAgent"One rigid four-link chain controlled by one shared hinge angle"
  UmiAgent         "UMI-style rotary gripper: 4-DOF (x, y, wrist, grip)"
  SuctionAgent     "Suction pad: 3-DOF (x, y, engage)"
  TriangleAgent    "Equilateral triangle pusher: 3-DOF (x, y, angle)"
  FlipperAgent     "Hinged flipper: 4-DOF (x, y, angle, grip)"
  SpringAgent      "Spring plunger ... contact is mediated by a spring"
  ScoopAgent       "Concave scoop: 3-DOF (x, y, angle)"

Colours are the validated categorical slots 1-5; bars carry visible value labels
because three light-mode slots sit under 3:1 on the light surface (relief rule).
"""
from __future__ import annotations

import argparse
import csv
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# contact mechanism -> embodiments, ordered within group
GROUPS = [
    ("Prehensile (form closure: latch or grasp)",
     ["u_socket", "gripper", "chain_gripper", "umi"]),
    ("Adhesion", ["suction"]),
    ("Non-prehensile (pushing)",
     ["triangle", "flipper", "spring", "scoop"]),
]
HELD_OUT = {"umi", "scoop"}
ARMS = ["dur_M56", "stk_M56", "dur_M16", "stk_M16", "DP"]

THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#78766f", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#96948c", grid="#33322f",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}


def load(path):
    rows = collections.defaultdict(dict)
    budget, sr = {}, collections.defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows[r["embodiment"]][r["arm"]] = float(r["peak_coverage_mean"])
            sr[r["embodiment"]][r["arm"]] = float(r["SR@0.80"])
            budget[r["embodiment"]] = int(r["budget"])
    return rows, sr, budget


def panel(ax, embs, rows, t, ylabel, ymax):
    n = len(ARMS)
    width = 0.155
    for i, arm in enumerate(ARMS):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(embs))]
        ys = [rows[e].get(arm, 0.0) for e in embs]
        ax.bar(xs, ys, width * 0.88, label=arm, color=t["series"][i],
               edgecolor=t["surface"], linewidth=1.1, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + ymax * 0.018, f"{y:.2f}".lstrip("0"), ha="center",
                    va="bottom", fontsize=6.1, color=t["secondary"], zorder=4)
    ax.set_xticks(range(len(embs)))
    ax.set_xticklabels(
        [e + ("\n(held out)" if e in HELD_OUT else "") for e in embs],
        fontsize=8.5, color=t["primary"])
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], length=0, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=t["secondary"])


def figure(rows, sr, budget, mode, out):
    t = THEME[mode]
    widths = [len(g[1]) for g in GROUPS]
    fig, axes = plt.subplots(
        1, len(GROUPS), figsize=(13.4, 4.9), sharey=True,
        gridspec_kw={"width_ratios": widths, "wspace": 0.07})
    fig.patch.set_facecolor(t["surface"])
    ymax = 0.85
    for ax, (title, embs) in zip(axes, GROUPS):
        ax.set_facecolor(t["surface"])
        panel(ax, embs, rows, t, "peak coverage (mean of 40 episodes)"
              if ax is axes[0] else None, ymax)
        med = sum(budget[e] for e in embs) / len(embs)
        ax.set_title(f"{title}\nmean frame budget {med:,.0f}",
                     fontsize=9.5, color=t["primary"], pad=10, linespacing=1.5)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=5, loc="upper left",
                   bbox_to_anchor=(0.0, 1.30), labelcolor=t["secondary"],
                   columnspacing=1.1, handlelength=1.2, handletextpad=0.5)
    fig.suptitle("Articulated co-train: rollout coverage by contact mechanism",
                 fontsize=13, color=t["primary"], x=0.5, y=1.045, weight="medium")
    fig.text(0.5, -0.055,
             "Mechanism and episode length are confounded: the prehensile group's frame budget is ~7x shorter than the pushing group's, "
             "and peak coverage is a maximum over time.\nThe groups therefore cannot separate 'grasping is hard' from 'these episodes are short'. "
             "Held-out tools appear in no training batch.",
             ha="center", fontsize=8, color=t["muted"], linespacing=1.6)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out}")


def confound(rows, budget, mode, out):
    """Budget vs coverage.

    Colour is NOT used for group here: the groups already separate perfectly on
    the x-axis, so a second colour meaning would collide with the arm hues used
    in the other figure (colour must follow one entity). Group is carried by the
    x-position and the point labels; marker fill carries held-out status, which
    is the thing that actually breaks the trend.
    """
    import statistics as st

    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    embs = sorted(rows, key=lambda e: budget[e])
    ys = {e: st.mean(rows[e].values()) for e in embs}
    ind = [e for e in embs if e not in HELD_OUT]

    mx = st.mean([budget[e] for e in ind]); my = st.mean([ys[e] for e in ind])
    num = sum((budget[e] - mx) * (ys[e] - my) for e in ind)
    den = (sum((budget[e] - mx) ** 2 for e in ind)
           * sum((ys[e] - my) ** 2 for e in ind)) ** 0.5
    r = num / den
    b = num / sum((budget[e] - mx) ** 2 for e in ind)
    xs_line = [min(budget[e] for e in ind), max(budget[e] for e in ind)]
    ax.plot(xs_line, [my + b * (x - mx) for x in xs_line], color=t["muted"],
            linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)

    col = t["series"][0]
    ax.scatter([budget[e] for e in ind], [ys[e] for e in ind], s=120, color=col,
               edgecolor=t["surface"], linewidth=1.6, zorder=3, label="in-domain")
    hel = [e for e in embs if e in HELD_OUT]
    ax.scatter([budget[e] for e in hel], [ys[e] for e in hel], s=130,
               facecolor=t["surface"], edgecolor=col, linewidth=2.2, zorder=4,
               label="held out (never trained on)")

    # The four prehensile tools span 626-752 frames, which is ~6 px on a 0-6100
    # axis: no text offset can separate them, and that near-coincidence is the
    # finding. Label them once, as a cluster, with a leader.
    CLUSTER = ["u_socket", "gripper", "chain_gripper", "umi"]
    offs = {"suction": (0, 15), "triangle": (0, 15),
            "flipper": (0, -21), "spring": (0, 15), "scoop": (0, 15)}
    for e in embs:
        if e in CLUSTER:
            continue
        ax.annotate(e, (budget[e], ys[e]), textcoords="offset points",
                    xytext=offs.get(e, (0, 13)), ha="center", fontsize=7.8,
                    color=t["secondary"], zorder=5)
    cx = sum(budget[e] for e in CLUSTER) / len(CLUSTER)
    cy = sum(ys[e] for e in CLUSTER) / len(CLUSTER)
    ax.annotate(
        "u_socket, gripper,\nchain_gripper, umi\nall within 626\u2013752 frames",
        xy=(cx + 130, cy), xytext=(1430, 0.115), fontsize=7.8,
        color=t["secondary"], ha="left", va="center", zorder=5, linespacing=1.5,
        arrowprops=dict(arrowstyle="-", color=t["grid"], linewidth=1.1,
                        shrinkA=0, shrinkB=6))

    ax.set_xlabel("p99 frame budget \u2014 episode length allowed at eval",
                  fontsize=9, color=t["secondary"])
    ax.set_ylabel("peak coverage (mean over the 5 arms)", fontsize=9,
                  color=t["secondary"])
    ax.set_xlim(0, 6100); ax.set_ylim(-0.04, 0.78)
    ax.grid(color=t["grid"], linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], length=0, labelsize=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=t["secondary"],
              loc="upper left", handletextpad=0.6)
    ax.set_title(f"In-domain coverage is almost entirely explained by episode length  "
                 f"(r = {r:.2f})",
                 fontsize=12, color=t["primary"], pad=12)
    fig.text(0.5, -0.14,
             "The seven trained tools sit on a line: longer demonstrations \u2192 longer eval budget \u2192 higher peak coverage, which is a maximum over time.\n"
             "So 'prehensile tools are hard' cannot be separated from 'their episodes are 7x shorter'. The two held-out tools are the exceptions that\n"
             "break the fit \u2014 scoop gets the longest budget of all and still scores 0.04 \u2014 which is a generalization failure, not a budget effect.",
             ha="center", fontsize=8, color=t["muted"], linespacing=1.7)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/artic_final.csv")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    rows, sr, budget = load(a.csv)
    os.makedirs(a.outdir, exist_ok=True)
    for mode in ("light", "dark"):
        sfx = "" if mode == "light" else "_dark"
        figure(rows, sr, budget, mode, f"{a.outdir}/artic_by_mechanism{sfx}.png")
        confound(rows, budget, mode, f"{a.outdir}/artic_budget_confound{sfx}.png")
    # the table view the relief rule asks for
    print(f"\n{'group':16s}{'embodiment':16s}{'budget':>8s}" + "".join(f"{x:>10s}" for x in ARMS))
    for title, embs in GROUPS:
        for e in embs:
            print(f"{title.split(chr(0x2003))[0]:16.16s}{e:16s}{budget[e]:>8,d}"
                  + "".join(f"{rows[e].get(x, float('nan')):>10.3f}" for x in ARMS))


if __name__ == "__main__":
    main()
