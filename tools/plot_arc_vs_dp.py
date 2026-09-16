#!/usr/bin/env python
"""ARC (4 arms pooled) vs the DP baseline, paired by seed, plus the pusher group alone.

Every cell has all 40 seeds, and all arms are scored on the SAME seeds, so ARC-DP
is a paired difference per episode rather than a difference of means. CIs are
20k-sample bootstrap over the 40 paired differences.

Arm hues are the same slots as plot_artic_by_mechanism.py -- colour follows the
entity across figures.
"""
from __future__ import annotations

import argparse, json, os, random, statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARC = ["arc_dur_D80_M56", "arc_stk_D80_M56", "arc_dur_D80_M16", "arc_stk_D80_M16"]
LABEL = {"arc_dur_D80_M56": "dur_M56", "arc_stk_D80_M56": "stk_M56",
         "arc_dur_D80_M16": "dur_M16", "arc_stk_D80_M16": "stk_M16",
         "dp_paper": "DP"}
ARMS = ARC + ["dp_paper"]
GROUPS = [("Prehensile", ["u_socket", "gripper", "chain_gripper", "umi"]),
          ("Adhesion", ["suction"]),
          ("Pusher", ["triangle", "flipper", "spring", "scoop"])]
PUSHERS = ["triangle", "flipper", "spring", "scoop"]
HELD_OUT = {"umi", "scoop"}
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#78766f", grid="#e4e3df", zero="#9a9891",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#96948c", grid="#33322f", zero="#6f6d66",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}


def boot(d, n=20000, seed=0):
    random.seed(seed)
    L = len(d)
    ms = sorted(st.mean([d[random.randrange(L)] for _ in range(L)]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def arc_dp(ep, e):
    dp = ep[f"dp_paper|{e}"]
    arc = [st.mean([ep[f"{a}|{e}"][i] for a in ARC]) for i in range(40)]
    d = [a - b for a, b in zip(arc, dp)]
    lo, hi = boot(d)
    return st.mean(arc), st.mean(dp), st.mean(d), lo, hi


def fig_delta(ep, mode, out):
    t = THEME[mode]
    order, ylab = [], []
    for g, embs in GROUPS:
        for e in embs:
            order.append(e); ylab.append((g, e))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    ax.axvline(0, color=t["zero"], linewidth=1.2, zorder=2)
    col = t["series"][0]
    ys = list(range(len(order)))[::-1]
    for y, e in zip(ys, order):
        _, _, d, lo, hi = arc_dp(ep, e)
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=col, linewidth=2.0, zorder=3,
                solid_capstyle="round")
        ax.scatter([d], [y], s=95, zorder=4, linewidth=2.0,
                   color=col if sig else t["surface"], edgecolor=col)
        ax.text(hi + 0.012, y, f"{d:+.3f}", va="center", fontsize=8,
                color=t["secondary"] if sig else t["muted"])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{e}" + ("  (held out)" if e in HELD_OUT else "")
                        for e in order], fontsize=9, color=t["primary"])
    for i, (g, e) in enumerate(ylab):
        if i == 0 or ylab[i - 1][0] != g:
            ax.text(-0.075, ys[i] + 0.46, g.upper(), fontsize=7.6,
                    color=t["muted"], transform=ax.get_yaxis_transform(),
                    ha="left", va="center")
    ax.set_xlabel("ARC − DP peak coverage  (paired on the same 40 seeds)",
                  fontsize=9, color=t["secondary"])
    ax.set_xlim(-0.085, 0.22)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], length=0, labelsize=8)
    ax.set_title("ARC beats DP — but only where the policy works at all",
                 fontsize=12.5, color=t["primary"], pad=26)
    ax.text(0.0, 1.045,
            "filled = 95% bootstrap CI excludes zero  hollow = not distinguishable from DP",
            transform=ax.transAxes, fontsize=8, color=t["muted"])
    fig.text(0.5, -0.045,
             "ARC is the mean of its four arms per episode. Every significant gain is in the pusher group; no prehensile tool separates from DP,\n"
             "because none of them get far enough off the ground for the representation to matter.",
             ha="center", fontsize=8, color=t["muted"], linespacing=1.7)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig); print(f"  wrote {out}")


def fig_pusher(ep, mode, out):
    t = THEME[mode]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [2.45, 1], "wspace": 0.22})
    for a in (ax, ax2): a.set_facecolor(t["surface"])
    fig.patch.set_facecolor(t["surface"])
    width = 0.155
    for i, arm in enumerate(ARMS):
        xs = [j + (i - 2) * width for j in range(len(PUSHERS))]
        ysv = [st.mean(ep[f"{arm}|{e}"]) for e in PUSHERS]
        ax.bar(xs, ysv, width * 0.88, color=t["series"][i], label=LABEL[arm],
               edgecolor=t["surface"], linewidth=1.1, zorder=3)
        for x, y in zip(xs, ysv):
            ax.text(x, y + 0.012, f"{y:.2f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=6.6, color=t["secondary"], zorder=4)
    ax.set_xticks(range(len(PUSHERS)))
    ax.set_xticklabels([e + ("\n(held out)" if e in HELD_OUT else "") for e in PUSHERS],
                       fontsize=9, color=t["primary"])
    ax.set_ylabel("peak coverage (40 episodes)", fontsize=9, color=t["secondary"])
    ax.set_ylim(0, 0.83)
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="upper left",
              bbox_to_anchor=(0, 1.13), labelcolor=t["secondary"],
              columnspacing=1.0, handlelength=1.2, handletextpad=0.5)
    ax.set_title("Per arm", fontsize=10, color=t["primary"], pad=30, loc="left")

    d_all = []
    for e in ["triangle", "flipper", "spring"]:
        dp = ep[f"dp_paper|{e}"]
        arc = [st.mean([ep[f"{a}|{e}"][i] for a in ARC]) for i in range(40)]
        d_all += [a - b for a, b in zip(arc, dp)]
    lo, hi = boot(d_all)
    arcm = st.mean([st.mean([ep[f"{a}|{e}"][i] for a in ARC])
                    for e in ["triangle", "flipper", "spring"] for i in range(40)])
    dpm = st.mean([v for e in ["triangle", "flipper", "spring"] for v in ep[f"dp_paper|{e}"]])
    # ARC here is an AGGREGATE of the four arms, not one of them, so it must not
    # borrow dur_M56's blue -- a hue means one entity across the whole figure.
    # DP keeps its slot-5 hue because it IS the same entity in both panels.
    ax2.bar([0, 1], [arcm, dpm], 0.52, color=[t["secondary"], t["series"][4]],
            edgecolor=t["surface"], linewidth=1.1, zorder=3)
    for x, v in zip([0, 1], [arcm, dpm]):
        ax2.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                 color=t["secondary"], zorder=4)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["ARC\n(4 arms)", "DP"], fontsize=9,
                                                color=t["primary"])
    ax2.set_ylim(0, 0.83); ax2.set_ylabel(None)
    ax2.set_title("In-domain pushers pooled", fontsize=10, color=t["primary"],
                  pad=30, loc="left")
    ax2.text(0.5, 0.80, f"ARC − DP = {st.mean(d_all):+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]",
             ha="center", fontsize=8.6, color=t["secondary"], transform=ax2.transData)
    for a in (ax, ax2):
        a.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0); a.set_axisbelow(True)
        for s in ("top", "right", "left"): a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(t["grid"])
        a.tick_params(colors=t["secondary"], length=0, labelsize=8)
    fig.suptitle("The pusher group in isolation — where every arm actually does the task",
                 fontsize=13, color=t["primary"], x=0.5, y=1.10)
    fig.text(0.5, -0.06,
             "scoop is a pusher too but was held out of training, so its 0.03-0.05 is the generalization result, not the pushing result.\n"
             "Pooling episodes across the three trained pushers treats them as independent; the per-embodiment CIs in the other figure are the conservative read.",
             ha="center", fontsize=8, color=t["muted"], linespacing=1.7)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig); print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="/tmp/percell/ep.json")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    ep = json.load(open(a.ep))
    os.makedirs(a.outdir, exist_ok=True)
    for mode in ("light", "dark"):
        sfx = "" if mode == "light" else "_dark"
        fig_delta(ep, mode, f"{a.outdir}/arc_vs_dp_delta{sfx}.png")
        fig_pusher(ep, mode, f"{a.outdir}/arc_vs_dp_pushers{sfx}.png")


if __name__ == "__main__":
    main()
