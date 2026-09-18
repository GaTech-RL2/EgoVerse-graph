#!/usr/bin/env python
"""Every condition per embodiment: BC, the DP baseline, and the four ARC arms.

ARC arms are shown individually with a rule marking their mean, rather than
collapsed into one bar -- the arms disagree (stk_M16 leads on the pushers,
stk_M56 on suction) and a single averaged bar hides that.

The right panel is the zero-shot pair: umi and scoop appear in no training
batch, so they have no BC baseline at all and their co-train numbers are
transfer, not fit.

Arm hues match the other figures in results/figures -- colour follows the
entity across the whole set.
"""
from __future__ import annotations

import argparse, json, os, statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARC = ["dur_D80_M56", "stk_D80_M56", "dur_D80_M16", "stk_D80_M16"]
LABEL = {"BC": "BC (single embodiment)", "DP": "DP baseline",
         "dur_D80_M56": "ARC dur M56", "stk_D80_M56": "ARC stk M56",
         "dur_D80_M16": "ARC dur M16", "stk_D80_M16": "ARC stk M16"}
ORDER = ["BC", "DP"] + ARC
TRAINED = ["u_socket", "gripper", "chain_gripper", "suction", "triangle", "flipper", "spring"]
ZEROSHOT = ["umi", "scoop"]
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#78766f", grid="#e4e3df", rule="#3d3c39",
                  col={"BC": "#008300", "DP": "#e87ba4", "dur_D80_M56": "#2a78d6",
                       "stk_D80_M56": "#eb6834", "dur_D80_M16": "#1baf7a",
                       "stk_D80_M16": "#eda100"}),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#96948c", grid="#33322f", rule="#dedcd2",
                 col={"BC": "#008300", "DP": "#d55181", "dur_D80_M56": "#3987e5",
                      "stk_D80_M56": "#d95926", "dur_D80_M16": "#199e70",
                      "stk_D80_M16": "#c98500"}),
}


def panel(ax, embs, ep, t, ymax, show_y):
    w = 0.135
    n = len(ORDER)
    for i, cond in enumerate(ORDER):
        xs, ys = [], []
        for j, e in enumerate(embs):
            k = f"{cond}|{e}"
            if k not in ep:
                continue
            xs.append(j + (i - (n - 1) / 2) * w)
            ys.append(st.mean(ep[k]))
        if not xs:
            continue
        ax.bar(xs, ys, w * 0.88, color=t["col"][cond], label=LABEL[cond],
               edgecolor=t["surface"], linewidth=1.0, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + ymax * 0.012, f"{y:.2f}".lstrip("0"), ha="center",
                    va="bottom", fontsize=5.6, color=t["secondary"], zorder=4)
    # the ARC mean, drawn as a rule across the four ARC bars it averages
    lo_i, hi_i = ORDER.index(ARC[0]), ORDER.index(ARC[-1])
    for j, e in enumerate(embs):
        vals = [st.mean(ep[f"{a}|{e}"]) for a in ARC if f"{a}|{e}" in ep]
        if not vals:
            continue
        m = st.mean(vals)
        x0 = j + (lo_i - (n - 1) / 2) * w - w * 0.5
        x1 = j + (hi_i - (n - 1) / 2) * w + w * 0.5
        ax.plot([x0, x1], [m, m], color=t["rule"], linewidth=1.6, zorder=5,
                solid_capstyle="butt")
        ax.text(x1 + 0.02, m, f"{m:.2f}".lstrip("0"), va="center", ha="left",
                fontsize=6.0, color=t["rule"], zorder=6)
    ax.set_xticks(range(len(embs)))
    ax.set_xticklabels(embs, fontsize=8.8, color=t["primary"])
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], length=0, labelsize=8)
    if show_y:
        ax.set_ylabel("peak coverage (40 episodes)", fontsize=9, color=t["secondary"])
    else:
        ax.tick_params(labelleft=False)


def figure(ep, mode, out):
    t = THEME[mode]
    tr = [e for e in TRAINED if any(f"{c}|{e}" in ep for c in ORDER)]
    zs = [e for e in ZEROSHOT if any(f"{c}|{e}" in ep for c in ORDER)]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15.2, 5.4), sharey=True,
        gridspec_kw={"width_ratios": [len(tr), max(len(zs), 1) + 0.5], "wspace": 0.045})
    fig.patch.set_facecolor(t["surface"])
    for a in (ax, ax2):
        a.set_facecolor(t["surface"])
    ymax = 0.84
    panel(ax, tr, ep, t, ymax, True)
    panel(ax2, zs, ep, t, ymax, False)
    ax.set_title("trained on (in-domain)", fontsize=9.6, color=t["muted"],
                 loc="left", pad=8)
    ax2.set_title("ZERO-SHOT — in no training batch", fontsize=9.6,
                  color=t["primary"], loc="left", pad=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=t["col"][c]) for c in ORDER]
    handles.append(Line2D([0], [0], color=t["rule"], linewidth=1.6))
    labels = [LABEL[c] for c in ORDER] + ["mean of the 4 ARC arms"]
    ax.legend(handles, labels, frameon=False, fontsize=8.4, ncol=7,
              loc="upper left", bbox_to_anchor=(0, 1.15),
              labelcolor=t["secondary"], columnspacing=1.0,
              handlelength=1.1, handletextpad=0.45)
    fig.suptitle("Every condition per embodiment — and what survives to an unseen tool",
                 fontsize=13.5, color=t["primary"], x=0.5, y=1.10)

    zline = ""
    if zs:
        bits = []
        for e in zs:
            arc = st.mean([st.mean(ep[f"{a}|{e}"]) for a in ARC if f"{a}|{e}" in ep])
            dp = st.mean(ep[f"DP|{e}"]) if f"DP|{e}" in ep else float("nan")
            best = max((st.mean(ep[f"{a}|{e}"]), LABEL[a]) for a in ARC if f"{a}|{e}" in ep)
            bits.append(f"{e}: ARC mean {arc:.3f} vs DP {dp:.3f}, best {best[1]} {best[0]:.3f}")
        zline = "Zero-shot — " + "  ".join(bits) + ".\n"
    missing = [e for e in TRAINED if f"BC|{e}" not in ep]
    mline = ("No BC bar for " + ", ".join(missing) + " (cell still running). "
             if missing else "")
    fig.text(0.5, -0.085,
             zline +
             mline +
             "umi and scoop have no BC bar by construction: a single-embodiment baseline cannot exist for a tool that is in no training batch.\n"
             "The ARC arms are shown individually because they disagree — stk_M16 leads on the pushers, stk_M56 on suction — and one averaged bar hides that.",
             ha="center", fontsize=8, color=t["muted"], linespacing=1.7)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print(f"  wrote {os.path.abspath(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="/tmp/percell/all.json")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    ep = json.load(open(a.ep))
    os.makedirs(a.outdir, exist_ok=True)
    for mode in ("light", "dark"):
        s = "" if mode == "light" else "_dark"
        figure(ep, mode, f"{a.outdir}/conditions_full{s}.png")


if __name__ == "__main__":
    main()
