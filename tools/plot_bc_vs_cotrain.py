#!/usr/bin/env python
"""BC (one embodiment) vs multi-embodiment co-train, and the codec on top of it.

Every condition is scored on the SAME 40 seeds per embodiment, so differences are
paired per episode; CIs are a 20k-sample bootstrap over the 40 paired deltas.

Only the seven TRAINED embodiments appear: umi and scoop are held out of
co-training, so no single-embodiment BC baseline exists for them.
"""
from __future__ import annotations

import argparse, json, os, random, statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARC = ["dur_D80_M56", "stk_D80_M56", "dur_D80_M16", "stk_D80_M16"]
CT = ARC + ["DP"]
EMBS = ["u_socket", "gripper", "chain_gripper", "suction", "triangle", "flipper", "spring"]
THEME = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#78766f", grid="#e4e3df", zero="#9a9891",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#96948c", grid="#33322f", zero="#6f6d66",
                 series=["#3987e5", "#d95926", "#199e70"]),
}


def boot(d, n=20000, seed=0):
    random.seed(seed); L = len(d)
    ms = sorted(st.mean([d[random.randrange(L)] for _ in range(L)]) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def have(ep, cond, e):
    return f"{cond}|{e}" in ep


def mean_of(ep, conds, e, i=None):
    vals = [ep[f"{c}|{e}"] for c in conds if have(ep, c, e)]
    if i is None:
        return st.mean([st.mean(v) for v in vals])
    return st.mean([v[i] for v in vals])


def axis_style(ax, t, ylab=None, ymax=0.83):
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], length=0, labelsize=8)
    if ylab: ax.set_ylabel(ylab, fontsize=9, color=t["secondary"])


def fig_bc_vs_ct(ep, mode, out):
    t = THEME[mode]
    embs = [e for e in EMBS if have(ep, "BC", e)]
    miss = [e for e in EMBS if not have(ep, "BC", e)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9),
                                  gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.24})
    fig.patch.set_facecolor(t["surface"])
    for a in (ax, ax2): a.set_facecolor(t["surface"])

    w = 0.34
    for i, (lab, col) in enumerate([("BC (one embodiment)", t["series"][0]),
                                    ("co-train (7 embodiments, mean of 5 arms)", t["series"][1])]):
        xs = [j + (i - 0.5) * w for j in range(len(embs))]
        ys = [st.mean(ep[f"BC|{e}"]) if i == 0 else mean_of(ep, CT, e) for e in embs]
        ax.bar(xs, ys, w * 0.9, color=col, label=lab, edgecolor=t["surface"],
               linewidth=1.1, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=7, color=t["secondary"], zorder=4)
    ax.set_xticks(range(len(embs)))
    ax.set_xticklabels(embs, fontsize=8.6, color=t["primary"], rotation=12, ha="right")
    axis_style(ax, t, "peak coverage (40 episodes)")
    ax.legend(frameon=False, fontsize=8.6, loc="upper left", labelcolor=t["secondary"],
              bbox_to_anchor=(0, 1.16), ncol=2, handlelength=1.2, handletextpad=0.5)

    ax2.axvline(0, color=t["zero"], linewidth=1.2, zorder=2)
    ys = list(range(len(embs)))[::-1]
    for y, e in zip(ys, embs):
        d = [ep[f"BC|{e}"][i] - mean_of(ep, CT, e, i) for i in range(40)]
        lo, hi = boot(d); m = st.mean(d); sig = lo > 0 or hi < 0
        ax2.plot([lo, hi], [y, y], color=t["series"][0], linewidth=2.0, zorder=3,
                 solid_capstyle="round")
        ax2.scatter([m], [y], s=85, zorder=4, linewidth=2.0,
                    color=t["series"][0] if sig else t["surface"], edgecolor=t["series"][0])
        # Park the value in the empty right half: at this x-range the deltas
        # cluster near zero and a label beside the marker lands on top of it.
        ax2.text(0.80, y, f"{m:+.3f}", va="center", ha="left", fontsize=7.8,
                 color=t["secondary"] if sig else t["muted"],
                 transform=ax2.get_yaxis_transform())
    ax2.set_yticks(ys); ax2.set_yticklabels(embs, fontsize=8.6, color=t["primary"])
    ax2.set_xlabel("BC − co-train  (paired on the same 40 seeds)", fontsize=9,
                   color=t["secondary"])
    ax2.set_xlim(-0.45, 0.42)
    ax2.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0); ax2.set_axisbelow(True)
    for s in ("top", "right", "left"): ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(t["grid"])
    ax2.tick_params(colors=t["secondary"], length=0, labelsize=8)
    ax2.set_title("filled = 95% CI excludes zero", fontsize=8.4, color=t["muted"],
                  loc="left", pad=8)

    fig.suptitle("Dropping the other six embodiments costs control — except on triangle",
                 fontsize=13, color=t["primary"], x=0.5, y=1.07)
    note = ("Same tokenizer, model, optimiser and 240k steps in both conditions; CombinedLoader draws one batch per domain per step, so each embodiment\n"
            "saw the same 7.68M of its own samples either way. The only difference is whether the other six tools were in the mixture.")
    if miss:
        note += "\n" + ", ".join(miss) + ": BC cell still running. umi and scoop have no BC baseline -- they are held out of training entirely."
    fig.text(0.5, -0.10, note, ha="center", fontsize=8, color=t["muted"], linespacing=1.7)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig); print(f"  wrote {out}")


def fig_three(ep, mode, out):
    t = THEME[mode]
    embs = [e for e in EMBS if have(ep, "BC", e)]
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    conds = [("BC (one embodiment)", lambda e: st.mean(ep[f"BC|{e}"]), t["series"][0]),
             ("co-train, DP baseline", lambda e: st.mean(ep[f"DP|{e}"]), t["series"][1]),
             ("co-train, ARC (mean of 4 arms)", lambda e: mean_of(ep, ARC, e), t["series"][2])]
    w = 0.26
    for i, (lab, fn, col) in enumerate(conds):
        xs = [j + (i - 1) * w for j in range(len(embs))]
        ys = [fn(e) for e in embs]
        ax.bar(xs, ys, w * 0.9, color=col, label=lab, edgecolor=t["surface"],
               linewidth=1.1, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=6.8, color=t["secondary"], zorder=4)
    ax.set_xticks(range(len(embs)))
    ax.set_xticklabels(embs, fontsize=9, color=t["primary"])
    axis_style(ax, t, "peak coverage (40 episodes)")
    ax.legend(frameon=False, fontsize=8.8, loc="upper left", labelcolor=t["secondary"],
              bbox_to_anchor=(0, 1.14), ncol=3, handlelength=1.2, handletextpad=0.5)
    ax.set_title("Two separate gains: the multi-embodiment mixture, then the ARC codec on top",
                 fontsize=12.5, color=t["primary"], pad=34)

    dd = [mean_of(ep, ARC, e, i) - ep[f"DP|{e}"][i] for e in embs for i in range(40)]
    db = [ep[f"DP|{e}"][i] - ep[f"BC|{e}"][i] for e in embs for i in range(40)]
    l1, h1 = boot(db); l2, h2 = boot(dd)
    fig.text(0.5, -0.07,
             f"Pooled over these {len(embs)} embodiments: co-train DP − BC = {st.mean(db):+.3f} [{l1:+.3f}, {h1:+.3f}]"
             f"  ARC − co-train DP = {st.mean(dd):+.3f} [{l2:+.3f}, {h2:+.3f}]\n"
             "Pooling episodes across embodiments treats them as independent, so these are indicative; the per-embodiment CIs in the other figure are the conservative read.",
             ha="center", fontsize=8, color=t["muted"], linespacing=1.7)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig); print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", default="/tmp/percell/all.json")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    ep = json.load(open(a.ep)); os.makedirs(a.outdir, exist_ok=True)
    for mode in ("light", "dark"):
        s = "" if mode == "light" else "_dark"
        fig_bc_vs_ct(ep, mode, f"{a.outdir}/bc_vs_cotrain{s}.png")
        fig_three(ep, mode, f"{a.outdir}/bc_vs_dp_vs_arc{s}.png")


if __name__ == "__main__":
    main()
