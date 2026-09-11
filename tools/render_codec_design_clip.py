#!/usr/bin/env python
"""A purely diagrammatic clip: what the two ARC variants actually STORE.

No dataset numbers. Hand-chosen values, six waypoints, one idea per act.

The load-bearing fact the diagram is built around: waypoints are equally spaced
in ARC LENGTH, so every spatial gap is the same size. Only the TIME to cross
each gap varies. That makes velocity and duration exact inverses of each other,
which is the whole design question.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle

BG, PANEL, INK, DIM, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
TR, ROT, DUR, VEL = "#58a6ff", "#f0883e", "#3fb950", "#bc8cff"
RED = "#f85149"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "font.size": 15,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
})

# Six waypoints on a gentle curve, EQUALLY spaced along it.
N = 6
_t = np.linspace(0, 1, 400)
_P = np.stack([_t * 10, 1.6 * np.sin(_t * 2.6) + 0.5 * _t], 1)
_s = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(_P, axis=0), axis=1))))
WP = np.stack([np.interp(np.linspace(0, _s[-1], N), _s, _P[:, k]) for k in (0, 1)], 1)

# Hand-chosen: the tool lingers over gaps 2 and 3, then hurries.
DT = np.array([0.05, 0.09, 0.34, 0.34, 0.07])
DS = float(np.linalg.norm(WP[1] - WP[0]))      # identical for every gap
V = DS / DT


def fig_new():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    t = fig.text(.05, .945, "", fontsize=30, fontweight="bold", va="top")
    u = fig.text(.05, .878, "", fontsize=17, color="#c9d1d9", va="top")
    return fig, t, u


def clear(fig, keep):
    for a in list(fig.axes):
        a.remove()
    for t in list(fig.texts):
        if t not in keep:
            t.remove()


def hold(w, n):
    for _ in range(n):
        w.grab_frame()


def cell(ax, x, y, w, h, label, face, edge, fs=17, bold=True, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=face, edgecolor=edge, lw=2.4))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.045, sub, ha="center", va="center",
                fontsize=12.5, color=DIM)


def path_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.0, 11.0); ax.set_ylim(-1.6, 2.9)
    ax.axis("off"); ax.set_aspect("equal")
    return ax


def draw_path(ax, upto=N, dots=True):
    ax.plot(_P[:, 0], _P[:, 1], color="#2d3748", lw=4, solid_capstyle="round")
    if dots:
        ax.scatter(WP[:upto, 0], WP[:upto, 1], s=190, color=TR, zorder=5,
                   edgecolors=BG, linewidths=2.5)
        for i in range(upto):
            ax.text(WP[i, 0], WP[i, 1] + .46, str(i), ha="center", fontsize=14,
                    color=TR, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "videos", "04_design.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fps = a.fps
    fig, T, U = fig_new()
    keep = (T, U)
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])

    with w.saving(fig, a.out, dpi=120):
        # ---- act 1: the shapes, as plain boxes --------------------------
        T.set_text("What the codec stores")
        U.set_text("a variable-length chunk becomes a fixed-size token, and decodes back")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        boxes = [(.07, "raw action chunk", "80 x 3", "x, y, theta  per frame", DIM),
                 (.40, "ARC token", "56 x 6", "fixed M rows, arc-spaced", TR),
                 (.73, "executed chunk", "16 x 3", "what the robot runs", DIM)]
        for x, name, shape, note, col in boxes:
            cell(ax, x, .40, .20, .22, shape, PANEL, col, fs=27)
            ax.text(x + .10, .655, name, ha="center", fontsize=17, color=col,
                    fontweight="bold")
            ax.text(x + .10, .355, note, ha="center", fontsize=13.5, color=DIM,
                    va="top")
        for x, lab in ((.315, "tokenize"), (.645, "detokenize")):
            ax.annotate("", xy=(x + .06, .51), xytext=(x, .51),
                        arrowprops=dict(arrowstyle="-|>", color=DIM, lw=3,
                                        mutation_scale=26))
            ax.text(x + .03, .555, lab, ha="center", fontsize=14, color=DIM)
        hold(w, fps * 5)
        clear(fig, keep)

        # ---- act 2: only two slots differ -------------------------------
        T.set_text("One token row, six slots")
        U.set_text("four slots are identical in both variants — the question is what fills "
                   "the other two")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        names = ["x", "y", "?", "cos", "sin", "?"]
        subs = ["position", "position", "TIMING", "heading", "heading", "TIMING"]
        for k, (nm, sb) in enumerate(zip(names, subs)):
            x = .085 + k * .142
            same = nm != "?"
            cell(ax, x, .42, .118, .20, nm, PANEL if same else "#1c1420",
                 GRID if same else RED, fs=26, sub=sb)
        ax.text(.5, .30, "geometry — the same arc-length resampling either way",
                ha="center", fontsize=16, color=DIM)
        ax.annotate("", xy=(.30, .40), xytext=(.30, .33),
                    arrowprops=dict(arrowstyle="-", color=DIM, lw=1.6))
        ax.text(.5, .215, "the two red slots are the entire difference between the "
                "variants", ha="center", fontsize=17, color=RED, fontweight="bold")
        hold(w, fps * 5)
        clear(fig, keep)

        # ---- act 3: equal in space, unequal in time ---------------------
        T.set_text("Every gap is the same size in space")
        U.set_text("waypoints are equally spaced along the path — only the time to cross "
                   "each gap varies")
        axp = path_axes(fig, [.06, .40, .88, .40])
        draw_path(axp)
        for i in range(N - 1):
            mid = (WP[i] + WP[i + 1]) / 2
            axp.annotate("", xy=WP[i + 1], xytext=WP[i],
                         arrowprops=dict(arrowstyle="<->", color=TR, lw=2.2,
                                         shrinkA=13, shrinkB=13))
            axp.text(mid[0], mid[1] - .75, r"$\Delta s$", ha="center", fontsize=16,
                     color=TR, fontweight="bold")
        fig.text(.5, .30, r"all five $\Delta s$ are equal  —  that is what arc-length "
                 "resampling guarantees", ha="center", fontsize=18, color=TR)
        fig.text(.5, .21, "so whatever goes in the timing slot only has to say "
                 "HOW LONG the tool took", ha="center", fontsize=17, color=INK)
        hold(w, fps * 5)
        clear(fig, keep)

        # ---- act 4: the two answers, side by side -----------------------
        T.set_text("Two ways to say the same thing")
        U.set_text("the tool lingers over the middle gaps, then hurries — read the bars")
        axp = path_axes(fig, [.06, .55, .88, .33])
        draw_path(axp)
        axv = fig.add_axes([.06, .30, .88, .18]); axd = fig.add_axes([.06, .06, .88, .18])
        # Put each bar under the GAP it describes, sharing the path's x-axis, so
        # the correspondence is visible instead of inferred.
        mids = (WP[:-1, 0] + WP[1:, 0]) / 2
        width = float(np.min(np.diff(mids))) * .72
        for ax_, vals, col, lab, unit in (
                (axv, V, VEL, "velocity token stores  SPEED", "px/s"),
                (axd, DT, DUR, "duration token stores  TIME", "s")):
            ax_.set_facecolor(PANEL)
            ax_.set_xlim(-1.0, 11.0); ax_.set_ylim(0, max(vals) * 1.5)
            ax_.set_xticks([]); ax_.set_yticks([])
            for sp in ax_.spines.values():
                sp.set_color(GRID)
            ax_.bar(mids, vals, width, color=col, alpha=.92)
            ax_.text(.008, .86, lab, transform=ax_.transAxes, fontsize=17,
                     color=col, fontweight="bold")
            for k, v in enumerate(vals):
                ax_.text(mids[k], v * 1.06, f"{v:.2f} {unit}", ha="center",
                         fontsize=13, color=INK)
        fig.text(.5, .012, "tall on one is short on the other — they are exact inverses, "
                 r"$v=\Delta s/\Delta t$", ha="center", fontsize=17, color=INK)
        hold(w, fps * 6)
        clear(fig, keep)

        # ---- act 5: decoding ---------------------------------------------
        T.set_text("What decoding has to do")
        U.set_text("the decoder needs TIME — one variant already has it")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        # velocity route
        ax.text(.07, .70, "velocity", fontsize=20, color=VEL, fontweight="bold")
        cell(ax, .07, .50, .15, .13, "v", PANEL, VEL, fs=24)
        ax.annotate("", xy=(.30, .565), xytext=(.225, .565),
                    arrowprops=dict(arrowstyle="-|>", color=VEL, lw=3, mutation_scale=24))
        cell(ax, .30, .50, .22, .13, r"$\Delta t = \Delta s\,/\,|v|$", "#1c1420", VEL, fs=20)
        ax.annotate("", xy=(.60, .565), xytext=(.525, .565),
                    arrowprops=dict(arrowstyle="-|>", color=VEL, lw=3, mutation_scale=24))
        cell(ax, .60, .50, .15, .13, r"$\Delta t$", PANEL, VEL, fs=24)
        ax.text(.78, .565, "a division, undone", fontsize=15, color=VEL, va="center")
        # duration route
        ax.text(.07, .33, "duration", fontsize=20, color=DUR, fontweight="bold")
        cell(ax, .07, .13, .15, .13, r"$\Delta t$", PANEL, DUR, fs=24)
        ax.annotate("", xy=(.60, .195), xytext=(.225, .195),
                    arrowprops=dict(arrowstyle="-|>", color=DUR, lw=3, mutation_scale=24))
        cell(ax, .60, .13, .15, .13, r"$\Delta t$", PANEL, DUR, fs=24)
        ax.text(.78, .195, "nothing to undo", fontsize=15, color=DUR, va="center")
        ax.text(.415, .245, "read it", ha="center", fontsize=14, color=DUR)
        hold(w, fps * 6)
        clear(fig, keep)

        # ---- act 6: the hold ---------------------------------------------
        T.set_text("And when the tool pauses")
        U.set_text("a hold is where the two representations come apart")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(.07, .70, "velocity", fontsize=20, color=VEL, fontweight="bold")
        cell(ax, .07, .48, .17, .14, "v = 0", "#1c1420", RED, fs=24)
        ax.annotate("", xy=(.33, .55), xytext=(.245, .55),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=3, mutation_scale=24))
        cell(ax, .33, .48, .26, .14, r"$\Delta s\,/\,0$", "#1c1420", RED, fs=22,
             sub="undefined")
        ax.annotate("", xy=(.67, .55), xytext=(.60, .55),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=3, mutation_scale=24))
        cell(ax, .67, .48, .26, .14, "synthetic stop_duration", "#1c1420", RED, fs=15,
             sub="a special case in the decoder")
        ax.text(.07, .31, "duration", fontsize=20, color=DUR, fontweight="bold")
        cell(ax, .07, .09, .17, .14, r"$\Delta t$ large", PANEL, DUR, fs=19)
        ax.annotate("", xy=(.67, .16), xytext=(.245, .16),
                    arrowprops=dict(arrowstyle="-|>", color=DUR, lw=3, mutation_scale=24))
        cell(ax, .67, .09, .26, .14, "a long interval", PANEL, DUR, fs=18,
             sub="no special case at all")
        hold(w, fps * 6)
        clear(fig, keep)

        # ---- act 7: the summary ------------------------------------------
        T.set_text("The whole difference")
        U.set_text("")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        rows = [("", "velocity", "duration"),
                ("timing slot holds", "a RATE   (px/s, rad/s)", "a TIME   (s, s)"),
                ("units across the two slots", "different", "the same"),
                ("decoder must", "divide to recover time", "read the value"),
                ("a hold needs", "a special case", "nothing"),
                ("sign of the rotation slot", "predicted, then discarded", "no sign to carry")]
        y = .74
        for k, (lab, va_, db) in enumerate(rows):
            head = k == 0
            if lab:
                ax.text(.06, y, lab, fontsize=16.5, color=DIM, va="center")
            ax.text(.47, y, va_, fontsize=18 if head else 16.5,
                    color=VEL, va="center", fontweight="bold" if head else "normal")
            ax.text(.76, y, db, fontsize=18 if head else 16.5,
                    color=DUR, va="center", fontweight="bold" if head else "normal")
            if head:
                ax.plot([.06, .95], [y - .045, y - .045], color=GRID, lw=1.4)
            y -= .115
        ax.text(.5, .045, "geometry is byte-identical either way — only these two slots "
                "change", ha="center", fontsize=16, color=INK)
        hold(w, fps * 7)
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
