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


def grab(w, rep=1):
    for _ in range(rep):
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


def _writer(fps):
    return FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                        extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])


def clip_hybrid(out, fps):
    """Timing mismatch between the streams, and how a row still aligns them."""
    fig, T, U = fig_new()
    keep = (T, U)
    w = _writer(fps)
    MD, PITCH, GAP0 = 8, .112, .10

    # Two plausible sample-time patterns over the same window. Translation
    # clumps where the tool crawls; rotation turns at a steadier rate. These are
    # schematic, but the SHAPE is what the real data does.
    T_END = 2.4
    t_tr = np.array([0.00, 0.62, 0.79, 0.88, 0.95, 1.34, 1.86, 2.31])
    t_rot = np.array([0.00, 0.21, 0.44, 0.70, 1.02, 1.38, 1.80, 2.26])
    X0, X1, Y_TR, Y_RO = .11, .90, .70, .49

    def tx(t):
        return X0 + (t / T_END) * (X1 - X0)

    def timelines(ax, upto=MD, mark=None):
        for y, col, name, ts in ((Y_TR, TR, "translation stream", t_tr),
                                 (Y_RO, ROT, "rotation stream", t_rot)):
            ax.plot([X0, X1], [y, y], color=GRID, lw=2.4)
            ax.text(X0, y + .062, name, fontsize=16, color=col, fontweight="bold")
            for j in range(upto):
                lit = (mark is not None and j == mark)
                ax.plot([tx(ts[j])] * 2, [y - .030, y + .030], color=col,
                        lw=4.0 if lit else 2.0, alpha=1.0 if lit else .40,
                        zorder=4 if lit else 2)
                if lit:
                    ax.add_patch(Rectangle((tx(ts[j]) - .008, y - .016), .016, .032,
                                           facecolor=col, edgecolor=BG, lw=1.2,
                                           zorder=5))
        ax.text(X1, Y_RO - .075, "time", fontsize=13.5, color=DIM, ha="right")

    with w.saving(fig, out, dpi=120):
        # ---- act 1: the timing mismatch ---------------------------------
        T.set_text("The two streams do not sample at the same moments")
        U.set_text("both produce sample i — but sample i happens at a different time "
                   "in each stream")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        for i in range(MD):
            ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            timelines(ax, mark=i)
            a, b = tx(t_tr[i]), tx(t_rot[i])
            lo, hi = min(a, b), max(a, b)
            ax.plot([a, a], [Y_TR - .030, Y_RO + .030], color=TR, lw=1.2,
                    ls=(0, (3, 3)), alpha=.65)
            ax.plot([b, b], [Y_TR - .030, Y_RO + .030], color=ROT, lw=1.2,
                    ls=(0, (3, 3)), alpha=.65)
            mid = (Y_TR + Y_RO) / 2
            if hi - lo > .004:
                ax.annotate("", xy=(hi, mid), xytext=(lo, mid),
                            arrowprops=dict(arrowstyle="<->", color=RED, lw=2.4))
                ax.text((lo + hi) / 2, mid + .026,
                        "%.2f s apart" % abs(t_tr[i] - t_rot[i]), ha="center",
                        fontsize=14.5, color=RED, fontweight="bold")
            ax.text(.11, .30, "sample  i = %d" % i, fontsize=22, color=INK,
                    fontweight="bold")
            ax.text(.11, .225, "translation at  %.2f s" % t_tr[i], fontsize=17,
                    color=TR, family="monospace")
            ax.text(.11, .16, "rotation    at  %.2f s" % t_rot[i], fontsize=17,
                    color=ROT, family="monospace")
            grab(w, 9)
        ax.text(.55, .225, "the mismatch is not a bug", fontsize=18, color=INK)
        ax.text(.55, .16, "each stream is sampled on its own arc clock, so its "
                "samples land\nwherever that quantity actually accumulated",
                fontsize=15.5, color=DIM, va="top")
        hold(w, fps * 5)
        clear(fig, keep)

        # ---- act 2: interpolation gives both the same count -------------
        T.set_text("Interpolation is what makes them the same length")
        U.set_text("each clock is cut into M equal steps — different units, different "
                   "totals, identical sample count")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        BARS = [(.70, TR, "translation arc", "0", "D = 80 px", .085, .80),
                (.47, ROT, "rotation arc", "0", "R = 26" + chr(176), .085, .52)]
        for k in range(1, MD + 1):
            ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            for y, col, name, lo, hi, x0, x1 in BARS:
                ax.add_patch(Rectangle((x0, y - .028), x1 - x0, .056,
                                       facecolor=PANEL, edgecolor=col, lw=2.2))
                ax.text(x0, y + .075, name, fontsize=17, color=col, fontweight="bold")
                ax.text(x0, y - .075, lo, fontsize=13.5, color=DIM, ha="center")
                ax.text(x1, y - .075, hi, fontsize=14.5, color=col, ha="center",
                        fontweight="bold")
                xs = np.linspace(x0, x1, MD)
                for j in range(k):
                    ax.plot([xs[j], xs[j]], [y - .028, y + .028], color=col, lw=2.6)
                    ax.add_patch(Rectangle((xs[j] - .0055, y - .0125), .011, .025,
                                           facecolor=col, edgecolor="none"))
                ax.text(x1 + .035, y, "%d / %d" % (k, MD), fontsize=17, color=col,
                        va="center", family="monospace", fontweight="bold")
            ax.text(.085, .30, "linspace(0, end, M) on each clock — the ONLY thing "
                    "they share is M", fontsize=17, color=INK)
            if k == MD:
                ax.text(.085, .21, "M samples each, whatever the budgets were",
                        fontsize=17, color=INK)
                ax.text(.085, .12, "(M = %d here so the ticks are countable; the real "
                        "token uses M = 56)" % MD, fontsize=13.5, color=DIM)
            grab(w, 6)
        hold(w, fps * 4)
        clear(fig, keep)

        # ---- act 3: alignment -- row i takes one sample from each -------
        T.set_text("A token row aligns them by INDEX")
        U.set_text("row i holds translation sample i and rotation sample i — sampled "
                   "at different moments")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ROW_X, ROW_W, ROW_TOP, ROW_H, ROW_GAP = .215, .55, .345, .025, .006
        for i in range(MD):
            ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            timelines(ax, mark=i)
            # the token, drawn as M thin rows of six cells
            for r in range(MD):
                y = ROW_TOP - r * (ROW_H + ROW_GAP)
                for c in range(6):
                    lit = (r == i)
                    col = TR if c < 3 else ROT
                    ax.add_patch(Rectangle((ROW_X + c * (ROW_W / 6), y),
                                           ROW_W / 6 - .004, ROW_H,
                                           facecolor=col if lit else PANEL,
                                           edgecolor=col if lit else GRID,
                                           alpha=1.0 if lit else .55, lw=1.4))
                ax.text(ROW_X - .022, y + ROW_H / 2, str(r), fontsize=12.5,
                        color=INK if r == i else DIM, ha="right", va="center",
                        family="monospace",
                        fontweight="bold" if r == i else "normal")
            yi = ROW_TOP - i * (ROW_H + ROW_GAP) + ROW_H / 2
            ax.annotate("", xy=(ROW_X + ROW_W / 12, yi + ROW_H),
                        xytext=(tx(t_tr[i]), Y_TR - .034),
                        arrowprops=dict(arrowstyle="-|>", color=TR, lw=2.4,
                                        connectionstyle="arc3,rad=-0.18",
                                        mutation_scale=20))
            ax.annotate("", xy=(ROW_X + ROW_W * 0.75, yi + ROW_H),
                        xytext=(tx(t_rot[i]), Y_RO - .034),
                        arrowprops=dict(arrowstyle="-|>", color=ROT, lw=2.4,
                                        connectionstyle="arc3,rad=0.18",
                                        mutation_scale=20))
            ax.text(ROW_X + ROW_W + .03, yi, "row %d" % i, fontsize=16, color=INK,
                    va="center", fontweight="bold")
            ax.text(ROW_X + ROW_W + .03, yi - .030,
                    "%.2f s  /  %.2f s" % (t_tr[i], t_rot[i]), fontsize=13,
                    color=DIM, va="center", family="monospace")
            grab(w, 9)
        ax.text(.11, .045, "the row index is a WAYPOINT number, not a timestamp — "
                "which is exactly why the two streams can share a row",
                fontsize=16.5, color=INK)
        hold(w, fps * 5)
        clear(fig, keep)

        # ---- act 4: and therefore they concatenate ----------------------
        T.set_text("Which is why they stack")
        U.set_text("two [M, 3] blocks on different clocks concatenate along the "
                   "feature axis")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        L3 = ["x", "y", "timing"]; R3 = ["cos", "sin", "timing"]
        for step in range(26):
            ax.clear(); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            f = min(1.0, step / 18.0)
            # Blocks must MEET, not overlap: the right block starts exactly one
            # block-width after the left one, plus the closing gap.
            gap = GAP0 * (1 - f)
            lx = .17
            rx = lx + 3 * PITCH + gap
            for x0, cols, col, name in ((lx, L3, TR, "translation"),
                                        (rx, R3, ROT, "rotation")):
                for j, c in enumerate(cols):
                    cell(ax, x0 + j * PITCH, .34, .105, .30, c, PANEL, col, fs=17)
                ax.text(x0 + 1.5 * PITCH - .003, .695, name + "   [M, 3]",
                        fontsize=16.5, color=col, ha="center", fontweight="bold")
            if f >= 1.0:
                right_edge = rx + 2 * PITCH + .105
                mid = (lx + right_edge) / 2
                ax.add_patch(FancyBboxPatch((lx - .018, .31),
                                            right_edge - lx + .036, .36,
                                            boxstyle="round,pad=0.012",
                                            facecolor="none", edgecolor=INK, lw=2.6))
                ax.text(mid, .245, "[M, 6]", fontsize=26, color=INK, ha="center",
                        fontweight="bold")
                ax.text(mid, .175, "one row, six numbers, two clocks",
                        fontsize=16, color=DIM, ha="center")
            grab(w, 3)
        hold(w, fps * 4)
        clear(fig, keep)

        # ---- act 5: why not one shared clock ----------------------------
        T.set_text("Why not one shared clock?")
        U.set_text("it was tried — the ablation is called carry")
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        rows = [("hybrid — two clocks", TR,
                 "rotation gets its own budget R, so a fast turn is not under-sampled "
                 "just because the tool barely moved"),
                ("carry — one clock", DIM,
                 "rotation rides the translation clock; there is no R at all, and a "
                 "turn-in-place has nowhere to be sampled")]
        y = .62
        for name, col, why in rows:
            ax.text(.09, y, name, fontsize=21, color=col, fontweight="bold")
            ax.text(.09, y - .085, why, fontsize=16.5, color=DIM)
            y -= .25
        # Do NOT imply a controlled head-to-head. carry (0.415) and the best
        # independent-clock cell (0.519) also differ in M, and carry actually
        # beat the row-split baseline (0.408). The ablation did not settle it.
        ax.text(.09, .145, "both were trained. carry scored 0.415 mean coverage; the "
                "best two-clock cell scored 0.519", fontsize=16, color=INK)
        ax.text(.09, .075, "but those cells differ in M as well, and at n = 40 the "
                "comparison is not conclusive", fontsize=15, color=DIM)
        hold(w, fps * 6)
    plt.close(fig)
    print("wrote", out)


def clip_design(out, fps):
    """What the two timing variants actually store."""
    fig, T, U = fig_new()
    keep = (T, U)
    w = _writer(fps)
    with w.saving(fig, out, dpi=120):
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
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "videos")
    ap.add_argument("--outdir", default=here)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--only", choices=["hybrid", "design"], default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.only in (None, "hybrid"):
        clip_hybrid(os.path.join(a.outdir, "04_hybrid.mp4"), a.fps)
    if a.only in (None, "design"):
        clip_design(os.path.join(a.outdir, "05_design.mp4"), a.fps)


if __name__ == "__main__":
    main()
