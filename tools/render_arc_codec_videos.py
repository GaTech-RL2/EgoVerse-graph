#!/usr/bin/env python
"""Render slide-ready MP4s explaining the ARC action codec on PushShapes.

Four clips, one idea each, from REAL data: a real u_socket episode, the real T
and socket geometry from Tsimulation, and the real tokenizers/decoders.

    01_chunk.mp4      what one action chunk is, on the PushT scene
    02_clocks.mp4     the two independent arc clocks and their D / R budgets
    03_resample.mp4   even in arc length is uneven in time
    04_token.mp4      velocity vs duration, and both decoding back

Usage:  python tools/render_arc_codec_videos.py [--outdir DIR] [--fps 30]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import ConnectionPatch, Polygon as MplPoly, Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from egomimic.rldb.zarr.planar_arc import (  # noqa: E402
    PadPlanarAction, TokenizeUSocketArcDuration, TokenizeUSocketArcVelocityStacked,
    _bracket_segment,
)
from egomimic.rldb.embodiment.usocket_arc_velocity import (  # noqa: E402
    USocketArcDurationNativeDecoder, USocketArcLocalVelocityStackedNativeDecoder,
)

PROBE = "/Users/rpunamiya/Desktop/GEAR/sim_run/usocket_probe"
D, M, R, DT, H, W0 = 80.0, 56, math.radians(26.0), 1.0 / 30.0, 16, 80

# Geometry, verbatim from Tsimulation.sim_v2.pushshapes.shapes
T_RECTS = [(0.0, -30.0, 120.0, 30.0), (0.0, 30.0, 30.0, 90.0)]
SOCKET_RECTS = [(5.0, -21.0, 30.0, 10.0), (5.0, 21.0, 30.0, 10.0), (-15.0, 0.0, 10.0, 52.0)]

BG, PANEL, INK, DIM, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
TR, ROT, DUR, VEL = "#58a6ff", "#f0883e", "#3fb950", "#bc8cff"
OBJ, GOAL = "#d29922", "#3fb950"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "axes.edgecolor": GRID, "font.size": 13,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
})


def rects_world(rects, x, y, th):
    c, s = math.cos(th), math.sin(th)
    out = []
    for cx, cy, w, h in rects:
        hw, hh = w / 2, h / 2
        loc = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
        out.append([(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in loc])
    return out


def load():
    best, score = None, -1
    for p in sorted(glob.glob(os.path.join(PROBE, "episode_*.zarr"))):
        g = zarr.open(p, mode="r")
        a, s = np.asarray(g["actions"]), np.asarray(g["observations.state"])
        if len(a) < W0 + 2:
            continue
        for st in range(0, len(a) - W0, 5):
            w, o = a[st:st + W0], s[st:st + W0, 3:6]
            tr = np.linalg.norm(np.diff(w[:, :2], axis=0), axis=-1).sum()
            ro = np.abs(np.diff(np.unwrap(w[:, 2]))).sum()
            early = np.linalg.norm(np.diff(w[:H, :2], axis=0), axis=-1).sum()
            obj = np.linalg.norm(o[-1, :2] - o[0, :2]) + 40 * abs(
                np.unwrap(o[:, 2])[-1] - o[0, 2])
            # Travel a little ABOVE D: the budget has to bite (that is the point)
            # but a path 4x longer than D leaves the waypoints in an unreadable
            # blob covering a quarter of it. Early motion keeps the 16-step
            # decode clip from being a 6-pixel wiggle.
            fit = math.exp(-((tr / D - 1.35) ** 2) / (2 * 0.45 ** 2))
            sc = (fit * min(ro / R, 1.4) * min(early / 25.0, 1.4)
                  * min(obj / 40.0, 1.3))
            if sc > score:
                score, best = sc, (p, st)
    path, st = best
    g = zarr.open(path, mode="r")
    return dict(
        name=os.path.basename(path), start=st,
        act=np.asarray(g["actions"])[st:st + W0].astype(float),
        obj=np.asarray(g["observations.state"])[st:st + W0, 3:6].astype(float),
        goal=np.asarray(g["goal_pose"])[st].astype(float),
    )


def tokenize(act):
    c5 = np.asarray(PadPlanarAction(["actions"]).transform({"actions": act.copy()})["actions"],
                    dtype=np.float64)
    v = TokenizeUSocketArcVelocityStacked(
        min_distance_unit=D, resampled_vector_length=M, dt=DT, rotation_distance_unit=R)
    d = TokenizeUSocketArcDuration(
        min_distance_unit=D, resampled_vector_length=M, dt=DT, rotation_distance_unit=R)
    return v.tokenize(c5), d.tokenize(c5)


def clocks(act):
    xy, th = act[:, :2], np.unwrap(act[:, 2])
    tr = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))))
    an = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(th)))))
    return tr, an, min(D, tr[-1]), min(R, an[-1])


def source_times(cum, end):
    out = []
    for t in np.linspace(0.0, end, M):
        i, a = _bracket_segment(cum, float(t))
        out.append((i + a) * DT)
    return np.array(out)


def view_box(E, pad=70.0):
    """Square window around everything that moves, so the scene fills the panel."""
    pts = [E["act"][:, :2], E["obj"][:, :2], E["goal"][None, :2]]
    p = np.concatenate(pts, axis=0)
    lo, hi = p.min(0) - pad, p.max(0) + pad
    c, half = (lo + hi) / 2, max(hi - lo) / 2
    return (c[0] - half, c[0] + half, c[1] - half, c[1] + half)


def scene(ax, E, i, trail_to=None, title=True, box=None):
    """Draw the PushShapes world at frame i."""
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    x0, x1, y0, y1 = box if box else view_box(E)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    gx, gy, gth = E["goal"]
    for poly in rects_world(T_RECTS, gx, gy, gth):
        ax.add_patch(MplPoly(poly, closed=True, facecolor="none", edgecolor=GOAL,
                             lw=2.0, ls=(0, (5, 4)), alpha=.85, zorder=1))
    ox, oy, oth = E["obj"][i]
    for poly in rects_world(T_RECTS, ox, oy, oth):
        ax.add_patch(MplPoly(poly, closed=True, facecolor=OBJ, edgecolor="#f2cc60",
                             lw=1.6, alpha=.92, zorder=3))
    k = i + 1 if trail_to is None else trail_to
    ax.plot(E["act"][:k, 0], E["act"][:k, 1], color=TR, lw=2.4, alpha=.95, zorder=4)
    ax.plot(E["act"][:, 0], E["act"][:, 1], color=TR, lw=1.0, alpha=.18, zorder=2)
    px, py, pth = E["act"][i]
    for poly in rects_world(SOCKET_RECTS, px, py, pth):
        ax.add_patch(MplPoly(poly, closed=True, facecolor="#1f6feb", edgecolor="#79c0ff",
                             lw=1.6, alpha=.95, zorder=5))
    L = 42
    ax.arrow(px, py, L * math.cos(pth), L * math.sin(pth), width=2.2, head_width=10,
             color=ROT, alpha=.95, zorder=6, length_includes_head=True)
    if title:
        ax.text(.02, .975, "PushShapes  ·  T object, U-socket pusher", transform=ax.transAxes,
                va="top", fontsize=12.5, color=DIM)
        ax.text(.98, .975, f"frame {i:>2}/{W0}", transform=ax.transAxes, va="top", ha="right",
                fontsize=12.5, color=DIM, family="monospace")


def chrome(fig, title, sub):
    fig.text(.035, .955, title, fontsize=28, fontweight="bold", va="top")
    return fig.text(.035, .898, sub, fontsize=15.5, color="#c9d1d9", va="top")


def hold(writer, n):
    for _ in range(n):
        writer.grab_frame()


def grab(writer, rep=1):
    """Repeat a frame so the animation reads at a followable pace."""
    for _ in range(rep):
        writer.grab_frame()


def gauge(ax, frac, budget_frac, value, budget, unit, label, col, done_at=None):
    """A horizontal fill bar with a hard stop at the budget.

    Reads as a clock filling up, which two cumulative line charts did not.
    """
    ax.clear(); ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Rectangle((0, .30), 1, .40, facecolor="#0b0f14",
                           edgecolor=GRID, lw=1.4, zorder=1))
    over = frac > budget_frac
    fill = min(frac, budget_frac)
    ax.add_patch(Rectangle((0, .30), fill, .40, facecolor=col, alpha=.95, zorder=2))
    if over:   # everything past the budget is NOT in the token
        ax.add_patch(Rectangle((budget_frac, .30), min(frac, 1.0) - budget_frac, .40,
                               facecolor=col, alpha=.20, zorder=2, hatch="///",
                               edgecolor=col))
    ax.plot([budget_frac, budget_frac], [.22, .78], color="#f85149", lw=3.0, zorder=4)
    ax.text(budget_frac, .84, f"budget {budget:g} {unit}", color="#f85149",
            fontsize=13.5, fontweight="bold", ha="center")
    ax.text(.005, .04, label, color=col, fontsize=15, fontweight="bold", va="bottom")
    ax.text(.995, .04, f"{value:.1f} {unit}", color=INK, fontsize=15, va="bottom",
            ha="right", family="monospace")
    if done_at is not None:
        ax.text(budget_frac, .12, f"full at frame {done_at}", color="#f85149",
                fontsize=12.5, ha="center")


def matrix(ax, T, cols, hi, title_txt, hicol):
    """Draw a token as a labelled column-normalised heatmap."""
    ax.clear(); ax.set_facecolor(PANEL)
    Z = np.zeros_like(T)
    for j in range(T.shape[1]):
        c = T[:, j]
        lo, up = c.min(), c.max()
        Z[:, j] = (c - lo) / (up - lo) if up > lo else 0.5
    ax.imshow(Z.T, aspect="auto", cmap="magma", origin="upper",
              extent=[0, T.shape[0], T.shape[1], 0], vmin=0, vmax=1)
    for j in range(T.shape[1] + 1):
        ax.plot([0, T.shape[0]], [j, j], color=BG, lw=1.2)
    for j in hi:
        ax.add_patch(Rectangle((0, j), T.shape[0], 1, facecolor="none",
                               edgecolor=hicol, lw=3.0, zorder=5))
    ax.set_yticks(np.arange(T.shape[1]) + .5)
    ax.set_yticklabels(cols, fontsize=12.5, family="monospace")
    for k, lab in enumerate(ax.get_yticklabels()):
        lab.set_color(hicol if k in hi else DIM)
    ax.set_xlabel("token row  i", fontsize=12.5)
    ax.set_title(title_txt, fontsize=14, color=INK, pad=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def sign_change_chunk():
    """A chunk whose omega actually changes sign.

    The chunk the other clips use turns at a constant rate, so omega never goes
    negative there and the 'discarded sign' slide would have read
    '0 of 56 rows' -- contradicting its own claim. Pick a window that reverses.
    """
    tok = TokenizeUSocketArcVelocityStacked(
        min_distance_unit=D, resampled_vector_length=M, dt=DT, rotation_distance_unit=R)
    best, score = None, -1
    for p in sorted(glob.glob(os.path.join(PROBE, "episode_*.zarr"))):
        a = np.asarray(zarr.open(p, mode="r")["actions"])
        if len(a) < W0 + 2:
            continue
        for st in range(0, len(a) - W0, 8):
            w = a[st:st + W0]
            c5 = np.asarray(PadPlanarAction(["actions"]).transform(
                {"actions": w.copy()})["actions"], dtype=np.float64)
            t = tok.tokenize(c5)
            neg = int((t[:, 5] < 0).sum())
            bal = min(neg, M - neg)            # prefer a clean reversal
            if bal > score:
                score, best = bal, (t, "%s +%d" % (os.path.basename(p)[8:-5], st))
    return best


# --------------------------------------------------------------- clip 1
def clip_chunk(E, out, fps):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    chrome(fig, "One action chunk", "80 commanded poses at 30 Hz — this is what the policy has to emit")
    ax = fig.add_axes([.05, .08, .44, .78])
    axx = fig.add_axes([.56, .53, .40, .33])
    axt = fig.add_axes([.56, .10, .40, .33])
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    t = np.arange(W0) * DT
    box = view_box(E)
    with w.saving(fig, out, dpi=120):
        hold(w, fps // 2)
        for i in range(W0):
            scene(ax, E, i, box=box)
            for a, ys, cols, lab in (
                (axx, (E["act"][:, 0], E["act"][:, 1]), (TR, "#79c0ff"), "position (px)"),
                (axt, (np.unwrap(E["act"][:, 2]),), (ROT,), "heading θ (rad)")):
                a.clear(); a.set_facecolor(PANEL)
                for sp in a.spines.values(): sp.set_color(GRID)
                a.grid(color=GRID, lw=.6, alpha=.5)
                for y, c in zip(ys, cols):
                    a.plot(t[:i + 1], y[:i + 1], color=c, lw=2.6)
                    a.plot(t, y, color=c, lw=1.0, alpha=.18)
                a.set_xlim(0, t[-1]); a.set_ylabel(lab, fontsize=12.5)
                a.set_xlabel("time (s)", fontsize=12)
            axx.text(.985, .06, "x", transform=axx.transAxes, ha="right", color=TR,
                     fontsize=14, fontweight="bold")
            axx.text(.94, .06, "y", transform=axx.transAxes, ha="right", color="#79c0ff",
                     fontsize=14, fontweight="bold")
            grab(w, 4)
        hold(w, fps * 2)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------- clip 2
def clip_clocks(E, out, fps):
    tr, an, tr_end, an_end = clocks(E["act"])
    an_deg, an_end_deg = np.degrees(an), math.degrees(an_end)
    i_tr = int(np.argmax(tr >= tr_end)) if tr[-1] >= tr_end else W0 - 1
    i_an = int(np.argmax(an >= an_end)) if an[-1] >= an_end else W0 - 1
    box = view_box(E)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    chrome(fig, "Two arc clocks, not one",
           "each stream accumulates its own distance and stops at its own budget")
    ax = fig.add_axes([.045, .09, .43, .77])
    g1 = fig.add_axes([.545, .62, .42, .17])
    g2 = fig.add_axes([.545, .36, .42, .17])
    tl = fig.add_axes([.545, .13, .42, .10])
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    # bars are scaled so the budget always sits at 65% of the width
    sc_tr = 0.65 / max(tr_end, 1e-9)
    sc_an = 0.65 / max(an_end_deg, 1e-9)
    with w.saving(fig, out, dpi=120):
        hold(w, fps // 2)
        for i in range(W0):
            # scene: path INSIDE the translation budget is bright, past it is dim
            scene(ax, E, i, trail_to=0, box=box)
            cut = min(i + 1, i_tr + 1)
            ax.plot(E["act"][:cut, 0], E["act"][:cut, 1], color=TR, lw=3.4, zorder=4)
            if i > i_tr:
                ax.plot(E["act"][i_tr:i + 1, 0], E["act"][i_tr:i + 1, 1],
                        color=TR, lw=2.0, alpha=.28, ls=(0, (4, 3)), zorder=4)
            ax.text(.02, .935, "bright = inside the budget, dashed = beyond it",
                    transform=ax.transAxes, va="top", fontsize=12, color=DIM)

            gauge(g1, tr[i] * sc_tr, 0.65, tr[i], tr_end, "px",
                  "translation   " + r"$\Sigma|\Delta xy|$", TR,
                  i_tr if i >= i_tr else None)
            gauge(g2, an_deg[i] * sc_an, 0.65, an_deg[i], an_end_deg, "deg",
                  "rotation   " + r"$\Sigma|\Delta\theta|$", ROT,
                  i_an if i >= i_an else None)

            tl.clear(); tl.set_facecolor(PANEL)
            tl.set_xlim(0, W0 - 1); tl.set_ylim(0, 1)
            tl.set_yticks([]); tl.set_xlabel("frame in the chunk", fontsize=12.5)
            for sp in tl.spines.values():
                sp.set_color(GRID)
            tl.axvspan(0, i, color="#21262d")
            # stagger the labels vertically: the two frames are often only a
            # couple apart and the strings overlapped into nonsense.
            for f, c, nm, yy in ((i_tr, TR, "translation full", 1.30),
                                 (i_an, ROT, "rotation full", 1.06)):
                if i >= f:
                    tl.plot([f, f], [0, 1], color=c, lw=3)
                    tl.text(f, yy, f"{nm} (frame {f})", color=c, fontsize=12,
                            ha="center")
            tl.plot([i, i], [0, 1], color=INK, lw=2)
            grab(w, 4)
        msg = fig.text(.755, .035,
                       f"the two clocks fill at different frames: {i_tr} and {i_an}",
                       fontsize=15, color="#f85149", ha="center")
        hold(w, fps * 3)
        msg.remove()
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------- clip 3
def clip_resample(E, out, fps):
    tr, an, tr_end, an_end = clocks(E["act"])
    tok_v, _ = tokenize(E["act"])
    t_tr, t_an = source_times(tr, tr_end), source_times(an, an_end)
    th_way = np.arctan2(tok_v[:, 4], tok_v[:, 3])
    t_raw = np.arange(W0) * DT
    th_raw = np.unwrap(E["act"][:, 2])
    wp = tok_v[:, :2]
    lo, hi = wp.min(0) - 55, wp.max(0) + 55
    c, half = (lo + hi) / 2, max(hi - lo) / 2
    box = (c[0] - half, c[0] + half, c[1] - half, c[1] + half)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    chrome(fig, "Even in arc length, uneven in time",
           "M = %d samples per stream, equally spaced along each stream's own path" % M)
    axp = fig.add_axes([.045, .40, .43, .44])
    axt = fig.add_axes([.545, .40, .42, .44])
    axtime = fig.add_axes([.045, .12, .87, .17])
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    tmax = max(t_tr[-1], t_an[-1]) * 1.03
    links = []
    with w.saving(fig, out, dpi=120):
        hold(w, fps // 2)
        for k in range(1, M + 1):
            for cp in links:
                cp.remove()
            links = []

            axp.clear(); axp.set_facecolor(PANEL); axp.set_aspect("equal")
            axp.set_xlim(box[0], box[1]); axp.set_ylim(box[2], box[3])
            axp.set_xticks([]); axp.set_yticks([])
            for sp in axp.spines.values():
                sp.set_color(GRID)
            ox, oy, oth = E["obj"][0]
            for poly in rects_world(T_RECTS, ox, oy, oth):
                axp.add_patch(MplPoly(poly, closed=True, facecolor=OBJ, alpha=.40,
                                      edgecolor="#f2cc60", lw=1.2))
            axp.plot(E["act"][:, 0], E["act"][:, 1], color="#2d3748", lw=2.4)
            axp.scatter(tok_v[:k, 0], tok_v[:k, 1], s=54, color=TR, zorder=6,
                        edgecolors=BG, linewidths=.9)
            axp.set_title("translation stream — equal steps along the PATH",
                          fontsize=13.5, color=TR, pad=8)

            axt.clear(); axt.set_facecolor(PANEL)
            for sp in axt.spines.values():
                sp.set_color(GRID)
            axt.grid(color=GRID, lw=.6, alpha=.45)
            axt.plot(t_raw, th_raw, color="#2d3748", lw=2.4)
            axt.scatter(t_an[:k], th_way[:k], s=48, color=ROT, zorder=6,
                        edgecolors=BG, linewidths=.9)
            axt.set_xlim(0, t_raw[-1]); axt.set_xlabel("time (s)", fontsize=12)
            axt.set_ylabel(r"heading $\theta$ (rad)", fontsize=12.5)
            axt.set_title(r"rotation stream — equal steps in $\theta$",
                          fontsize=13.5, color=ROT, pad=8)

            axtime.clear(); axtime.set_facecolor(PANEL)
            axtime.set_xlim(0, tmax); axtime.set_ylim(0, 1)
            axtime.set_yticks([]); axtime.set_xlabel("source time (s)", fontsize=13)
            for sp in axtime.spines.values():
                sp.set_color(GRID)
            axtime.vlines(t_tr[:k], .55, .95, color=TR, lw=2.0)
            axtime.vlines(t_an[:k], .05, .45, color=ROT, lw=2.0)
            axtime.text(.004, .75, " translation", transform=axtime.transAxes,
                        color=TR, fontsize=13, va="center")
            axtime.text(.004, .25, " rotation", transform=axtime.transAxes,
                        color=ROT, fontsize=13, va="center")

            # Connect only every STRIDE-th waypoint. All 56 per stream drew 112
            # lines across the whole figure and read as moire rather than as a
            # mapping; a sparse fan shows the same thing.
            STRIDE = 6
            for j in range(0, k, STRIDE):
                cp = ConnectionPatch(
                    xyA=(tok_v[j, 0], tok_v[j, 1]), coordsA=axp.transData,
                    xyB=(t_tr[j], .95), coordsB=axtime.transData,
                    color=TR, lw=1.3, alpha=.45, zorder=3)
                fig.add_artist(cp); links.append(cp)
                cp = ConnectionPatch(
                    xyA=(t_an[j], th_way[j]), coordsA=axt.transData,
                    xyB=(t_an[j], .45), coordsB=axtime.transData,
                    color=ROT, lw=1.3, alpha=.45, zorder=3)
                fig.add_artist(cp); links.append(cp)
            fig.texts[-1].set_text(
                "M = %d samples per stream — waypoint %d/%d" % (M, k, M))
            grab(w, 5)
        msg = fig.text(.5, .035,
                       "equally spaced above, visibly clumped below — and the two "
                       "streams clump in different places",
                       fontsize=15, color=INK, ha="center")
        hold(w, fps * 3)
        msg.remove()
        for cp in links:
            cp.remove()
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------- clip 4
def clip_token(E, out, fps):
    """Shapes, then the timing design, then both decoding back."""
    tok_v, tok_d = tokenize(E["act"])
    tr, an, tr_end, an_end = clocks(E["act"])
    t_tr = source_times(tr, tr_end)
    dv = USocketArcLocalVelocityStackedNativeDecoder(M, H, native_action_dim=3, dt=DT)
    dd = USocketArcDurationNativeDecoder(M, H, native_action_dim=3, dt=DT)
    rv = dv.decode(torch.as_tensor(tok_v, dtype=torch.float32).unsqueeze(0))[0].numpy()
    rd = dd.decode(torch.as_tensor(tok_d, dtype=torch.float32).unsqueeze(0))[0].numpy()
    ref = E["act"][:H]

    COLS_V = ["x", "y", "v_xy", "cos", "sin", "omega"]
    COLS_D = ["x", "y", "dt_tr", "cos", "sin", "dt_rot"]
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])

    fig = plt.figure(figsize=(16, 9), dpi=120)
    title = fig.text(.035, .955, "", fontsize=28, fontweight="bold", va="top")
    sub = fig.text(.035, .898, "", fontsize=15.5, color="#c9d1d9", va="top")

    def act(t, u):
        title.set_text(t); sub.set_text(u)

    with w.saving(fig, out, dpi=120):
        # ---- act 1: the shapes -------------------------------------------
        act("What the codec actually produces",
            "one 80-step action chunk becomes a fixed-size token, and decodes back to the "
            "executed chunk")
        axL = fig.add_axes([.05, .16, .25, .62])
        axM = fig.add_axes([.375, .16, .25, .62])
        axR = fig.add_axes([.70, .16, .25, .62])
        matrix(axL, E["act"], ["x", "y", "theta"], [], "raw chunk", DIM)
        matrix(axM, tok_d, COLS_D, [2, 5], "ARC token", DUR)
        matrix(axR, ref, ["x", "y", "theta"], [], "decoded chunk (executed)", DIM)
        for a, sh, note in (
                (axL, (W0, 3), "80 steps at 30 Hz"),
                (axM, (M, 6), "fixed M rows, arc-spaced"),
                (axR, (H, 3), "action_horizon = 16")):
            a.text(.5, -.155, "shape %s" % (sh,), transform=a.transAxes, ha="center",
                   fontsize=17, fontweight="bold", color=INK, family="monospace")
            a.text(.5, -.225, note, transform=a.transAxes, ha="center",
                   fontsize=12.5, color=DIM)
        for x in (.335, .66):
            fig.text(x, .47, r"$\rightarrow$", fontsize=34, color=DIM, ha="center")
        fig.text(.335, .40, "tokenize", fontsize=13, color=DIM, ha="center")
        fig.text(.66, .40, "detokenize", fontsize=13, color=DIM, ha="center")
        hold(w, fps * 5)
        for a in (axL, axM, axR):
            a.remove()
        for t in list(fig.texts):
            if t not in (title, sub):
                t.remove()

        # ---- act 2: only two columns differ ------------------------------
        act("Only the two timing columns differ",
            "columns 0, 1, 3, 4 are byte-identical between the variants — the geometry is "
            "the same resampling")
        a1 = fig.add_axes([.06, .14, .40, .64])
        a2 = fig.add_axes([.55, .14, .40, .64])
        matrix(a1, tok_v, COLS_V, [2, 5], "velocity token   (%d, 6)" % M, VEL)
        matrix(a2, tok_d, COLS_D, [2, 5], "duration token   (%d, 6)" % M, DUR)
        same = np.abs(tok_v[:, [0, 1, 3, 4]] - tok_d[:, [0, 1, 3, 4]]).max()
        fig.text(.5, .055, "max difference across the four geometry columns: %.0e" % same,
                 fontsize=15, color=INK, ha="center")
        hold(w, fps * 4)
        for a in (a1, a2):
            a.remove()
        for t in list(fig.texts):
            if t not in (title, sub):
                t.remove()

        # ---- act 3: the worked example -----------------------------------
        j = int(np.argmax(np.abs(np.diff(t_tr))))     # the most interesting interval
        ds = float(np.linalg.norm(tok_v[j + 1, :2] - tok_v[j, :2]))
        dt = float(t_tr[j + 1] - t_tr[j])
        v = float(tok_v[j, 2])
        act("The timing design, at one interval",
            "between waypoint %d and %d the tool covers a fixed arc step in a variable "
            "amount of time" % (j, j + 1))
        ax = fig.add_axes([.05, .10, .90, .72]); ax.axis("off")
        rows = [
            ("both variants first compute the same two quantities", "", INK, 15.5, True),
            (r"   $\Delta s$  = arc step between waypoints", "%.3f px" % ds, TR, 15, False),
            (r"   $\Delta t$  = source time between them", "%.4f s" % dt, INK, 15, False),
            ("", "", INK, 8, False),
            ("velocity token stores the QUOTIENT", "", VEL, 15.5, True),
            (r"   $v = \Delta s / \Delta t$", "%.2f px/s" % v, VEL, 15, False),
            (r"   decode must undo it:  $\Delta t = \Delta s / |v|$", "", VEL, 15, False),
            (r"   and needs a synthetic stop_duration when $v \approx 0$", "", VEL, 14, False),
            ("", "", INK, 8, False),
            ("duration token stores the QUANTITY ITSELF", "", DUR, 15.5, True),
            (r"   $\Delta t$", "%.4f s" % dt, DUR, 15, False),
            ("   decode reads it, clamped non-negative", "", DUR, 15, False),
            ("   a hold is just a long interval — no special case", "", DUR, 14, False),
        ]
        y = .95
        for txt, val, col, fs, bold in rows:
            if txt:
                ax.text(.02, y, txt, transform=ax.transAxes, fontsize=fs, color=col,
                        va="top", fontweight="bold" if bold else "normal")
                if val:
                    ax.text(.52, y, val, transform=ax.transAxes, fontsize=fs, color=col,
                            va="top", family="monospace", fontweight="bold")
            y -= .073 if txt else .035
        hold(w, fps * 7)
        ax.remove()
        for t in list(fig.texts):
            if t not in (title, sub):
                t.remove()

        # ---- act 4: the discarded sign -----------------------------------
        sign_tok, sign_name = sign_change_chunk()
        om = sign_tok[:, 5]
        act("The velocity codec predicts a sign nothing reads",
            r"decode uses $|rate|$ — direction already lives in the cos/sin waypoints")
        ax = fig.add_axes([.08, .14, .84, .64])
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(color=GRID, lw=.6, alpha=.45)
        idx = np.arange(M)
        ax.axhline(0, color=GRID, lw=1.4)
        ax.plot(idx, om, color=VEL, lw=3.0,
                label=r"$\omega$ the model is trained to predict")
        ax.plot(idx, np.abs(om), color=DUR, lw=2.6, ls=(0, (5, 3)),
                label=r"$|\omega|$ — all the decoder ever uses")
        ax.set_xlim(0, M - 1); ax.set_xlabel("token row  i", fontsize=13)
        ax.set_ylabel(r"$\omega$  (rad/s)", fontsize=13)
        ax.legend(loc="best", facecolor=PANEL, edgecolor=GRID, fontsize=14)
        neg = int((om < 0).sum())
        fig.text(.5, .055,
                 "%d of %d rows carry a sign that decoding throws away        "
                 "(chunk: %s — the tool reverses its turn here)" % (neg, M, sign_name),
                 fontsize=14, color=INK, ha="center")
        hold(w, fps * 5)
        ax.remove()
        for t in list(fig.texts):
            if t not in (title, sub):
                t.remove()

        # ---- act 5: both decode back -------------------------------------
        act("Both decode to the same trajectory",
            "duration is not a better fit to the data — it is a better thing to ask a "
            "network to predict")
        aL = fig.add_axes([.07, .12, .38, .68])
        aR = fig.add_axes([.56, .12, .38, .68])
        for k in range(2, H + 1):
            for a, rec, col, name in ((aL, rv, VEL, "velocity decode"),
                                      (aR, rd, DUR, "duration decode")):
                a.clear(); a.set_facecolor(PANEL); a.set_aspect("equal")
                for sp in a.spines.values():
                    sp.set_color(GRID)
                a.grid(color=GRID, lw=.6, alpha=.45)
                a.plot(ref[:, 0], ref[:, 1], color=DIM, lw=7, alpha=.5, label="source")
                a.plot(rec[:k, 0], rec[:k, 1], color=col, lw=3.0, label=name)
                a.scatter(ref[:k, 0], ref[:k, 1], s=26, color=DIM, zorder=4)
                err = np.abs(rec[:, :2] - ref[:, :2]).max()
                a.set_title("%s   ·   max error %.3f px" % (name, err),
                            fontsize=14.5, color=col, pad=8)
                a.set_xlabel("x (px)", fontsize=12); a.set_ylabel("y (px)", fontsize=12)
                a.legend(loc="best", facecolor=PANEL, edgecolor=GRID, fontsize=12)
            grab(w, 8)
        fig.text(.5, .045,
                 "at 260M the two are statistically indistinguishable in rollout "
                 "(paired t-test, p = 0.78)", fontsize=15, color=INK, ha="center")
        hold(w, fps * 4)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "results", "videos"))
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E = load()
    print(f"episode {E['name']} frames [{E['start']},{E['start']+W0})")
    clip_chunk(E, os.path.join(a.outdir, "01_chunk.mp4"), a.fps)
    clip_clocks(E, os.path.join(a.outdir, "02_clocks.mp4"), a.fps)
    clip_resample(E, os.path.join(a.outdir, "03_resample.mp4"), a.fps)
    clip_token(E, os.path.join(a.outdir, "04_token.mp4"), a.fps)


if __name__ == "__main__":
    main()
