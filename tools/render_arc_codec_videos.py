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
from matplotlib.patches import Polygon as MplPoly, Rectangle

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
    fig = plt.figure(figsize=(16, 9), dpi=120)
    chrome(fig, "Two arc clocks, not one",
           "translation accumulates |Δxy|, rotation accumulates |Δθ| — separate sums, separate budgets")
    ax = fig.add_axes([.05, .08, .42, .78])
    a1 = fig.add_axes([.55, .53, .41, .32])
    a2 = fig.add_axes([.55, .10, .41, .32])
    t = np.arange(W0) * DT
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    box = view_box(E)
    i_tr = int(np.argmax(tr >= tr_end)) if tr[-1] >= tr_end else W0 - 1
    i_an = int(np.argmax(an >= an_end)) if an[-1] >= an_end else W0 - 1
    with w.saving(fig, out, dpi=120):
        hold(w, fps // 2)
        for i in range(W0):
            scene(ax, E, i, box=box)
            for a, cum, end, col, unit, lab, icut in (
                (a1, tr, tr_end, TR, "px", "translation arc  Σ|Δxy|", i_tr),
                (a2, np.degrees(an), math.degrees(an_end), ROT, "deg",
                 "rotation arc  Σ|Δθ|", i_an)):
                a.clear(); a.set_facecolor(PANEL)
                for sp in a.spines.values(): sp.set_color(GRID)
                a.grid(color=GRID, lw=.6, alpha=.5)
                a.plot(t, cum, color=col, lw=1.0, alpha=.2)
                a.plot(t[:i + 1], cum[:i + 1], color=col, lw=3.0)
                a.axhline(end, color="#f85149", ls="--", lw=2.0)
                a.text(t[-1], end, f"  budget = {end:.0f} {unit}", color="#f85149",
                       va="bottom", ha="right", fontsize=13, fontweight="bold")
                a.set_xlim(0, t[-1]); a.set_ylim(0, max(cum[-1], end) * 1.12)
                a.set_ylabel(f"{lab}  ({unit})", fontsize=12.5)
                a.set_xlabel("time (s)", fontsize=12)
                if i >= icut:
                    a.axvline(t[icut], color="#f85149", lw=1.4, alpha=.65)
                    a.text(t[icut], max(cum[-1], end) * 1.02, f" saturates at frame {icut}",
                           color="#f85149", fontsize=12.5, va="top")
            grab(w, 4)
        # caption the punchline and hold
        msg = fig.text(.5, .022,
                       f"Both budgets fill at DIFFERENT frames ({i_tr} and {i_an}) "
                       f"— the token describes a prefix of the chunk",
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
    fig = plt.figure(figsize=(16, 9), dpi=120)
    chrome(fig, "Even in arc length is uneven in time",
           f"sample each clock at M={M} equal steps — waypoints crowd where the motion is slow")
    ax = fig.add_axes([.05, .08, .42, .78])
    a1 = fig.add_axes([.55, .50, .41, .35])
    a2 = fig.add_axes([.55, .10, .41, .30])
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    tmax = max(t_tr[-1], t_an[-1])
    with w.saving(fig, out, dpi=120):
        hold(w, fps // 2)
        for k in range(1, M + 1):
            ax.clear(); ax.set_facecolor(PANEL); ax.set_aspect("equal")
            bx = view_box(E)
            ax.set_xlim(bx[0], bx[1]); ax.set_ylim(bx[2], bx[3])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(GRID)
            gx, gy, gth = E["goal"]
            for poly in rects_world(T_RECTS, gx, gy, gth):
                ax.add_patch(MplPoly(poly, closed=True, facecolor="none", edgecolor=GOAL,
                                     lw=2.0, ls=(0, (5, 4)), alpha=.7))
            ox, oy, oth = E["obj"][0]
            for poly in rects_world(T_RECTS, ox, oy, oth):
                ax.add_patch(MplPoly(poly, closed=True, facecolor=OBJ, alpha=.55,
                                     edgecolor="#f2cc60", lw=1.4))
            ax.plot(E["act"][:, 0], E["act"][:, 1], color="#2d3748", lw=2.2)
            ax.scatter(tok_v[:k, 0], tok_v[:k, 1], s=46, color=TR, zorder=5,
                       edgecolors="#0d1117", linewidths=.8)
            ax.text(.02, .975, f"waypoint {k}/{M}   — equally spaced along the PATH",
                    transform=ax.transAxes, va="top", fontsize=13, color=TR)
            ax.text(.02, .935,
                    f"D={D:.0f} px covers {100*tr_end/max(tr[-1],1e-9):.0f}% of this chunk",
                    transform=ax.transAxes, va="top", fontsize=12, color=DIM)

            a1.clear(); a1.set_facecolor(PANEL)
            for sp in a1.spines.values(): sp.set_color(GRID)
            a1.grid(color=GRID, lw=.6, alpha=.5)
            a1.plot(np.arange(M), t_tr, color=TR, lw=1.0, alpha=.2)
            a1.plot(np.arange(M), t_an, color=ROT, lw=1.0, alpha=.2)
            a1.plot(np.arange(k), t_tr[:k], color=TR, lw=3.0, label="translation")
            a1.plot(np.arange(k), t_an[:k], color=ROT, lw=3.0, label="rotation")
            a1.set_xlim(0, M - 1); a1.set_ylim(0, tmax * 1.08)
            a1.set_xlabel("waypoint index  i", fontsize=12.5)
            a1.set_ylabel("source time (s)", fontsize=12.5)
            a1.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, fontsize=12.5)
            a1.text(.99, .05, "same row i, different times", transform=a1.transAxes,
                    ha="right", fontsize=12.5, color=DIM)

            a2.clear(); a2.set_facecolor(PANEL)
            for sp in a2.spines.values(): sp.set_color(GRID)
            a2.set_xlim(0, tmax * 1.02); a2.set_ylim(0, 1); a2.set_yticks([])
            a2.set_xlabel("source time (s)", fontsize=12.5)
            a2.vlines(t_tr[:k], .55, .95, color=TR, lw=1.8)
            a2.vlines(t_an[:k], .05, .45, color=ROT, lw=1.8)
            a2.text(.005, .74, " translation", transform=a2.transAxes, color=TR, fontsize=12.5,
                    va="center")
            a2.text(.005, .24, " rotation", transform=a2.transAxes, color=ROT, fontsize=12.5,
                    va="center")
            grab(w, 5)
        hold(w, fps * 3)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------- clip 4
def clip_token(E, out, fps):
    tok_v, tok_d = tokenize(E["act"])
    dv = USocketArcLocalVelocityStackedNativeDecoder(M, H, native_action_dim=3, dt=DT)
    dd = USocketArcDurationNativeDecoder(M, H, native_action_dim=3, dt=DT)
    rv = dv.decode(torch.as_tensor(tok_v, dtype=torch.float32).unsqueeze(0))[0].numpy()
    rd = dd.decode(torch.as_tensor(tok_d, dtype=torch.float32).unsqueeze(0))[0].numpy()
    ref = E["act"][:H]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    sub = chrome(fig, "Velocity token vs duration token",
                 "identical geometry in columns 0,1,3,4 — only the two timing columns differ")
    aL = fig.add_axes([.05, .10, .41, .74])
    aR = fig.add_axes([.54, .10, .42, .74])
    idx = np.arange(M)
    w = FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                     extra_args=["-pix_fmt", "yuv420p", "-preset", "slow"])
    n_build, n_dec = M, H
    with w.saving(fig, out, dpi=120):
        hold(w, fps // 2)
        for k in range(1, n_build + 1):
            for a, (c2, c5), cols, names, unit in (
                (aL, (tok_v[:, 2], tok_v[:, 5]), (VEL, "#d2a8ff"),
                 ("v_xy   (px/s)", "ω   (rad/s, signed)"), "velocity token   [x, y, v_xy, cos, sin, ω]"),
                (aR, (tok_d[:, 2], tok_d[:, 5]), (DUR, "#7ee787"),
                 ("Δt translation (s)", "Δt rotation (s)"),
                 "duration token   [x, y, Δt_tr, cos, sin, Δt_rot]")):
                a.clear(); a.set_facecolor(PANEL)
                for sp in a.spines.values(): sp.set_color(GRID)
                a.grid(color=GRID, lw=.6, alpha=.5)
                sc = 20 if a is aL else 1
                a.plot(idx, c2, color=cols[0], lw=1.0, alpha=.2)
                a.plot(idx, c5 * sc, color=cols[1], lw=1.0, alpha=.2)
                a.plot(idx[:k], c2[:k], color=cols[0], lw=3.0, label=names[0])
                a.plot(idx[:k], c5[:k] * sc, color=cols[1], lw=2.4,
                       label=names[1] + (f"  ×{sc}" if sc != 1 else ""))
                a.set_xlim(0, M - 1)
                lo = min(c2.min(), (c5 * sc).min()); hi = max(c2.max(), (c5 * sc).max())
                a.set_ylim(lo - abs(hi - lo) * .12, hi + abs(hi - lo) * .22)
                a.set_xlabel("token row  i", fontsize=12.5)
                a.set_title(unit, fontsize=14, color=INK, pad=10)
                a.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, fontsize=12)
            aR.text(.02, .04, "both timing channels in the SAME unit: seconds",
                    transform=aR.transAxes, fontsize=12.5, color=DUR)
            aL.text(.02, .04, "two different units; decode divides the rate back out",
                    transform=aL.transAxes, fontsize=12.5, color=VEL)
            grab(w, 4)
        hold(w, fps)

        # second act: decode both back onto the source path.
        # sub.remove() first -- drawing a second string at the same coordinates
        # renders both on top of each other.
        sub.remove()
        cap = fig.text(.035, .898,
                       "both decode back to the same trajectory — the codec is not what limits the policy",
                       fontsize=15.5, color="#c9d1d9", va="top")
        for k in range(2, n_dec + 1):
            for a, rec, col, name in ((aL, rv, VEL, "velocity decode"),
                                      (aR, rd, DUR, "duration decode")):
                a.clear(); a.set_facecolor(PANEL); a.set_aspect("equal")
                for sp in a.spines.values(): sp.set_color(GRID)
                a.grid(color=GRID, lw=.6, alpha=.5)
                a.plot(ref[:, 0], ref[:, 1], color=DIM, lw=6, alpha=.55, label="source")
                a.plot(rec[:k, 0], rec[:k, 1], color=col, lw=2.8, label=name)
                a.scatter(ref[:k, 0], ref[:k, 1], s=22, color=DIM, zorder=4)
                err = np.abs(rec[:, :2] - ref[:, :2]).max()
                a.set_title(f"{name}   ·   max error {err:.3f} px", fontsize=14, color=INK, pad=10)
                a.legend(loc="best", facecolor=PANEL, edgecolor=GRID, fontsize=12)
                a.set_xlabel("x (px)", fontsize=12); a.set_ylabel("y (px)", fontsize=12)
            grab(w, 8)
        hold(w, fps * 3)
        cap.remove()
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
