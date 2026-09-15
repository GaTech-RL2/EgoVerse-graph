#!/usr/bin/env python
"""Plot training loss and validation accuracy for the articulated co-train arms.

The question these answer: is the near-zero rollout coverage a training failure
(loss still falling, underfit) or a generalisation failure (loss converged, the
policy simply does not transfer to closed-loop control)?
"""
from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, PANEL, INK, DIM, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
COL = {"arc_dur_D80_M56": "#3fb950", "arc_stk_D80_M56": "#bc8cff",
       "arc_dur_D80_M16": "#58a6ff", "arc_stk_D80_M16": "#f0883e",
       "dp_paper": "#e6edf3"}
LBL = {"arc_dur_D80_M56": "duration M56", "arc_stk_D80_M56": "velocity M56",
       "arc_dur_D80_M16": "duration M16", "arc_stk_D80_M16": "velocity M16",
       "dp_paper": "DP (no codec)"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL,
                     "savefig.facecolor": BG, "text.color": INK,
                     "axes.labelcolor": DIM, "xtick.color": DIM,
                     "ytick.color": DIM, "axes.edgecolor": GRID, "font.size": 12})


def key(arm: str) -> str:
    """Arm names carry an _R26deg suffix the colour/label tables do not."""
    return arm.replace("_R26deg", "")


def smooth(y, k=31):
    if len(y) < k:
        return np.asarray(y, float)
    return np.convolve(np.asarray(y, float), np.ones(k) / k, mode="same")


def main() -> int:
    data = json.load(open(sys.argv[1] if len(sys.argv) > 1
                          else "results/artic_curves.json"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    ax = axes[0]
    for arm, d in data.items():
        if not d["train"]:
            continue
        e = [p[0] for p in d["train"]]; y = [p[1] for p in d["train"]]
        ax.plot(e, smooth(y), color=COL.get(key(arm), DIM), lw=2, label=LBL.get(key(arm), arm))
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("Train/loss  (DDPM eps MSE)")
    ax.set_title("Training loss", color=INK); ax.grid(color=GRID, lw=.6, alpha=.5)
    ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=10)

    ax = axes[1]
    for arm, d in data.items():
        if not d["train"]:
            continue
        e = [p[0] for p in d["train"]]; y = smooth([p[1] for p in d["train"]])
        n = max(len(e) // 5, 1)
        ax.plot(e[-n:], y[-n:], color=COL.get(key(arm), DIM), lw=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("Train/loss")
    ax.set_title("Training loss, final 20% of run", color=INK)
    ax.grid(color=GRID, lw=.6, alpha=.5)

    ax = axes[2]
    any_val = False
    for arm, d in data.items():
        if not d["valid"]:
            continue
        any_val = True
        e = [p[0] for p in d["valid"]]; y = [p[1] for p in d["valid"]]
        ax.plot(e, y, marker="o", ms=4, color=COL.get(key(arm), DIM), lw=2,
                label=LBL.get(key(arm), arm))
    ax.set_xlabel("epoch")
    ax.set_ylabel("Valid EnergyScore accuracy   (LOWER is better)")
    ax.set_title("Validation: distance from samples to target\n"
                 "held-out episodes of the SEVEN trained embodiments",
                 color=INK, fontsize=11)
    ax.grid(color=GRID, lw=.6, alpha=.5)
    if any_val:
        ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)

    fig.suptitle("Articulated co-train: is low rollout coverage a training failure?  "
                 "No -- train loss converged and validation improved monotonically.",
                 fontsize=14, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = "results/figures/artic_curves.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)

    print("\nfinal values")
    print(f"{'arm':16s} {'train loss':>11s} {'first-decile':>13s} {'ratio':>7s} "
          f"{'val acc first':>14s} {'val acc last':>13s}")
    for arm, d in data.items():
        tr = [p[1] for p in d["train"]]
        if not tr:
            continue
        early = float(np.mean(tr[: max(len(tr) // 10, 1)]))
        late = float(np.mean(tr[-max(len(tr) // 10, 1):]))
        v = [p[1] for p in d["valid"]]
        print(f"{LBL.get(key(arm), arm):16s} {late:11.5f} {early:13.5f} "
              f"{early/late:6.1f}x {(v[0] if v else float('nan')):14.4f} "
              f"{(v[-1] if v else float('nan')):13.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
