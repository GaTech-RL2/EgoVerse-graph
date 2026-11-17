"""Render exact level schematics alongside the completed coverage comparison."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT = ROOT.parent
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
FAMILIES = {
    "no_obstacles": "No obstacles", "edge_wall": "Edge wall", "edge_l": "Edge L",
    "wide_gate": "Wide gate", "floating_l": "Floating L",
    "middle_diagonal": "Diagonal stick", "middle_axis": "Axis-aligned stick",
    "diagonal_gate": "Diagonal gate", "corner_square_gate": "Corner-post gate",
    "t_obstacle": "Centered T",
}


def draw_layout(ax, design, radius):
    ax.set_xlim(-8, 520)
    ax.set_ylim(520, -8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 512, 512, facecolor="#f5f7fa", edgecolor="#bac3ce", lw=.8))
    for start, end in design["segments"]:
        a, b = np.asarray(start, float), np.asarray(end, float)
        normal = np.array([-(b-a)[1], (b-a)[0]]) / np.linalg.norm(b-a) * radius
        ax.add_patch(Polygon([a+normal, b+normal, b-normal, a-normal], color="#26394b", lw=0, snap=False, antialiased=True))
        for center in (a, b):
            ax.add_patch(Circle(center, radius, color="#26394b", lw=0, snap=False, antialiased=True))


def render(designs, coverage, delta, suffix, title):
    n = len(designs)
    fig = plt.figure(figsize=(14, 2.7 + n * .46), facecolor="white")
    height = fig.get_figheight()
    bottom, top = 1.05 / height, 1 - 1.6 / height
    body = top-bottom
    row_h = body/n
    ax = fig.add_axes([.30, bottom, .42, body])
    dax = fig.add_axes([.79, bottom, .17, body])
    absolute_norm = Normalize(0, 100)
    delta_norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    assert np.abs(delta).max() <= 30
    absolute_map = plt.get_cmap("YlGnBu")
    delta_map = plt.get_cmap("RdBu")
    im = ax.imshow(coverage, aspect="auto", cmap=absolute_map, norm=absolute_norm,
                   interpolation="nearest", origin="upper")
    dim = dax.imshow(delta, aspect="auto", cmap=delta_map, norm=delta_norm,
                     interpolation="nearest", origin="upper")
    for axes, matrix, norm, cmap in ((ax, coverage, absolute_norm, absolute_map),
                                    (dax, delta, delta_norm, delta_map)):
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values(): spine.set_visible(False)
        for i in range(n):
            for j in range(matrix.shape[1]):
                value = float(matrix[i, j])
                rgb = cmap(norm(value))[:3]
                luminance = .2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]
                text = ("0.0" if abs(value) < .05 else f"{value:+.1f}") if axes is dax else f"{value:.1f}"
                axes.text(j, i, text, ha="center", va="center", fontsize=10,
                          color="white" if luminance < .48 else "#17222f")
        axes.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
        axes.set_yticks(np.arange(-.5, n, 1), minor=True)
        axes.grid(which="minor", color="white", linewidth=1.0)
        axes.tick_params(which="minor", bottom=False, left=False)
    ax.axvline(1.5, color="white", lw=6)
    for i, design in enumerate(designs):
        center = top-(i+.5)*row_h
        fig.text(.03, center, f"L{design['level']:02}", va="center", fontsize=10,
                 fontweight="bold" if design['level']==0 else "normal", color="#17222f")
        thumb_h = row_h * .87
        thumb_w = thumb_h * height / fig.get_figwidth()
        thumb = fig.add_axes([.083, center-thumb_h/2, thumb_w, thumb_h])
        draw_layout(thumb, design, 4)
        fig.text(.137, center, FAMILIES[design["family"]], va="center", fontsize=9, color="#475467")
        if i < n-1 and design['family'] != designs[i+1]['family']:
            y = top-(i+1)*row_h
            fig.add_artist(plt.Line2D([.025, .963], [y,y], transform=fig.transFigure,
                                     color="#9aa9b8", lw=1.4 if design['level']==0 else .65))
    fig.text(.03, 1-.42/height, title, fontsize=21, fontweight="bold", color="#17222f")
    fig.text(.03, 1-.73/height,
             "Triangle holdout  |  Mean peak coverage (%)  |  3 training seeds x 50 resets per level and model",
             fontsize=10, color="#475467")
    fig.text(.03, top+.23/height, "Level", fontsize=10, fontweight="bold")
    fig.text(.083, top+.23/height, "Layout", fontsize=10, fontweight="bold")
    fig.text(.137, top+.23/height, "Level design", fontsize=10, fontweight="bold")
    for x, label in ((.405,"Clean-only training"),(.615,"Obstacle co-training"),(.875,"ARC minus DP")):
        fig.text(x, top+.62/height, label, ha="center", fontsize=11, fontweight="bold", color="#17222f")
    for j, label in enumerate(("DP16", "ARC abs duration", "DP16", "ARC abs duration")):
        fig.text(.30+(j+.5)*.42/4, top+.23/height, label, ha="center", fontsize=9)
    for j, label in enumerate(("Clean only", "Co-training")):
        fig.text(.79+(j+.5)*.17/2, top+.23/height, label, ha="center", fontsize=9)
    cb = fig.colorbar(im, cax=fig.add_axes([.30, .66/height, .42, .12/height]), orientation="horizontal")
    cb.set_ticks([0,20,40,60,80,100]); cb.ax.tick_params(labelsize=8, length=2)
    cb.set_label("Mean peak coverage (%) — higher is better", fontsize=9, labelpad=2)
    dcb = fig.colorbar(dim, cax=fig.add_axes([.79, .66/height, .17, .12/height]), orientation="horizontal")
    dcb.set_ticks([-30,0,30]);dcb.ax.tick_params(labelsize=8,length=2)
    dcb.set_label("Difference (percentage points)", fontsize=8, labelpad=2)
    fig.text(.03, .13/height,
             "Layouts use exact frozen obstacle geometry; object and goal poses vary across resets. L0 has no obstacles. Coverage is not SR@80.",
             fontsize=8, color="#475467")
    filename = ROOT / f"coverage-by-level-and-design{suffix}"
    fig.savefig(filename.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(filename.with_suffix(".pdf"), facecolor="white", metadata={"Title":title, "Creator":"Matplotlib"})
    if not suffix:
        fig.savefig(ROOT / "coverage-preview.png", dpi=105, facecolor="white")
    plt.close(fig)


def main():
    data = json.loads((ROOT / "level-designs.json").read_text())
    designs = data["levels"]
    assert [r["level"] for r in designs] == list(range(31))
    rows = list(csv.DictReader((REPORT / "per_level_comparison.csv").open()))
    lookup = {(r["training_regime"], int(r["level"])): r for r in rows}
    assert len(lookup)==62
    coverage = np.array([[float(lookup[regime,level][f"{model}_peak_coverage_pct"])
                         for regime in ("clean","cotrain") for model in ("dp16","arc_abs_duration")]
                        for level in range(31)])
    delta = coverage[:, [1,3]] - coverage[:, [0,2]]
    assert np.isfinite(coverage).all() and np.all((coverage>=0)&(coverage<=100))
    render(designs, coverage, delta, "", "Pusher: level design vs coverage")
    for start, end in ((0,15),(15,31)):
        render(designs[start:end], coverage[start:end], delta[start:end], f"-L{start:02}-L{end-1:02}",
               f"Pusher: level design vs coverage  |  L{start:02}–L{end-1:02}")
    manifest = {"metric":"mean peak coverage (%)", "episodes_per_cell":150,
                "training_seeds":[42,43,44], "levels":list(range(31)),
                "absolute_color_limits":[0,100], "difference_color_limits_pp":[-30,30],
                "geometry":"Exact obstacle endpoints and wall radius from the pinned simulator and frozen evaluation seed bank; poses omitted.",
                "plot_dependencies":{"matplotlib":matplotlib.__version__,"numpy":np.__version__},
                "input_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT/"level-designs.json",REPORT/"per_level_comparison.csv")}}
    (ROOT/"figure-provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Rendered all 31 verified level designs and 124 coverage cells, plus 62 paired differences.")


if __name__ == "__main__":
    main()
