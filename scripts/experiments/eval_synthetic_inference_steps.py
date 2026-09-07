#!/usr/bin/env python3
"""Re-evaluate trained synthetic checkpoints at several Euler step counts (no retraining).

For every config in a manifest whose variant is selected, load the terminal
checkpoint, integrate the frozen evaluation particles with each step count and
report symmetric NN MSE, energy distance and torus-surface RMSE, mean +- sample
std over seeds."""
# ruff: noqa: E402
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from egomimic.eval.synthetic_trajectory_eval import SyntheticTrajectoryEval
from egomimic.synthetic.action_adapter_flow import SyntheticActionAdapterFlow
from egomimic.synthetic.shared_latent_flow import SyntheticDirectFlow, SyntheticSharedLatentFlow


def energy_distance(s, t):
    return 2.0 * torch.cdist(s, t).mean() - torch.cdist(s, s).mean() - torch.cdist(t, t).mean()


def build(config):
    arch = config.get("architecture", "shared_latent")
    if arch == "shared_latent": return SyntheticSharedLatentFlow(**config["model"])
    if arch == "direct_flow": return SyntheticDirectFlow(**config["model"])
    if arch == "action_adapter_flow": return SyntheticActionAdapterFlow(**config["model"])
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[32, 16, 8, 4, 2, 1])
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-markdown", type=Path, required=True)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    rows = {}
    for cpath in man["configs"]:
        cfg = json.loads(Path(cpath).read_text())
        if cfg["variant"] not in a.variants: continue
        ckpts = sorted(glob.glob(str(Path(cfg["output_dir"]) / "checkpoints" / "*.pt")))
        if not ckpts: raise FileNotFoundError(cfg["output_dir"])
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        model = build(state["config"]); model.load_state_dict(state["model"], strict=True); model.eval()
        src, tgt = SyntheticTrajectoryEval.load_validation_data(
            cfg["evaluation_dataset"], cfg.get("source_key", "source_2d"), int(cfg["evaluation_particles"]))
        for steps in a.steps:
            with torch.inference_mode():
                gen = model.trajectory(src, steps=steps)[-1]
            rec = rows.setdefault((cfg["variant"], steps), {"nn": [], "ed": [], "surf": []})
            rec["nn"].append(float(SyntheticTrajectoryEval.symmetric_nearest_neighbor_mse(gen, tgt)))
            rec["ed"].append(float(energy_distance(gen, tgt)))
            rec["surf"].append(float(SyntheticTrajectoryEval.torus_surface_rmse(
                gen, major_radius=float(cfg.get("torus_major_radius", 2.0)), minor_radius=float(cfg.get("torus_minor_radius", 0.65)))))
    def ms(v):
        v = np.asarray(v, float); return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"
    lines = ["| variant | steps | NN-MSE | ED | surface RMSE | n |", "|---|---:|---:|---:|---:|---:|"]
    out = []
    for (variant, steps), rec in sorted(rows.items(), key=lambda kv: (a.variants.index(kv[0][0]), -kv[0][1])):
        lines.append(f"| {variant} | {steps} | {ms(rec['nn'])} | {ms(rec['ed'])} | {ms(rec['surf'])} | {len(rec['nn'])} |")
        out.append({"variant": variant, "steps": steps, **{k: v for k, v in rec.items()}})
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, indent=1) + "\n")
    a.output_markdown.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
