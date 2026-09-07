#!/usr/bin/env python3
"""Follow-ups to the sg x action-velocity 2x2 (rec 100 only).

(Run 2026-09-06 as ICE array 5714392 at a861d4a with the pre-merge key names
detach_flow_target / detach_action_velocity_target, mapped here onto
clean_gradient_mode / action_velocity_clean_gradient_mode "all_stopgrad" | "full".
Results: docs/experiments/action-flow-stop-gradient.md.)
  exp1: which generative term attaches to the encoder — flow-sg + AV-attached
        (Elmo's described configuration) and its mirror;
  exp2: UNITE noise-augmented reconstruction (t~U[0.7,1], p=0.5) for E and G,
        attached and stop-gradient.
Same model, data, budget and 2048-particle eval contract as the 2x2."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

BASE_MODEL = {"latent_dim": 8, "adapter_family": "nonlinear", "residual_width": 32,
              "residual_depth": 2, "field_width": 128, "field_depth": 4}

def base(objective, rec=100.0):
    return {"architecture": "action_adapter_flow", "source_key": "source_gaussian_latent",
            "adapter_objective": objective, "lambda_reconstruction": float(rec), "lambda_scale": 1.0,
            "lambda_path": 0.0, "lambda_action_velocity": 1.0 if objective == "action_velocity" else 0.0,
            "model": dict(BASE_MODEL)}

VARIANTS = {
    # exp1
    "g-rec100-flowsg-avattached": {**base("action_velocity"), "clean_gradient_mode": "all_stopgrad", "action_velocity_clean_gradient_mode": "full"},
    "g-rec100-flowattached-avsg": {**base("action_velocity"), "clean_gradient_mode": "full", "action_velocity_clean_gradient_mode": "all_stopgrad"},
    # exp2
    "e-rec100-attached-naug": {**base("reconstruction"), "clean_gradient_mode": "full", "reconstruction_noise_aug": True},
    "e-rec100-sg-naug": {**base("reconstruction"), "clean_gradient_mode": "all_stopgrad", "reconstruction_noise_aug": True},
    "g-rec100-attached-naug": {**base("action_velocity"), "clean_gradient_mode": "full", "reconstruction_noise_aug": True},
    "g-rec100-sg-naug": {**base("action_velocity"), "clean_gradient_mode": "all_stopgrad", "reconstruction_noise_aug": True},
}

def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as s:
        for chunk in iter(lambda: s.read(1 << 20), b""): d.update(chunk)
    return d.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--experiment-root", type=Path, required=True)
    ap.add_argument("--training-dataset", type=Path, required=True)
    ap.add_argument("--evaluation-dataset", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-steps", type=int, default=60_000)
    ap.add_argument("--evaluation-particles", type=int, default=2048)
    ap.add_argument("--run-suffix", default="")
    a = ap.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    if a.run_suffix and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", a.run_suffix): raise ValueError("bad suffix")
    for p in (a.training_dataset, a.evaluation_dataset):
        if not p.is_file(): raise FileNotFoundError(p)
    a.output_dir.mkdir(parents=True)
    suffix = f"-{a.run_suffix}" if a.run_suffix else ""
    written = []
    for variant, ov in VARIANTS.items():
        for seed in a.seeds:
            run_id = f"sgfu-{variant}-seed{seed}{suffix}"
            cfg = {"variant": variant, "seed": seed, "dataset": str(a.training_dataset),
                   "evaluation_dataset": str(a.evaluation_dataset),
                   "output_dir": str(a.experiment_root / "runs" / run_id),
                   "flow_samples": 14, "learning_rate": 3e-4, "batch_size": 512, "max_steps": a.max_steps,
                   "inference_steps": 32, "evaluation_particles": a.evaluation_particles,
                   "diagnostic_noise_samples": 4096, "angular_bins": 16, "torus_major_radius": 2.0,
                   "torus_minor_radius": 0.65, "log_every": 100, "checkpoint_every": a.max_steps,
                   "wandb": {"entity": "rl2-group", "project": "synthetic-action-flow-affine",
                             "group": "aidan-sg-followup", "id": run_id, "name": run_id, "resume": "never",
                             "mode": "offline", "dir": str(a.experiment_root / "wandb")},
                   **ov}
            p = a.output_dir / f"{run_id}.json"; p.write_text(json.dumps(cfg, indent=2) + "\n"); written.append(str(p))
    manifest = {"schema_version": 1, "variants": list(VARIANTS), "seeds": a.seeds,
                "primary_metric": "validation_generation_symmetric_nn_mse",
                "comparison_contract": f"identical frozen seed-4242 cloud, first {a.evaluation_particles} validation particles; mean and sample std over seeds",
                "pass_threshold": None, "checkpoint_every": a.max_steps, "run_suffix": a.run_suffix,
                "datasets": {"training": {"path": str(a.training_dataset.resolve()), "sha256": sha256(a.training_dataset)},
                             "evaluation": {"path": str(a.evaluation_dataset.resolve()), "sha256": sha256(a.evaluation_dataset)}},
                "configs": written}
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (a.output_dir / "configs.txt").write_text("\n".join(written) + "\n")
    print(f"wrote {len(written)} configs to {a.output_dir}")

if __name__ == "__main__":
    main()
