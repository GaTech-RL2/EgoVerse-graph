"""Calibrate action representations on held-out control traces and physics.

This measures reconstruction, not learned-policy success. Selection never sees
the benchmark's evaluation reset bank or trained-model scores.
"""
from __future__ import annotations

import argparse
import itertools
import json
import hashlib
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from egomimic.rldb.goal_replay import sha256_file


def _restore(env, raw, index, seed):
    # Some OGBench task/goal choices use NumPy's global RNG as well as np_random.
    np.random.seed(seed)
    env.reset(seed=seed)
    base = env.unwrapped
    if "button_states" in raw:
        base.set_state(raw["qpos"][index].copy(), raw["qvel"][index].copy(),
                       raw["button_states"][index].copy())
    else:
        base.set_state(raw["qpos"][index].copy(), raw["qvel"][index].copy())


def physics_replay(env, raw, indices, original, decoded, lengths, preview_dir=None):
    rows = []
    for i, idx in enumerate(indices):
        states, frames = [], []
        for actions in (original[i], decoded[i]):
            _restore(env, raw, int(idx), 1700000 + i)
            render = []
            for action in actions[:int(lengths[i])]:
                env.step(action)
                if preview_dir and i < 3:
                    render.append(env.render())
            base = env.unwrapped
            qpos = (base._data.qpos if hasattr(base, "_data") else base.data.qpos).copy()
            states.append(qpos)
            frames.append(render)
        row = {"index": int(idx), "native_steps": int(lengths[i]),
               "qpos_rmse": float(np.sqrt(np.mean((states[0] - states[1]) ** 2))),
               "native_recorded_qpos_rmse": float(np.sqrt(np.mean((states[0] - raw["qpos"][idx + int(lengths[i])]) ** 2)))}
        rows.append(row)
        if preview_dir and i < 3:
            import imageio.v2 as imageio
            path = Path(preview_dir)
            path.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(path / f"replay-{i}.mp4",
                [np.concatenate([a, b], axis=1) for a, b in zip(*frames)],
                fps=env.metadata.get("render_fps", 30))
    return {"rows": rows,
            "qpos_rmse_p90": float(np.percentile([r["qpos_rmse"] for r in rows], 90)),
            "native_recorded_qpos_rmse_p90": float(np.percentile([r["native_recorded_qpos_rmse"] for r in rows], 90))}


def calibrate(cfg):
    import ogbench
    torch.set_num_threads(1)
    raw = dict(np.load(cfg.validation_path))
    final = np.flatnonzero(raw["terminals"])
    starts = np.r_[0, final[:-1] + 1]
    max_horizon = max(cfg.grid.native_horizon)
    valid = np.concatenate([np.arange(start, end - max_horizon + 1) for start, end in zip(starts, final)])
    if not len(valid):
        raise ValueError("validation file has no complete calibration windows")
    rng = np.random.RandomState(cfg.seed)
    indices = rng.choice(valid, size=min(cfg.windows, len(valid)), replace=False)
    actions = torch.from_numpy(raw["actions"][indices[:, None] + np.arange(max_horizon)].astype(np.float32))
    cfg.codec.action_dim = int(actions.shape[-1])
    candidates = []
    grid = OmegaConf.to_container(cfg.grid)
    for values in itertools.product(*(grid[key] for key in grid)):
        choice = dict(zip(grid, values))
        spec = OmegaConf.merge(cfg.codec, choice, {"kind": "arc", "action_dim": actions.shape[-1]})
        codec = hydra.utils.instantiate(spec)
        latent, lengths = codec.encode(actions)
        recovered, decoded_lengths = codec.decode(latent)
        if not torch.equal(lengths, decoded_lengths):
            raise AssertionError("round-trip changed native execution duration")
        mask = torch.arange(codec.native_horizon)[None] < lengths[:, None]
        error = recovered - actions[:, :codec.native_horizon]
        rms = ((error.square().mean(-1) * mask).sum(1) / lengths).sqrt()
        compression = float(lengths.float().mean() * actions.shape[-1] / codec.encoded_dim)
        row = {"codec": OmegaConf.to_container(spec, resolve=True),
               "action_rmse_mean": float(rms.mean()), "action_rmse_p90": float(torch.quantile(rms, .9)),
               "mean_native_steps": float(lengths.float().mean()),
               "native_steps_p10": float(torch.quantile(lengths.float(), .1)),
               "scalar_compression_ratio": compression,
               "roundtrip_length_errors": 0}
        row["passes_trace_gates"] = (row["action_rmse_p90"] <= cfg.gates.action_rmse_p90
            and row["mean_native_steps"] >= cfg.gates.mean_native_steps
            and compression >= cfg.gates.scalar_compression_ratio)
        candidates.append(row)
    eligible = sorted([r for r in candidates if r["passes_trace_gates"]],
                      key=lambda r: (-r["scalar_compression_ratio"], r["action_rmse_p90"]))
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = ogbench.make_env_and_datasets(cfg.env_name, env_only=True, terminate_at_goal=False)
    selected = None
    physics_cache = {}
    try:
        for row in eligible:
            codec = hydra.utils.instantiate(row["codec"])
            count = min(cfg.physics_windows, len(actions))
            latent, lengths = codec.encode(actions[:count])
            recovered, _ = codec.decode(latent)
            fingerprint = hashlib.sha256(lengths.numpy().tobytes())
            for index, length in enumerate(lengths):
                fingerprint.update(np.round(recovered[index, :int(length)].numpy(), 5).tobytes())
            key = fingerprint.hexdigest()
            if key not in physics_cache:
                if len(physics_cache) >= cfg.physics_candidates:
                    break
                physics_cache[key] = physics_replay(env, raw, indices[:count], actions[:count].numpy(),
                                                   recovered.numpy(), lengths.numpy())
            physics = physics_cache[key]
            row["physics"] = physics
            if physics["qpos_rmse_p90"] <= cfg.gates.qpos_rmse_p90:
                selected = row
                physics_replay(env, raw, indices[:3], actions[:3].numpy(),
                               recovered[:3].numpy(), lengths[:3].numpy(), output / "previews")
                break
    finally:
        env.close()
    report = {"status": "PASS" if selected else "NO_CANDIDATE_PASSED",
              "compression_achieved": bool(selected and selected["scalar_compression_ratio"] > 1),
              "env_name": cfg.env_name, "source_validation_sha256": sha256_file(cfg.validation_path),
              "calibration_config": OmegaConf.to_container(cfg, resolve=True),
              "sample_indices": indices.tolist(), "selected": selected, "candidates": candidates,
              "selection_uses_policy_evaluation": False, "canonical_policy_score": False}
    (output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "candidates": len(candidates),
                      "eligible": len(eligible), "selected": selected}), flush=True)
    if selected is None:
        raise RuntimeError("ARC calibration gates failed; investigate before training")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    calibrate(OmegaConf.load(args.config))
