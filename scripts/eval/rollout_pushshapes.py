#!/usr/bin/env python3
"""Closed-loop PushShapes (Sim V2) rollouts for chunked Pipeline policies.

Generalises egomimic/eval/core/ckpt_loading.py (arc waypoint-zero, replan every
step) to any Pipeline checkpoint whose inference graph emits ``pred_action``:
Paper-DP (obs horizon 2, execute 8 of 16), UNITE (register policy), direct /
arc BC. Protocol follows Elmo's DP evaluation doc: obstacle level, seeded
resets, 1,800 max env steps, peak object-goal coverage per episode,
SR@0.80 = fraction of episodes whose peak coverage >= 0.80.

ChainGripper (2026-09-08): ``--pusher chain_gripper`` with
``--chain-control-mode points`` feeds the six-point prediction straight to the
simulator's point mode (the env projects it onto the 4-DOF manifold with
orientation continuity, exactly as Elmo's adapter does), so no native decoding
happens on the policy side. Cotrain checkpoints carry per-embodiment decoders
under ``evaluator.native_decoders``; ``--sim-root`` selects the simulator tree
(the chain-capable sim at ``2133a92`` is not the one on the rollout branch).
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if os.environ.get("SIM_ROOT"):
    # Simulator tree (must contain ``Tsimulation/``); inserted after the repo so
    # egomimic still resolves from the checkout that owns this script.
    sys.path.insert(1, os.environ["SIM_ROOT"])

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.eval.core.ckpt_loading import _apply_stats, _checkpoint_config
from egomimic.pl_utils.pl_model import ModelWrapper

ORIENTED = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def load_stats(config, output_dir: Path, embodiment_id: int):
    """Find the normalisation artefact: precomputed path, else the run's cache."""
    candidates = []
    raw = OmegaConf.select(config, "norm_stats.precomputed_norm_path")
    if raw:
        candidates.append(Path(str(raw)))
    candidates += [output_dir / "norm_stats.json", output_dir / "norm_stats"]
    cache = OmegaConf.select(config, "norm_stats.save_cache_dir")
    if cache:
        candidates.append(Path(str(cache)))
    files = []
    for c in candidates:
        if c.is_dir():
            files += sorted(c.rglob("*.json"))
        elif c.is_file():
            files.append(c)
    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        stats = (payload.get("stats") or {}).get(str(int(embodiment_id)))
        if isinstance(stats, dict) and "actions" in stats and (
            "state_agent_obj" in stats or "state_agent_model" in stats
        ):
            mode = str(OmegaConf.select(config, "norm_stats.norm_mode", default="quantile"))
            return f, mode, stats
    raise RuntimeError(f"no normalisation artefact for embodiment {embodiment_id} under {candidates}")


def frame_state(obs: dict, oriented: bool) -> np.ndarray:
    parts = [obs["agent_pos"]]
    if oriented:
        parts.append(obs["agent_angle"])
    parts.append(obs["object_pose"])
    return np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])


class ChunkPolicy:
    def __init__(self, ckpt: Path, config_path: Path, output_dir: Path, embodiment_name: str,
                 embodiment_id: int, use_ema: bool, device: torch.device, raw_action_width: int | None = None):
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.ckpt_epoch, self.ckpt_global_step = checkpoint.get("epoch"), checkpoint.get("global_step")
        embedded, resolved = _checkpoint_config(checkpoint, str(config_path))
        # Inference is eager for every row: a run trained with torch.compile
        # (compile_backbones=true) is rolled out through the same uncompiled
        # graph as the others, so the evaluation path is identical across rows.
        self.trained_with_compile = False
        stages = OmegaConf.select(embedded, "model.pipeline.stages") or []
        for stage in stages:
            if isinstance(stage, DictConfig) and "compile_backbones" in stage:
                self.trained_with_compile = bool(stage.compile_backbones) or self.trained_with_compile
                with open_dict(stage):
                    stage.compile_backbones = False
        wrapper = ModelWrapper(config_tree=embedded)
        strict_load_pipeline_checkpoint(wrapper.model, checkpoint, use_ema=use_ema)
        wrapper.to(device)
        wrapper.eval()
        self.algo = wrapper.model
        self.algo.device = next(self.algo.nets.parameters()).device
        self.algo.nets.eval()
        self.algo.pipeline.eval()
        self.device = device
        self.embodiment_name = str(embodiment_name)
        self.embodiment_id = int(embodiment_id)
        self.oriented = self.embodiment_name in ORIENTED
        self.norm_path, self.norm_mode, self.stats = load_stats(resolved, output_dir, self.embodiment_id)
        decoder_cfg = None
        for key in (f"evaluator.native_decoders.{self.embodiment_name}", "planar.eval_native_decoder", "evaluator.native_decoder"):
            decoder_cfg = OmegaConf.select(resolved, key)
            if decoder_cfg is not None:
                break
        self.raw_action_width = raw_action_width
        if raw_action_width is not None:
            # Emit the model's action space untouched (e.g. six ChainGripper points
            # for the env's point mode); the decoder config is only recorded.
            self.decoder = None
            self.decoder_name = f"raw{raw_action_width}"
        else:
            if decoder_cfg is None:
                raise RuntimeError("no native decoder config under evaluator.native_decoders.<embodiment>, planar.eval_native_decoder or evaluator.native_decoder")
            self.decoder = instantiate(decoder_cfg)
            self.decoder_name = type(self.decoder).__name__
        obs_stages = [s for s in self.algo.pipeline.stages if hasattr(s, "n_obs_steps")]
        self.n_obs = int(obs_stages[0].n_obs_steps) if obs_stages else 1
        self.stage_names = [type(s).__name__ for s in self.algo.pipeline.stages]
        # Which batch keys does this graph read?  FusedObsEncoder.inputs maps
        # batch_key -> packed_key; KeyedFeatureProjection reads input_key.
        keys, produced = set(), set()
        for stage in self.algo.pipeline.stages:
            inputs = getattr(stage, "inputs", None)
            if isinstance(inputs, dict):
                keys.update(str(k) for k in inputs.keys())
            input_key = getattr(stage, "input_key", None)
            if isinstance(input_key, str):
                keys.add(input_key)
            for attr in ("writes", "outputs"):
                value = getattr(stage, attr, None)
                if isinstance(value, (list, tuple, set)):
                    produced.update(str(v) for v in value)
            output_key = getattr(stage, "output_key", None)
            if isinstance(output_key, str):
                produced.add(output_key)
        # keep only the graph's external inputs (not keys another stage writes)
        self.input_keys = sorted(keys - produced) or ["front_img_1", "state_agent_obj"]
        for key in self.input_keys:
            if key.startswith("state_") and key not in self.stats:
                raise RuntimeError(f"graph reads {key} but the normalisation artefact has no stats for it")
        self.horizon = None

    def _with_frame_axis(self, value: np.ndarray) -> torch.Tensor:
        """(n_obs, ...) -> (1, n_obs, ...) for multi-frame graphs, (1, ...) otherwise."""
        tensor = torch.from_numpy(np.ascontiguousarray(value)).float()
        if self.n_obs == 1:
            tensor = tensor[0]
        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_chunk(self, frames: list[dict]) -> np.ndarray:
        raw = np.stack([frame_state(f, self.oriented) for f in frames])             # (n_obs, D) = [x, y, th, ox, oy, oth]
        batch = {"embodiment": self.embodiment_name}
        for key in self.input_keys:
            if key.startswith("front_img") or key.endswith("_img") or "image" in key:
                image = np.stack([np.transpose(f["image"], (2, 0, 1)).astype(np.float32) / 255.0 for f in frames])
                batch[key] = self._with_frame_axis(image)                            # (1[, n_obs], 3, H, W)
            elif key == "state_agent_obj":
                batch[key] = _apply_stats(self._with_frame_axis(raw), self.stats[key], self.norm_mode, inverse=False)
            elif key == "state_agent_model":
                # PlanarAgentStateToRotVec4: agent [x, y, theta] -> [x, y, cos, sin]
                rot = np.concatenate([raw[:, :2], np.cos(raw[:, 2:3]), np.sin(raw[:, 2:3])], axis=1)
                batch[key] = _apply_stats(self._with_frame_axis(rot), self.stats[key], self.norm_mode, inverse=False)
            else:
                raise RuntimeError(f"unsupported graph input key {key}")
        out = self.algo.forward_eval({"rollout": batch})["rollout"]
        if "pred_action" not in out:
            raise RuntimeError(f"inference graph produced no pred_action; keys={list(out)}")
        pred = _apply_stats(out["pred_action"], self.stats["actions"], self.norm_mode, inverse=True)
        if self.decoder is None:
            if pred.shape[-1] != self.raw_action_width:
                raise RuntimeError(f"raw action width {pred.shape[-1]} != expected {self.raw_action_width}")
            native = pred
        else:
            native = self.decoder.decode(pred, context={})
        native = torch.as_tensor(native)
        if native.ndim != 3 or native.shape[0] != 1:
            raise RuntimeError(f"native decoder returned {tuple(native.shape)}")
        chunk = native[0].detach().float().cpu().numpy()
        self.horizon = chunk.shape[0]
        return chunk


class _VideoWriter:
    """MP4 from the env's world render; a coverage / step caption is burned in."""

    def __init__(self, path: Path, fps: int, every: int):
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.every = max(1, int(every))
        self.writer = imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=7, macro_block_size=1)

    def capture(self, env, step: int, coverage: float) -> None:
        if step % self.every:
            return
        frame = env.render()
        if frame is None:
            return
        frame = np.ascontiguousarray(frame)
        try:
            import cv2

            cv2.putText(frame, f"t={step:4d}  IoU={coverage:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"t={step:4d}  IoU={coverage:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        self.writer.append_data(frame)

    def close(self) -> None:
        try:
            self.writer.close()
        except Exception:
            pass


def run_episode(policy: ChunkPolicy, seed: int, args) -> dict:
    from Tsimulation.pushshapes import PushShapesEnv

    env_kwargs = dict(object_shape="T", pusher_shape=args.pusher, obstacle_level=args.obstacle_level,
                      image_size=args.image_size)
    if args.pusher == "chain_gripper":
        env_kwargs["chain_gripper_control_mode"] = args.chain_control_mode
    video = None
    if args.video_dir is not None:
        env_kwargs["render_mode"] = "rgb_array"
    env = PushShapesEnv(**env_kwargs)
    obs, reset_info = env.reset(seed=int(seed))
    if args.video_dir is not None:
        video = _VideoWriter(Path(args.video_dir) / f"seed_{int(seed):03d}.mp4", int(args.video_fps), int(args.video_every))
        video.capture(env, 0, float(reset_info.get("coverage", 0.0)))
    frames = deque([obs] * policy.n_obs, maxlen=policy.n_obs)
    peak, steps, coverages, predictions = float(reset_info.get("coverage", 0.0)), 0, [], 0
    proj_rmse, proj_wrong, proj_degenerate = [], 0, 0
    trace = [] if args.trace else None
    if args.trace:
        trace.append({"t": 0, "agent_pos": [float(v) for v in obs["agent_pos"]], "object_pose": [float(v) for v in obs["object_pose"]], "coverage": peak})
    terminated = False
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(args.episode_timeout))
    t0 = time.time()
    try:
        while steps < args.max_steps and not terminated:
            chunk = policy.predict_chunk(list(frames))
            predictions += 1
            start = int(getattr(args, "chunk_start", 0))
            if start + args.replan_every > len(chunk):
                raise RuntimeError(f"chunk of {len(chunk)} cannot execute [{start},{start + args.replan_every})")
            for k in range(start, start + args.replan_every):
                action = np.asarray(chunk[k], dtype=np.float64).reshape(-1)
                if not np.isfinite(action).all():
                    return {"seed": seed, "peak": 0.0, "steps": steps, "predictions": predictions,
                            "status": "nonfinite_action", "seconds": time.time() - t0}
                obs, _r, terminated, _tr, info = env.step(action)
                if getattr(args, "full_horizon", False):
                    terminated = False  # protocol: peak is measured over the whole budget
                cov = float(info.get("coverage", 0.0))
                if "point_projection_rmse" in info:
                    proj_rmse.append(float(info["point_projection_rmse"]))
                    proj_wrong += int(bool(info.get("point_wrong_chirality", False)))
                    proj_degenerate += int(bool(info.get("point_degenerate", False)))
                peak = max(peak, cov)
                coverages.append(cov)
                frames.append(obs)
                steps += 1
                if video is not None:
                    video.capture(env, steps, cov)
                if trace is not None:
                    trace.append({"t": steps, "action": [float(v) for v in action], "agent_pos": [float(v) for v in obs["agent_pos"]], "object_pose": [float(v) for v in obs["object_pose"]], "coverage": cov})
                if terminated or steps >= args.max_steps:
                    break
        status = "terminated" if terminated else "max_steps"
    except _Timeout:
        status = "timeout"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if video is not None:
            video.close()
        env.close()
    record = {"seed": seed, "peak": float(peak), "final": float(coverages[-1]) if coverages else 0.0,
              "steps": steps, "predictions": predictions, "status": status, "seconds": time.time() - t0}
    if proj_rmse:
        record["point_projection"] = {"mean_rmse": float(np.mean(proj_rmse)), "max_rmse": float(np.max(proj_rmse)),
                                      "wrong_chirality_steps": proj_wrong, "degenerate_steps": proj_degenerate}
    if trace is not None:
        record["trace"] = trace
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True, help="run dir (for cached norm stats)")
    ap.add_argument("--embodiment-name", default="pushshapes_sim_u_socket")
    ap.add_argument("--embodiment-id", type=int, default=19)
    ap.add_argument("--pusher", default="u_socket")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--replan-every", type=int, default=8)
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--obstacle-level", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--success-threshold", type=float, default=0.80)
    ap.add_argument("--episode-timeout", type=int, default=900)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trace", action="store_true", help="store per-step action / pose / coverage")
    ap.add_argument("--cfg-scale", type=float, default=None, help="override classifier-free-guidance scale on stages that have cfg_scale")
    ap.add_argument("--sampler-steps", type=int, default=None, help="override num_inference_steps on stages that have it")
    ap.add_argument("--cfg-interval", default=None, help="override cfg_interval as lo,hi (flow time range where guidance applies)")
    ap.add_argument("--video-dir", type=Path, default=None, help="write <video-dir>/seed_<k>.mp4 from the env's 512x512 world render")
    ap.add_argument("--video-fps", type=int, default=10, help="MP4 frame rate (Elmo's sim_v2 protocol: 10)")
    ap.add_argument("--video-every", type=int, default=3, help="capture one frame every N env steps (sim is 30 Hz; 3 at 10 FPS is real time)")
    ap.add_argument("--budget-json", type=Path, default=None,
                    help="protocol rev-3 budget file (episode_budget.py / derive_budget.py); sets max_steps = by_level[level].budget")
    ap.add_argument("--full-horizon", action="store_true", help="protocol: ignore the env's terminated flag and always run the full budget")
    ap.add_argument("--chunk-start", type=int, default=0, help="first decoded action index to execute (protocol default 0; registered post-step Paper-DP rows use 1)")
    ap.add_argument("--seeds", default=None, help="explicit comma-separated seed list (e.g. an OEC-56 level); overrides --seed-base/--n-episodes")
    ap.add_argument("--label", default=None, help="comparability label recorded in the result (e.g. CANONICAL_LEVEL0_SEEDS_0_39)")
    ap.add_argument("--chain-control-mode", default="points", choices=("pose", "points"),
                    help="chain_gripper only: 'points' feeds the 6-D prediction to the env's point mode (no policy-side IK); 'pose' decodes to [x, y, theta, grip] first")
    args = ap.parse_args(argv)
    raw_width = 6 if (args.pusher == "chain_gripper" and args.chain_control_mode == "points") else None
    budget_provenance = None
    if args.budget_json is not None:
        payload = json.loads(Path(args.budget_json).read_text())
        if payload.get("statistic") != "p99":
            raise RuntimeError(f"protocol horizon revision 3 requires statistic=p99, got {payload.get('statistic')!r}")
        row = (payload.get("by_level") or {}).get(str(int(args.obstacle_level)))
        if row is None or "budget" not in row:
            raise RuntimeError(f"budget file has no level {args.obstacle_level}; never fall back to a default")
        args.max_steps = int(row["budget"])
        budget_provenance = {
            "path": str(args.budget_json), "dataset_name": payload.get("dataset_name"), "statistic": payload["statistic"],
            "multiplier": payload.get("multiplier"), "content_sha256": payload.get("content_sha256"),
            "action_space": payload.get("action_space", "cursor"), "level": int(args.obstacle_level), "budget": int(row["budget"]),
            "derived": bool(row.get("derived", False)), "derivation": payload.get("derivation"),
        }
    if args.chunk_start < 0:
        raise ValueError("--chunk-start must be non-negative")
    if args.seeds:
        seed_list = [int(v) for v in args.seeds.split(",") if v.strip()]
    else:
        seed_list = [args.seed_base + ep for ep in range(args.n_episodes)]
    from Tsimulation.pushshapes import env as env_module
    assert getattr(env_module, "SIM_VERSION", 2) == 2, "expected Sim V2"
    device = torch.device(args.device)
    policy = ChunkPolicy(args.ckpt, args.config_path, args.output_dir, args.embodiment_name,
                         args.embodiment_id, args.use_ema, device, raw_action_width=raw_width)
    overrides = []
    for stage in policy.algo.pipeline.stages:
        if args.cfg_scale is not None and hasattr(stage, "cfg_scale"):
            overrides.append(f"{type(stage).__name__}.cfg_scale {stage.cfg_scale} -> {args.cfg_scale}"); stage.cfg_scale = float(args.cfg_scale)
        if args.cfg_interval is not None and hasattr(stage, "cfg_interval"):
            lo, hi = (float(v) for v in args.cfg_interval.split(","))
            overrides.append(f"{type(stage).__name__}.cfg_interval {stage.cfg_interval} -> {(lo, hi)}"); stage.cfg_interval = (lo, hi)
        if args.sampler_steps is not None and hasattr(stage, "num_inference_steps"):
            overrides.append(f"{type(stage).__name__}.num_inference_steps {stage.num_inference_steps} -> {args.sampler_steps}"); stage.num_inference_steps = int(args.sampler_steps)
    if (args.cfg_scale is not None or args.sampler_steps is not None or args.cfg_interval is not None) and not overrides:
        raise RuntimeError("requested an inference override but no stage exposes that attribute")
    print("[override]", overrides if overrides else "none", flush=True)
    print(f"[load] stages={policy.stage_names} n_obs={policy.n_obs} norm={policy.norm_path} "
          f"mode={policy.norm_mode} ema={args.use_ema} device={device} decoder={policy.decoder_name} "
          f"pusher={args.pusher} chain_mode={args.chain_control_mode if args.pusher == 'chain_gripper' else '-'} "
          f"sim={env_module.__file__}", flush=True)
    episodes = []
    for ep, seed in enumerate(seed_list):
        rec = run_episode(policy, seed, args)
        rec["episode"] = ep
        episodes.append(rec)
        print(f"[sim] ep{ep} seed={seed} peak={rec['peak']:.3f} steps={rec['steps']} "
              f"preds={rec['predictions']} {rec['status']} {rec['seconds']:.0f}s", flush=True)
    peaks = np.array([e["peak"] for e in episodes])
    sim_root = Path(str(env_module.__file__)).resolve().parents[3]
    import hashlib, subprocess
    def _git(*cmd):
        try:
            return subprocess.run(["git", "-C", str(sim_root), *cmd], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    sim_head = _git("rev-parse", "HEAD"); sim_dirty = bool(_git("status", "--porcelain=v1")) if sim_head else None
    origin_file = sim_root / "ORIGIN.txt"
    sim_origin = origin_file.read_text().strip() if origin_file.exists() else None
    pkg = Path(str(env_module.__file__)).parent
    sim_file_sha = {name: hashlib.sha256((pkg / name).read_bytes()).hexdigest() for name in ("env.py", "obstacles.py", "shapes.py") if (pkg / name).exists()}
    try:
        import subprocess
        driver_head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        driver_head = ""
    ckpt_epoch, ckpt_step = policy.ckpt_epoch, policy.ckpt_global_step
    sampler = {}
    for stage in policy.algo.pipeline.stages:
        for attr in ("num_inference_steps", "dopri5_output_points", "cfg_scale", "cfg_interval"):
            if hasattr(stage, attr):
                sampler[f"{type(stage).__name__}.{attr}"] = getattr(stage, attr) if not isinstance(getattr(stage, attr), tuple) else list(getattr(stage, attr))
    protocol = {
        "document": "PushShapes eval protocol sim_v2 (EVAL_PROTOCOL_2026-09-08), horizon revision 3" if budget_provenance else "rev-1 flat horizon (non-protocol)",
        "horizon_revision": 3 if budget_provenance else 1,
        "budget": budget_provenance, "max_steps": int(args.max_steps), "full_horizon": bool(args.full_horizon),
        "coverage_threshold_early_stop": None if args.full_horizon else "env terminated at 0.95",
        "init_mode": "seeds", "seeds": seed_list, "level": int(args.obstacle_level),
        "replan_every": int(args.replan_every), "chunk_start": int(args.chunk_start), "chunk_stop": int(args.chunk_start + args.replan_every),
        "execution_horizon": int(args.replan_every), "use_ema": bool(args.use_ema),
        "sampler_effective": sampler, "cfg_scale_override": args.cfg_scale, "sampler_steps_override": args.sampler_steps,
        "checkpoint": {"path": str(args.ckpt), "epoch": ckpt_epoch, "global_step": ckpt_step},
        "simulator": {"root": str(sim_root), "module": str(env_module.__file__), "git_head": sim_head or None, "dirty": sim_dirty, "origin": sim_origin, "file_sha256": sim_file_sha},
        "driver": {"path": str(Path(__file__).resolve()), "git_head": driver_head},
        "label": args.label,
        "metric": "peak coverage; SR recomputed from per-episode peaks",
    }
    summary = {
        "protocol": protocol,
        "sr_0p80": float((peaks >= 0.80).mean()) if len(peaks) else None,
        "sr_0p95": float((peaks >= 0.95).mean()) if len(peaks) else None,
        "ckpt": str(args.ckpt), "config_path": str(args.config_path), "use_ema": args.use_ema,
        "embodiment": args.embodiment_name, "replan_every": args.replan_every, "n_obs": policy.n_obs,
        "horizon": policy.horizon, "obstacle_level": args.obstacle_level, "max_steps": args.max_steps,
        "n_episodes": len(episodes), "seed_base": args.seed_base, "success_threshold": args.success_threshold,
        "cfg_scale_override": args.cfg_scale, "sampler_steps_override": args.sampler_steps, "cfg_interval_override": args.cfg_interval,
        "pusher": args.pusher, "chain_control_mode": args.chain_control_mode if args.pusher == "chain_gripper" else None,
        "trained_with_compile": policy.trained_with_compile, "compiled_inference": False,
        "decoder": policy.decoder_name, "sim_module": str(env_module.__file__),
        "success_rate": float((peaks >= args.success_threshold).mean()) if len(peaks) else None,
        "successes": int((peaks >= args.success_threshold).sum()),
        "mean_peak": float(peaks.mean()) if len(peaks) else None,
        "median_peak": float(np.median(peaks)) if len(peaks) else None,
        "timeouts": int(sum(e["status"] == "timeout" for e in episodes)),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[done] SR@{args.success_threshold:.2f} = {summary['successes']}/{len(episodes)} "
          f"= {summary['success_rate']:.3f} | mean peak {summary['mean_peak']:.3f} | median {summary['median_peak']:.3f} "
          f"| timeouts {summary['timeouts']} | wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
