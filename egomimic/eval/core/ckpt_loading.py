"""Strict standalone PushShapes rollout for route-agnostic Pipeline checkpoints.

Training keeps :class:`PipelineAlgo` free of environment, embodiment, and
normalization policy.  This module supplies those data-bound concerns only at
evaluation time and executes the checkpoint's unchanged inference graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


class _RolloutTimeout(Exception):
    pass


def _alarm_handler(_signum, _frame):
    raise _RolloutTimeout()


def _plain(value):
    return OmegaConf.to_container(value, resolve=True)


def _checkpoint_config(checkpoint: dict, config_path: str):
    hparams = checkpoint.get("hyper_parameters") or checkpoint.get("hparams") or {}
    embedded_tree = hparams.get("config_tree")
    if embedded_tree is None:
        raise RuntimeError("checkpoint has no embedded hyper_parameters.config_tree")
    embedded = OmegaConf.create(embedded_tree)
    resolved = OmegaConf.load(config_path)
    embedded_model = OmegaConf.select(embedded, "model")
    resolved_model = OmegaConf.select(resolved, "model")
    if embedded_model is None or resolved_model is None:
        raise RuntimeError("checkpoint and resolved config must both define model")
    if _plain(embedded_model) != _plain(resolved_model):
        raise RuntimeError("resolved config model differs from checkpoint-embedded model")
    if OmegaConf.select(embedded, "model.pipeline") is None:
        raise RuntimeError("checkpoint model has no model.pipeline graph")
    return embedded, resolved


def _load_stats(config, embodiment_id: int):
    raw_path = OmegaConf.select(config, "norm_stats.precomputed_norm_path")
    if not raw_path:
        raise RuntimeError("resolved config has no norm_stats.precomputed_norm_path")
    path = Path(str(raw_path)).resolve()
    if path.is_dir():
        path = path / "norm_stats.json"
    if not path.is_file():
        raise RuntimeError(f"normalization artifact not found: {path}")
    payload = json.loads(path.read_text())
    stats = payload.get("stats", {}).get(str(int(embodiment_id)))
    if not isinstance(stats, dict):
        raise RuntimeError(
            f"normalization artifact has no embodiment {embodiment_id}: {path}"
        )
    mode = str(OmegaConf.select(config, "norm_stats.norm_mode", default="quantile"))
    if mode not in {"quantile", "zscore", "minmax"}:
        raise RuntimeError(f"unsupported normalization mode {mode!r}")
    return path, mode, stats


def _apply_stats(value: torch.Tensor, stats: dict, mode: str, *, inverse: bool):
    value = value.float()
    if mode == "quantile":
        low_name, high_name = "quantile_1", "quantile_99"
    elif mode == "minmax":
        low_name, high_name = "min", "max"
    else:
        mean = torch.as_tensor(stats["mean"], device=value.device, dtype=torch.float32)
        std = torch.as_tensor(stats["std"], device=value.device, dtype=torch.float32)
        return value * (std + 1e-6) + mean if inverse else (value - mean) / (std + 1e-6)
    low = torch.as_tensor(stats[low_name], device=value.device, dtype=torch.float32)
    high = torch.as_tensor(stats[high_name], device=value.device, dtype=torch.float32)
    if inverse:
        return (value + 1.0) * 0.5 * (high - low + 1e-6) + low
    return 2.0 * ((value - low) / (high - low + 1e-6)) - 1.0


def _env_to_batch(obs: dict, device: torch.device, oriented: bool):
    parts = [obs["agent_pos"]]
    if oriented:
        parts.append(obs["agent_angle"])
    parts.append(obs["object_pose"])
    state = np.concatenate(parts, axis=0).astype(np.float32)
    image = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
    return {
        "state_agent_obj": torch.from_numpy(state).unsqueeze(0).to(device),
        "front_img_1": torch.from_numpy(image).unsqueeze(0).to(device),
    }


class PipelineRolloutGraph:
    """Bind normalization and native decoding around an immutable Pipeline graph."""

    def __init__(self, algo, config, embodiment_name: str, embodiment_id: int):
        self.algo = algo
        self.config = config
        self.embodiment_name = str(embodiment_name)
        self.embodiment_id = int(embodiment_id)
        self.device = next(algo.nets.parameters()).device
        self.norm_path, self.norm_mode, self.stats = _load_stats(
            config, self.embodiment_id
        )
        if "state_agent_obj" not in self.stats or "actions" not in self.stats:
            raise RuntimeError("normalization artifact needs state_agent_obj and actions")
        self.decoder = instantiate(config.planar.eval_native_decoder)
        self.pipeline = algo.pipeline
        self.pipeline.eval()
        self._validate_plan()

    def _validate_plan(self):
        seed_keys = ("front_img_1", "state_agent_obj", "embodiment")
        runnable, excluded = self.pipeline.plan(seed_keys, mode="inference")
        blocked = [
            (type(stage).__name__, missing)
            for stage, missing in excluded
            if missing != ["<train-only>"]
        ]
        if blocked:
            raise RuntimeError(f"Pipeline rollout graph has blocked stages: {blocked}")
        names = [type(stage).__name__ for stage in runnable]
        required = {"FusedObsEncoder", "GaussianLatentNoise", "PlanarFlowSampler"}
        if not required.issubset(names):
            raise RuntimeError(
                f"arc rollout graph lacks required inference stages: {names}"
            )
        if names[-1] != "PlanarFlowSampler":
            raise RuntimeError(f"arc rollout graph must end at PlanarFlowSampler: {names}")
        self.stage_names = names

    def set_sampler_steps(self, value: int):
        stages = [stage for stage in self.pipeline.stages if hasattr(stage, "num_inference_steps")]
        if len(stages) != 1:
            raise RuntimeError(
                f"sampler override needs exactly one stage, found {[type(s).__name__ for s in stages]}"
            )
        original = int(stages[0].num_inference_steps)
        stages[0].num_inference_steps = int(value)
        return type(stages[0]).__name__, original

    @torch.inference_mode()
    def inference_step(self, obs_zarr: dict, t: int, emb_id: int, T_max=None):
        del t, T_max
        if int(emb_id) != self.embodiment_id:
            raise RuntimeError(
                f"rollout embodiment {emb_id} != configured {self.embodiment_id}"
            )
        batch = dict(obs_zarr)
        batch["state_agent_obj"] = _apply_stats(
            batch["state_agent_obj"],
            self.stats["state_agent_obj"],
            self.norm_mode,
            inverse=False,
        )
        batch["embodiment"] = self.embodiment_name
        result = self.algo.forward_eval({"rollout": batch})["rollout"]
        if "pred_action" not in result:
            raise RuntimeError("Pipeline inference graph produced no pred_action")
        tokens = _apply_stats(
            result["pred_action"], self.stats["actions"], self.norm_mode, inverse=True
        )
        native = self.decoder.decode(tokens, context={})
        if not torch.is_tensor(native):
            native = torch.as_tensor(native)
        if native.ndim != 3 or native.shape[0] != 1 or native.shape[1] != 1:
            raise RuntimeError(
                f"arc decoder must return one replanned action, got {tuple(native.shape)}"
            )
        return native[0, 0].detach().float().cpu().numpy()


def _load_rollout_graph(
    ckpt_path: str,
    config_path: str,
    embodiment_name: str,
    embodiment_id: int,
    use_ema: bool,
):
    from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
    from egomimic.pl_utils.pl_model import ModelWrapper

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    embedded, resolved = _checkpoint_config(checkpoint, config_path)
    wrapper = ModelWrapper(config_tree=embedded)
    strict_load_pipeline_checkpoint(wrapper.model, checkpoint, use_ema=use_ema)
    wrapper.model.device = next(wrapper.model.nets.parameters()).device
    graph = PipelineRolloutGraph(
        wrapper.model, resolved, embodiment_name, embodiment_id
    )
    return graph, checkpoint


def strict_no_rollout_preflight(
    ckpt_path: str,
    config_path: str,
    selected_embodiment_name: str,
    selected_embodiment_id: int,
    expected_native_action_dim: int,
    use_ema: bool = False,
    action_chunk_start_index: int = 0,
    replan_every: int | None = None,
):
    if int(action_chunk_start_index) != 0:
        raise RuntimeError("arc rollout requires action_chunk_start_index=0")
    if replan_every not in (None, 1):
        raise RuntimeError("waypoint-zero arc rollout replans every step")
    graph, _checkpoint = _load_rollout_graph(
        ckpt_path,
        config_path,
        selected_embodiment_name,
        selected_embodiment_id,
        use_ema,
    )
    sampler = [
        stage for stage in graph.pipeline.stages if hasattr(stage, "num_inference_steps")
    ]
    action_stats = graph.stats["actions"]
    token_shape = np.asarray(action_stats["mean"]).shape
    if len(token_shape) != 2:
        raise RuntimeError(f"action normalization shape must be (H,D), got {token_shape}")
    probe = torch.zeros((1, *token_shape), device=graph.device)
    decoded = graph.decoder.decode(probe, context={})
    decoded = decoded if torch.is_tensor(decoded) else torch.as_tensor(decoded)
    if decoded.shape != (1, 1, int(expected_native_action_dim)):
        raise RuntimeError(
            "arc decoder/native action contract mismatch: "
            f"{tuple(decoded.shape)} != {(1, 1, int(expected_native_action_dim))}"
        )
    report = {
        "status": "audited_strict_no_rollout_ok",
        "embodiment_id": int(selected_embodiment_id),
        "embodiment_name": str(selected_embodiment_name),
        "model_token_horizon": int(token_shape[0]),
        "model_token_dim": int(token_shape[1]),
        "decoded_action_horizon": 1,
        "action_chunk_start_index": 0,
        "execution_horizon": 1,
        "execution_stop_index_exclusive": 1,
        "native_action_dim": int(expected_native_action_dim),
        "state_tensor_count": len(graph.algo.nets.state_dict()),
        "weights": "ema" if use_ema else "raw",
        "inference_graph": graph.stage_names,
        "sampler_inference_steps": int(sampler[0].num_inference_steps),
        "normalization_path": str(graph.norm_path),
    }
    print("[strict-preflight] " + json.dumps(report, sort_keys=True))
    return report


def _parse_seeds(raw: str | None, base: int, count: int):
    if raw:
        seeds = [int(piece) for piece in raw.split(",") if piece.strip()]
    else:
        seeds = list(range(int(base), int(base) + int(count)))
    if len(seeds) != int(count) or len(set(seeds)) != len(seeds):
        raise RuntimeError("seed list must contain exactly n-episodes unique seeds")
    return seeds


def _write_episode_artifacts(root: Path, emb_id: int, ep: int, actions, coverages):
    artifact_dir = root / "videos" / "sim"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    action_array = np.asarray(actions, dtype=np.float32)
    np.savez_compressed(artifact_dir / f"actions_emb{emb_id}_ep{ep}.npz", actions=action_array)
    with (artifact_dir / f"coverage_emb{emb_id}_ep{ep}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "coverage", "action_x", "action_y"])
        for index, coverage in enumerate(coverages):
            action = action_array[index]
            writer.writerow([index, f"{coverage:.6f}", f"{action[0]:.4f}", f"{action[1]:.4f}"])


def _rollout_episode(graph, args, seed: int, ep: int):
    from Tsimulation.pushshapes import PushShapesEnv

    env = PushShapesEnv(
        object_shape="T",
        pusher_shape=args.pusher,
        obstacle_level=args.obstacle_level,
        render_mode="rgb_array" if args.per_episode_videos else None,
    )
    old_handler = None
    actions, coverages = [], []
    peak = 0.0
    try:
        env.reset(seed=int(seed))
        if args.rollout_timeout > 0:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(args.rollout_timeout)
        for step in range(args.max_steps):
            obs = _env_to_batch(
                env._get_obs(),
                graph.device,
                oriented=graph.embodiment_name in {
                    "pushshapes_sim_u_socket",
                    "pushshapes_sim_chain_gripper",
                },
            )
            action = np.asarray(
                graph.inference_step(obs, step, graph.embodiment_id, T_max=args.max_steps),
                dtype=np.float32,
            ).reshape(-1)
            if not np.isfinite(action).all():
                print(
                    f"[sim] WARNING: non-finite action t={step} emb{graph.embodiment_id} "
                    f"ep{ep} (raw={action.tolist()}); abort 0-cov."
                )
                return 0.0, [], []
            _obs, _reward, terminated, _truncated, info = env.step(action)
            coverage = float(info.get("coverage", 0.0))
            peak = max(peak, coverage)
            actions.append(action)
            coverages.append(coverage)
            if terminated and not args.full_horizon:
                break
    except _RolloutTimeout:
        print(
            f"[sim] WATCHDOG: rollout emb{graph.embodiment_id} ep{ep} exceeded "
            f"{args.rollout_timeout}s; logging 0-coverage and continuing."
        )
        return 0.0, [], []
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
        env.close()
    return peak if args.max_coverage else (coverages[-1] if coverages else 0.0), actions, coverages


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--eval-class", choices=("packed", "hpt"), default="packed")
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--obs-stride", type=int)
    parser.add_argument("--per-episode-videos", action="store_true")
    parser.add_argument("--obstacle-level", type=int, default=0)
    parser.add_argument("--coverage-threshold", type=float, default=1.01)
    parser.add_argument("--full-horizon", action="store_true")
    parser.add_argument("--max-coverage", action="store_true")
    parser.add_argument("--rollout-timeout", type=int, default=120)
    parser.add_argument("--replan-every", type=int)
    parser.add_argument("--action-chunk-start-index", type=int, default=0)
    parser.add_argument("--sampler-inference-steps", type=int)
    parser.add_argument("--embodiment-name", required=True)
    parser.add_argument("--only-emb", type=int, required=True)
    parser.add_argument("--pusher", required=True)
    parser.add_argument("--init-mode", choices=("seeds",), required=True)
    parser.add_argument("--init-seed-base", type=int, default=0)
    parser.add_argument("--init-seeds")
    parser.add_argument("--rng-pairing", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    if args.eval_class != "packed":
        raise RuntimeError("Pipeline rollout currently requires eval-class=packed")
    if args.obs_stride not in (None, 1):
        raise RuntimeError("arc waypoint-zero rollout requires obs-stride 1")
    if args.replan_every not in (None, 1):
        raise RuntimeError("arc waypoint-zero rollout requires replan-every 1")
    if args.action_chunk_start_index != 0:
        raise RuntimeError("arc waypoint-zero rollout requires action start index 0")
    if args.per_episode_videos:
        raise RuntimeError("per-episode video export is not yet supported for Pipeline arc rollout")

    graph, checkpoint = _load_rollout_graph(
        args.ckpt, args.config_path, args.embodiment_name, args.only_emb, args.use_ema
    )
    if args.sampler_inference_steps is not None:
        stage_name, original = graph.set_sampler_steps(args.sampler_inference_steps)
        print(
            f"[load] sampler override stage={stage_name} checkpoint={original} "
            f"effective={args.sampler_inference_steps}"
        )
    graph.algo.nets.eval()
    seeds = _parse_seeds(args.init_seeds, args.init_seed_base, args.n_episodes)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    peaks = []
    for ep, seed in enumerate(seeds):
        devices = [graph.device.index or 0] if graph.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if args.rng_pairing:
                rng_seed = 20000 + 97 * ep
                torch.manual_seed(rng_seed)
                if graph.device.type == "cuda":
                    torch.cuda.manual_seed_all(rng_seed)
            peak, actions, coverages = _rollout_episode(graph, args, seed, ep)
        _write_episode_artifacts(output, graph.embodiment_id, ep, actions, coverages)
        peaks.append(float(peak))
        print(f"[sim] emb{graph.embodiment_id} ep{ep} seed={seed} peak={peak:.6f}")
    if len(peaks) != args.n_episodes or not all(math.isfinite(v) for v in peaks):
        raise RuntimeError("rollout did not produce the expected finite episode peaks")
    print(
        f"[sim] emb{graph.embodiment_id} ep_coverages: "
        + ", ".join(f"{value:.4g}" for value in peaks)
    )
    print(
        f"[sim] emb{graph.embodiment_id} SR@0.80="
        f"{sum(v >= 0.80 for v in peaks) / len(peaks):.6f} "
        f"SR@0.95={sum(v >= 0.95 for v in peaks) / len(peaks):.6f}"
    )
    print(
        f"[sim] completed emb{graph.embodiment_id} count={len(peaks)} "
        f"checkpoint_epoch={int(checkpoint.get('epoch', -1))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
