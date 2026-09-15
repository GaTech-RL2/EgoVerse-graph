"""Strict PushShapes evaluator for the Paper-DP and ARC pipeline families.

This is intentionally a small integration layer.  Model construction, strict
checkpoint loading, normalization, ARC detokenization, and simulator semantics
remain owned by the existing EgoVerse source modules; this module only joins
those interfaces for the live paired rollout launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes_sim import (
    _env_to_zarr_pushshapes_oriented,
)
from egomimic.rldb.zarr.planar_arc import arc_token_rows

_EMBODIMENTS = {
    "pushshapes_sim_u_socket": {"id": 19, "native_action_dim": 3},
    "pushshapes_sim_chain_gripper": {"id": 20, "native_action_dim": 4},
}
_OBSERVATION_HORIZON = 2
_COMMON_ACTION_DIM = 5
_NETS_PREFIX = "nets."


class _RolloutTimeout(Exception):
    """Raised by the per-episode simulator watchdog."""


def _rollout_alarm_handler(_signum, _frame):
    raise _RolloutTimeout()


def _checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint has no mapping-valued state_dict")
    return state_dict


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> DictConfig:
    hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams"))
    if not isinstance(hparams, Mapping):
        raise TypeError("checkpoint has no mapping-valued hyper_parameters")
    config_tree = hparams.get("config_tree")
    if config_tree is None:
        raise RuntimeError("checkpoint hyper_parameters has no config_tree")
    config_tree = OmegaConf.create(config_tree)
    if OmegaConf.select(config_tree, "model.pipeline") is None:
        raise RuntimeError("checkpoint config_tree has no model.pipeline")
    return config_tree


def _assert_model_config_match(config_path: str | Path, embedded: DictConfig) -> DictConfig:
    supplied = OmegaConf.load(str(config_path))
    supplied_model = OmegaConf.select(supplied, "model")
    embedded_model = OmegaConf.select(embedded, "model")
    if supplied_model is None or embedded_model is None:
        raise RuntimeError("both supplied and checkpoint configs must contain model")
    # Historical checkpoints may store only the model subtree while retaining
    # interpolations to the full training config.  Resolve that frozen model
    # against the supplied config only as interpolation context, then compare
    # the resolved model trees exactly.
    embedded_context = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(supplied, resolve=False)),
        OmegaConf.create({"model": embedded_model}),
    )
    expected = OmegaConf.to_container(embedded_context.model, resolve=True)
    actual = OmegaConf.to_container(supplied_model, resolve=True)
    if actual != expected:
        raise RuntimeError("supplied config model does not exactly match checkpoint model")
    return supplied


def _stats_file(config: DictConfig) -> Path:
    raw = Path(str(OmegaConf.select(config, "norm_stats.precomputed_norm_path")))
    path = raw / "norm_stats.json" if raw.is_dir() else raw
    if not path.is_file():
        raise RuntimeError(f"recorded normalizer is missing: {path}")
    expected = OmegaConf.select(config, "run_provenance.normalization_sha256")
    if expected:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected):
            raise RuntimeError(
                f"normalizer SHA mismatch: actual={actual} expected={expected}"
            )
    return path


def _normalizer_from_config(config: DictConfig, embodiment_id: int):
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    payload = json.loads(_stats_file(config).read_text())
    stats = payload.get("stats", {}).get(str(embodiment_id))
    if not isinstance(stats, Mapping) or {"state_agent_obj", "actions"} - set(stats):
        raise RuntimeError(
            f"recorded normalizer lacks state/action stats for embodiment {embodiment_id}"
        )
    norm_mode = str(OmegaConf.select(config, "norm_stats.norm_mode"))
    return MultiDataset.from_state(
        {
            "norm_mode": norm_mode,
            "embodiments": [embodiment_id],
            "key_types": {
                embodiment_id: {
                    "state_agent_obj": "proprio_keys",
                    "actions": "action_keys",
                }
            },
            "zarr_keys": {
                embodiment_id: {
                    "state_agent_obj": "state_agent_obj",
                    "actions": "actions",
                }
            },
            "norm_stats": {embodiment_id: dict(stats)},
        }
    )


def _pipeline_stage_names(algo) -> list[str]:
    return [type(stage).__name__ for stage in algo.pipeline.stages]


def _is_arc(config: DictConfig, stage_names: Sequence[str]) -> bool:
    return "ArcTokenizeStage" in stage_names or int(
        OmegaConf.select(config, "planar.arc_token_rows", default=0)
    ) > 0


def _token_shape(config: DictConfig, stage_names: Sequence[str]) -> tuple[int, int, str]:
    arc = _is_arc(config, stage_names)
    if arc:
        waypoints = int(OmegaConf.select(config, "planar.arc_waypoints", default=0))
        velocity_mode = str(
            OmegaConf.select(config, "planar.arc_velocity_mode", default="")
        )
        horizon = arc_token_rows(waypoints, velocity_mode)
        representation = f"planar_arc_{velocity_mode}"
    else:
        horizon = int(OmegaConf.select(config, "planar.action_horizon", default=0))
        representation = "planar_common5"
    if horizon <= 0:
        raise RuntimeError("config has no positive model token horizon")
    return horizon, _COMMON_ACTION_DIM, representation


def _sampler_steps(algo) -> int:
    values = []
    for stage in algo.pipeline.stages:
        policy = getattr(stage, "policy", None)
        value = getattr(policy, "num_inference_steps", None)
        if value is not None:
            values.append(int(value))
    if not values or len(set(values)) != 1:
        raise RuntimeError(f"expected one unambiguous diffusion sampler step count, got {values}")
    return values[0]


def _native_decoder(config: DictConfig, embodiment_name: str):
    decoder_cfg = OmegaConf.select(config, f"model.native_decoders.{embodiment_name}")
    if decoder_cfg is None:
        decoder_cfg = OmegaConf.select(config, f"native_decoders.{embodiment_name}")
    if decoder_cfg is None:
        # Older configs used this location.  It is accepted only as a fallback;
        # current Paper-DP/ARC configs keep one decoder per embodiment.
        decoder_cfg = OmegaConf.select(config, "planar.eval_native_decoder")
    if decoder_cfg is None:
        raise RuntimeError(f"config has no native decoder for {embodiment_name}")
    decoder = instantiate(decoder_cfg)
    if not hasattr(decoder, "decode"):
        raise RuntimeError("configured native decoder has no decode method")
    return decoder


def _load_policy(
    *,
    ckpt_path: str | Path,
    config_path: str | Path,
    selected_embodiment_name: str,
    selected_embodiment_id: int,
    expected_native_action_dim: int,
    use_ema: bool,
    device: torch.device,
):
    if selected_embodiment_name not in _EMBODIMENTS:
        raise ValueError(f"unsupported PushShapes embodiment {selected_embodiment_name!r}")
    expected = _EMBODIMENTS[selected_embodiment_name]
    if selected_embodiment_id != expected["id"]:
        raise ValueError(
            f"embodiment id mismatch: expected {expected['id']}, got {selected_embodiment_id}"
        )
    if expected_native_action_dim != expected["native_action_dim"]:
        raise ValueError(
            "native action dim mismatch: "
            f"expected {expected['native_action_dim']}, got {expected_native_action_dim}"
        )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint payload is not a mapping")
    embedded = _checkpoint_config(checkpoint)
    config = _assert_model_config_match(config_path, embedded)
    pipeline_cfg = OmegaConf.select(config, "model.pipeline")
    if pipeline_cfg is None:
        raise RuntimeError("config has no model.pipeline")
    algo = instantiate(pipeline_cfg)
    if not hasattr(algo, "pipeline") or not hasattr(algo, "nets"):
        raise TypeError("model.pipeline did not instantiate PipelineAlgo")
    strict_load_pipeline_checkpoint(algo, checkpoint, use_ema=use_ema)
    algo.device = device
    algo.nets.to(device)
    algo.nets.eval()

    stage_names = _pipeline_stage_names(algo)
    runnable, excluded = algo.pipeline.plan(
        ["front_img_1", "state_agent_obj", "embodiment"], mode="inference"
    )
    blocked = [
        (type(stage).__name__, missing)
        for stage, missing in excluded
        if missing not in (["<train-only>"], ["<inference-only>"])
    ]
    if blocked:
        raise RuntimeError(f"inference graph has blocked stages: {blocked}")
    runnable_names = [type(stage).__name__ for stage in runnable]
    if "DiffusionDenoiserStage" not in runnable_names:
        raise RuntimeError(f"inference graph has no DiffusionDenoiserStage: {runnable_names}")

    token_horizon, token_dim, representation = _token_shape(config, stage_names)
    decoder = _native_decoder(config, selected_embodiment_name)
    stats = _normalizer_from_config(config, selected_embodiment_id)
    action_stats = stats.norm_stats[selected_embodiment_id]["actions"]
    action_mean = action_stats["mean"] if isinstance(action_stats, Mapping) else None
    if action_mean is None:
        raise RuntimeError("normalizer action stats have no mean")
    stats_shape = tuple(np.asarray(action_mean).shape)
    expected_stats_shape = (
        (int(getattr(decoder, "action_horizon", 0)), token_dim)
        if getattr(decoder, "requires_common5_unnormalization", False)
        else (token_horizon, token_dim)
    )
    if stats_shape != expected_stats_shape:
        raise RuntimeError(
            f"action stats shape {stats_shape} does not match model tokens "
            f"{expected_stats_shape} for the configured decoder"
        )

    token_probe = torch.zeros(1, token_horizon, token_dim)
    decoded = decoder.decode(token_probe)
    if not torch.is_tensor(decoded):
        decoded = torch.as_tensor(decoded)
    if decoded.ndim != 3 or decoded.shape[0] != 1:
        raise RuntimeError(f"native decoder returned unexpected shape {tuple(decoded.shape)}")
    if int(decoded.shape[-1]) != expected_native_action_dim:
        raise RuntimeError(
            f"native decoder width {decoded.shape[-1]} does not match "
            f"expected {expected_native_action_dim}"
        )
    observation_horizon = int(
        OmegaConf.select(config, "planar.observation_horizon", default=0)
    )
    if observation_horizon != _OBSERVATION_HORIZON:
        raise RuntimeError(
            f"current integrated evaluator requires two observations, got {observation_horizon}"
        )
    configured_replan = OmegaConf.select(
        config, "run_provenance.action_contract.replan_every", default=None
    )
    configured_replan = int(configured_replan) if configured_replan is not None else 1
    configured_start = int(
        OmegaConf.select(
            config,
            "run_provenance.action_contract.rollout_action_chunk_start_index",
            default=0,
        )
    )
    execution_horizon = int(
        OmegaConf.select(
            config,
            "run_provenance.action_contract.execution_horizon",
            default=configured_replan,
        )
    )
    return {
        "algo": algo,
        "normalizer": stats,
        "decoder": decoder,
        "checkpoint": checkpoint,
        "config": config,
        "stage_names": stage_names,
        "runnable_stage_names": runnable_names,
        "token_horizon": token_horizon,
        "token_dim": token_dim,
        "representation": representation,
        "decoded_horizon": int(decoded.shape[1]),
        "sampler_inference_steps": _sampler_steps(algo),
        "configured_replan": configured_replan,
        "configured_start": configured_start,
        "execution_horizon": execution_horizon,
    }


def _contract_args(
    config: DictConfig,
    *,
    action_chunk_start_index: int,
    replan_every: int | None,
) -> tuple[int, int]:
    configured_start = int(
        OmegaConf.select(
            config,
            "run_provenance.action_contract.rollout_action_chunk_start_index",
            default=0,
        )
    )
    configured_replan = int(
        OmegaConf.select(
            config,
            "run_provenance.action_contract.replan_every",
            default=1,
        )
    )
    if action_chunk_start_index != configured_start:
        raise ValueError(
            "action chunk start does not match recorded config contract: "
            f"{action_chunk_start_index} != {configured_start}"
        )
    actual_replan = configured_replan if replan_every is None else int(replan_every)
    if actual_replan != configured_replan:
        raise ValueError(
            f"replan cadence does not match recorded config contract: {actual_replan} != {configured_replan}"
        )
    if actual_replan <= 0:
        raise ValueError("replan cadence must be positive")
    return action_chunk_start_index, actual_replan


def strict_no_rollout_preflight(
    ckpt_path: str,
    config_path: str,
    selected_embodiment_name: str,
    selected_embodiment_id: int,
    expected_native_action_dim: int,
    use_ema: bool = False,
    action_chunk_start_index: int = 0,
    replan_every: int | None = None,
) -> dict[str, Any]:
    """Strictly prove model, pipeline, normalizer, and decoder compatibility."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_policy(
        ckpt_path=ckpt_path,
        config_path=config_path,
        selected_embodiment_name=selected_embodiment_name,
        selected_embodiment_id=selected_embodiment_id,
        expected_native_action_dim=expected_native_action_dim,
        use_ema=use_ema,
        device=device,
    )
    start, cadence = _contract_args(
        loaded["config"],
        action_chunk_start_index=action_chunk_start_index,
        replan_every=replan_every,
    )
    if start + cadence > loaded["decoded_horizon"]:
        raise ValueError(
            f"execution slice [{start},{start + cadence}) exceeds decoded horizon "
            f"{loaded['decoded_horizon']}"
        )
    parameter_keys = len(loaded["algo"].nets.state_dict())
    checkpoint = loaded["checkpoint"]
    return {
        "status": "audited_strict_no_rollout_ok",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "weights": "ema" if use_ema else "raw",
        "model_family": loaded["representation"],
        "model_token_horizon": loaded["token_horizon"],
        "model_token_dim": loaded["token_dim"],
        "decoded_action_horizon": loaded["decoded_horizon"],
        "native_action_dim": expected_native_action_dim,
        "state_tensor_count": parameter_keys,
        "action_chunk_start_index": start,
        "execution_stop_index_exclusive": start + cadence,
        "execution_horizon": cadence,
        "execution_horizon_semantics": "replan_every_steps_from_decoded_native_chunk",
        "replan_every": cadence,
        "timing_semantics": getattr(loaded["decoder"], "timing_semantics", None),
        "sampler_inference_steps": loaded["sampler_inference_steps"],
        "pipeline_stages": loaded["stage_names"],
        "inference_stages": loaded["runnable_stage_names"],
        "bridge": "pipeline_paper_dp_arc_rollout_v1",
    }


class _Policy:
    def __init__(self, loaded: Mapping[str, Any], *, embodiment_name: str, embodiment_id: int, device: torch.device, start: int, cadence: int):
        self.algo = loaded["algo"]
        self.normalizer = loaded["normalizer"]
        self.decoder = loaded["decoder"]
        self.embodiment_name = embodiment_name
        self.embodiment_id = embodiment_id
        self.device = device
        self.start = start
        self.cadence = cadence
        self.token_shape = (int(loaded["token_horizon"]), int(loaded["token_dim"]))
        self.requires_common5_unnormalization = bool(
            getattr(self.decoder, "requires_common5_unnormalization", False)
        )
        self._history: list[dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self._history.clear()

    def _window(self, obs_env: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        current = _env_to_zarr_pushshapes_oriented(dict(obs_env), self.device)
        self._history.append(current)
        self._history = self._history[-_OBSERVATION_HORIZON:]
        history = [self._history[0]] * (_OBSERVATION_HORIZON - len(self._history))
        history += self._history
        return {
            key: torch.stack([item[key] for item in history], dim=1)
            for key in current
        }

    @torch.inference_mode()
    def predict_native_actions(self, obs_env: Mapping[str, Any]) -> np.ndarray:
        window = self._window(obs_env)
        normalized = self.normalizer.normalize(window, self.embodiment_id)
        normalized = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in normalized.items()
        }
        prediction = self.algo.forward_eval({self.embodiment_name: normalized})
        result = prediction.get(self.embodiment_name)
        if not isinstance(result, Mapping) or "pred_action" not in result:
            raise RuntimeError("PipelineAlgo inference did not produce pred_action")
        tokens = result["pred_action"]
        if not torch.is_tensor(tokens) or tuple(tokens.shape[-2:]) != self.token_shape:
            raise RuntimeError(
                f"PipelineAlgo returned unexpected token shape {getattr(tokens, 'shape', None)}; expected {self.token_shape}"
            )
        if self.requires_common5_unnormalization:
            common = self.decoder.decode_common(tokens)
            common = self.normalizer.unnormalize(
                {"actions": common}, self.embodiment_id
            )["actions"]
            native = self.decoder.decode_common_to_native(common)
        else:
            actions = self.normalizer.unnormalize(
                {"actions": tokens}, self.embodiment_id
            )["actions"]
            native = self.decoder.decode(actions)
        if not torch.is_tensor(native):
            native = torch.as_tensor(native, device=self.device)
        if native.ndim == 2:
            native = native.unsqueeze(0)
        if native.ndim != 3 or native.shape[0] != 1:
            raise RuntimeError(f"native decoder returned unexpected batch shape {tuple(native.shape)}")
        value = native.detach().float().cpu().numpy()
        if not np.all(np.isfinite(value)):
            raise RuntimeError("native decoder produced non-finite actions")
        return value[0].astype(np.float32, copy=False)


def _parse_seed_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.split(",") if item]


def _write_video(path: Path, frames: Sequence[np.ndarray]) -> None:
    if not frames:
        raise RuntimeError("cannot write empty rollout video")
    from torchvision.io import write_video

    video = torch.from_numpy(np.ascontiguousarray(np.stack(frames, axis=0)))
    write_video(str(path), video, fps=30, video_codec="h264")


def _validate_runtime_args(args) -> None:
    if args.eval_class not in {"packed", "hpt"}:
        raise ValueError("eval-class must be packed or hpt")
    if args.embodiment_name not in _EMBODIMENTS:
        raise ValueError(f"unsupported embodiment {args.embodiment_name!r}")
    expected = _EMBODIMENTS[args.embodiment_name]
    if args.only_emb != expected["id"] or args.pusher != args.embodiment_name.removeprefix("pushshapes_sim_"):
        raise ValueError("embodiment name, id, and pusher do not agree")
    if args.n_episodes <= 0 or args.max_steps <= 0:
        raise ValueError("n-episodes and max-steps must be positive")
    if args.obs_stride not in (None, 1):
        raise ValueError("integrated evaluator requires obs-stride=1")
    if args.action_chunk_start_index < 0:
        raise ValueError("action-chunk-start-index must be non-negative")
    if args.sampler_inference_steps is not None and args.sampler_inference_steps != 100:
        raise ValueError("current Paper-DP checkpoints require 100 sampler steps")
    if args.init_mode != "seeds":
        raise ValueError("canonical evaluator requires init-mode=seeds")
    if not args.init_seeds:
        args.init_seeds = [args.init_seed_base + index for index in range(args.n_episodes)]
    if len(args.init_seeds) < args.n_episodes:
        raise ValueError("init-seeds must include one seed per requested episode")


def _rollout_one(args, policy: _Policy, seed: int, ep_idx: int):
    from Tsimulation.pushshapes import PushShapesEnv

    env = PushShapesEnv(
        object_shape="T",
        pusher_shape=args.pusher,
        obstacle_level=args.obstacle_level,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    chunk_lengths: list[int] = []
    coverage = 0.0
    max_coverage = 0.0
    previous_handler = None
    if args.rollout_timeout > 0:
        previous_handler = signal.signal(signal.SIGALRM, _rollout_alarm_handler)
        signal.alarm(args.rollout_timeout)
    try:
        env.reset(seed=seed)
        policy.reset()
        action_chunk = None
        chunk_offset = 0
        for _step in range(args.max_steps):
            if action_chunk is None or chunk_offset >= min(len(action_chunk), policy.start + policy.cadence):
                action_chunk = policy.predict_native_actions(env._get_obs())
                chunk_offset = policy.start
                chunk_lengths.append(len(action_chunk))
            action = action_chunk[chunk_offset]
            chunk_offset += 1
            actions.append(action.copy())
            _obs, _reward, terminated, _truncated, info = env.step(action)
            coverage = float(info.get("coverage", 0.0))
            max_coverage = max(max_coverage, coverage)
            if args.per_episode_videos:
                frame = env.render()
                if frame is not None:
                    frames.append(np.ascontiguousarray(frame))
            if terminated and not args.full_horizon:
                break
    except _RolloutTimeout:
        coverage = 0.0
        max_coverage = 0.0
        print(
            f"[sim] WATCHDOG: rollout emb{policy.embodiment_id} ep{ep_idx} exceeded "
            f"{args.rollout_timeout}s; logging invalid 0-coverage and continuing."
        )
    finally:
        if args.rollout_timeout > 0:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
        env.close()
    return max_coverage if args.max_coverage else coverage, actions, frames, chunk_lengths


def run(args) -> None:
    _validate_runtime_args(args)
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise RuntimeError(f"canonical launcher did not create output directory {out_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_policy(
        ckpt_path=args.ckpt,
        config_path=args.config_path,
        selected_embodiment_name=args.embodiment_name,
        selected_embodiment_id=args.only_emb,
        expected_native_action_dim=_EMBODIMENTS[args.embodiment_name]["native_action_dim"],
        use_ema=args.use_ema,
        device=device,
    )
    start, cadence = _contract_args(
        loaded["config"],
        action_chunk_start_index=args.action_chunk_start_index,
        replan_every=args.replan_every,
    )
    if start + cadence > loaded["decoded_horizon"]:
        raise ValueError("configured execution slice exceeds decoded action horizon")
    policy = _Policy(
        loaded,
        embodiment_name=args.embodiment_name,
        embodiment_id=args.only_emb,
        device=device,
        start=start,
        cadence=cadence,
    )
    coverages: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    videos_dir = out_dir / "videos"
    if args.per_episode_videos:
        videos_dir.mkdir()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    for ep_idx, seed in enumerate(args.init_seeds[: args.n_episodes]):
        rng_context = torch.random.fork_rng(devices=cuda_devices) if args.rng_pairing else nullcontext()
        with rng_context:
            if args.rng_pairing:
                torch.manual_seed(20000 + 97 * ep_idx)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(20000 + 97 * ep_idx)
            result = _rollout_one(args, policy, seed, ep_idx)
        coverage, actions, frames, chunk_lengths = result
        coverages.append(float(coverage))
        if actions:
            np.save(out_dir / f"episode_{ep_idx:02d}_actions.npy", np.stack(actions, axis=0))
        video_path = None
        if args.per_episode_videos:
            video_path = videos_dir / f"episode_{ep_idx:02d}.mp4"
            _write_video(video_path, frames)
        episode_rows.append(
            {
                "episode_index": ep_idx,
                "init_seed": int(seed),
                "peak_coverage": float(coverage),
                "action_count": len(actions),
                "predicted_chunk_count": len(chunk_lengths),
                "decoded_chunk_length_min": min(chunk_lengths) if chunk_lengths else 0,
                "decoded_chunk_length_max": max(chunk_lengths) if chunk_lengths else 0,
                "decoded_chunk_length_mean": float(np.mean(chunk_lengths)) if chunk_lengths else 0.0,
                "video": str(video_path) if video_path else None,
            }
        )
    print(
        f"[sim] emb{args.only_emb} ep_coverages: "
        + ",".join(f"{value:.4f}" for value in coverages)
    )
    summary = {
        "bridge": "pipeline_paper_dp_arc_rollout_v1",
        "comparability": "source_integrated_checkpoint_bound_evaluator",
        "weights": "ema" if args.use_ema else "raw",
        "embodiment": args.embodiment_name,
        "embodiment_id": args.only_emb,
        "model_family": loaded["representation"],
        "model_token_horizon": loaded["token_horizon"],
        "model_token_dim": loaded["token_dim"],
        "decoded_action_horizon": loaded["decoded_horizon"],
        "action_chunk_start_index": start,
        "execution_slice": f"[{start},{start + cadence})",
        "execution_horizon": cadence,
        "replan_every": cadence,
        "timing_semantics": getattr(loaded["decoder"], "timing_semantics", None),
        "sampler_inference_steps": loaded["sampler_inference_steps"],
        "episodes": episode_rows,
    }
    (out_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--eval-class", required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--obs-stride", type=int, default=None)
    parser.add_argument("--per-episode-videos", action="store_true")
    parser.add_argument("--obstacle-level", type=int, required=True)
    parser.add_argument("--coverage-threshold", type=float, required=True)
    parser.add_argument("--full-horizon", action="store_true")
    parser.add_argument("--max-coverage", action="store_true")
    parser.add_argument("--rollout-timeout", type=int, required=True)
    parser.add_argument("--replan-every", type=int, default=None)
    parser.add_argument("--action-chunk-start-index", type=int, default=0)
    parser.add_argument("--sampler-inference-steps", type=int, default=None)
    parser.add_argument("--chunk-stitch-weights", default=None)
    parser.add_argument("--chunk-seam-artifact", default=None)
    parser.add_argument("--rollout-noise-shift-tokens", type=int, default=None)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=None)
    parser.add_argument("--rtc-inference-delay", type=int, default=None)
    parser.add_argument("--rtc-prefix-attention-schedule", default=None)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=None)
    parser.add_argument("--embodiment-name", required=True)
    parser.add_argument("--only-emb", type=int, required=True)
    parser.add_argument("--pusher", required=True)
    parser.add_argument("--init-mode", required=True)
    parser.add_argument("--init-seed-base", type=int, required=True)
    parser.add_argument("--init-seeds", type=_parse_seed_list, default=[])
    parser.add_argument("--rng-pairing", action="store_true")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
