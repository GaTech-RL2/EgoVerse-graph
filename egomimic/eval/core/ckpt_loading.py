"""Strict rollout bridge for the recorded U-Socket duration-ARC checkpoints.

These September 2026 checkpoints predate the current rollout inference-step
interface. This module keeps their original PipelineAlgo, checkpoint
normalizer, diffusion sampler, and native duration decoder, and supplies only
the missing simulator loop. It is deliberately restricted to the recorded
U-Socket contract: two observations and a 32-row prediction (16 spline-support
waypoints plus 16 per-waypoint interval-duration rows). The decoder reconstructs
the represented 40-step fixed-rate trajectory before the simulator executes it.
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

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes_sim import _env_to_zarr_pushshapes_oriented
from egomimic.rldb.zarr.planar_arc import arc_token_rows
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

_NETS_PREFIX = "nets."
_LEGACY_EMBODIMENT = "pushshapes_sim_u_socket"
_LEGACY_OBSERVATION_HORIZON = 2
_DURATION_ACTION_HORIZON = 32
_LEGACY_WAYPOINT_HORIZON = 16
_LEGACY_RAW_ACTION_HORIZON = 40
_LEGACY_NATIVE_ACTION_DIM = 3
_DURATION_TIMING_SEMANTICS = "per_waypoint_duration_cubic_v1"


class _RolloutTimeout(Exception):
    """Raised by the per-episode simulator watchdog."""


def _rollout_alarm_handler(_signum, _frame):
    raise _RolloutTimeout()


def _checkpoint_state_dict(checkpoint: dict[str, Any]) -> Mapping[str, torch.Tensor]:
    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint has no state_dict")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint state_dict is not a mapping")
    return state_dict


def _extract_nets_state(
    source: Mapping[str, torch.Tensor], *, source_name: str
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(source, Mapping):
        raise TypeError(f"checkpoint {source_name} is not a mapping")
    extracted = OrderedDict(
        (str(key)[len(_NETS_PREFIX) :], value)
        for key, value in source.items()
        if str(key).startswith(_NETS_PREFIX)
    )
    if not extracted:
        raise ValueError(f"{source_name} contains no Pipeline nets parameters")
    return extracted


def extract_pipeline_nets_state(checkpoint: dict[str, Any], use_ema: bool = False):
    """Extract the historical PipelineAlgo parameter namespace."""
    if use_ema:
        if "ema_state_dict" not in checkpoint:
            raise KeyError("checkpoint has no EMA state")
        return _extract_nets_state(
            checkpoint["ema_state_dict"], source_name="ema_state_dict"
        )
    return _extract_nets_state(_checkpoint_state_dict(checkpoint), source_name="state_dict")


def _require_exact_keys(actual, expected, *, label: str) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} key mismatch: missing={sorted(expected_keys - actual_keys)[:8]} "
            f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )


def strict_load_pipeline_checkpoint(algo, checkpoint: dict[str, Any], use_ema: bool = False):
    """Strictly load legacy online weights, optionally overlaying legacy EMA."""
    online = extract_pipeline_nets_state(checkpoint)
    expected = algo.nets.state_dict()
    _require_exact_keys(online, expected, label="Pipeline checkpoint")
    state = OrderedDict(online)
    if use_ema:
        averaged = extract_pipeline_nets_state(checkpoint, use_ema=True)
        parameter_keys = set(dict(algo.nets.named_parameters()))
        _require_exact_keys(averaged, parameter_keys, label="EMA parameter")
        state.update((key, averaged[key]) for key in parameter_keys)
    algo.nets.load_state_dict(state, strict=True)
    return algo


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint payload is not a dictionary")
    return checkpoint


def _checkpoint_hparams(checkpoint: Mapping[str, Any]) -> DictConfig:
    hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams"))
    if not isinstance(hparams, Mapping):
        raise TypeError("checkpoint has no mapping-valued hyper_parameters")
    return OmegaConf.create(hparams)


def _legacy_config_tree(hparams: DictConfig) -> DictConfig:
    config_tree = OmegaConf.select(hparams, "config_tree")
    if config_tree is None:
        raise RuntimeError("checkpoint hyper_parameters has no config_tree")
    config_tree = OmegaConf.create(config_tree)
    pipeline = OmegaConf.select(config_tree, "model.pipeline")
    if pipeline is None:
        raise RuntimeError("checkpoint config_tree has no model.pipeline")
    return config_tree


def _assert_model_config_match(config_path: str | Path, embedded: DictConfig) -> DictConfig:
    supplied = OmegaConf.load(str(config_path))
    supplied_model = OmegaConf.select(supplied, "model")
    embedded_model = OmegaConf.select(embedded, "model")
    if supplied_model is None or embedded_model is None:
        raise RuntimeError("both supplied and checkpoint configs must contain model")
    # Some historical checkpoints retained only ``model`` in config_tree while
    # its fields still interpolate the complete training config (for example
    # ``${planar.action_horizon}``).  Resolve that frozen model against the
    # supplied full config only as interpolation context; its model subtree
    # itself remains the checkpoint's and is compared before instantiation.
    embedded_context = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(supplied, resolve=False)),
        OmegaConf.create({"model": embedded_model}),
    )
    expected = OmegaConf.to_container(embedded_context.model, resolve=True)
    actual = OmegaConf.to_container(supplied_model, resolve=True)
    if actual != expected:
        raise RuntimeError("supplied config model does not exactly match checkpoint model")
    return supplied


def _strict_contract(
    cfg: DictConfig,
    *,
    selected_embodiment_name: str,
    selected_embodiment_id: int,
    expected_native_action_dim: int,
    action_chunk_start_index: int = 0,
    replan_every: int | None = None,
):
    if selected_embodiment_name != _LEGACY_EMBODIMENT:
        raise ValueError(
            f"legacy arc bridge supports only {_LEGACY_EMBODIMENT!r}, got "
            f"{selected_embodiment_name!r}"
        )
    expected_id = int(get_embodiment_id(_LEGACY_EMBODIMENT))
    if selected_embodiment_id != expected_id:
        raise ValueError(
            f"embodiment id mismatch: expected {expected_id}, got {selected_embodiment_id}"
        )
    if expected_native_action_dim != _LEGACY_NATIVE_ACTION_DIM:
        raise ValueError(
            f"native action dim mismatch: expected {_LEGACY_NATIVE_ACTION_DIM}, "
            f"got {expected_native_action_dim}"
        )
    action_horizon = int(OmegaConf.select(cfg, "planar.action_horizon", default=0))
    observation_horizon = int(
        OmegaConf.select(cfg, "planar.observation_horizon", default=0)
    )
    action_dim = int(
        OmegaConf.select(cfg, f"planar.action_dims.{_LEGACY_EMBODIMENT}", default=0)
    )
    waypoint_count = int(OmegaConf.select(cfg, "planar.arc_waypoints", default=0))
    raw_action_horizon = int(
        OmegaConf.select(cfg, "planar.raw_action_horizon", default=0)
    )
    velocity_mode = str(OmegaConf.select(cfg, "planar.arc_velocity_mode", default=""))
    contract = OmegaConf.select(cfg, "run_provenance.action_contract")
    if observation_horizon != _LEGACY_OBSERVATION_HORIZON:
        raise ValueError(
            "duration ARC bridge requires two observations; "
            f"got obs={observation_horizon}"
        )
    if action_dim != 5:
        raise ValueError(
            f"duration ARC bridge requires common Planar action width 5, got {action_dim}"
        )
    if waypoint_count != _LEGACY_WAYPOINT_HORIZON:
        raise ValueError(
            "duration ARC bridge requires 16 waypoints; "
            f"got {waypoint_count}"
        )
    if raw_action_horizon != _LEGACY_RAW_ACTION_HORIZON:
        raise ValueError(
            "duration ARC bridge requires a 40-step native horizon; "
            f"got {raw_action_horizon}"
        )
    expected_token_horizon = arc_token_rows(waypoint_count, velocity_mode)
    if action_horizon != expected_token_horizon or action_horizon != _DURATION_ACTION_HORIZON:
        raise ValueError(
            "duration ARC bridge requires 32 token rows; "
            f"got velocity_mode={velocity_mode!r}, tokens={action_horizon}"
        )
    if velocity_mode != "duration":
        raise ValueError(
            "duration ARC bridge refuses non-duration timing; "
            f"got {velocity_mode!r}"
        )
    if contract is None:
        raise ValueError("config has no run_provenance.action_contract")
    required_contract = {
        "representation": "planar_arc_supports_xytheta_plus_per_waypoint_duration",
        "prediction_horizon": _DURATION_ACTION_HORIZON,
        "observation_horizon": _LEGACY_OBSERVATION_HORIZON,
        "training_action_target_offset": 1,
        "waypoint_count": _LEGACY_WAYPOINT_HORIZON,
        "timing_rows": _LEGACY_WAYPOINT_HORIZON,
        "timing_semantics": "per_waypoint_interval_duration_seconds",
        "rollout_action_chunk_start_index": 0,
        "execution_horizon": _LEGACY_RAW_ACTION_HORIZON,
        "execution_slice": "[0,40)",
    }
    for key, expected in required_contract.items():
        actual = OmegaConf.select(contract, key, default=None)
        if actual != expected:
            raise ValueError(
                f"duration ARC contract mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}"
            )
    if action_chunk_start_index != 0:
        raise ValueError("duration ARC bridge requires action_chunk_start_index=0")
    if replan_every is not None:
        raise ValueError(
            "duration ARC timing is carried inside the predicted chunk; "
            "explicit replan_every is invalid"
        )
    decoder_cfg = OmegaConf.select(cfg, "planar.eval_native_decoder")
    if decoder_cfg is None:
        raise ValueError("config has no planar.eval_native_decoder")
    decoder = instantiate(decoder_cfg)
    token_probe = torch.zeros(1, action_horizon, action_dim)
    if not hasattr(decoder, "decode"):
        raise ValueError("duration ARC rollout requires a native trajectory decoder")
    decoded = decoder.decode(token_probe)
    decoded_horizon = int(getattr(decoder, "action_horizon", 0))
    if decoded_horizon != _LEGACY_RAW_ACTION_HORIZON:
        raise ValueError(
            "duration ARC decoder must expose the reconstructed raw horizon; "
            f"expected {_LEGACY_RAW_ACTION_HORIZON}, got {decoded_horizon}"
        )
    if tuple(decoded.shape) != (1, decoded_horizon, expected_native_action_dim):
        raise ValueError(
            "native decoder output mismatch: expected "
            f"(1, {decoded_horizon}, {expected_native_action_dim}), "
            f"got {tuple(decoded.shape)}"
        )
    if getattr(decoder, "timing_semantics", None) != _DURATION_TIMING_SEMANTICS:
        raise ValueError(
            "duration ARC decoder does not declare per-waypoint duration timing"
        )
    if str(getattr(decoder, "velocity_mode", "")) != "duration":
        raise ValueError("duration ARC decoder is not configured for duration timing")
    return decoder


def _normalizer_from_config(cfg: DictConfig) -> MultiDataset:
    """Load the recorded, data-owned normalizer for these legacy checkpoints."""
    path = Path(str(OmegaConf.select(cfg, "norm_stats.precomputed_norm_path")))
    expected = str(OmegaConf.select(cfg, "run_provenance.normalization_sha256"))
    if not path.is_file():
        raise RuntimeError(f"recorded normalizer is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"normalizer SHA mismatch: actual={actual} expected={expected}"
        )
    payload = json.loads(path.read_text())
    stats = payload.get("stats", {}).get("19")
    if not isinstance(stats, dict) or {"state_agent_obj", "actions"} - set(stats):
        raise RuntimeError("recorded normalizer lacks U-socket state/action stats")
    return MultiDataset.from_state(
        {
            "norm_mode": str(OmegaConf.select(cfg, "norm_stats.norm_mode")),
            "embodiments": [19],
            "key_types": {19: {"state_agent_obj": "proprio_keys", "actions": "action_keys"}},
            "zarr_keys": {19: {"state_agent_obj": "state_agent_obj", "actions": "actions"}},
            "norm_stats": {19: stats},
        }
    )


def _load_legacy_policy(
    *,
    ckpt_path: str | Path,
    config_path: str | Path,
    selected_embodiment_name: str,
    selected_embodiment_id: int,
    expected_native_action_dim: int,
    use_ema: bool,
    device: torch.device,
    action_chunk_start_index: int = 0,
    replan_every: int | None = None,
):
    checkpoint = _load_checkpoint(ckpt_path)
    hparams = _checkpoint_hparams(checkpoint)
    embedded = _legacy_config_tree(hparams)
    cfg = _assert_model_config_match(config_path, embedded)
    decoder = _strict_contract(
        cfg,
        selected_embodiment_name=selected_embodiment_name,
        selected_embodiment_id=selected_embodiment_id,
        expected_native_action_dim=expected_native_action_dim,
        action_chunk_start_index=action_chunk_start_index,
        replan_every=replan_every,
    )
    # The serialized training ``model`` also carries optimizer/scheduler
    # settings, which are not rollout state and are not constructor arguments
    # for the historical ModelWrapper.  The checkpoint state itself is exactly
    # ``nets.pipeline.*``, so build only that recorded policy graph.
    pipeline_cfg = OmegaConf.select(cfg, "model.pipeline")
    if pipeline_cfg is None:
        raise RuntimeError("supplied config has no model.pipeline")
    algo = instantiate(pipeline_cfg)
    if not hasattr(algo, "nets"):
        raise TypeError("model.pipeline did not instantiate a PipelineAlgo")
    strict_load_pipeline_checkpoint(algo, checkpoint, use_ema=use_ema)
    algo.device = device
    algo.nets.to(device)
    algo.nets.eval()
    normalizer = _normalizer_from_config(cfg)
    return algo, normalizer, decoder, checkpoint, cfg


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
    """Strictly prove loading/decoding compatibility without simulator rollout."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo, _normalizer, decoder, checkpoint, cfg = _load_legacy_policy(
        ckpt_path=ckpt_path,
        config_path=config_path,
        selected_embodiment_name=selected_embodiment_name,
        selected_embodiment_id=selected_embodiment_id,
        expected_native_action_dim=expected_native_action_dim,
        use_ema=use_ema,
        device=device,
        action_chunk_start_index=action_chunk_start_index,
        replan_every=replan_every,
    )
    action_horizon = int(OmegaConf.select(cfg, "planar.action_horizon"))
    decoded_horizon = int(getattr(decoder, "action_horizon", 0))
    parameter_keys = len(algo.nets.state_dict())
    return {
        "status": "audited_strict_no_rollout_ok",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "weights": "ema" if use_ema else "raw",
        "model_token_horizon": action_horizon,
        "decoded_action_horizon": decoded_horizon,
        "native_action_dim": expected_native_action_dim,
        "state_tensor_count": parameter_keys,
        "action_chunk_start_index": int(action_chunk_start_index),
        "execution_stop_index_exclusive": decoded_horizon,
        "execution_horizon": decoded_horizon,
        "execution_horizon_semantics": "fixed_40_step_duration_decoded_chunk",
        "replan_every": decoded_horizon,
        "replan_semantics": "decoded_fixed_duration_chunk_horizon",
        "timing_semantics": getattr(decoder, "timing_semantics", None),
        "timing_rows": _LEGACY_WAYPOINT_HORIZON,
        "timing_representation": "per_waypoint_interval_duration_seconds",
        "sampler_inference_steps": int(
            OmegaConf.select(
                cfg,
                "model.pipeline.stages.3.policy.num_inference_steps",
                default=0,
            )
        ),
        "bridge": "pipeline_arc_duration_rollout_v1",
    }


class _LegacyArcPolicy:
    def __init__(
        self,
        *,
        algo,
        normalizer: MultiDataset,
        decoder,
        embodiment_id: int,
        device: torch.device,
    ):
        self.algo = algo
        self.normalizer = normalizer
        self.decoder = decoder
        self.embodiment_id = embodiment_id
        self.device = device
        self.token_horizon = arc_token_rows(
            int(getattr(decoder, "num_waypoints", _LEGACY_WAYPOINT_HORIZON)),
            str(getattr(decoder, "velocity_mode", "duration")),
        )
        self.decoded_horizon = int(getattr(decoder, "action_horizon", 0))
        self.timing_semantics = str(getattr(decoder, "timing_semantics", ""))
        if self.timing_semantics != _DURATION_TIMING_SEMANTICS:
            raise ValueError(
                "duration ARC policy requires per-waypoint duration decoder timing"
            )
        if self.decoded_horizon != _LEGACY_RAW_ACTION_HORIZON:
            raise ValueError(
                "duration ARC policy requires a 40-step decoded action horizon"
            )
        self._history: list[dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self._history.clear()

    def _window(self, obs_env: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        current = _env_to_zarr_pushshapes_oriented(dict(obs_env), self.device)
        self._history.append(current)
        self._history = self._history[-_LEGACY_OBSERVATION_HORIZON:]
        history = [self._history[0]] * (_LEGACY_OBSERVATION_HORIZON - len(self._history))
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
        prediction = self.algo.forward_eval({_LEGACY_EMBODIMENT: normalized})
        result = prediction.get(_LEGACY_EMBODIMENT)
        if not isinstance(result, Mapping) or "pred_action" not in result:
            raise RuntimeError("legacy PipelineAlgo inference did not produce pred_action")
        tokens = result["pred_action"]
        if not torch.is_tensor(tokens) or tuple(tokens.shape[-2:]) != (self.token_horizon, 5):
            raise RuntimeError(
                "legacy PipelineAlgo returned unexpected arc token shape "
                f"{getattr(tokens, 'shape', None)}"
            )
        actions = self.normalizer.unnormalize({"actions": tokens}, self.embodiment_id)[
            "actions"
        ]
        native = self.decoder.decode(actions)
        decoded_horizon = self.decoded_horizon
        if not torch.is_tensor(native):
            native = torch.as_tensor(native, device=self.device)
        if native.ndim == 2:
            native = native.unsqueeze(0)
        if native.ndim != 3 or native.shape[0] != 1:
            raise RuntimeError(
                f"duration ARC decoder returned unexpected batch shape {native.shape}"
            )
        value = native.detach().float().cpu().numpy()
        if value.ndim != 3 or value.shape[-2:] != (
            decoded_horizon,
            _LEGACY_NATIVE_ACTION_DIM,
        ):
            raise RuntimeError(
                "legacy PipelineAlgo returned unexpected native chunk shape "
                f"{value.shape}; expected (batch, {decoded_horizon}, "
                f"{_LEGACY_NATIVE_ACTION_DIM})"
            )
        length = decoded_horizon
        if length <= 0 or length > decoded_horizon:
            raise RuntimeError("duration ARC decoder returned an invalid trajectory length")
        if not np.all(np.isfinite(value)):
            raise RuntimeError("duration ARC decoder produced non-finite actions")
        chunk = value[0, :length]
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError(
                "legacy PipelineAlgo produced non-finite native action chunk "
                f"{chunk.tolist()}"
            )
        return chunk.astype(np.float32, copy=False)

    def predict_native_action(self, obs_env: Mapping[str, Any]) -> np.ndarray:
        """Compatibility helper returning the first action of a decoded chunk."""
        return self.predict_native_actions(obs_env)[0]


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


def _rollout_one(args, policy: _LegacyArcPolicy, seed: int, ep_idx: int):
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
            if action_chunk is None or chunk_offset >= len(action_chunk):
                action_chunk = policy.predict_native_actions(env._get_obs())
                chunk_offset = 0
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
            f"{args.rollout_timeout}s; logging 0-coverage and continuing."
        )
    finally:
        if args.rollout_timeout > 0:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
        env.close()
    return (
        max_coverage if args.max_coverage else coverage,
        actions,
        frames,
        chunk_lengths,
    )


def _validate_runtime_args(args) -> None:
    if args.eval_class not in {"packed", "hpt"}:
        raise ValueError("eval-class must be packed or hpt")
    if args.embodiment_name != _LEGACY_EMBODIMENT:
        raise ValueError(f"only {_LEGACY_EMBODIMENT} is supported")
    expected_id = int(get_embodiment_id(_LEGACY_EMBODIMENT))
    if args.only_emb != expected_id:
        raise ValueError(f"only-emb must be {expected_id}")
    if args.n_episodes <= 0 or args.max_steps <= 0:
        raise ValueError("n-episodes and max-steps must be positive")
    if args.replan_every is not None:
        raise ValueError(
            "arc spline timing chooses replan length from terminal speed; "
            "do not pass --replan-every"
        )
    if args.action_chunk_start_index != 0:
        raise ValueError("the legacy arc bridge only permits action-chunk-start-index=0")
    if args.sampler_inference_steps is not None and args.sampler_inference_steps != 100:
        raise ValueError("the legacy Paper-DP bridge only permits its recorded 100 sampler steps")
    if args.init_mode != "seeds":
        raise ValueError("the canonical bridge requires init-mode=seeds")
    if not args.init_seeds:
        args.init_seeds = [args.init_seed_base + index for index in range(args.n_episodes)]
    if len(args.init_seeds) < args.n_episodes:
        raise ValueError("init-seeds must include one seed per requested episode")


def run(args) -> None:
    _validate_runtime_args(args)
    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise RuntimeError(f"canonical launcher did not create output directory {out_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo, normalizer, decoder, _checkpoint, _cfg = _load_legacy_policy(
        ckpt_path=args.ckpt,
        config_path=args.config_path,
        selected_embodiment_name=args.embodiment_name,
        selected_embodiment_id=args.only_emb,
        expected_native_action_dim=_LEGACY_NATIVE_ACTION_DIM,
        use_ema=args.use_ema,
        device=device,
        action_chunk_start_index=args.action_chunk_start_index,
        replan_every=args.replan_every,
    )
    policy = _LegacyArcPolicy(
        algo=algo,
        normalizer=normalizer,
        decoder=decoder,
        embodiment_id=args.only_emb,
        device=device,
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
            coverage, actions, frames, chunk_lengths = _rollout_one(
                args, policy, seed, ep_idx
            )
        coverages.append(float(coverage))
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
                "decoded_chunk_length_min": min(chunk_lengths),
                "decoded_chunk_length_max": max(chunk_lengths),
                "decoded_chunk_length_mean": float(np.mean(chunk_lengths)),
                "video": str(video_path) if video_path else None,
            }
        )
    print(
        f"[sim] emb{args.only_emb} ep_coverages: "
        + ",".join(f"{value:.4f}" for value in coverages)
    )
    summary = {
        "bridge": "pipeline_arc_duration_rollout_v1",
        "comparability": "noncanonical_historical_evaluator_bridge",
        "weights": "ema" if args.use_ema else "raw",
        "embodiment": args.embodiment_name,
        "embodiment_id": args.only_emb,
        "model_token_horizon": policy.token_horizon,
        "decoded_action_horizon": int(getattr(policy.decoder, "action_horizon", 0)),
        "action_chunk_start_index": int(args.action_chunk_start_index),
        "waypoint_count": _LEGACY_WAYPOINT_HORIZON,
        "execution_slice": "[0,40)",
        "execution_horizon": 40,
        "replan_every": "decoded_fixed_duration_chunk_horizon",
        "timing_semantics": getattr(policy.decoder, "timing_semantics", None),
        "sampler_inference_steps": 100,
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
