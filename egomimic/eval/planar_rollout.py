"""Shared planar graph inference with control-rate observation history.

This module owns inference and action queuing, not the canonical evaluation
protocol. Simulator settings, seeds, episode limits and scoring belong to the
calling evaluator's verified protocol.
"""

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.eval.normalization import load_normalizer
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes_sim import _ENV_TO_ZARR


def require_file_hash(path, expected):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


@dataclass(frozen=True)
class PlanarActionPrediction:
    native_actions: np.ndarray
    tokens: torch.Tensor


class PlanarArcExecutionSelector:
    """Choose supports first, then execute their control-rate time prefix.

    Distance is accumulated between unnormalized geometric waypoints using
    the decoder/tokenizer metric. It is independent of the native time horizon.
    The largest prefix within the distance budget includes holds at its boundary.
    Control samples at or before the last selected support time are retained;
    off-grid endpoints are rounded down, with at least the anchored first action.
    """

    def __init__(self, *, mode, fraction=0.5, distance_budget=None):
        if mode not in {"waypoint_fraction", "distance_fraction"}:
            raise ValueError("Select waypoint_fraction or distance_fraction")
        self.mode = mode
        self.fraction = float(fraction)
        if not math.isfinite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError("fraction must be finite and in (0, 1]")
        self.distance_budget = (
            None if distance_budget is None else float(distance_budget)
        )
        if mode == "distance_fraction":
            if self.distance_budget is None or not (
                math.isfinite(self.distance_budget) and self.distance_budget > 0
            ):
                raise ValueError("distance_fraction requires a positive distance_budget")
        elif self.distance_budget is not None:
            raise ValueError("waypoint_fraction does not use a distance_budget")

    def select(self, tokens, decoder):
        distances, elapsed = decoder.waypoint_schedule(tokens)
        if not torch.isfinite(distances).all() or not torch.isfinite(elapsed).all():
            raise ValueError("Nonfinite ARC waypoint schedule")
        threshold = None
        if self.mode == "waypoint_fraction":
            count = max(1, math.floor(len(distances) * self.fraction))
        else:
            threshold = self.distance_budget * self.fraction
            count = int((distances <= threshold).sum().item())
        endpoint = count - 1
        end_time = float(elapsed[endpoint].item())
        stage = decoder.detokenizer
        # Match the decoder's actual floating-point time grid. The tolerance
        # only absorbs accumulation error at an exact control-period boundary.
        times = torch.arange(
            decoder.action_horizon, device=elapsed.device, dtype=elapsed.dtype
        ) * stage.dt
        steps = max(
            1, int((times <= elapsed[endpoint] + stage.dt * 1e-5).sum().item())
        )
        return {
            "mode": self.mode,
            "fraction": self.fraction,
            "distance_budget": self.distance_budget,
            "distance_threshold": threshold,
            "distance_metric": "translation_plus_weighted_chordal_rotation",
            "rotation_radius": stage.rotation_radius,
            "waypoint_count": len(distances),
            "selected_waypoint_count": count,
            "last_selected_waypoint_index": endpoint,
            "selected_distance": float(distances[endpoint].item()),
            "total_distance": float(distances[-1].item()),
            "waypoint_distances": distances.detach().cpu().tolist(),
            "waypoint_times_seconds": elapsed.detach().cpu().tolist(),
            "selected_endpoint_seconds": end_time,
            "execution_start_index": 0,
            "execution_steps": steps,
            "last_control_sample_seconds": float(times[steps - 1].item()),
            "native_horizon_clipped": end_time > float(times[-1].item()),
        }


class PlanarTimedArcExecutionSelector:
    """Stop at the first active stream's half-support endpoint.

    Translation/grip and rotation may have different clocks. Constant streams
    impose no constraint; when both are stationary, use their duration clocks.
    The fraction counts geometric supports, never native control samples.
    """

    def __init__(self, fraction=0.5):
        self.fraction = float(fraction)
        if not math.isfinite(self.fraction) or not 0 < self.fraction <= 1:
            raise ValueError("fraction must be finite and in (0, 1]")

    def select(self, tokens, decoder):
        clocks = decoder.waypoint_clocks(tokens)
        count = max(1, math.floor(decoder.num_waypoints * self.fraction))
        active = [row["seconds"] for row in clocks.values() if row["active"]]
        selected = active or [row["seconds"] for row in clocks.values()]
        endpoint = torch.stack([seconds[count - 1] for seconds in selected]).min()
        times = torch.arange(decoder.action_horizon, device=endpoint.device,
                             dtype=endpoint.dtype) * decoder.dt
        steps = max(1, int((times <= endpoint + decoder.dt * 1e-5).sum().item()))
        return {
            "mode": "independent_clocks_waypoint_fraction",
            "fraction": self.fraction,
            "waypoint_count": decoder.num_waypoints,
            "selected_waypoint_count": count,
            "selected_endpoint_seconds": float(endpoint.item()),
            "execution_start_index": 0,
            "execution_steps": steps,
            "clocks": {name: {"active": row["active"],
                                "seconds": row["seconds"].detach().cpu().tolist()}
                       for name, row in clocks.items()},
        }


class PlanarGraphPolicy:
    """Normalize observations and decode predictions with the training contract.

    Call observe once at reset and after EVERY simulator step, including steps
    between predictions. Explicit consecutive indices reject stale history.
    """

    def __init__(
        self,
        *,
        graph,
        normalizer,
        decoder,
        embodiment_name,
        observation_horizon,
        token_shape,
        native_shape,
    ):
        if embodiment_name not in _ENV_TO_ZARR:
            raise ValueError(f"No simulator observation adapter for {embodiment_name}")
        self.graph, self.normalizer, self.decoder = graph, normalizer, decoder
        self.embodiment_name = embodiment_name
        self.embodiment_id = get_embodiment_id(embodiment_name)
        self.observation_horizon = int(observation_horizon)
        self.token_shape = tuple(int(x) for x in token_shape)
        self.native_shape = tuple(int(x) for x in native_shape)
        if (
            self.observation_horizon <= 0
            or min(*self.token_shape, *self.native_shape) <= 0
        ):
            raise ValueError("Observation and action shapes must be positive")
        if len(self.token_shape) != 2 or len(self.native_shape) != 2:
            raise ValueError("Action shapes must be (horizon, width)")
        if self.embodiment_id not in normalizer.embodiments:
            raise ValueError("Normalizer does not contain the selected embodiment")
        for key in ("state_agent_obj", "actions"):
            if key not in normalizer.key_types[self.embodiment_id]:
                raise ValueError(f"Normalizer is missing {key}")
            if (
                normalizer.norm_mode != "none"
                and key not in normalizer.norm_stats[self.embodiment_id]
            ):
                raise ValueError(f"Normalizer is missing statistics for {key}")
        if (
            tuple(normalizer.key_shape("actions", self.embodiment_id))
            != self.token_shape
        ):
            raise ValueError("Normalizer action shape differs from model token shape")
        self._history = deque(maxlen=self.observation_horizon)
        self._last_step = None

    def reset(self):
        self._history.clear()
        self._last_step = None

    def observe(self, observation, *, step):
        expected = 0 if self._last_step is None else self._last_step + 1
        if step != expected:
            raise ValueError(
                f"Expected consecutive control observation {expected}, got {step}"
            )
        current = _ENV_TO_ZARR[self.embodiment_name](observation, self.graph.device)
        if any(not torch.isfinite(value).all() for value in current.values()):
            raise ValueError("Nonfinite simulator observation")
        # Some simulators reuse observation buffers. Store a snapshot.
        self._history.append(
            {key: value.detach().clone() for key, value in current.items()}
        )
        self._last_step = int(step)

    def observation_window(self):
        if not self._history:
            raise RuntimeError("Observe the reset state before inference")
        history = [self._history[0]] * (self.observation_horizon - len(self._history))
        history += list(self._history)
        return {
            key: torch.stack([row[key] for row in history], dim=1)
            for key in history[-1]
        }

    @torch.inference_mode()
    def predict_action_plan(self):
        normalized = self.normalizer.normalize(
            self.observation_window(), self.embodiment_id
        )
        batch = self.graph.process_batch_for_training(
            {self.embodiment_name: normalized}
        )
        prediction = self.graph.forward_eval(batch)[self.embodiment_name]["pred_action"]
        if not torch.is_tensor(prediction) or tuple(prediction.shape) != (
            1,
            *self.token_shape,
        ):
            raise ValueError(
                f"Unexpected prediction shape: {getattr(prediction, 'shape', None)}"
            )
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite model prediction")
        actions = self.normalizer.unnormalize(
            {"actions": prediction}, self.embodiment_id
        )["actions"]
        native = torch.as_tensor(self.decoder.decode(actions))
        if (
            tuple(native.shape) != (1, *self.native_shape)
            or not torch.isfinite(native).all()
        ):
            raise ValueError(
                f"Invalid decoded native trajectory: {tuple(native.shape)}"
            )
        return PlanarActionPrediction(
            native_actions=native[0].detach().float().cpu().numpy().copy(),
            tokens=actions[0].detach().clone(),
        )

    def predict_native_actions(self):
        return self.predict_action_plan().native_actions


class PlanarActionQueue:
    """Execute a decoded prefix from index zero while observing every step."""

    def __init__(self, policy, *, execution_horizon=None, execution_selector=None):
        self.policy = policy
        if (execution_horizon is None) == (execution_selector is None):
            raise ValueError("Supply exactly one execution horizon or selector")
        self.execution_selector = execution_selector
        self.execution_horizon = (
            None if execution_horizon is None else int(execution_horizon)
        )
        if self.execution_horizon is not None:
            if not 0 < self.execution_horizon <= policy.native_shape[0]:
                raise ValueError("Execution horizon must fit within the decoded trajectory")
        self.prediction_count = 0
        self.last_prediction = None
        self.last_execution = None
        self._chunk_horizon = None
        self._pending_observation = False
        self._chunk = None
        self._offset = 0
        self._step = None

    def reset(self, observation):
        self.policy.reset()
        self.policy.observe(observation, step=0)
        self._pending_observation = False
        self._chunk = None
        self._offset = 0
        self._step = 0
        self.prediction_count = 0
        self.last_prediction = None
        self.last_execution = None
        self._chunk_horizon = None

    def next_action(self):
        if self._step is None:
            raise RuntimeError("Reset the action queue first")
        if self._pending_observation:
            raise RuntimeError(
                "Observe the last executed action before requesting another"
            )
        if self._chunk is None or self._offset == self._chunk_horizon:
            if self.execution_selector is None:
                self._chunk = self.policy.predict_native_actions()
                receipt = {
                    "mode": "fixed_native_steps",
                    "execution_start_index": 0,
                    "execution_steps": self.execution_horizon,
                }
            else:
                self.last_prediction = self.policy.predict_action_plan()
                self._chunk = self.last_prediction.native_actions
                receipt = self.execution_selector.select(
                    self.last_prediction.tokens, self.policy.decoder
                )
            self._chunk_horizon = int(receipt["execution_steps"])
            if not 0 < self._chunk_horizon <= len(self._chunk):
                raise ValueError("Selected execution prefix is outside the native chunk")
            self.last_execution = {**receipt, "replan_control_step": self._step}
            self.prediction_count += 1
            self._offset = 0
        action = self._chunk[self._offset].copy()
        self._offset += 1
        self._pending_observation = True
        return action

    def observe(self, observation):
        if not self._pending_observation:
            raise RuntimeError("No executed action awaits an observation")
        self.policy.observe(observation, step=self._step + 1)
        self._step += 1
        self._pending_observation = False


def load_planar_graph_policy(
    *,
    checkpoint_path,
    config_path,
    normalizer_path,
    checkpoint_sha256,
    config_sha256,
    normalizer_sha256,
    embodiment_name,
    device,
    use_ema,
):
    """Restore exact model weights, decoder and exported normalization state."""
    if type(use_ema) is not bool:
        raise TypeError("Select raw/EMA explicitly with a boolean")
    for path, expected in (
        (checkpoint_path, checkpoint_sha256),
        (config_path, config_sha256),
        (normalizer_path, normalizer_sha256),
    ):
        require_file_hash(path, expected)
    cfg = OmegaConf.load(config_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    embedded = OmegaConf.create(checkpoint["hyper_parameters"]["config_tree"])
    if OmegaConf.to_container(embedded.model, resolve=True) != OmegaConf.to_container(
        cfg.model, resolve=True
    ):
        raise ValueError("Checkpoint model differs from the supplied training config")
    contract = OmegaConf.select(cfg, "run_provenance.action_contract")
    saved_contract = OmegaConf.select(embedded, "run_provenance.action_contract")
    if contract is None or contract != saved_contract:
        raise ValueError("Checkpoint action contract differs from training config")
    if contract.rollout_action_chunk_start_index != 0:
        raise ValueError(
            "Planar queue requires decoded execution starting at index zero"
        )
    if int(cfg.planar.action_target_offset) != int(cfg.planar.observation_horizon) - 1:
        raise ValueError(
            "Planar rollout requires aligned pre-step observation/action windows"
        )
    if embodiment_name not in cfg.data.train_datasets:
        raise ValueError("Embodiment was not present in the training config")
    normalizer = load_normalizer(normalizer_path)
    if normalizer.norm_mode != cfg.norm_stats.norm_mode:
        raise ValueError("Normalizer mode differs from the training config")
    graph = instantiate(cfg.model.pipeline, device=str(device))
    if not isinstance(graph, PipelineAlgo):
        raise TypeError("Planar inference requires the shared PipelineAlgo")
    graph.bind_data_context(normalizer=normalizer)
    strict_load_pipeline_checkpoint(graph, checkpoint, use_ema=use_ema)
    graph.nets.eval()
    decoder_config = cfg.planar.eval_native_decoder
    if decoder_config is None:
        decoder_config = OmegaConf.select(cfg, f"evaluator.native_decoders.{embodiment_name}")
    if decoder_config is None:
        raise ValueError(f"No native decoder configured for {embodiment_name}")
    decoder = instantiate(decoder_config)
    return PlanarGraphPolicy(
        graph=graph,
        normalizer=normalizer,
        decoder=decoder,
        embodiment_name=embodiment_name,
        observation_horizon=cfg.planar.observation_horizon,
        token_shape=(
            cfg.planar.action_horizon,
            cfg.planar.action_dims[embodiment_name],
        ),
        native_shape=(decoder.action_horizon, decoder.native_action_dim),
    )
