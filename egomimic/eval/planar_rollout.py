"""Shared planar graph inference with control-rate observation history.

This module owns inference and action queuing, not the canonical evaluation
protocol. Simulator settings, seeds, episode limits and scoring belong to the
calling evaluator's verified protocol.
"""

import hashlib
from collections import deque
from pathlib import Path

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
    def predict_native_actions(self):
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
        return native[0].detach().float().cpu().numpy().copy()


class PlanarActionQueue:
    """Execute a decoded prefix from index zero while observing every step."""

    def __init__(self, policy, *, execution_horizon):
        self.policy = policy
        self.execution_horizon = int(execution_horizon)
        if not 0 < self.execution_horizon <= policy.native_shape[0]:
            raise ValueError("Execution horizon must fit within the decoded trajectory")
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

    def next_action(self):
        if self._step is None:
            raise RuntimeError("Reset the action queue first")
        if self._pending_observation:
            raise RuntimeError(
                "Observe the last executed action before requesting another"
            )
        if self._chunk is None or self._offset == self.execution_horizon:
            self._chunk = self.policy.predict_native_actions()
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
    decoder = instantiate(cfg.planar.eval_native_decoder)
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
