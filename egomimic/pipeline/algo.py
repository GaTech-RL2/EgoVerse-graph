"""Training and teacher-forced evaluation adapter for policy pipelines."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.nn as nn

from egomimic.algo.algo import Algo
from egomimic.pipeline.core import Pipeline, Stage, sum_losses
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class PipelineAlgo(Algo):
    """Expose a dependency-aware :class:`Pipeline` through ``Algo``.

    Policy behavior belongs to stages. This adapter only maps normalized loader
    batches into the pipeline namespace and reduces explicit ``loss/*`` values.
    """

    def __init__(
        self,
        stages: Iterable[Stage],
        norm_stats,
        domains: list[str],
        ac_keys: dict[str, str],
        auxiliary_ac_keys: dict | None = None,
        action_horizon: int = 1,
        rollout_adapter=None,
        rollout_adapters: dict | None = None,
        rollout_observation_adapters: dict | None = None,
        device=None,
    ):
        super().__init__()
        self.norm_stats = norm_stats
        self.domains = [str(domain) for domain in domains]
        self.ac_keys = {str(domain): str(key) for domain, key in ac_keys.items()}
        self.auxiliary_ac_keys = dict(auxiliary_ac_keys or {})
        self.action_horizon = int(action_horizon)
        self.rollout_adapter = rollout_adapter
        self.rollout_adapters = dict(rollout_adapters or {})
        self.rollout_observation_adapters = dict(rollout_observation_adapters or {})
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not self.domains:
            raise ValueError("PipelineAlgo requires at least one domain")
        if set(self.ac_keys) != set(self.domains):
            raise ValueError(
                "ac_keys must exactly match domains: "
                f"domains={sorted(self.domains)} ac_keys={sorted(self.ac_keys)}"
            )
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        unknown_adapters = set(self.rollout_adapters) - set(self.domains)
        if unknown_adapters:
            raise ValueError(
                "Rollout adapters configured for unknown domains: "
                f"{sorted(unknown_adapters)}"
            )
        unknown_observation_adapters = set(self.rollout_observation_adapters) - set(
            self.domains
        )
        if unknown_observation_adapters:
            raise ValueError(
                "Rollout observation adapters configured for unknown domains: "
                f"{sorted(unknown_observation_adapters)}"
            )

        self.domain_by_id = {
            get_embodiment_id(domain): domain for domain in self.domains
        }
        if len(self.domain_by_id) != len(self.domains):
            raise ValueError("domains resolve to duplicate embodiment IDs")

        self._resolve_keys()
        self.nets = nn.ModuleDict({"policy": Pipeline(list(stages))})
        self.nets.to(self.device)

    @property
    def policy(self) -> Pipeline:
        return self.nets["policy"]

    def _domain_name(self, domain: str | int) -> str:
        if isinstance(domain, int):
            if domain not in self.domain_by_id:
                raise KeyError(f"Unknown rollout embodiment id {domain}")
            return self.domain_by_id[domain]
        name = str(domain)
        if name not in self.domains:
            raise KeyError(
                f"Unknown rollout domain {name!r}; configured={self.domains}"
            )
        return name

    def rollout_adapter_for(self, domain: str | int):
        """Return a domain adapter, with the legacy singleton as fallback."""
        name = self._domain_name(domain)
        return self.rollout_adapters.get(name, self.rollout_adapter)

    def rollout_observation_adapter_for(self, domain: str | int):
        """Return the optional adapter applied before rollout normalization."""
        return self.rollout_observation_adapters.get(self._domain_name(domain))

    def _resolve_keys(self) -> None:
        self.proprio_keys: dict[int, list[str]] = {}
        self.lang_keys: dict[int, list[str]] = {}
        self.camera_keys: dict[int, list[str]] = {}
        self.resolved_ac_keys: dict[int, str] = {}

        for domain in self.domains:
            emb_id = get_embodiment_id(domain)
            for key_type, destination in (
                ("proprio_keys", self.proprio_keys),
                ("lang_keys", self.lang_keys),
                ("camera_keys", self.camera_keys),
            ):
                destination[emb_id] = [
                    key
                    for key in self.norm_stats.keys_of_type(key_type, emb_id)
                    if self.norm_stats.is_key_with_embodiment(key, emb_id)
                ]

            available_actions = {
                key
                for key in self.norm_stats.keys_of_type("action_keys", emb_id)
                if self.norm_stats.is_key_with_embodiment(key, emb_id)
            }
            requested = self.ac_keys[domain]
            if requested not in available_actions:
                raise KeyError(
                    f"Action key {requested!r} is unavailable for {domain!r}; "
                    f"available={sorted(available_actions)}"
                )
            self.resolved_ac_keys[emb_id] = requested

    def _prepare_loader_batch(self, batch: dict, emb_id: int) -> dict:
        """Translate zarr keys; standard ``MultiDataset`` output is normalized."""
        prepared = {}
        for key, value in batch.items():
            mapped = self.norm_stats.zarr_key_to_keyname(key, emb_id)
            if mapped is not None:
                prepared[mapped] = value
        return prepared

    def _build_obs(self, batch: dict, emb_id: int) -> dict:
        keys = (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        )
        return {key: batch[key] for key in keys if key in batch}

    def _seed(self, emb_id: int, batch: dict, *, include_actions: bool = True) -> dict:
        seed = {"embodiment": self.domain_by_id[emb_id]}
        if include_actions:
            action_key = self.resolved_ac_keys[emb_id]
            seed["actions"] = batch[action_key]
        seed.update(
            {
                f"obs/{key}": value
                for key, value in self._build_obs(batch, emb_id).items()
            }
        )
        return seed

    def _move_to_device(self, batch: dict) -> dict:
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                value = value.to(self.device)
                if value.is_floating_point():
                    value = value.float()
            moved[key] = value
        return moved

    def process_batch_for_training(self, batch: dict) -> dict:
        processed = {}
        for domain, loader_batch in batch.items():
            emb_id = get_embodiment_id(domain)
            if emb_id not in self.domain_by_id:
                raise KeyError(f"Unexpected embodiment batch {domain!r}")
            prepared = self._prepare_loader_batch(loader_batch, emb_id)
            processed[emb_id] = self._move_to_device(prepared)
        return processed

    def process_batch_for_rollout(self, batch: dict) -> dict:
        """Adapt, normalize, and map observations created outside a dataset."""
        processed = {}
        for domain, loader_batch in batch.items():
            emb_id = get_embodiment_id(domain)
            if emb_id not in self.domain_by_id:
                raise KeyError(f"Unexpected embodiment batch {domain!r}")
            rollout_input = dict(loader_batch)
            observation_adapter = self.rollout_observation_adapter_for(domain)
            if observation_adapter is not None:
                rollout_input = observation_adapter.encode(rollout_input)
            for action_key in self.norm_stats.keys_of_type("action_keys", emb_id):
                rollout_input.pop(action_key, None)
                zarr_key = self.norm_stats.keyname_to_zarr_key(action_key, emb_id)
                if zarr_key is not None:
                    rollout_input.pop(zarr_key, None)
            normalized = self.norm_stats.normalize(rollout_input, emb_id)
            prepared = self._prepare_loader_batch(normalized, emb_id)
            processed[emb_id] = self._move_to_device(prepared)
        return processed

    def _rollout_observation_horizon(self) -> int:
        horizons = {
            int(stage.rollout_obs_steps)
            for stage in self.policy.stages
            if hasattr(stage, "rollout_obs_steps")
        }
        if len(horizons) != 1:
            raise RuntimeError(
                "Pipeline rollout requires exactly one observation horizon; "
                f"found {sorted(horizons)}"
            )
        return horizons.pop()

    def _rollout_seed(self, emb_id: int, batch: dict, rollout_t: int) -> dict:
        seed = self._seed(emb_id, batch, include_actions=False)
        n_obs_steps = self._rollout_observation_horizon()
        for key, value in list(seed.items()):
            if not key.startswith("obs/") or not torch.is_tensor(value):
                continue
            if n_obs_steps == 1:
                value = value.unsqueeze(1)
            elif value.ndim < 2 or int(value.shape[1]) != n_obs_steps:
                raise ValueError(
                    f"Rollout {key} must carry {n_obs_steps} observations, got "
                    f"shape {tuple(value.shape)}"
                )
            seed[key] = value
        seed["rollout_t"] = int(rollout_t)
        return seed

    def _run_rollout_policy(self, seed: dict) -> dict:
        runnable, excluded = self.policy.plan(seed.keys(), mode="rollout")
        blocked = [
            (type(stage).__name__, missing)
            for stage, missing in excluded
            if missing != ["<train-only>"]
        ]
        if blocked:
            raise RuntimeError(f"Pipeline rollout graph has blocked stages: {blocked}")
        result = seed
        for stage in runnable:
            result = stage(result)
        return result

    @torch.inference_mode()
    def forward_rollout(self, batch: dict, rollout_t: int = 0) -> OrderedDict:
        """Run the action-free graph, unnormalize tokens, then decode them."""
        predictions = OrderedDict()
        for emb_id, loader_batch in batch.items():
            result = self._run_rollout_policy(
                self._rollout_seed(emb_id, loader_batch, rollout_t)
            )
            action_key = self.resolved_ac_keys[emb_id]
            tokens = self.norm_stats.unnormalize(
                {action_key: result["pred_action"]}, emb_id
            )[action_key]
            context = self.norm_stats.unnormalize(loader_batch, emb_id)
            adapter = self.rollout_adapter_for(emb_id)
            actions = (
                adapter.decode(tokens, context=context)
                if adapter is not None
                else tokens
            )
            domain = self.domain_by_id[emb_id]
            predictions[f"emb{emb_id}_{action_key}"] = actions
            predictions[f"{domain}_{action_key}"] = actions
            predictions[f"{domain}_{action_key}_tokens"] = tokens
            predictions.setdefault(action_key, actions)
        return predictions

    def forward_training(self, batch: dict) -> OrderedDict:
        predictions = OrderedDict()
        for emb_id, loader_batch in batch.items():
            result = self.policy(self._seed(emb_id, loader_batch))
            total = sum_losses(result)
            predictions[f"{emb_id}_action_loss"] = total
            for key, value in result.items():
                if not (key.startswith("loss/") or key.startswith("log/")):
                    continue
                if not torch.is_tensor(value):
                    value = torch.tensor(float(value), device=total.device)
                predictions[f"{emb_id}_{key.replace('/', '_')}"] = value
        return predictions

    def compute_losses(self, predictions: dict, batch: dict) -> OrderedDict:
        per_domain = [predictions[f"{emb_id}_action_loss"] for emb_id in batch]
        if not per_domain:
            raise RuntimeError("PipelineAlgo received an empty multi-dataset batch")
        losses = OrderedDict(action_loss=torch.stack(per_domain).mean())
        losses.update(
            (key, value)
            for key, value in predictions.items()
            if torch.is_tensor(value) and value.ndim == 0
        )
        return losses

    def forward_eval(self, batch: dict) -> OrderedDict:
        predictions = OrderedDict()
        for emb_id, loader_batch in batch.items():
            result = self.policy(self._seed(emb_id, loader_batch))
            action_key = self.resolved_ac_keys[emb_id]
            prediction = result["pred_action"]
            domain = self.domain_by_id[emb_id]
            predictions[f"emb{emb_id}_{action_key}"] = prediction
            predictions[f"{domain}_{action_key}"] = prediction
            predictions.setdefault(action_key, prediction)
        return predictions

    def log_info(self, info: dict) -> OrderedDict:
        losses = info["losses"]
        overall = losses["action_loss"].item()
        logged = OrderedDict(Loss=overall, MSE=overall)
        for emb_id, domain in self.domain_by_id.items():
            key = f"{emb_id}_action_loss"
            if key in losses:
                logged[f"MSE/{domain}"] = losses[key].item()
        logged.update((key, value.item()) for key, value in losses.items())
        return logged
