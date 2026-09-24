"""Zarr-specific construction and normalization behind the data interface.

The shared trainer never inspects a resolver, keymap, episode path or concrete
normalizer. Standalone evaluation restores the full data context and opens
only its explicitly configured validation datasets.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from egomimic.pl_utils.data_context import DataContext
from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper, as_valid_groups
from egomimic.rldb.resolve_memo import resolve_once
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            _json_value(value), sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def normalizer_from_state(state):
    state = copy.deepcopy(state)
    required = {
        "norm_mode",
        "embodiments",
        "key_types",
        "zarr_keys",
        "shapes",
        "norm_stats",
    }
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError(
            "Evaluation requires a complete immutable normalizer_state, including schema"
        )
    for name in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        state[name] = {int(key): value for key, value in state[name].items()}
    state["embodiments"] = [int(key) for key in state["embodiments"]]
    normalizer = MultiDataset.from_state(state)
    for source, keys in normalizer.norm_stats.items():
        for key, stats in keys.items():
            for value in stats.values():
                tensor = torch.as_tensor(value)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite normalizer value at {source}/{key}")
                torch.broadcast_to(tensor, tuple(normalizer.key_shape(key, source)))
    return normalizer


def load_normalizer(path):
    """Restore this adapter's immutable export without resolving any episode."""
    payload = json.loads(Path(path).read_text())
    if "data_context" in payload:
        payload = payload["data_context"]
    state = payload.get("normalizer_state", payload)
    if "sha256" in payload and payload["sha256"] != _digest(state):
        raise ValueError("Data context normalizer hash mismatch")
    return normalizer_from_state(state)


class ZarrDataModule(MultiDataModuleWrapper):
    """Configured Zarr adapter; construction is lazy until mode is known."""

    def __init__(
        self,
        train_datasets,
        valid_datasets,
        train_dataloader_params,
        valid_dataloader_params,
        **loader_options,
    ):
        # Do not instantiate the train branch for standalone evaluation.
        super().__init__(
            {}, {}, train_dataloader_params, valid_dataloader_params, **loader_options
        )
        self._train_configs = train_datasets
        self._valid_configs = valid_datasets
        self._loader_options = loader_options
        self.context = None

    @resolve_once()
    def prepare_context(
        self, *, mode, normalization, normalizer=None, restored_state=None
    ):
        if mode not in {"train", "eval", "normalization"}:
            raise ValueError(f"Unsupported data preparation mode: {mode!r}")
        options = dict(normalization or {})
        if mode == "normalization" and not options.get("save_cache_dir"):
            raise ValueError(
                "Normalization-only mode requires norm_stats.save_cache_dir"
            )
        saved = restored_state
        path = options.get("precomputed_norm_path")
        if saved is None and path:
            path = Path(path)
            if path.is_dir():
                path /= "norm_stats.json"
            payload = json.loads(path.read_text())
            if "data_context" in payload:
                saved = payload["data_context"]
            elif "normalizer_state" in payload:
                saved = {
                    "kind": "zarr-normalizer-v1",
                    "normalizer_state": payload["normalizer_state"],
                }
        if mode == "eval" and saved is None:
            raise ValueError(
                "Standalone evaluation requires checkpoint data_context or a full "
                "normalizer_state at norm_stats.precomputed_norm_path. Export it from "
                "the training run; evaluation will not reopen the training corpus."
            )
        train = (
            {}
            if mode == "eval"
            else {
                name: hydra.utils.instantiate(config)
                for name, config in self._train_configs.items()
                if config is not None
            }
        )
        valid = (
            {}
            if mode == "normalization"
            else {
                group: {
                    name: hydra.utils.instantiate(config)
                    for name, config in members.items()
                    if config is not None
                }
                for group, members in as_valid_groups(self._valid_configs).items()
            }
        )
        super().__init__(
            train,
            valid,
            self.train_dataloader_params,
            self.valid_dataloader_params,
            **self._loader_options,
        )
        if saved is not None:
            if saved.get("kind") != "zarr-normalizer-v1":
                raise ValueError(
                    "Data context kind is not supported by the configured Zarr adapter"
                )
            state = saved["normalizer_state"]
            if "sha256" in saved and saved["sha256"] != _digest(state):
                raise ValueError("Data context normalizer hash mismatch")
            owner = normalizer_from_state(state)
            if options.get("norm_mode", owner.norm_mode) != owner.norm_mode:
                raise ValueError(
                    "Requested normalization mode differs from saved data context"
                )
        else:
            kwargs = {"state": {}, "norm_mode": options.get("norm_mode", "quantile")}
            owner = (
                hydra.utils.instantiate(normalizer, **kwargs)
                if normalizer
                else MultiDataset(**kwargs)
            )
            owner.populate_from_datasets(train)
            seen_identities = set()
            for name, dataset in train.items():
                sample = dataset[0]
                identity = int(sample["embodiment"])
                if identity in seen_identities:
                    raise ValueError(
                        "Combine datasets sharing a normalization identity in one source before fitting statistics"
                    )
                seen_identities.add(identity)
                owner.infer_shapes_from_batch(sample)
                config = OmegaConf.create(copy.deepcopy(self._train_configs[name]))
                if OmegaConf.select(config, "resolver.key_map", default=None) is None:
                    raise ValueError(
                        f"Zarr normalization needs a configured resolver.key_map: {name}"
                    )
                config.resolver.key_map.norm_mode = True
                norm_dataset = hydra.utils.instantiate(config)
                owner.infer_norm_from_dataset(
                    norm_dataset,
                    identity,
                    sample_frac=options.get("sample_frac", 1.0),
                    num_workers=options.get("num_workers", 4),
                    precomputed_norm_path=path,
                )
        if options.get("save_cache_dir") and mode != "eval":
            owner.cache_stats(save_cache_dir=str(options["save_cache_dir"]))
        all_datasets = [(f"train/{k}", v) for k, v in train.items()] + [
            (f"{group}/{name}", ds) for group, name, ds in self.iter_valid_datasets()
        ]
        for name, dataset in all_datasets:
            dataset.set_norm_stats_from(owner)
            if dataset.norm_stats is not owner.norm_stats:
                raise RuntimeError(f"Normalization context was not bound to {name}")
            sample = dataset[0]
            if int(sample["embodiment"]) not in owner.embodiments:
                raise ValueError(
                    f"Validation source {name} is absent from the normalization context"
                )
            identity = int(sample["embodiment"])
            required = {
                owner.keyname_to_zarr_key(key, identity)
                for key in owner.norm_stats.get(identity, {})
            }
            if missing := required - set(sample):
                raise ValueError(
                    f"Saved normalization requires missing keys at {name}: {sorted(missing)}"
                )
            for key, shape in owner.shapes.get(int(sample["embodiment"]), {}).items():
                zarr_key = owner.keyname_to_zarr_key(key, int(sample["embodiment"]))
                if zarr_key in sample and tuple(np.shape(sample[zarr_key])) != tuple(
                    shape
                ):
                    raise ValueError(
                        f"Saved normalization schema differs at {name}/{zarr_key}"
                    )
        state = owner.to_state()
        snapshot = {
            "kind": "zarr-normalizer-v1",
            "normalizer_state": state,
            "sha256": _digest(state),
        }
        self.context = DataContext(
            owner, copy.deepcopy(owner.shapes), tuple(self.valid_group_names), snapshot
        )
        return self.context

    def configure_evaluation(self, requirements):
        if requirements.ordered or requirements.complete_episodes:
            for _, _, dataset in self.iter_valid_datasets():
                dataset.require_ordered_samples()
        super().configure_evaluation(requirements)

    def frame_counts(self):
        return [
            ("train", name, len(ds)) for name, ds in self.train_datasets.items()
        ] + [(group, name, len(ds)) for group, name, ds in self.iter_valid_datasets()]
