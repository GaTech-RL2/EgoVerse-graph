"""PI bounds semantics and bounded recovery on the graph dataset implementation.

Source method bodies: d5f72068; graph owns loading, grouping and normalization.
"""
import logging
import json
import math
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.campaigns.pi05.pose import bimanual_cartesian_layout
from egomimic.campaigns.pi05.embodiment import get_embodiment_id

logger = logging.getLogger(__name__)


class PI05Dataset(MultiDataset):
    GLOBAL_FALLBACK_ATTEMPTS = 25
    MAX_FALLBACK_ATTEMPTS = 1000

    def _check_bounds(
        self, data: dict, dataset, idx: int, dataset_name: str
    ) -> str | None:
        """Return a violation message if any tracked key in ``data`` has NaN/Inf
        or values outside per-key quantile bounds. ``None`` means the sample
        passes. Logs each (episode, key) violation once.
        """
        embodiment_id = data.get("embodiment")
        if embodiment_id is None:
            return None
        per_emb_stats = self.norm_stats.get(embodiment_id, {})
        if not per_emb_stats:
            return None

        episode_name = self._episode_name_for_dataset(dataset, dataset_name)

        for key_name, stats in per_emb_stats.items():
            zarr_key = self.zarr_keys.get(embodiment_id, {}).get(key_name)
            if zarr_key is None or zarr_key not in data:
                continue
            v = data[zarr_key]
            if isinstance(v, torch.Tensor):
                arr = v.float()
            elif isinstance(v, np.ndarray):
                arr = torch.from_numpy(v).float()
            else:
                continue

            q_low = stats.get(
                "quantile_0_01", stats.get("quantile_0_1", stats["quantile_1"])
            )
            q_high = stats.get(
                "quantile_99_99", stats.get("quantile_99_9", stats["quantile_99"])
            )
            q_low = torch.as_tensor(q_low, device=arr.device, dtype=torch.float32)
            q_high = torch.as_tensor(q_high, device=arr.device, dtype=torch.float32)
            try:
                q_low = torch.broadcast_to(q_low, arr.shape)
                q_high = torch.broadcast_to(q_high, arr.shape)
            except RuntimeError:
                # Stats were computed for a different layout than this sample
                # (e.g. a stale precomputed norm_stats.json). Say so once
                # instead of silently disabling the bounds check for the key;
                # normalize() will raise on the same mismatch anyway.
                warn_key = f"bounds-shape:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    logger.warning(
                        f"[MultiDataset] bounds check skipped for {zarr_key}: "
                        f"stats shape {tuple(q_low.shape)} does not broadcast to "
                        f"sample shape {tuple(arr.shape)} (norm stats computed "
                        "for a different layout?)"
                    )
                continue

            if torch.any(torch.isnan(arr)) or torch.any(torch.isinf(arr)):
                prefix = f"NaN/Inf in {zarr_key} ep={episode_name} frame={idx}"
                warn_key = f"naninf:{episode_name}:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    logger.warning(prefix)
                return prefix

            # The bimanual cartesian action chunk and the ee_pose proprio share
            # a [L | R] layout whose rotation channels are either Euler ypr
            # (wraps at ±π) or continuous 6D columns. In both cases quantile
            # bounds on the rotation channels are meaningless and reject
            # otherwise-valid frames, so only the translation (and gripper)
            # channels are bounds-checked. Unrecognized widths fall through to
            # a full-vector check; NaN/Inf above still covers the full vector.
            cartesian_layout = None
            if zarr_key in ("actions_cartesian", "observations.state.ee_pose"):
                cartesian_layout = bimanual_cartesian_layout(arr.shape[-1])
            if cartesian_layout is not None:
                check_idx = list(cartesian_layout["xyz"]) + list(
                    cartesian_layout["grip"]
                )
                arr_q = arr[..., check_idx]
                q_low = q_low[..., check_idx]
                q_high = q_high[..., check_idx]
            else:
                arr_q = arr

            # Absolute slack on the quantile bounds. Wrist-frame action chunks
            # are the identity pose at t=0 (the reference IS the obs pose), so
            # those cells' bounds collapse to [0, 0] and a strict compare would
            # reject every frame on any roundoff (today the cells are exactly
            # 0.0, so this only guards against a different BLAS/dtype path).
            # 1e-6 (m / normalized grip) is far below any real outlier.
            tol = 1e-6
            below = arr_q < q_low - tol
            above = arr_q > q_high + tol
            if torch.any(below) or torch.any(above):
                prefix = f"Bounds violation in {zarr_key} ep={episode_name} frame={idx}"
                warn_key = f"bounds:{episode_name}:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    n_below = int(below.sum().item())
                    n_above = int(above.sum().item())
                    logger.warning(
                        f"{prefix} | n_below={n_below} n_above={n_above} "
                        f"arr_range=[{arr_q.min().item():.4f}, {arr_q.max().item():.4f}]"
                    )
                return prefix
        return None

    def _next_after_failure(
        self, idx: int, dataset_name: str, attempts: int | None, *, reason: str
    ) -> tuple[int, int]:
        attempts = (attempts or 0) + 1
        if attempts >= self.MAX_FALLBACK_ATTEMPTS:
            raise RuntimeError(
                f"{self.MAX_FALLBACK_ATTEMPTS} consecutive bad samples "
                f"(systemic data/norm-stats problem?); last: {reason}"
            )
        if attempts <= self.GLOBAL_FALLBACK_ATTEMPTS:
            candidates = [
                c for c in self._global_indices_by_dataset[dataset_name] if c != idx
            ]
        else:
            candidates = None
        if candidates:
            next_idx = random.choice(candidates)
        else:
            next_idx = random.randrange(len(self.index_map))
        next_dataset_name, next_local_idx = self.index_map[next_idx]
        logger.warning(
            f"{reason} | attempt {attempts}, "
            f"trying {next_dataset_name}[{next_local_idx}]"
        )
        return next_idx, attempts

    def infer_norm_from_dataset(
        self,
        dataset,
        dataset_name,
        sample_frac: float = 0.10,
        seed: int = 42,
        max_samples: int | None = None,
        batch_size: int = 512,
        num_workers: int = 4,
        precomputed_norm_path: str | None = None,
    ):
        embodiment = dataset_name
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        norm_keys = list(self.keys_of_type("proprio_keys", embodiment))
        norm_keys.extend(self.keys_of_type("action_keys", embodiment))
        if not norm_keys:
            logger.warning(
                f"[MultiDataset] No proprio/action keys for embodiment={embodiment}"
            )
            return

        self.norm_stats.setdefault(embodiment, {})

        if precomputed_norm_path is not None:
            if os.path.isdir(precomputed_norm_path):
                precomputed_file = os.path.join(
                    precomputed_norm_path, "norm_stats.json"
                )
            elif os.path.isfile(precomputed_norm_path):
                precomputed_file = precomputed_norm_path
            else:
                logger.warning(
                    f"[MultiDataset] precomputed_norm_path={precomputed_norm_path} is not valid"
                )
                return
            if os.path.isfile(precomputed_file):
                self._load_precomputed_stats(precomputed_file, embodiment, norm_keys)
                logger.info(
                    f"[MultiDataset] Loaded precomputed stats for embodiment={embodiment}"
                )
                return

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        N = len(dataset)
        if N <= 0:
            raise ValueError("Dataset is empty")
        n_samples = int(math.ceil(sample_frac * N))
        n_samples = max(1, min(n_samples, N))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        logger.info(f"[MultiDataset] embodiment={embodiment} norm_keys={norm_keys}")
        logger.info(
            f"[MultiDataset] sampling {n_samples}/{N} (~{100 * sample_frac:.1f}%)"
        )

        loading_start = time.time()
        collected = self._collect_norm_samples(
            loader, norm_keys, embodiment, n_samples, batch_size, num_workers
        )
        for k in [k for k, v in collected.items() if not v]:
            del collected[k]
            norm_keys.remove(k)
        loading_time = time.time() - loading_start

        computing_start = time.time()
        for k in norm_keys:
            collected[k] = np.concatenate(collected[k], axis=0)
            stats_np = self._compute_stats_for_array(collected[k])
            self.norm_stats[embodiment][k] = {
                name: np.asarray(arr, dtype=np.float32)
                for name, arr in stats_np.items()
            }
            logger.info(
                f"[MultiDataset] key={k} samples={collected[k].shape[0]} stat_shape={stats_np['mean'].shape}"
            )
        computing_time = time.time() - computing_start

        self._norm_run_metadata = {
            "loading_time": loading_time,
            "computing_time": computing_time,
            "frames": n_samples,
        }
        logger.info(
            f"[MultiDataset] Finished norm inference, loading={loading_time:.2f}s, computing={computing_time:.2f}s"
        )

    def _load_precomputed_stats(
        self, precomputed_file: str, embodiment: int, norm_keys: list[str]
    ) -> None:
        """Load ``norm_stats.json`` for one embodiment, refusing a file whose
        provenance does not match this dataset.

        The stats are only meaningful for the exact (norm_mode, key set)
        they were computed under; the payload's ``provenance`` block (written
        by :meth:`cache_stats`) carries both. Files written before provenance
        existed load as before, with a warning.
        """
        with open(precomputed_file, "r") as f:
            payload = json.load(f)
        if str(embodiment) not in payload["stats"]:
            raise ValueError(
                f"norm_stats file {precomputed_file} has no entry for "
                f"embodiment id {embodiment} (available: "
                f"{sorted(payload['stats'])}). Stats are keyed by numeric "
                "EMBODIMENT id, and ids were renumbered by the human/eva "
                "embodiment collapse — recompute norm stats instead of "
                "reusing a pre-collapse norm_stats.json."
            )
        provenance = payload.get("provenance")
        if provenance is None:
            logger.warning(
                f"[MultiDataset] {precomputed_file} carries no provenance block "
                "(written by an older cache_stats); cannot verify it matches "
                f"norm_mode={self.norm_mode!r} and this dataset's keys."
            )
        else:
            file_mode = provenance.get("norm_mode")
            if file_mode != self.norm_mode:
                raise ValueError(
                    f"norm_stats file {precomputed_file} was computed with "
                    f"norm_mode={file_mode!r} but this dataset uses "
                    f"norm_mode={self.norm_mode!r}; recompute the stats."
                )
        file_keys = set(payload["stats"][str(embodiment)])
        want_keys = set(norm_keys)
        if norm_keys and file_keys != want_keys:
            raise ValueError(
                f"norm_stats file {precomputed_file} keys for embodiment "
                f"{embodiment} are {sorted(file_keys)} but this dataset "
                f"normalizes {sorted(want_keys)}; the file was computed for a "
                "different keymap/transform mode — recompute the stats."
            )
        self.norm_stats[embodiment] = payload["stats"][str(embodiment)]
        self._norm_run_metadata = payload.get("norm_run_metadata", None)

    def _collect_norm_samples(
        self, loader, norm_keys, embodiment, n_samples, batch_size, num_workers
    ):
        collected = {k: [] for k in norm_keys}
        cur = 0
        with tqdm(total=n_samples, unit="sample") as pbar:
            for batch in loader:
                remaining = n_samples - cur
                if remaining <= 0:
                    break
                batch_len = None
                for value in batch.values():
                    if hasattr(value, "shape") and len(value.shape) > 0:
                        batch_len = int(value.shape[0])
                        break
                if batch_len is None:
                    raise ValueError(
                        "[MultiDataset] Could not infer batch size from DataLoader batch"
                    )
                take = min(remaining, batch_len)
                for k in norm_keys:
                    zarr_key = self.keyname_to_zarr_key(k, embodiment)
                    if zarr_key is None or zarr_key not in batch:
                        continue
                    x = batch[zarr_key][:take]
                    if hasattr(x, "detach"):
                        x = x.detach().cpu().numpy()
                    # float32: stats are consumed as float32 anyway, and the
                    # float64 poses double the stacked-sample footprint (an
                    # (N, 100, 18) action stack at large N is tens of GB).
                    collected[k].append(np.asarray(x, dtype=np.float32))
                cur += take
                pbar.update(take)
        return collected

    def cache_stats(self, save_cache_dir: str):
        cache_dir = os.path.join(save_cache_dir, "norm_stats")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, "norm_stats.json")

        stats_out: dict[str, dict[str, dict[str, list]]] = {}
        for emb, keys_dict in self.norm_stats.items():
            stats_out[str(emb)] = {
                k: {name: np.asarray(arr).tolist() for name, arr in stat_dict.items()}
                for k, stat_dict in keys_dict.items()
            }
        payload = {
            "stats": stats_out,
            # What the stats are valid for — checked by _load_precomputed_stats
            # so a cached file from another norm_mode / keymap / transform mode
            # (same dims, different meaning) is refused instead of applied.
            "provenance": {
                "norm_mode": self.norm_mode,
                "stat_shapes": {
                    str(emb): {
                        k: list(np.asarray(next(iter(sd.values()))).shape)
                        for k, sd in keys_dict.items()
                    }
                    for emb, keys_dict in self.norm_stats.items()
                },
            },
            "loading_time": None,
            "computing_time": None,
            "frames": None,
        }
        if self._norm_run_metadata is not None:
            for k in ("loading_time", "computing_time", "frames"):
                if k in self._norm_run_metadata:
                    payload[k] = self._norm_run_metadata[k]
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"[MultiDataset] Cached stats to {out_path}")

    def _apply_norm_one(self, tensor, stats):
        if self.norm_mode == "none":
            return tensor
        if self.norm_mode == "zscore":
            mean = torch.as_tensor(
                stats["mean"], device=tensor.device, dtype=torch.float32
            )
            std = torch.as_tensor(
                stats["std"], device=tensor.device, dtype=torch.float32
            )
            return (tensor - mean) / (std + 1e-6)
        if self.norm_mode == "minmax":
            mn = torch.as_tensor(
                stats["min"], device=tensor.device, dtype=torch.float32
            )
            mx = torch.as_tensor(
                stats["max"], device=tensor.device, dtype=torch.float32
            )
            return 2.0 * ((tensor - mn) / (mx - mn + 1e-6)) - 1.0
        if self.norm_mode == "quantile":
            q1 = torch.as_tensor(
                stats["quantile_1"], device=tensor.device, dtype=torch.float32
            )
            q99 = torch.as_tensor(
                stats["quantile_99"], device=tensor.device, dtype=torch.float32
            )
            return 2.0 * ((tensor - q1) / (q99 - q1 + 1e-6)) - 1.0
        raise ValueError(f"Invalid normalization mode: {self.norm_mode}")

    def _apply_unnorm_one(self, tensor, stats):
        if self.norm_mode == "none":
            return tensor
        if self.norm_mode == "zscore":
            mean = torch.as_tensor(
                stats["mean"], device=tensor.device, dtype=torch.float32
            )
            std = torch.as_tensor(
                stats["std"], device=tensor.device, dtype=torch.float32
            )
            return tensor * (std + 1e-6) + mean
        if self.norm_mode == "minmax":
            mn = torch.as_tensor(
                stats["min"], device=tensor.device, dtype=torch.float32
            )
            mx = torch.as_tensor(
                stats["max"], device=tensor.device, dtype=torch.float32
            )
            return (tensor + 1) * 0.5 * (mx - mn + 1e-6) + mn
        if self.norm_mode == "quantile":
            q1 = torch.as_tensor(
                stats["quantile_1"], device=tensor.device, dtype=torch.float32
            )
            q99 = torch.as_tensor(
                stats["quantile_99"], device=tensor.device, dtype=torch.float32
            )
            return (tensor + 1) * 0.5 * (q99 - q1 + 1e-6) + q1
        raise ValueError(f"Invalid normalization mode: {self.norm_mode}")

_IMAGE_TARGET_HW = (480, 640)

def _resize_image_keys(batch, size=_IMAGE_TARGET_HW):
    """In-place resize of every camera-image tensor in *batch* to a fixed (H, W).

    Camera images are identified by the substring ``"images"`` in the key
    (dataset-style keymaps, ``observations.images.*``) or the ``"_rgb"`` suffix
    (PI/PaliGemma-style keymaps, ``base_0_rgb`` / ``*_wrist_0_rgb``). Each image
    is ``(C, H, W)`` or ``(T, C, H, W)``; only the trailing two spatial dims are
    resized (bilinear, per-channel — equivalent to resizing the image). Tensors
    already at the target size are skipped.
    """
    th, tw = size
    for sample in batch:
        for k in list(sample.keys()):
            if "images" not in k and not k.endswith("_rgb"):
                continue
            v = sample[k]
            if not isinstance(v, torch.Tensor) or v.ndim < 2:
                continue
            if v.shape[-2] == th and v.shape[-1] == tw:
                continue
            orig_dtype = v.dtype
            x = v.float()
            lead = x.shape[:-2]  # leading (non-spatial) dims, e.g. (C,) or (T, C)
            x4 = x.reshape(-1, 1, x.shape[-2], x.shape[-1])  # (N, 1, H, W)
            x4 = torch.nn.functional.interpolate(
                x4, size=(th, tw), mode="bilinear", align_corners=False
            )
            sample[k] = x4.reshape(*lead, th, tw).to(orig_dtype)
