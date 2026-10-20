"""Offline, paired t-SNE of native action representations (not learned latents)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    bimanual_arc_token_shape,
    stack_arc_token,
)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(np.abs(result) >= 1e8):
        raise ValueError("Nonfinite values or invalid-pose sentinels in sample")
    return result


def arc_features(token, codec, clock=True):
    """Canonicalize layouts; omit ALL timing rows before scaling/projection."""
    token = array(token)
    expected = bimanual_arc_token_shape(
        codec.M, codec.velocity_mode, codec.velocity_layout
    )
    if token.shape != expected:
        raise ValueError(f"ARC shape {token.shape} does not match {expected}")
    stacked = stack_arc_token(token)
    return (stacked if clock else stacked[: codec.M]).reshape(-1)


def baseline_features(raw, rows=100):
    """Native-cadence fixed horizon; repeat-last padding, never ARC detokenization."""
    raw = array(raw)
    if raw.ndim != 2 or raw.shape[1] != 14 or not len(raw):
        raise ValueError("Baseline requires a nonempty canonical (T,14) action chunk")
    if rows < 1:
        raise ValueError("Baseline horizon must be positive")
    return raw[np.minimum(np.arange(rows), len(raw) - 1)].reshape(-1)


class CaptureActions:
    """Tap native geometry immediately before the training ARC transform."""

    def __init__(self, key):
        self.key = key
        self.value = None

    def transform(self, batch):
        self.value = array(batch[self.key]).copy()
        return batch


def baseline_leaf(leaf, transforms, codec_index, rows):
    """Read a genuine fixed-time window without mutating the ARC leaf's keymap."""
    result = copy.copy(leaf)
    result.key_map = copy.deepcopy(leaf.key_map)
    for spec in result.key_map.values():
        if (
            isinstance(spec.get("horizon"), dict)
            or spec.get("key_type") == "action_keys"
        ):
            spec["horizon"] = rows
    result.transform = list(transforms[:codec_index])
    return result


def collect(
    leaves,
    *,
    max_episodes=12,
    samples_per_episode=64,
    seed=42,
    representations="both",
    clock="both",
    baseline_rows=100,
):
    if max_episodes < 1 or samples_per_episode < 1:
        raise ValueError("Sampling limits must be positive")
    if max_episodes * samples_per_episode > 10000:
        raise ValueError(
            "Limit exports to 10,000 anchors to keep t-SNE and offline HTML bounded"
        )
    rng = np.random.default_rng(seed)
    names = sorted(leaves)
    names = sorted(rng.choice(names, min(max_episodes, len(names)), replace=False))
    features, records, codecs = {}, [], []
    expected_codec = None
    for name in names:
        leaf = leaves[name]
        original = list(leaf.transform or [])
        matches = [
            (i, t)
            for i, t in enumerate(original)
            if isinstance(t, TokenizeBimanualArcLengthCartesian)
        ]
        if len(matches) > 1:
            raise ValueError("Multiple ARC transforms on one leaf are unsupported")
        codec = matches[0][1] if matches else None
        time_leaf = (
            baseline_leaf(leaf, original, matches[0][0], baseline_rows)
            if codec and representations != "arc" and hasattr(leaf, "key_map")
            else None
        )
        if representations != "baseline" and codec is None:
            raise ValueError(
                "ARC output requires a data config using TokenizeBimanualArcLengthCartesian"
            )
        if codec:
            spec = {
                k: getattr(codec, k)
                for k in (
                    "M",
                    "velocity_mode",
                    "velocity_layout",
                    "arc_chunking_mode",
                    "rotation_distance_unit",
                    "action_key",
                    "output_action_key",
                )
            }
            spec["tokenizer_config"] = vars(codec.tokenizer.config).copy()
            if expected_codec is not None and spec != expected_codec:
                raise ValueError("Mixed ARC contracts: export each dataset separately")
            expected_codec = spec
            codecs.append(dict(episode=str(name), **spec))
        tap = CaptureActions(codec.action_key if codec else "actions_cartesian")
        transforms = original.copy()
        transforms.insert(matches[0][0] if codec else len(transforms), tap)
        leaf.transform = transforms
        anchors = sorted(
            rng.choice(len(leaf), min(samples_per_episode, len(leaf)), replace=False)
        )
        try:
            for frame in anchors:
                tap.value = None
                sample = leaf[int(frame)]
                actual_frame = int(sample.get("frame_index", frame))
                if actual_frame != frame:
                    raise ValueError(
                        f"Loader substituted {name}:{frame} with {actual_frame}; refusing biased pairing"
                    )
                raw = tap.value
                if raw is None:
                    raise ValueError("Action capture was not invoked")
                variants = {}
                if representations != "arc":
                    time_raw = raw
                    if time_leaf is not None:
                        time_sample = time_leaf[int(frame)]
                        if int(time_sample.get("frame_index", frame)) != frame:
                            raise ValueError(
                                "Baseline loader substituted the paired anchor"
                            )
                        time_raw = time_sample[codec.action_key]
                    variants["baseline"] = baseline_features(time_raw, baseline_rows)
                if representations != "baseline":
                    for enabled in (
                        [True, False] if clock == "both" else [clock == "include"]
                    ):
                        label = "arc_clock" if enabled else "arc_no_clock"
                        variants[label] = arc_features(
                            sample[codec.output_action_key], codec, enabled
                        )
                for label, vector in variants.items():
                    features.setdefault(label, []).append(vector)
                records.append(
                    {
                        "episode": str(name),
                        "frame": actual_frame,
                        "embodiment": str(sample.get("embodiment", "")),
                        "task": str(
                            leaf.metadata.get(
                                "task", leaf.metadata.get("task_name", "unknown")
                            )
                        ),
                        "source_rows": len(raw),
                        "raw_xyz": raw[
                            np.linspace(0, len(raw) - 1, min(100, len(raw))).astype(int)
                        ][:, [0, 1, 2, 7, 8, 9]].tolist(),
                    }
                )
        finally:
            leaf.transform = original
    if len(records) < 3:
        raise ValueError("t-SNE requires at least three sampled anchors")
    return {k: np.stack(v) for k, v in features.items()}, records, codecs


def drop_gripper(features):
    """Remove both gripper columns from every 14-wide row of every representation.

    Human grippers are zero-padded, so in a joint fit those columns would
    separate the embodiments by construction rather than by motion.
    """
    keep = [c for c in range(14) if c not in (6, 13)]
    out = {}
    for name, values in features.items():
        values = array(values)
        if values.shape[1] % 14:
            raise ValueError(f"{name}: width {values.shape[1]} is not a multiple of 14")
        out[name] = values.reshape(len(values), -1, 14)[:, :, keep].reshape(
            len(values), -1
        )
    return out


def mixing(x, groups, k=10):
    """Neighbour indices plus how mixed the groups are among k nearest neighbours.

    Score = observed cross-group neighbour fraction / fraction expected under
    random labels: 1 means fully mixed, 0 means every group is its own island.
    """
    from sklearn.neighbors import NearestNeighbors

    groups = np.asarray(groups)
    k = min(k, len(x) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)[1][:, 1:]
    observed = (groups[idx] != groups[:, None]).mean()
    counts = {g: (groups == g).sum() for g in set(groups)}
    expected = np.mean([(len(groups) - counts[g]) / (len(groups) - 1) for g in groups])
    score = float(observed / expected) if expected > 0 else None
    return idx[:, :8].tolist(), score


def project(features, seed=42, perplexity=30.0, scaling="standard", groups=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    if not np.isfinite(perplexity) or perplexity <= 0:
        raise ValueError("Perplexity must be finite and positive")
    projections, diagnostics, neighbors = {}, {}, {}
    for name, values in features.items():
        values = array(values)
        if values.ndim != 2 or len(values) < 3:
            raise ValueError("Expected at least three feature vectors")
        varying = np.ptp(values, axis=0) > 1e-12
        x = values[:, varying]
        if not x.shape[1]:
            raise ValueError(
                f"{name}: all samples identical; t-SNE would be misleading"
            )
        if scaling == "standard":
            x = StandardScaler().fit_transform(x)
        effective = min(float(perplexity), max(1.0, (len(x) - 1) / 3))
        with threadpool_limits(limits=1):
            components = min(50, x.shape[0] - 1, x.shape[1])
            if x.shape[1] > components:
                x = PCA(n_components=components, random_state=seed).fit_transform(x)
            fit = TSNE(
                n_components=2,
                perplexity=effective,
                random_state=seed,
                init="random",
                learning_rate="auto",
                n_jobs=1,
            )
            projections[name] = fit.fit_transform(x).tolist()
        diagnostics[name] = {
            "input_features": values.shape[1],
            "varying_features": int(varying.sum()),
            "pca_features": x.shape[1],
            "perplexity": effective,
            "kl_divergence": float(fit.kl_divergence_),
        }
        if groups is not None:
            neighbors[name], diagnostics[name]["mixing"] = mixing(x, groups)
    if groups is not None:
        return projections, diagnostics, neighbors
    return projections, diagnostics


def load_data(path, overrides, dataset_name, split, allow_remote):
    """Compose trusted Hydra YAML; instantiate only the selected data resolver."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from egomimic.rldb.zarr.zarr_dataset_multi import (
        _validate_episode_name_pin,
        split_dataset_names,
    )

    path = Path(path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = compose(config_name=path.stem, overrides=overrides)
    data = cfg.get("data", cfg)
    group = data[f"{split}_datasets"]
    if dataset_name is None:
        if len(group) != 1:
            raise ValueError("Select --dataset from: " + ", ".join(group))
        dataset_name = next(iter(group))
    selected = group[dataset_name]
    resolved = OmegaConf.to_container(selected, resolve=True)
    resolver_cfg = resolved["resolver"]
    target = resolver_cfg.get("_target_", "")
    if not allow_remote and "Local" not in target.rsplit(".", 1)[-1]:
        raise ValueError(
            "Non-local resolver refused. Use a local data config, or explicitly pass --allow-remote-resolution (may query/download data)."
        )
    resolver = instantiate(resolver_cfg)
    filters = instantiate(resolved.get("filters"))
    leaves = resolver.resolve(filters=filters)
    train, valid = split_dataset_names(
        leaves,
        valid_ratio=resolved.get("valid_ratio", 0.2),
        seed=resolved.get("split_seed", 42),
    )
    for label, ids, prefix in [
        ("train", train, "train"),
        ("validation", valid, "valid"),
    ]:
        _validate_episode_name_pin(
            label,
            ids,
            resolved.get(f"expected_{prefix}_episode_count"),
            resolved.get(f"expected_{prefix}_episode_names_sha256"),
        )
    chosen = train if split == "train" else valid
    if resolved.get("mode", split) != split:
        raise ValueError(
            "Selected dataset mode disagrees with --split; refusing to silently change it"
        )
    if not chosen:
        raise ValueError("No episodes in selected split")
    provenance = {
        "config": resolved,
        "dataset": dataset_name,
        "split": split,
        "inventory_sha256": hashlib.sha256(
            "\n".join(sorted(leaves)).encode()
        ).hexdigest(),
        "split_episodes": sorted(chosen),
    }
    return {k: leaves[k] for k in sorted(chosen)}, provenance


def write_report(output, features, records, projections, provenance, neighbors=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}; choose a new output directory"
        )
    output.mkdir(parents=True)
    payload = {"records": records, "projections": projections, "provenance": provenance}
    if neighbors:
        payload["neighbors"] = neighbors
    encoded = json.dumps(payload, allow_nan=False, default=str)
    (output / "report.json").write_text(encoded)
    np.savez_compressed(output / "features.npz", **features)
    # Escape HTML parser delimiters, including user-provided task/episode names.
    safe = (
        encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    )
    template = Path(__file__).with_name("arc_tsne.html").read_text()
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", safe))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument(
        "--override", action="append", default=[], help="Hydra override, repeatable"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument(
        "--representations", choices=["arc", "baseline", "both"], default="both"
    )
    parser.add_argument(
        "--clock", choices=["include", "exclude", "both"], default="both"
    )
    parser.add_argument("--max-episodes", type=int, default=12)
    parser.add_argument("--samples-per-episode", type=int, default=64)
    parser.add_argument("--baseline-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--scaling", choices=["standard", "none"], default="standard")
    parser.add_argument("--allow-remote-resolution", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    leaves, provenance = load_data(
        args.data_config,
        args.override,
        args.dataset,
        args.split,
        args.allow_remote_resolution,
    )
    features, records, codecs = collect(
        leaves,
        max_episodes=args.max_episodes,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
        representations=args.representations,
        clock=args.clock,
        baseline_rows=args.baseline_rows,
    )
    projections, diagnostics = project(
        features, args.seed, args.perplexity, args.scaling
    )
    provenance.update(
        arguments=vars(args),
        codecs=codecs,
        projections=diagnostics,
        semantics="Native token features, not learned latents. Independent t-SNE fits; axes are not comparable.",
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "hydra-core")
        },
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(
            Path(__file__).with_name("arc_tsne.html").read_bytes()
        ).hexdigest(),
    )
    repo = Path(__file__).resolve().parents[3]
    try:
        provenance["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        provenance["source_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        provenance["source_commit"] = "unavailable"
    write_report(args.output, features, records, projections, provenance)
    print(f"Wrote {len(records)} matched anchors: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
