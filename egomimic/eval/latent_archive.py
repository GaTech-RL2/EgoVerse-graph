"""Model-independent token archives and reproducible dimensionality reductions."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch


def safe_name(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe diagnostic artifact name: {value!r}")
    return value


def reduce_features(
    features, *, methods, pca_components=50, seed=0, pca_for_downstream=False
):
    """Use explicit CPU backends; optional UMAP is loaded only when requested."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or min(features.shape) < 1 or not np.isfinite(features).all():
        raise ValueError("Token reductions require a nonempty finite matrix")
    supported = {"pca", "umap", "pca_umap", "tsne2d", "tsne3d"}
    if not set(methods) <= supported:
        raise ValueError(f"Unknown reductions: {set(methods) - supported}")
    if type(pca_components) is not int or pca_components < 1:
        raise ValueError("pca_components must be positive")
    count, width = features.shape
    need_pca = pca_for_downstream or bool({"pca", "pca_umap"} & set(methods))
    pca = None
    receipt = {
        "seed": seed,
        "methods": list(methods),
        "backend": "cpu",
        "rows": count,
        "width": width,
    }
    if need_pca:
        reducer = PCA(n_components=min(pca_components, count, width), random_state=seed)
        if count < 2 or np.all(features == features[0]):
            pca = np.zeros((count, reducer.n_components), dtype=np.float32)
            explained = np.zeros(reducer.n_components)
        else:
            pca = reducer.fit_transform(features).astype(np.float32)
            explained = reducer.explained_variance_ratio_
        receipt["pca_explained_variance"] = explained.tolist()
    downstream = pca if pca_for_downstream else features
    coordinates = {}
    for method in methods:
        dimensions = 2 if method == "tsne2d" else 3
        if method == "pca":
            values = pca[:, :3]
            coordinates[method] = np.pad(values, ((0, 0), (0, 3 - values.shape[1])))
        elif count < 4:
            coordinates[method] = np.zeros((count, dimensions), dtype=np.float32)
            receipt.setdefault("degenerate_reductions", []).append(method)
        elif method in {"umap", "pca_umap"}:
            from umap import UMAP

            selected = pca if method == "pca_umap" else downstream
            coordinates[method] = UMAP(
                n_components=3,
                n_neighbors=min(15, count - 1),
                metric="euclidean",
                random_state=seed,
                init="random",
                n_jobs=1,
            ).fit_transform(selected)
        else:
            coordinates[method] = TSNE(
                n_components=dimensions,
                perplexity=min(30, max(2, count // 3), count - 1),
                random_state=seed,
                init="pca" if min(downstream.shape) >= dimensions else "random",
            ).fit_transform(downstream)
        if not np.isfinite(coordinates[method]).all():
            raise ValueError(f"{method} produced nonfinite coordinates")
    return coordinates, receipt


def write_archive(directory, layer, keys, rows, *, reductions, receipt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layer = safe_name(layer)
    csv_path = directory / f"{layer}.csv"
    key_path = directory / f"{layer}_keys.pt"
    receipt_path = directory / f"{layer}.json"
    if any(path.exists() for path in (csv_path, key_path, receipt_path)):
        raise FileExistsError(f"Refusing to overwrite token archive {csv_path}")
    if len(rows) != len(keys) or any(
        len(value) != len(rows) for value in reductions.values()
    ):
        raise ValueError(
            "Token metadata, keys and reductions must have identical row counts"
        )
    fields = ["video_hash", "embodiment", "frame_idx", "token_idx"]
    extra = [
        (f"{name}_{axis}", values[:, i])
        for name, values in reductions.items()
        for i, axis in enumerate("xyz"[: values.shape[1]])
    ]
    with csv_path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields + [name for name, _ in extra])
        for i, row in enumerate(rows):
            writer.writerow(
                [row[key] for key in fields] + [float(values[i]) for _, values in extra]
            )
    with key_path.open("xb") as handle:
        torch.save(torch.as_tensor(keys).float().contiguous(), handle)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return csv_path


def read_archive(path):
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sidecar = path.with_name(f"{path.stem}_keys.pt")
    if sidecar.exists():
        keys = (
            torch.load(sidecar, map_location="cpu", weights_only=True).float().numpy()
        )
    else:
        # Preserve rebuilds of older, pre-sidecar CSV exports.
        columns = (
            sorted(
                (key for key in rows[0] if re.fullmatch(r"k\d+", key)),
                key=lambda key: int(key[1:]),
            )
            if rows
            else []
        )
        if not columns:
            raise ValueError(f"No raw keys in {path} or {sidecar}")
        keys = np.array(
            [[float(row[key]) for key in columns] for row in rows], dtype=np.float32
        )
    if len(keys) != len(rows) or keys.ndim != 2 or not np.isfinite(keys).all():
        raise ValueError("Token archive row counts or values are invalid")
    return keys, rows


def plot_reductions(directory, layer, rows, reductions, *, color_by="embodiment"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = {"embodiment": "embodiment", "hash": "video_hash"}.get(color_by)
    if key is None:
        raise ValueError("color_by must be embodiment or hash")
    labels = np.array([row[key] for row in rows])
    for name, points in reductions.items():
        dimensions = points.shape[1]
        fig = plt.figure(figsize=(8, 6))
        ax = (
            fig.add_subplot(111, projection="3d")
            if dimensions == 3
            else fig.add_subplot(111)
        )
        for label in sorted(set(labels)):
            selected = points[labels == label]
            ax.scatter(
                *selected.T, s=8, alpha=0.5, label=f"{label} (n={len(selected)})"
            )
        ax.set_title(f"{layer}: {name}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(
            Path(directory) / f"{safe_name(layer)}_{safe_name(name)}.png", dpi=140
        )
        plt.close(fig)


def rebuild_archives(
    source, destination, *, methods, save_plots=True, color_by="embodiment", **options
):
    """Recompute plots/CSV from raw archives without loading any policy or data."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Rebuild destination must differ from the immutable source")
    paths = [
        path for path in sorted(source.glob("*.csv")) if path.name != "comparison.csv"
    ]
    if not paths:
        raise ValueError(f"No layer CSVs in {source}")
    for path in paths:
        keys, rows = read_archive(path)
        coordinates, receipt = reduce_features(keys, methods=methods, **options)
        receipt["source_archive"] = str(path.resolve())
        write_archive(
            destination, path.stem, keys, rows, reductions=coordinates, receipt=receipt
        )
        if save_plots:
            plot_reductions(
                destination, path.stem, rows, coordinates, color_by=color_by
            )
