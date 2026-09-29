"""Shared PCA/UMAP backend for existing and UNITE latent visualizations."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def _features(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"latent features must have shape (rows, dims), got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("latent features must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("latent features contain non-finite values")
    return array


def project_pca(features, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Fit deterministic PCA, preferring cuML when available."""

    values = _features(features)
    rows, dims = values.shape
    count = min(int(n_components), rows, dims)
    if count <= 0:
        raise ValueError("PCA component count must be positive")
    if rows < 2:
        return (
            np.zeros((rows, count), dtype=np.float32),
            np.zeros((count,), dtype=np.float32),
        )
    try:
        from cuml.decomposition import PCA as CuPCA

        reducer = CuPCA(n_components=count, output_type="numpy")
        transformed = reducer.fit_transform(values)
        ratio = np.asarray(reducer.explained_variance_ratio_, dtype=np.float32)
    except ImportError:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=count, random_state=0)
        transformed = reducer.fit_transform(values)
        ratio = reducer.explained_variance_ratio_
    return np.asarray(transformed, dtype=np.float32), np.asarray(
        ratio, dtype=np.float32
    )


def project_umap(
    features,
    n_components: int = 3,
    *,
    n_neighbors: int = 15,
    random_state: int = 0,
) -> np.ndarray:
    """Fit UMAP with stable settings, preferring cuML when available."""

    values = _features(features)
    rows = values.shape[0]
    count = int(n_components)
    if count <= 0:
        raise ValueError("UMAP component count must be positive")
    if rows < 3:
        return np.zeros((rows, count), dtype=np.float32)
    neighbors = max(2, min(int(n_neighbors), rows - 1))
    try:
        from cuml.manifold import UMAP as CuUMAP

        reducer = CuUMAP(
            n_components=count,
            n_neighbors=neighbors,
            metric="euclidean",
            random_state=int(random_state),
            output_type="numpy",
        )
    except ImportError:
        try:
            from umap import UMAP
        except ImportError as error:
            raise RuntimeError(
                "UMAP requested but neither cuML nor umap-learn is installed. "
                "Install the project's 'umap-learn' dependency or disable UMAP."
            ) from error
        reducer = UMAP(
            n_components=count,
            n_neighbors=neighbors,
            metric="euclidean",
            random_state=int(random_state),
        )
    return np.asarray(reducer.fit_transform(values), dtype=np.float32)


def _pad_three(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[1] >= 3:
        return coords[:, :3]
    return np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))


def _scatter(
    coords: np.ndarray,
    embodiments: np.ndarray,
    latent_kinds: np.ndarray,
    output: Path,
    *,
    title: str,
    axis_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        name: color
        for name, color in zip(
            sorted(set(embodiments.tolist())),
            plt.get_cmap("tab10").colors,
        )
    }
    markers = {"clean": "o", "generated": "^"}
    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    for embodiment in sorted(set(embodiments.tolist())):
        for kind in sorted(set(latent_kinds.tolist())):
            mask = (embodiments == embodiment) & (latent_kinds == kind)
            if not np.any(mask):
                continue
            points = coords[mask]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=colors[embodiment],
                marker=markers.get(kind, "x"),
                s=10,
                alpha=0.5,
                label=f"{embodiment} / {kind} (n={int(mask.sum())})",
            )
    axis.set_title(title)
    axis.set_xlabel(f"{axis_prefix}_x")
    axis.set_ylabel(f"{axis_prefix}_y")
    axis.set_zlabel(f"{axis_prefix}_z")
    axis.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_latent_projection(
    features,
    embodiments: Sequence[str],
    latent_kinds: Sequence[str],
    sample_ids: Sequence[str],
    output_dir: str | Path,
    *,
    prefix: str = "shared_latent",
    compute_umap: bool = True,
    random_state: int = 0,
) -> dict[str, Path]:
    """Write reusable PCA/UMAP coordinates, CSV metadata, and scatter plots."""

    values = _features(features)
    embodiments_array = np.asarray(embodiments, dtype=np.str_)
    kinds_array = np.asarray(latent_kinds, dtype=np.str_)
    ids_array = np.asarray(sample_ids, dtype=np.str_)
    if any(
        array.shape != (values.shape[0],)
        for array in (embodiments_array, kinds_array, ids_array)
    ):
        raise ValueError("latent metadata length must equal the feature row count")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pca, ratio = project_pca(values, 3)
    pca_xyz = _pad_three(pca)
    umap_xyz = (
        project_umap(values, 3, random_state=random_state) if compute_umap else None
    )

    npz_path = output / f"{prefix}.npz"
    arrays = {
        "features": values,
        "embodiment": embodiments_array,
        "latent_kind": kinds_array,
        "sample_id": ids_array,
        "pca_xyz": pca_xyz,
        "pca_explained_variance_ratio": ratio,
    }
    if umap_xyz is not None:
        arrays["umap_xyz"] = umap_xyz
    np.savez_compressed(npz_path, **arrays)

    csv_path = output / f"{prefix}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "sample_id",
            "embodiment",
            "latent_kind",
            "pca_x",
            "pca_y",
            "pca_z",
        ]
        if umap_xyz is not None:
            header += ["umap_x", "umap_y", "umap_z"]
        writer.writerow(header)
        for index in range(values.shape[0]):
            row = [
                ids_array[index],
                embodiments_array[index],
                kinds_array[index],
                *pca_xyz[index].tolist(),
            ]
            if umap_xyz is not None:
                row += umap_xyz[index].tolist()
            writer.writerow(row)

    pca_path = output / f"{prefix}_pca.png"
    _scatter(
        pca_xyz,
        embodiments_array,
        kinds_array,
        pca_path,
        title="Shared UNITE latent space: clean vs generated",
        axis_prefix="pca",
    )
    paths = {"arrays": npz_path, "csv": csv_path, "pca": pca_path}
    if umap_xyz is not None:
        umap_path = output / f"{prefix}_umap.png"
        _scatter(
            umap_xyz,
            embodiments_array,
            kinds_array,
            umap_path,
            title="Shared UNITE latent space: clean vs generated",
            axis_prefix="umap",
        )
        paths["umap"] = umap_path
    return paths
