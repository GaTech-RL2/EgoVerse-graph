"""Immutable capture and offline rendering for UNITE diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from egomimic.eval.diagnostics._io import (
    as_numpy,
    atomic_json,
    atomic_npz,
    sha256,
    utc_now,
)
from egomimic.eval.diagnostics.latent_projection import write_latent_projection

SCHEMA_VERSION = 1


def _episode_hashes(value: Any, count: int) -> np.ndarray:
    if value is None:
        raise ValueError("UNITE diagnostics require episode_hash in every batch")
    values = as_numpy(value).reshape(-1).astype(np.str_)
    if len(values) < count:
        raise ValueError(f"episode_hash count {len(values)} is smaller than {count}")
    return values[:count]


def _float_array(value: Any, name: str, ndim: int) -> np.ndarray:
    array = as_numpy(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


class UniteDiagnosticArtifactWriter:
    """Write one immutable artifact containing every requested UNITE view."""

    def __init__(self, artifact_dir: str, *, request_path: str):
        self.final_dir = Path(artifact_dir).resolve()
        if self.final_dir.exists():
            raise FileExistsError(f"UNITE diagnostic artifact exists: {self.final_dir}")
        self.request_path = Path(request_path).resolve(strict=True)
        self.request = json.loads(self.request_path.read_text())
        self.staging_dir = self.final_dir.with_name(
            f".{self.final_dir.name}.tmp-{os.getpid()}"
        )
        self.staging_dir.mkdir(parents=True, exist_ok=False)
        (self.staging_dir / "parts").mkdir()
        self.parts: list[dict[str, Any]] = []
        self._counts: dict[int, int] = {}
        self._finalized = False

    def append(
        self,
        *,
        embodiment_id: int,
        embodiment_name: str,
        action_key: str,
        target_action: Any,
        reconstructed_action: Any,
        generated_action: Any,
        clean_latent: Any,
        generated_latent: Any,
        denoising_latents: Any,
        denoising_actions: Any,
        diversity_latents: Any,
        diversity_actions: Any,
        episode_hashes: Any,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to a finalized UNITE artifact")
        arrays = {
            "target_action": _float_array(target_action, "target_action", 3),
            "reconstructed_action": _float_array(
                reconstructed_action, "reconstructed_action", 3
            ),
            "generated_action": _float_array(generated_action, "generated_action", 3),
            "clean_latent": _float_array(clean_latent, "clean_latent", 3),
            "generated_latent": _float_array(generated_latent, "generated_latent", 3),
            "denoising_latents": _float_array(
                denoising_latents, "denoising_latents", 4
            ),
            "denoising_actions": _float_array(
                denoising_actions, "denoising_actions", 4
            ),
            "diversity_latents": _float_array(
                diversity_latents, "diversity_latents", 4
            ),
            "diversity_actions": _float_array(
                diversity_actions, "diversity_actions", 4
            ),
        }
        count = arrays["target_action"].shape[0]
        if any(array.shape[0] != count for array in arrays.values()):
            raise ValueError("UNITE diagnostic arrays have different batch sizes")
        if arrays["reconstructed_action"].shape != arrays["target_action"].shape:
            raise ValueError("reconstruction and target action shapes differ")
        if arrays["generated_action"].shape != arrays["target_action"].shape:
            raise ValueError("generation and target action shapes differ")
        if arrays["clean_latent"].shape != arrays["generated_latent"].shape:
            raise ValueError("clean and generated latent shapes differ")
        if arrays["denoising_actions"].shape[2:] != arrays["target_action"].shape[1:]:
            raise ValueError("denoising action shape is inconsistent with target")
        if arrays["diversity_actions"].shape[2:] != arrays["target_action"].shape[1:]:
            raise ValueError("diversity action shape is inconsistent with target")
        arrays["episode_hash"] = _episode_hashes(episode_hashes, count)

        part_index = self._counts.get(int(embodiment_id), 0)
        self._counts[int(embodiment_id)] = part_index + 1
        filename = f"emb{int(embodiment_id):03d}_part{part_index:06d}.npz"
        path = self.staging_dir / "parts" / filename
        atomic_npz(path, **arrays)
        self.parts.append(
            {
                "file": f"parts/{filename}",
                "sha256": sha256(path),
                "count": int(count),
                "embodiment_id": int(embodiment_id),
                "embodiment_name": str(embodiment_name),
                "action_key": str(action_key),
                "action_horizon": int(arrays["target_action"].shape[1]),
                "action_dim": int(arrays["target_action"].shape[2]),
                "latent_tokens": int(arrays["clean_latent"].shape[1]),
                "latent_dim": int(arrays["clean_latent"].shape[2]),
                "denoising_steps": int(arrays["denoising_actions"].shape[1]),
                "diversity_samples": int(arrays["diversity_actions"].shape[1]),
            }
        )

    def finalize(self) -> Path:
        if self._finalized:
            return self.final_dir
        if not self.parts:
            raise RuntimeError("cannot finalize an empty UNITE diagnostic artifact")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "request": self.request,
            "request_path": str(self.request_path),
            "parts": self.parts,
            "sample_count": int(sum(part["count"] for part in self.parts)),
        }
        manifest_path = self.staging_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        atomic_json(
            self.staging_dir / "COMPLETE.json",
            {"manifest_sha256": sha256(manifest_path), "completed_at": utc_now()},
        )
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            raise FileExistsError(
                f"UNITE artifact appeared during write: {self.final_dir}"
            )
        os.rename(self.staging_dir, self.final_dir)
        self._finalized = True
        return self.final_dir


def validate_unite_artifact(
    artifact_dir: str | Path, request_path: str | Path | None = None
) -> dict[str, Any]:
    root = Path(artifact_dir).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("UNITE diagnostic schema version mismatch")
    if complete.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("UNITE diagnostic manifest hash mismatch")
    if request_path is not None:
        request = json.loads(Path(request_path).resolve(strict=True).read_text())
        if manifest.get("request") != request:
            raise ValueError("UNITE diagnostic request/provenance mismatch")
    required = {
        "target_action",
        "reconstructed_action",
        "generated_action",
        "clean_latent",
        "generated_latent",
        "denoising_latents",
        "denoising_actions",
        "diversity_latents",
        "diversity_actions",
        "episode_hash",
    }
    for part in manifest.get("parts", []):
        path = root / part["file"]
        if not path.is_file() or sha256(path) != part["sha256"]:
            raise ValueError(f"UNITE diagnostic part hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            missing = required - set(arrays.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
    return manifest


def _new_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _render_action_comparison(
    target: np.ndarray,
    reconstruction: np.ndarray,
    generation: np.ndarray,
    output: Path,
    title: str,
) -> None:
    plt = _new_figure()
    dims = target.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(target[:, 0], target[:, 1], "o-", label="demonstrated")
    axes[0].plot(
        reconstruction[:, 0], reconstruction[:, 1], "o-", label="reconstructed"
    )
    axes[0].plot(generation[:, 0], generation[:, 1], "o-", label="generated")
    axes[0].set(
        title="first two action coordinates", xlabel="dimension 0", ylabel="dimension 1"
    )
    for dim in range(dims):
        axes[1].plot(target[:, dim], label=f"target d{dim}")
        axes[1].plot(
            reconstruction[:, dim], linestyle="--", alpha=0.8, label=f"recon d{dim}"
        )
        axes[1].plot(
            generation[:, dim], linestyle=":", alpha=0.9, label=f"generated d{dim}"
        )
    axes[1].set(title="all action components", xlabel="action index")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _render_diversity(
    actions: np.ndarray, target: np.ndarray, output: Path, title: str
) -> None:
    plt = _new_figure()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for sample in actions:
        axes[0].plot(sample[:, 0], sample[:, 1], color="#1f77b4", alpha=0.2)
    axes[0].plot(
        target[:, 0], target[:, 1], color="black", linewidth=2, label="demonstrated"
    )
    axes[0].set(
        title=f"{len(actions)} noise samples",
        xlabel="dimension 0",
        ylabel="dimension 1",
    )
    axes[0].legend()
    mean = actions.mean(axis=0)
    std = actions.std(axis=0)
    time = np.arange(actions.shape[1])
    for dim in range(actions.shape[2]):
        axes[1].plot(time, mean[:, dim], label=f"mean d{dim}")
        axes[1].fill_between(
            time, mean[:, dim] - std[:, dim], mean[:, dim] + std[:, dim], alpha=0.15
        )
    axes[1].set(title="mean ± one standard deviation", xlabel="action index")
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _denoising_frames(
    actions: np.ndarray, target: np.ndarray, title: str
) -> np.ndarray:
    plt = _new_figure()
    frames = []
    for step, prediction in enumerate(actions, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].plot(
            target[:, 0], target[:, 1], "o-", color="black", label="demonstrated"
        )
        axes[0].plot(
            prediction[:, 0],
            prediction[:, 1],
            "o-",
            color="#d62728",
            label="decoded latent",
        )
        axes[0].set(title="trajectory", xlabel="dimension 0", ylabel="dimension 1")
        for dim in range(prediction.shape[1]):
            axes[1].plot(prediction[:, dim], label=f"d{dim}")
        axes[1].set(title="decoded action components", xlabel="action index")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        fig.suptitle(f"{title} — denoising step {step}/{len(actions)}")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        frames.append(frame)
        plt.close(fig)
    return np.stack(frames)


def _write_video(path: Path, frames: np.ndarray, fps: int = 2) -> None:
    import torch
    import torchvision.io as tvio

    tvio.write_video(str(path), torch.from_numpy(frames), fps=fps, video_codec="h264")


def render_unite_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    compute_umap: bool = True,
    max_examples_per_embodiment: int = 4,
) -> dict[str, Any]:
    """Derive every requested UNITE visualization without model inference."""

    root = Path(artifact_dir).resolve(strict=True)
    manifest = validate_unite_artifact(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latent_features = []
    latent_embodiments: list[str] = []
    latent_kinds: list[str] = []
    latent_ids: list[str] = []
    outputs: dict[str, Any] = {"examples": []}
    seen: dict[str, int] = {}

    for part in manifest["parts"]:
        embodiment = part["embodiment_name"]
        domain_dir = output / embodiment
        domain_dir.mkdir(parents=True, exist_ok=True)
        with np.load(root / part["file"], allow_pickle=False) as arrays:
            hashes = arrays["episode_hash"].astype(str)
            for kind, key in (
                ("clean", "clean_latent"),
                ("generated", "generated_latent"),
            ):
                values = arrays[key]
                latent_features.append(values.reshape(-1, values.shape[-1]))
                for sample_id in hashes:
                    latent_embodiments.extend([embodiment] * values.shape[1])
                    latent_kinds.extend([kind] * values.shape[1])
                    latent_ids.extend([str(sample_id)] * values.shape[1])

            allowed = max(0, int(max_examples_per_embodiment) - seen.get(embodiment, 0))
            count = min(int(part["count"]), allowed)
            for index in range(count):
                ordinal = seen.get(embodiment, 0)
                stem = f"sample_{ordinal:04d}_{hashes[index]}"
                comparison = domain_dir / f"{stem}_reconstruction_generation.png"
                diversity = domain_dir / f"{stem}_noise_diversity.png"
                animation = domain_dir / f"{stem}_denoising.mp4"
                _render_action_comparison(
                    arrays["target_action"][index],
                    arrays["reconstructed_action"][index],
                    arrays["generated_action"][index],
                    comparison,
                    f"{embodiment}: reconstruction vs generation",
                )
                _render_diversity(
                    arrays["diversity_actions"][index],
                    arrays["target_action"][index],
                    diversity,
                    f"{embodiment}: fixed observation, varied noise",
                )
                frames = _denoising_frames(
                    arrays["denoising_actions"][index],
                    arrays["target_action"][index],
                    embodiment,
                )
                _write_video(animation, frames)
                outputs["examples"].append(
                    {
                        "embodiment": embodiment,
                        "episode_hash": str(hashes[index]),
                        "comparison": comparison,
                        "diversity": diversity,
                        "denoising_animation": animation,
                    }
                )
                seen[embodiment] = ordinal + 1

    outputs["latent_projection"] = write_latent_projection(
        np.concatenate(latent_features),
        latent_embodiments,
        latent_kinds,
        latent_ids,
        output / "latent_visualization",
        compute_umap=compute_umap,
    )
    atomic_json(
        output / "render_manifest.json",
        {
            "schema_version": 1,
            "source_artifact": str(root),
            "source_manifest_sha256": sha256(root / "manifest.json"),
            "compute_umap": bool(compute_umap),
            "max_examples_per_embodiment": int(max_examples_per_embodiment),
        },
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True)
    validate_parser.add_argument("--request")
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--artifact", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.add_argument("--max-examples-per-embodiment", type=int, default=4)
    render_parser.add_argument("--no-umap", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        print(
            json.dumps(validate_unite_artifact(args.artifact, args.request), indent=2)
        )
    else:
        outputs = render_unite_artifact(
            args.artifact,
            args.output_dir,
            compute_umap=not args.no_umap,
            max_examples_per_embodiment=args.max_examples_per_embodiment,
        )
        print(json.dumps(outputs, indent=2, default=str))


if __name__ == "__main__":
    main()
