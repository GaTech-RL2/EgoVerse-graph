"""Bounded SQL-inventory export: no directory scans, images, or bulk episode sync."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from egomimic.scripts.data_visualization.arc_tsne import collect, project, write_report


def native_source_transforms(transforms):
    """Explicit viewer recipe: preserve native cadence before ARC tokenization."""
    from egomimic.rldb.zarr.action_chunk_transforms import (
        InterpolateLinear,
        InterpolatePose,
    )

    result = [copy.copy(t) for t in transforms]
    for transform in result:
        if isinstance(transform, (InterpolatePose, InterpolateLinear)):
            transform.new_chunk_length = None
            transform.stride = 1
    return result


def select_episode_ids(names, valid_ratio, split_seed, sample_seed, limit):
    from egomimic.rldb.zarr.zarr_dataset_multi import split_dataset_names

    train, valid = split_dataset_names(names, valid_ratio=valid_ratio, seed=split_seed)
    names = sorted(train)
    rng = np.random.default_rng(sample_seed)
    selected = sorted(rng.choice(names, min(limit, len(names)), replace=False).tolist())
    if not selected:
        raise ValueError("No training episodes in the requested inventory")
    return selected, sorted(train), sorted(valid)


def stage_action_arrays(uri, destination, keys, budget):
    """Fetch only required Zarr arrays, sequentially; never modify the shared cache."""
    from boto3.s3.transfer import TransferConfig

    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// episode URI, got {uri!r}")
    client = get_boto3_s3_client()
    prefix = parsed.path.strip("/") + "/"
    objects = []
    # Root metadata plus only explicitly selected non-camera array prefixes.
    for name in ("zarr.json", ".zgroup", ".zattrs", ".zmetadata"):
        try:
            obj = client.head_object(Bucket=parsed.netloc, Key=prefix + name)
        except client.exceptions.ClientError as exc:
            if str(exc.response["Error"]["Code"]) in ("404", "NoSuchKey", "NotFound"):
                continue
            raise
        objects.append((prefix + name, obj["ContentLength"]))
    for key in sorted(set(keys)):
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=parsed.netloc, Prefix=prefix + key + "/"
        ):
            objects.extend(
                (obj["Key"], obj["Size"]) for obj in page.get("Contents", [])
            )
    size = sum(size for _, size in objects)
    if not objects or size > budget["remaining_bytes"] or len(objects) > 2000:
        raise ValueError(
            f"Sparse staging refused: {size} bytes / {len(objects)} objects"
        )
    print(
        f"STAGE {destination.name}: {size} bytes, {len(objects)} objects, one transfer worker",
        flush=True,
    )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite sparse cache {destination}")
    destination.mkdir(parents=True)
    for key, _ in objects:
        relative = Path(key[len(prefix) :])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe object path")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(
            parsed.netloc, key, str(path), Config=TransferConfig(use_threads=False)
        )
    budget["remaining_bytes"] -= size
    budget["downloaded_bytes"] += size
    budget["downloaded_objects"] += len(objects)
    return destination


def run(
    config_path,
    output,
    max_episodes,
    samples_per_episode,
    seed,
    action_cache,
    max_download_mb,
):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sqlalchemy import text

    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset, episode_names_sha256
    from egomimic.utils.aws.aws_sql import create_default_engine

    if output.exists():
        raise FileExistsError(output)
    if (
        min(max_episodes, samples_per_episode) < 1
        or max_episodes * samples_per_episode > 10000
    ):
        raise ValueError("Invalid sampling budget")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    data = config["data"]
    engine = create_default_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            rows = [
                dict(row)
                for row in connection.execute(
                    text(config["inventory"]["query"]),
                    config["inventory"]["parameters"],
                ).mappings()
            ]
    finally:
        engine.dispose()
    filters = instantiate(data["filters"])
    if any(not filters.matches(row) for row in rows):
        raise ValueError("SQL inventory and YAML filters disagree")
    inventory = {row["episode_hash"]: row for row in rows}
    if len(inventory) != len(rows):
        raise ValueError("Duplicate episode IDs")
    selected, train, valid = select_episode_ids(
        inventory, data["valid_ratio"], data["split_seed"], seed, max_episodes
    )
    print(
        json.dumps(
            {
                "dataset": config["name"],
                "inventory": len(rows),
                "train": len(train),
                "valid": len(valid),
                "selected": selected,
            }
        ),
        flush=True,
    )
    key_map = instantiate(data["key_map"])
    key_map = {
        k: v
        for k, v in key_map.items()
        if v.get("key_type")
        not in ("camera_keys", "annotation_keys", "episode_metadata")
    }
    transforms = instantiate(data["transform_list"])
    budget = {
        "remaining_bytes": int(max_download_mb * 1024**2),
        "downloaded_bytes": 0,
        "downloaded_objects": 0,
    }
    leaves = {}
    for name in selected:
        if Path(name).name != name:
            raise ValueError("Unsafe episode ID")
        candidates = [
            Path(root) / (name + suffix)
            for root in config["cache_roots"]
            for suffix in ("", ".zarr")
        ]
        path = next((p for p in candidates if p.is_dir()), None)
        if path is None:
            if action_cache is None:
                raise FileNotFoundError(
                    f"Sampled episode {name} is not cached; no resampling or automatic full-episode download"
                )
            path = stage_action_arrays(
                inventory[name]["zarr_processed_path"],
                action_cache / name,
                [v["zarr_key"] for v in key_map.values()],
                budget,
            )
        leaf = ZarrDataset(
            path, copy.deepcopy(key_map), transform_list=copy.deepcopy(transforms)
        )
        fps = float(leaf.metadata.get("fps", config["source_fps"]))
        if not np.isclose(fps, config["source_fps"]):
            raise ValueError(
                f"Episode {name} fps={fps} differs from configured {config['source_fps']}"
            )
        if leaf.embodiment.lower() != config["inventory"]["parameters"]["embodiment"]:
            raise ValueError(f"Episode {name} embodiment mismatch: {leaf.embodiment}")
        leaves[name] = leaf
    features, records, codecs = collect(
        leaves,
        max_episodes=max_episodes,
        samples_per_episode=samples_per_episode,
        seed=seed,
        representations="both",
        clock="both",
    )
    print(
        f"TOKENIZED {config['name']}: {len(records)} anchors; fitting t-SNE", flush=True
    )
    projections, diagnostics = project(features, seed=seed)
    repo = Path(__file__).resolve().parents[3]
    provenance = {
        "config": config,
        "split": "train",
        "valid_ratio": data["valid_ratio"],
        "inventory_sha256": episode_names_sha256(inventory),
        "train_episodes": train,
        "valid_episodes": valid,
        "selected_episodes": selected,
        "seed": seed,
        "sample_limit": {
            "episodes": max_episodes,
            "anchors_per_episode": samples_per_episode,
        },
        "codecs": codecs,
        "projections": diagnostics,
        "transfer": budget,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "semantics": "Native token features, not learned latents. Independent t-SNE fits. Human grippers are zero-padded.",
    }
    write_report(output, features, records, projections, provenance)
    print(f"REPORT_READY {output / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=12)
    parser.add_argument("--samples-per-episode", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-cache",
        type=Path,
        help="Opt in to sparse missing-array staging; images are excluded",
    )
    parser.add_argument("--max-download-mb", type=float, default=64)
    args = parser.parse_args()
    run(
        args.data_config,
        args.output,
        args.max_episodes,
        args.samples_per_episode,
        args.seed,
        args.action_cache,
        args.max_download_mb,
    )


if __name__ == "__main__":
    main()
