"""Run the same fixed data probe in a pinned legacy or graph source checkout.

Use PYTHONPATH pointing at exactly the selected checkout. This probe never
loads a model, contacts a dataset service, or changes its input episodes.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import egomimic
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset


def probe(root, source_root, vendor, representation, frame, api):
    if not Path(egomimic.__file__).resolve().is_relative_to(source_root.resolve()):
        raise RuntimeError("Probe imported a different source checkout")
    embodiment = Eva if vendor == "eva" else Human
    stride = 3 if vendor == "aria" else 1
    if api == "legacy":
        if representation == "keypoints":
            mode = (
                "keypoints_wristframe_ypr"
                if frame == "eef_frame"
                else "keypoints_headframe_ypr"
            )
        else:
            mode = "cartesian_wristframe_ypr" if frame == "eef_frame" else "cartesian"
        arguments = {"mode": mode}
        if vendor != "eva":
            arguments["stride"] = stride
    else:
        arguments = {
            "action_mode": representation,
            "coord_frame": frame,
            "rotation_mode": "euler",
        }
        if vendor != "eva":
            arguments["stride"] = stride
    transforms = embodiment.get_transform_list(**arguments)

    def dataset(norm_mode=False):
        keys = {"keymap_mode": representation, "norm_mode": norm_mode}
        if vendor != "eva":
            # Retained 138D targets use MANO keypoints. The separate Aria hand
            # tracker is not an input to these recipes or this fixture.
            keys.update(has_head_pose=True, include_aria_keypoints=False)
        resolver = LocalEpisodeResolver(
            folder_path=str(root / vendor),
            key_map=embodiment.get_keymap(**keys),
            transform_list=transforms,
        )
        return MultiDataset._from_resolver(resolver, mode="total", bounds_check=False)

    data = dataset()
    owner = MultiDataset(state={}, norm_mode="quantile")
    owner.populate_from_datasets({"source": data})
    first = data[0]
    identity = int(first["embodiment"])
    owner.infer_shapes_from_batch(first)
    owner.infer_norm_from_dataset(
        dataset(norm_mode=True),
        identity,
        sample_frac=0.37,
        seed=42,
        batch_size=11,
        num_workers=0,
    )
    arrays = {}
    indices = [0, 7, 29, 47, 50]
    model_keys = sorted(
        key for key in first if key.startswith(("actions_", "observations."))
    )
    for key in model_keys:
        arrays["raw/" + key] = np.stack([np.asarray(data[i][key]) for i in indices])
    for key, stats in owner.norm_stats[identity].items():
        for name, values in stats.items():
            arrays[f"stats/{key}/{name}"] = np.asarray(values)
    batch = {key: torch.as_tensor(arrays["raw/" + key]) for key in model_keys}
    for mode in ("quantile", "minmax", "zscore"):
        owner.norm_mode = mode
        normalized = owner.normalize(batch, identity)
        restored = owner.unnormalize(normalized, identity)
        for key in owner.norm_stats[identity]:
            arrays[f"normalized/{mode}/{key}"] = normalized[key].numpy()
            arrays[f"restored/{mode}/{key}"] = restored[key].numpy()
    action_key = (
        "actions_keypoints" if representation == "keypoints" else "actions_cartesian"
    )
    target = batch[action_key].float()
    generator = torch.Generator().manual_seed(714)
    predictions = target + torch.randn(target.shape, generator=generator) * 0.03
    samples = torch.stack([predictions, predictions + 0.02, predictions - 0.01])
    if api == "legacy":
        from egomimic.utils.metrics import (
            frechet_gaussian_over_time,
            reverse_kl_from_samples,
        )

        metrics = {
            "paired_mse_avg": (predictions - target).square().mean(),
            "final_mse_avg": (predictions[:, -1] - target[:, -1]).square().mean(),
        }
        frechet = frechet_gaussian_over_time(predictions, target)
        metrics.update(
            frechet_gauss_avg=frechet.mean(),
            frechet_gauss_min=frechet.min(),
            frechet_gauss_max=frechet.max(),
        )
        metrics["reverse_kl_M3"] = reverse_kl_from_samples(samples, target)
    else:
        from egomimic.eval.cartesian_metrics import cartesian_metrics, sample_metrics

        metrics = cartesian_metrics(predictions, target)
        metrics.update(sample_metrics(samples, target))
    for key in (
        "paired_mse_avg",
        "final_mse_avg",
        "frechet_gauss_avg",
        "frechet_gauss_min",
        "frechet_gauss_max",
        "reverse_kl_M3",
    ):
        arrays["metrics/" + key] = torch.as_tensor(metrics[key]).numpy()
    metadata = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "vendor": vendor,
        "representation": representation,
        "frame": frame,
        "identity": identity,
        "indices": indices,
        "model_keys": model_keys,
        "api": api,
        "transform_arguments": arguments,
        "normalization_frames": owner._norm_run_metadata["frames"],
    }
    return arrays, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--vendor", choices=["eva", "aria", "mecka", "scale"], required=True
    )
    parser.add_argument(
        "--representation", choices=["cartesian", "keypoints"], required=True
    )
    parser.add_argument("--frame", choices=["camframe", "eef_frame"], required=True)
    parser.add_argument("--api", choices=["legacy", "graph"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    arrays, metadata = probe(**args)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata["archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    with output.with_suffix(".json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
