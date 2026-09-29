"""Convert every task in a pinned LIBERO suite into one OAT-format replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from egomimic.benchmarks.libero.catalog import (
    LIBERO_COMMIT,
    OAT_COMMIT,
    TASK_IDS,
    get_tasks,
)


def convert_suite(input_root, output, suite, *, seed=42, demos_per_task=None):
    root, output = Path(input_root), Path(output)
    tasks = get_tasks(suite)
    if demos_per_task is not None and demos_per_task < 1:
        raise ValueError("demos_per_task must be positive")
    sources = {}
    for task in tasks:
        paths = list(root.rglob(f"{task}_demo.hdf5"))
        if len(paths) != 1:
            raise ValueError(
                f"Expected exactly one HDF5 for {task}, found {len(paths)}"
            )
        sources[task] = paths[0]
    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w", zarr_format=2)
    data = group.create_group("data")
    rng = np.random.default_rng(seed)
    pending, source_hashes = {}, {}
    for task, path in sources.items():
        with path.open("rb") as handle:
            source_hashes[task] = hashlib.file_digest(handle, "sha256").hexdigest()
        with h5py.File(path, "r") as handle:
            names = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
            count = (
                len(names)
                if demos_per_task is None
                else min(demos_per_task, len(names))
            )
            pending[task] = list(rng.choice(names, count, replace=False))
    total, ends, episodes = 0, [], []
    while pending:
        task = list(pending)[int(rng.integers(len(pending)))]
        demo_name = pending[task].pop(0)
        if not pending[task]:
            del pending[task]
        with h5py.File(sources[task], "r") as handle:
            raw = handle["data"]
            source_task = Path(raw.attrs["bddl_file_name"]).stem
            if source_task != task:
                raise ValueError(f"HDF5 task metadata differs: {source_task} != {task}")
            demo = raw[demo_name]
            action = np.asarray(demo["actions"], dtype=np.float32)
            length = len(action)
            values = {
                "action": action,
                "agentview_rgb": np.flip(
                    np.asarray(demo["obs/agentview_rgb"], dtype=np.uint8), axis=1
                ),
                "robot0_eye_in_hand_rgb": np.flip(
                    np.asarray(demo["obs/eye_in_hand_rgb"], dtype=np.uint8), axis=1
                ),
                "robot0_eef_pos": np.asarray(demo["obs/ee_pos"], dtype=np.float32),
                "robot0_eef_quat": Rotation.from_rotvec(np.asarray(demo["obs/ee_ori"]))
                .as_quat()
                .astype(np.float32),
                "robot0_gripper_qpos": np.asarray(
                    demo["obs/gripper_states"], dtype=np.float32
                ),
                "task_uid": np.full((length, 1), TASK_IDS[task][2], dtype=np.int64),
            }
            if "joint_states" in demo["obs"]:
                values["robot0_joint_pos"] = np.asarray(
                    demo["obs/joint_states"], dtype=np.float32
                )
            prompt = json.loads(raw.attrs["problem_info"])["language_instruction"]
            if action.shape != (length, 7) or length == 0:
                raise ValueError(f"Invalid action shape in {task}/{demo_name}")
            for key, value in values.items():
                if len(value) != length or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key} in {task}/{demo_name}")
                if key not in data:
                    chunks = (1 if value.ndim == 4 else 1024, *value.shape[1:])
                    data.create_array(
                        key,
                        shape=(0, *value.shape[1:]),
                        chunks=chunks,
                        dtype=value.dtype,
                    )
                array = data[key]
                array.resize((total + length, *value.shape[1:]))
                array[total : total + length] = value
            total += length
            ends.append(total)
            episodes.append(
                {"task": task, "demo": demo_name, "prompt": prompt, "length": length}
            )
    group.create_group("meta").create_array(
        "episode_ends", data=np.asarray(ends, dtype=np.int64)
    )
    manifest = {
        "suite": suite,
        "seed": seed,
        "oat_commit": OAT_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "hdf5_sha256": source_hashes,
        "episodes": episodes,
        "complete": True,
    }
    (output / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demos-per-task", type=int)
    args = parser.parse_args()
    convert_suite(
        args.input_root,
        args.output,
        args.suite,
        seed=args.seed,
        demos_per_task=args.demos_per_task,
    )


if __name__ == "__main__":
    main()
