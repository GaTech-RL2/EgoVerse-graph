"""Write ABC-style ``<arm>.{obs,cmd}_ee_pose_arm_local`` into local rl2 YAM episodes.

rl2 ``processed_v3/yam`` stores ``<arm>.{obs,cmd}_ee_pose`` as YAM ``tcp_site`` FK
with both arms in ``left_base`` (``pose_world_frame``). The ABC
``arm_local_fk_v1`` conversion stores each arm in its own base at ``grasp_site``.
This adds the ABC-convention arrays next to the originals, so a data config can
read rl2 episodes with ``pose_zarr_suffix: _arm_local`` exactly as in ABC
pretraining:

    left  = P_left · O_left
    right = inv(left_base_T_right_base) · P_right · O_right

``O`` is a pure translation in the tcp frame (orientation is identical). It was
fit on 4554 ABC frames per arm as mean(FK_tcp(q)^-1 · ABC_pose) with
``yam_arm.xml``; the per-frame std is about 3 mm because ABC used a slightly
different FK model. All-zero padding rows stay zero. Source arrays are not
modified; only the local episode copies gain new arrays and feature entries.

Usage: python scripts/data/rl2_tcp_to_arm_local.py <dataset_dir> [--workers N]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

TCP_TO_GRASP_M = {
    "left": np.array([0.0062, 0.0029, 0.1386]),
    "right": np.array([0.0061, -0.0027, 0.1388]),
}
SUFFIX = "_arm_local"


def convert(pose, base_T_frame, offset):
    """pose: (N, 7) xyz + wxyz in the stored frame -> (N, 7) in the arm base at grasp."""
    out = np.zeros_like(pose)
    valid = np.linalg.norm(pose[:, 3:], axis=1) > 0.5
    if not valid.any():
        return out
    p = pose[valid]
    rot = Rotation.from_quat(p[:, 3:], scalar_first=True)
    pos = p[:, :3] + rot.apply(offset)
    base_rot = Rotation.from_matrix(base_T_frame[:3, :3])
    pos = base_rot.apply(pos) + base_T_frame[:3, 3]
    quat = (base_rot * rot).as_quat(scalar_first=True)
    out[valid] = np.concatenate([pos, quat], axis=1)
    return out


def process(episode: Path):
    group = zarr.open_group(str(episode), mode="r+")
    attrs = group.attrs.asdict()
    if attrs.get("pose_world_frame") != "left_base":
        raise ValueError(f"{episode.name}: unexpected pose_world_frame {attrs.get('pose_world_frame')!r}")
    left_T_right = np.asarray(attrs["left_base_T_right_base"], dtype=np.float64)
    to_base = {"left": np.eye(4), "right": np.linalg.inv(left_T_right)}
    features = dict(attrs["features"])
    for arm in ("left", "right"):
        for kind in ("obs", "cmd"):
            src_key = f"{arm}.{kind}_ee_pose"
            src = group[src_key]
            out = convert(np.asarray(src[:], dtype=np.float64), to_base[arm], TCP_TO_GRASP_M[arm])
            group.create_array(
                src_key + SUFFIX,
                shape=out.shape,
                dtype="float64",
                chunks=src.chunks,
                shards=src.shards,
                overwrite=True,
            )[:] = out
            features[src_key + SUFFIX] = features[src_key]
    attrs["features"] = features
    attrs["arm_local_conversion"] = {
        "script": "scripts/data/rl2_tcp_to_arm_local.py",
        "pose_frame": "per_arm_base",
        "pose_endpoint": "grasp_site_abc_fit",
        "tcp_to_grasp_m": {k: v.tolist() for k, v in TCP_TO_GRASP_M.items()},
    }
    group.attrs.update(attrs)
    return episode.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    episodes = sorted(p for p in args.dataset_dir.iterdir() if (p / "zarr.json").exists())
    with ThreadPoolExecutor(args.workers) as pool:
        done = list(pool.map(process, episodes))
    print(f"converted {len(done)} episodes")


if __name__ == "__main__":
    main()
