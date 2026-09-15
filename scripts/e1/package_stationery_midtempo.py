#!/usr/bin/env python3
"""Package one stationery mid-tempo checkpoint as a rollout-ready artefact for the YAM station.

  <dest-root>/stationery_midtempo_<row>_graph_step<N>/
    checkpoint.ckpt              copied (.partial -> rename), sha256 == source
    norm_stats/norm_stats.json   the training run's cache (carries normalizer_state)
    hydra/{config,hydra,overrides}.yaml   the training run's composed config
    yam_rollout_<row>.yaml       graph robot/yam_rollout.yaml with the policy block filled
    rollout_info.yaml            provenance, decode + frame contract, selection and eval reads
    .copy_complete               timestamp, written last

Frame contract: the rl2 zarrs store both arms in the left arm's base frame
(attrs pose_world_frame = left_base) with left_base_T_right_base. CartesianGraphAdapter's
base_T_model maps model-frame poses into each arm's base, so left = I and
right = inv(left_base_T_right_base), read from --calib-episode.

usage: package_stationery_midtempo.py --row time|arcdur --run <runs/.../<row>_s42>
         --ckpt <checkpoint> --step <N> --graph <EgoVerse-graph> --dest-root <EgoVerse/rollouts>
         --calib-episode <mirror/<rl2 hash>> [--selection <selection.json>] [--split-manifest <manifest.json>]
         [--train-job <slurm id>]
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import yaml
import zarr
from omegaconf import OmegaConf

CAMERA_KEYS = {
    "front_img_1": "observations.images.front_img_1",
    "left_wrist_img": "observations.images.left_wrist_img",
    "right_wrist_img": "observations.images.right_wrist_img",
}
ARC_DECODER = {
    "_target_": "egomimic.robot.arc_decoder.BimanualArcDecoder",
    "token_layout": "e1_dur",
    "min_distance_unit": 0.40,
    "resampled_vector_length": 100,
    "dt": 1.0 / 30.0,
    "action_horizon": 100,
}
EXECUTE_STEPS = 25  # the open-loop evaluator's execute_fraction 0.25 of a 100-step chunk


def sha256(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def git(graph, *args):
    return subprocess.run(["git", "-C", graph, *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=("time", "arcdur"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", required=True, type=int)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dest-root", required=True)
    ap.add_argument("--calib-episode", required=True)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--split-manifest", default=None)
    ap.add_argument("--train-job", default=None)
    args = ap.parse_args()

    name = f"stationery_midtempo_{args.row}_graph_step{args.step}"
    dest = os.path.join(args.dest_root, name)
    if os.path.exists(os.path.join(dest, ".copy_complete")):
        raise SystemExit(f"{dest} is already packaged; refusing to overwrite")
    os.makedirs(os.path.join(dest, "norm_stats"), exist_ok=True)
    os.makedirs(os.path.join(dest, "hydra"), exist_ok=True)

    # Checkpoint: copy to .partial, verify, rename.
    ckpt_dst = os.path.join(dest, "checkpoint.ckpt")
    partial = ckpt_dst + ".partial"
    shutil.copyfile(args.ckpt, partial)
    src_sha, dst_sha = sha256(args.ckpt), sha256(partial)
    if src_sha != dst_sha:
        raise SystemExit(f"sha256 mismatch: {src_sha} != {dst_sha}")
    os.replace(partial, ckpt_dst)

    train_dir = os.path.join(args.run, "train")
    norm_src = os.path.join(train_dir, "norm_stats", "norm_stats.json")
    norm_dst = os.path.join(dest, "norm_stats", "norm_stats.json")
    shutil.copyfile(norm_src, norm_dst)
    if "normalizer_state" not in json.load(open(norm_dst)):
        raise SystemExit("norm_stats.json lacks normalizer_state; graph_policy cannot load it")
    for f in ("config.yaml", "hydra.yaml", "overrides.yaml"):
        src = os.path.join(train_dir, ".hydra", f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dest, "hydra", f))

    # Frame contract from the station's own calibration.
    attrs = dict(zarr.open_group(args.calib_episode, mode="r").attrs)
    if attrs.get("pose_world_frame") != "left_base":
        raise SystemExit(f"calibration episode pose_world_frame={attrs.get('pose_world_frame')!r}, expected left_base")
    left_T_right = np.asarray(attrs["left_base_T_right_base"], dtype=float)
    base_T_model = {"left": np.eye(4).tolist(), "right": np.linalg.inv(left_T_right).tolist()}

    robot = OmegaConf.load(os.path.join(args.graph, "egomimic/hydra_configs/robot/yam_rollout.yaml"))
    robot.execute_steps = EXECUTE_STEPS
    robot.policy.training_config = os.path.join(dest, "hydra", "config.yaml")
    robot.policy.checkpoint = ckpt_dst
    robot.policy.normalizer_path = norm_dst
    robot.policy.adapter.base_T_model = base_T_model
    robot.policy.adapter.camera_keys = CAMERA_KEYS
    robot.policy.adapter.prompt = ""
    robot.policy.adapter.decoder = dict(ARC_DECODER) if args.row == "arcdur" else None
    robot_yaml = os.path.join(dest, f"yam_rollout_{args.row}.yaml")
    OmegaConf.save(robot, robot_yaml)

    selection = json.load(open(args.selection)).get(args.row) if args.selection else None
    split = None
    if args.split_manifest:
        man = json.load(open(args.split_manifest))
        split = {"manifest": args.split_manifest, "version": man.get("version"),
                 "tertile_cuts_m_per_s": man["pool"]["tertile_cuts"],
                 "sets": {k: {kk: v[kk] for kk in ("n", "hours", "sources")} for k, v in man["sets"].items()
                          if k in ("train", "val", "test_mid", "test_in")}}
    info = {
        "artefact": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "row": args.row,
        "task": "sort the stationery into containers (ABC) + organize_stationary (rl2 YAM station)",
        "design": "middle tau_ep tertile held out of training; see split",
        "repo": "GaTech-RL2/EgoVerse-graph",
        "branch": git(args.graph, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git(args.graph, "rev-parse", "HEAD"),
        "worktree_dirty_files": len([l for l in git(args.graph, "status", "--porcelain").splitlines() if l.strip()]),
        "run_dir": args.run,
        "train_job": args.train_job,
        "checkpoint": {"source": args.ckpt, "step": args.step, "sha256": dst_sha},
        "decode": ({"layout": "e1_dur", "decoder": ARC_DECODER,
                    "note": "decode the native (100, 16) token before frame conversion; the open-loop evaluator used the same codec"}
                   if args.row == "arcdur" else {"layout": "time", "note": "(100, 14) eef_frame chunk at 30 Hz; no decoder"}),
        "frames": {"action_frame": "eef_frame", "rotation": "xyz + intrinsic ZYX euler (radians), gripper in [0, 1]",
                   "model_frame": "left arm base (rl2 zarr pose_world_frame)",
                   "base_T_model": base_T_model, "calibration_episode": args.calib_episode,
                   "calibration_source": {k: attrs.get(k) for k in ("calibration_source_repository", "calibration_source_revision", "front_camera_serial")}},
        "inputs": {"cameras": CAMERA_KEYS, "image_hw": [480, 640], "proprio": "observations.state.ee_pose (14)", "embodiment_id": 7},
        "execute_steps": EXECUTE_STEPS,
        "selection": selection,
        "split": split,
        "robot_config": robot_yaml,
    }
    with open(os.path.join(dest, "rollout_info.yaml"), "w") as f:
        yaml.safe_dump(json.loads(json.dumps(info, default=str)), f, sort_keys=False)
    with open(os.path.join(dest, ".copy_complete"), "w") as f:
        f.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
    print(json.dumps({"dest": dest, "sha256": dst_sha, "robot_yaml": robot_yaml}, indent=1))


if __name__ == "__main__":
    main()
