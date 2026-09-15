#!/usr/bin/env python3
"""Offline replay check of a packaged stationery mid-tempo rollout artefact (no robot I/O).

Loads the package exactly as the station would (egomimic.robot.graph_policy.load_graph_policy
on its yam_rollout_<row>.yaml policy block), rebuilds robot observations from a recorded rl2
episode, and walks the episode the way the open-loop evaluator does: predict at frame t,
execute the first EXECUTE_STEPS commands, jump to t + EXECUTE_STEPS.

Robot observations are synthesized in each arm's base frame (model-frame zarr pose mapped
through base_T_model), cameras as uint8 BGR HWC, gripper in joint_positions[offset + 6] --
the contract CartesianGraphAdapter reads. Images are decoded straight with simplejpeg:
zarr's object arrays hand back a 0-d array per frame (slice to get bytes), and the repo's
decode_jpeg_payload returns CHW float in [0, 1], which is not what the adapter wants.
Commands come back in arm base frames and are compared with the recorded cmd_ee_pose mapped
the same way.

CartesianGraphAdapter refuses to emit a gripper command outside [0, 1] -- a real deployment
guard. An undertrained checkpoint trips it on every segment, so that is caught and reported
(guard_failures, native_action_ranges, verdict) rather than raised: the point of this check is
to describe what the station would do with this package, including refusing to drive it.

Reports:
  proprio round trip  adapter proprio vs the zarr model-frame pose (checks the ZYX / quaternion
                      and base_T_model code path; it cannot validate the calibration itself)
  replay_xyz_mse      mean squared xyz error over executed steps, both arms (6 columns)
  eval_xyz_mse        the open-loop evaluator's xyz_mse for the same episode, if --eval-json holds it
A mis-wired camera, proprio, normalizer or decoder shows up as replay_xyz_mse far above eval_xyz_mse.
Flow sampling is stochastic here (the evaluator pins its seed), so expect agreement, not equality.

usage: replay_check_stationery_midtempo.py --robot-yaml <package>/yam_rollout_<row>.yaml
         --episode <mirror>/<rl2 hash> [--eval-json <eval dir>/open_loop_sim.json] [--out <json>]
"""

import argparse
import json
import os

import numpy as np
import simplejpeg
import torch
import zarr
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from egomimic.robot.graph_policy import load_graph_policy
from egomimic.robot.interface import ARM_OFFSET, pose_vector

ZARR_CAMERAS = {"front_img_1": "images.front_1", "left_wrist_img": "images.left_wrist", "right_wrist_img": "images.right_wrist"}
EXECUTE_STEPS = 25


def matrix_from_xyzwxyz(p):
    m = np.eye(4)
    m[:3, 3] = p[:3]
    w, x, y, z = p[3:7]
    m[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    return m


def bgr_frame(array, index):
    """One frame of a zarr JPEG object array as uint8 BGR HWC, the adapter's input."""
    payload = array[index : index + 1][0]
    if isinstance(payload, np.ndarray):
        payload = payload.item()
    rgb = simplejpeg.decode_jpeg(bytes(payload), colorspace="RGB")
    return np.ascontiguousarray(rgb[..., ::-1])


@torch.no_grad()
def native_action(policy, obs):
    """The decoded (B, H, 14) prediction, before the adapter's frame conversion and guards.

    Same path as GraphRobotPolicy.predict, stopped one step earlier so an out-of-range
    prediction can be described instead of only refused.
    """
    adapter = policy.adapter
    values = policy.normalizer.normalize(adapter.observation(obs), adapter.embodiment_id)
    batch = policy.graph.process_batch_for_training({"robot": values})
    prediction = policy.graph.forward_eval(batch)["robot"]["pred_action"]
    native = policy.normalizer.unnormalize({adapter.action_key: prediction}, adapter.embodiment_id)[adapter.action_key]
    if adapter.decoder is not None:
        native = adapter.decoder(native)
    if torch.is_tensor(native):
        native = native.detach().cpu().numpy()
    return np.asarray(native, dtype=float)


def action_ranges(native):
    rows = native.reshape(-1, native.shape[-1])
    width = rows.shape[-1]
    half = width // 2
    grip = np.concatenate([rows[:, half - 1], rows[:, width - 1]])
    xyz = np.concatenate([rows[:, 0:3], rows[:, half : half + 3]], axis=0)
    return {
        "width": int(width),
        "gripper_min": float(grip.min()),
        "gripper_max": float(grip.max()),
        "xyz_abs_max_m": float(np.abs(xyz).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-yaml", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-segments", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)

    cfg = OmegaConf.to_container(OmegaConf.load(args.robot_yaml).policy, resolve=True)
    cfg["device"] = args.device
    policy = load_graph_policy(cfg)
    adapter = policy.adapter

    g = zarr.open_group(args.episode, mode="r")
    n = int(g.attrs["total_frames"])
    obs_pose = {a: np.asarray(g[f"{a}.obs_ee_pose"][:n], dtype=float) for a in ARM_OFFSET}
    cmd_pose = {a: np.asarray(g[f"{a}.cmd_ee_pose"][:n], dtype=float) for a in ARM_OFFSET}
    obs_grip = {a: np.asarray(g[f"{a}.obs_gripper"][:n], dtype=float).reshape(-1) for a in ARM_OFFSET}
    cmd_grip = {a: np.asarray(g[f"{a}.cmd_gripper"][:n], dtype=float).reshape(-1) for a in ARM_OFFSET}
    cameras = {camera: g[ZARR_CAMERAS[camera]] for camera in adapter.camera_keys}

    sq_xyz = sq_grip = 0.0
    steps = segments = scored = guard_failures = 0
    prop_xyz_max = prop_rot_max_deg = 0.0
    shapes_ok = finite_ok = True
    first_guard_error = None
    ranges = None
    for t in range(0, n, EXECUTE_STEPS):
        if args.max_segments and segments >= args.max_segments:
            break
        k = min(EXECUTE_STEPS, n - t)
        ee, joints = np.zeros(14), np.zeros(14)
        for arm, off in ARM_OFFSET.items():
            base_pose = adapter.base_T_model[arm] @ matrix_from_xyzwxyz(obs_pose[arm][t])
            ee[off : off + 6] = pose_vector(base_pose)
            ee[off + 6] = obs_grip[arm][t]
            joints[off + 6] = obs_grip[arm][t]
        obs = {"ee_poses": ee, "joint_positions": joints}
        for camera, array in cameras.items():
            obs[camera] = bgr_frame(array, t)

        proprio = adapter.observation(obs)[adapter.proprio_key][0].numpy().astype(float)
        for arm, off in ARM_OFFSET.items():
            prop_xyz_max = max(prop_xyz_max, float(np.abs(proprio[off : off + 3] - obs_pose[arm][t][:3]).max()))
            w, x, y, z = obs_pose[arm][t][3:7]
            rel = Rotation.from_euler("ZYX", proprio[off + 3 : off + 6]).inv() * Rotation.from_quat([x, y, z, w])
            prop_rot_max_deg = max(prop_rot_max_deg, float(np.degrees(rel.magnitude())))

        segments += 1
        if ranges is None:
            ranges = action_ranges(native_action(policy, obs))
        try:
            actions = policy.predict(obs)
        except ValueError as error:
            # A deployment guard (gripper range, degenerate rotation, shape): the station
            # would refuse to drive this package. Record it and keep walking the episode.
            guard_failures += 1
            if first_guard_error is None:
                first_guard_error = str(error)
            continue
        shapes_ok &= actions.ndim == 2 and actions.shape[1] == 14 and actions.shape[0] >= k
        finite_ok &= bool(np.isfinite(actions).all())
        for arm, off in ARM_OFFSET.items():
            gt = np.stack([pose_vector(adapter.base_T_model[arm] @ matrix_from_xyzwxyz(cmd_pose[arm][t + j])) for j in range(k)])
            sq_xyz += float(np.square(actions[:k, off : off + 3] - gt[:, :3]).sum())
            sq_grip += float(np.square(actions[:k, off + 6] - cmd_grip[arm][t : t + k]).sum())
        steps += k
        scored += 1

    episode = os.path.basename(os.path.normpath(args.episode))
    if scored == 0:
        verdict = f"package loads and predicts, but every segment tripped a deployment guard: {first_guard_error}"
    elif guard_failures:
        verdict = f"{guard_failures} of {segments} segments tripped a deployment guard: {first_guard_error}"
    else:
        verdict = "every segment produced valid commands"
    result = {
        "robot_yaml": args.robot_yaml,
        "episode": episode,
        "frames": n,
        "segments": segments,
        "segments_scored": scored,
        "guard_failures": guard_failures,
        "first_guard_error": first_guard_error,
        "native_action_ranges": ranges,
        "executed_steps": steps,
        "shapes_ok": bool(shapes_ok),
        "finite_ok": bool(finite_ok),
        "proprio_roundtrip_xyz_max_m": prop_xyz_max,
        "proprio_roundtrip_rot_max_deg": prop_rot_max_deg,
        "replay_xyz_mse": sq_xyz / (steps * 6) if steps else None,
        "replay_grip_mse": sq_grip / (steps * 2) if steps else None,
        "eval_xyz_mse": None,
        "verdict": verdict,
    }
    if args.eval_json:
        for item in json.load(open(args.eval_json)).get("episode_results", []):
            if str(item.get("episode")) == episode:
                result["eval_xyz_mse"] = float(item["metrics"]["xyz_mse"])
                result["eval_grip_mse"] = float(item["metrics"]["grip_mse"])
                result["eval_coverage"] = float(item["coverage"])
    if result["eval_xyz_mse"] and result["replay_xyz_mse"]:
        result["replay_over_eval_xyz"] = result["replay_xyz_mse"] / result["eval_xyz_mse"]
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
