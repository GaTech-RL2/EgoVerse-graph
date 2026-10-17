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


def native_episode_values(
    demo, task, *, action_dim=7, hand_dim=2, flip_rows=True, native_quaternion=False
):
    """Read the declared native replay view without changing controller units."""
    action = np.asarray(demo["actions"], dtype=np.float32)
    length = len(action)
    if length == 0 or action.shape != (length, action_dim):
        raise ValueError("Native action shape differs from the explicit contract")

    def image(key):
        rgb = np.asarray(demo["obs/" + key], dtype=np.uint8)
        return np.flip(rgb, axis=1) if flip_rows else rgb

    quat = (
        np.asarray(demo["obs/ee_quat"], dtype=np.float32)
        if native_quaternion
        else Rotation.from_rotvec(np.asarray(demo["obs/ee_ori"]))
        .as_quat()
        .astype(np.float32)
    )
    values = {
        "action": action,
        "agentview_rgb": image("agentview_rgb"),
        "robot0_eye_in_hand_rgb": image("eye_in_hand_rgb"),
        "robot0_eef_pos": np.asarray(demo["obs/ee_pos"], dtype=np.float32),
        "robot0_eef_quat": quat,
        "robot0_gripper_qpos": np.asarray(demo["obs/gripper_states"], dtype=np.float32),
        "task_uid": np.full((length, 1), TASK_IDS[task][2], dtype=np.int64),
    }
    if values["robot0_gripper_qpos"].shape != (length, hand_dim):
        raise ValueError("Native hand-state width differs from the explicit contract")
    if values["robot0_eef_pos"].shape != (length, 3) or quat.shape != (length, 4):
        raise ValueError("Native EEF pose shape differs")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-4):
        raise ValueError("Native EEF quaternion must be unit XYZW")
    if "joint_states" in demo["obs"]:
        values["robot0_joint_pos"] = np.asarray(
            demo["obs/joint_states"], dtype=np.float32
        )
    for key, value in values.items():
        if len(value) != length or not np.isfinite(value).all():
            raise ValueError(f"Invalid native {key}")
    return values


def recorded_hand_control_is_valid(handle, metadata, step):
    """The recorder's control flag also marks post-action IK pauses as false.

    Quest packet validity is checked independently by the caller. The uploaded
    Mink V4 collector executes the saved action, then sets paused for a failed
    IK result; recording uses packet.tracking_valid AND not command.paused.
    """
    if bool(handle["tracking_valid"][step]):
        return True
    if "control_reason" not in handle:
        return False
    reason = handle["control_reason"][step]
    if isinstance(reason, bytes):
        reason = reason.decode("utf-8")
    arm = metadata.get("action_contract", {}).get("arm_control", {})
    return (
        reason == "arm_ik_infeasible"
        and arm.get("name") == "mink_direct_joint_servo_v4"
        and arm.get("mode") == "mink"
    )


def aligned_hand_keypoint_actions(handle, metadata):
    """Recover causal operator targets using the recorded pre-filter wrist goal.

    An asynchronous latest-packet guess is prohibited: control computation can
    span multiple packet arrivals. Match the independently saved intended wrist
    target against calibration instead, rejecting every ambiguous/missing step.
    An unrecorded first input may be trimmed; later gaps cannot be stitched.
    """
    packets = [json.loads(raw) for raw in handle["packet_json"][:]]
    received = np.asarray(handle["packet_receive_ns"], dtype=np.int64)
    applied = np.asarray(handle["action_host_ns"], dtype=np.int64)
    goals = np.asarray(handle["intent_target_position_world_m"], dtype=np.float64)
    calibration = metadata["hand_calibration"]
    if json.loads(handle.attrs.get("recenter_events_json", "[]")):
        raise ValueError(
            "Keypoint alignment requires explicit per-recenter calibration"
        )
    basis = np.asarray(metadata["world_from_quest"], dtype=np.float64)
    if basis.shape != (3, 3) or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-8):
        raise ValueError("Unknown Quest-to-world frame")
    settings = metadata["control_settings"]
    position = np.asarray([packet["wrist"]["position_m"] for packet in packets])
    offset = (
        (position - calibration["operator_wrist_position_quest_m"])
        @ basis.T
        * settings["translation_gain"]
    )
    offset[:, 2] *= settings["vertical_gain"]
    expected = np.clip(offset, -0.30, 0.30) + calibration["position_world_m"]
    selected = []
    for step, goal in enumerate(goals):
        matches = np.flatnonzero(
            (received <= applied[step])
            & (np.max(np.abs(expected - goal), axis=-1) <= 1e-10)
        )
        if len(matches) != 1:
            if step == 0 and len(matches) == 0:
                selected.append(None)
                continue
            raise ValueError(
                f"Unproven keypoint packet correspondence at step {step}: {len(matches)} candidates"
            )
        selected.append(int(matches[0]))
    start = int(bool(selected and selected[0] is None))
    rows, sequence = [], []
    for step, index in enumerate(selected[start:], start):
        packet = packets[index]
        if (
            packet.get("source") != "quest_webxr"
            or packet.get("hand") != "right"
            or packet.get("joint_layout") != "mediapipe21.v2"
            or not packet.get("tracking_valid")
            or not all(packet.get("joint_valid", []))
            or len(packet.get("joint_valid", [])) != 21
            or not recorded_hand_control_is_valid(handle, metadata, step)
        ):
            raise ValueError("Invalid or unsupported hand tracking target")
        points = np.asarray(packet["joints_m"], dtype=np.float64)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("Hand target must contain 21 finite XYZ points")
        rows.append((points - points[0]).reshape(63))
        sequence.append(packet["sequence"])
    if not rows:
        raise ValueError("No aligned hand keypoint steps")
    actions = np.concatenate(
        [np.asarray(handle["actions"])[start:, :6], np.asarray(rows)], axis=-1
    )
    return actions.astype(np.float32), start, np.asarray(sequence, dtype=np.int64)


def append_replay(data, values, offset, *, image_chunk_frames=1):
    if type(image_chunk_frames) is not int or image_chunk_frames < 1:
        raise ValueError("Image storage chunks require a positive frame count")
    length = len(values["action"])
    for key, value in values.items():
        if key not in data:
            data.create_array(
                key,
                shape=(0, *value.shape[1:]),
                chunks=(
                    image_chunk_frames if value.ndim == 4 else 1024,
                    *value.shape[1:],
                ),
                dtype=value.dtype,
            )
        array = data[key]
        if tuple(array.shape[1:]) != tuple(value.shape[1:]):
            raise ValueError(f"Replay source shape changed within {key}")
        array.resize((offset + length, *value.shape[1:]))
        array[offset : offset + length] = value
    return offset + length


def causal_collection_episode_values(
    handle, metadata, task, targets, start, sequence, *, resume_proof=None
):
    """Pair action[t] with the lossless observation recorded before action[t].

    The collection's convenience demo view is post-action. Native replay
    observations end at t while targets start at t, so using that view directly
    leaks a future frame. The immutable raw timeline contains T+1 observations.
    """
    if metadata.get("observation_order") != "before_action_and_terminal":
        raise ValueError("Raw pre-action observation order is unproven")
    if (
        handle["data"].attrs.get("rgb_row_transform")
        != "vertical_flip_from_raw_opengl_to_original_zarr_upright"
    ):
        raise ValueError("Raw OpenGL-to-upright row convention is unproven")
    count = len(handle["actions"])
    before = np.asarray(handle["observation_host_ns"], dtype=np.int64)
    action_time = np.asarray(handle["action_host_ns"], dtype=np.int64)
    if (
        before.shape != (count + 1,)
        or action_time.shape != (count,)
        or np.any(np.diff(before) <= 0)
        or np.any(np.diff(action_time) <= 0)
        or np.any(before[:-1] > action_time)
        or np.any(action_time > before[1:])
    ):
        raise ValueError("Raw observation/action timeline is not causal")
    if not 0 <= start < count or targets.shape != (count - start, 69):
        raise ValueError("Keypoint target trim/width differs from raw timeline")
    if start > 1:
        if (
            resume_proof is None
            or resume_proof.get("status")
            != "PASS_UNIQUE_SOURCE_BACKED_RESUME_CALIBRATION"
            or resume_proof.get("source_start_step") != start
            or resume_proof.get("source_path") != str(Path(handle.filename).resolve())
            or len(resume_proof.get("solutions", [])) != 1
            or resume_proof.get("no_later_gap_discarded") is not True
        ):
            raise ValueError(
                "Initial hold prefix requires an exact unique source-backed resume proof"
            )
        reasons = np.asarray(handle["control_reason"]).astype(str)
        if (
            not set(reasons[:start]) <= {"stale_3d_view", "operator_stop"}
            or not np.all(np.asarray(handle["actions"][:start, :6]) == 0)
            or not np.all(reasons[start:] == "active")
        ):
            raise ValueError(
                "Resume proof cannot discard active controls or later gaps"
            )
    values = {
        "action": targets,
        "target_packet_sequence": sequence,
        "action_host_ns": action_time[start:],
        "observation_host_ns": before[start:-1],
    }
    for output, source, width in (
        ("robot0_eef_pos", "robot0_eef_pos", 3),
        ("robot0_eef_quat", "robot0_eef_quat", 4),
        ("robot0_gripper_qpos", "robot0_gripper_qpos", 22),
    ):
        dataset = handle["numeric_observations"][source]
        if dataset.shape != (count + 1, width):
            raise ValueError(f"Raw pre-action {source} shape differs")
        values[output] = np.asarray(dataset[start:-1], dtype=np.float32)
    for output, source in (
        ("agentview_rgb", "agentview"),
        ("robot0_eye_in_hand_rgb", "robot0_eye_in_hand"),
    ):
        dataset = handle["rgb"][source]
        if dataset.shape[0] != count + 1 or dataset.ndim != 4 or dataset.shape[-1] != 3:
            raise ValueError(f"Raw pre-action camera {source} shape differs")
        values[output] = np.ascontiguousarray(np.asarray(dataset[start:-1])[:, ::-1])
    values["task_uid"] = np.full((count - start, 1), TASK_IDS[task][2], dtype=np.int64)
    for name, value in values.items():
        if len(value) != count - start or not np.isfinite(value).all():
            raise ValueError(f"Invalid causal pre-action {name}")
    return values


def verified_hand_target_cache(audit_path, manifest_path, root, episodes):
    """Reuse a complete scheduled alignment audit, bound to the unchanged corpus."""
    audit_path, manifest_path, root = (
        Path(audit_path),
        Path(manifest_path),
        Path(root).resolve(),
    )
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("status") != "PASS_ALL_500_EXACT_CAUSAL_HAND_TARGETS"
        or audit.get("source_episode_count") != len(episodes)
        or audit.get("passed_episode_count") != len(episodes)
        or audit.get("failed_episode_count") != 0
        or audit.get("source_manifest_sha256")
        != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Hand target audit is incomplete or bound to a different manifest"
        )
    cache = Path(audit["target_cache"])
    digest = hashlib.sha256()
    with cache.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != audit["target_cache_sha256"]:
        raise ValueError("Hand target cache SHA differs from its audit")
    rows = {row["index"]: row for row in audit["passed"]}
    if set(rows) != set(range(len(episodes))) or len(rows) != len(audit["passed"]):
        raise ValueError("Hand target audit episode coverage differs")
    for index, record in enumerate(episodes):
        row = rows[index]
        source = (root / record["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Hand target source escapes the corpus root")
        stat = source.stat()
        if (
            row["path"] != record["path"]
            or row["task"] != record["task"]
            or row["source_bytes"] != stat.st_size
            or row["source_mtime_ns"] != stat.st_mtime_ns
            or not 0 <= row["source_start_step"] < row["source_control_count"]
            or row["target_shape"]
            != [row["source_control_count"] - row["source_start_step"], 69]
        ):
            raise ValueError("Hand target source identity changed after audit")
        if row["source_start_step"] > 1:
            descriptor = row.get("source_start_proof")
            if not descriptor:
                raise ValueError(
                    "Initial hold prefix has no source-backed resume proof"
                )
            raw = Path(descriptor["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
                raise ValueError("Resume proof SHA differs")
            proof = json.loads(raw)
            if (
                proof.get("status") != "PASS_UNIQUE_SOURCE_BACKED_RESUME_CALIBRATION"
                or proof.get("index") != index
                or proof.get("source_path") != str(source)
                or proof.get("source_start_step") != row["source_start_step"]
                or len(proof.get("solutions", [])) != 1
            ):
                raise ValueError("Resume proof identity differs")
            row["_validated_resume_proof"] = proof
    return audit, rows


def cached_hand_keypoint_actions(cache, row, handle):
    """Load the already verified target without repeating packet alignment."""
    if handle["actions"].shape != (row["source_control_count"], 28):
        raise ValueError("Audited source control shape changed")
    target = np.asarray(cache[row["cache_key"] + "_actions"])
    sequence = np.asarray(cache[row["cache_key"] + "_sequence"])
    if (
        target.dtype != np.float32
        or list(target.shape) != row["target_shape"]
        or not np.isfinite(target).all()
        or sequence.dtype != np.int64
        or sequence.shape != (len(target),)
        or np.any(sequence < 0)
    ):
        raise ValueError("Audited hand target tensor contract differs")
    return target, row["source_start_step"], sequence


def convert_collection_manifest(
    input_root,
    output,
    manifest_path,
    *,
    seed=42,
    action_representation="native",
    target_audit=None,
    image_chunk_frames=1,
):
    """Convert an immutable Quest Sharpa upload through the shared replay writer.

    Keypoint training uses raw pre-action observations with one RGB row flip.
    The historical native28 post-action conversion remains unchanged.
    Source upload checksums must be independently verified before this stage.
    """
    if action_representation not in ("native", "wrist_hand_keypoints"):
        raise ValueError("Unknown collection action representation")
    root, output = Path(input_root).resolve(), Path(output)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    suite = manifest["suite"]
    if manifest["embodiment"] != "custom_Panda_SharpaWaveRight_LIBERO":
        raise ValueError("Collection contract is not native Panda Sharpa Wave right")
    episodes = manifest["episodes"]
    if len(episodes) != manifest["episode_count"] or not episodes:
        raise ValueError("Collection manifest episode count differs")
    tasks = set(get_tasks(suite))
    counts, paths = {}, set()
    for record in episodes:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or path in paths or not path.is_file():
            raise ValueError(
                "Collection episode path is missing, duplicated or outside root"
            )
        paths.add(path)
        if (
            record["task"] not in tasks
            or record["suite"] != suite
            or record["action_dimension"] != 28
            or not record["automatic_success"]
        ):
            raise ValueError("Collection task/action/success contract differs")
        counts[record["task"]] = counts.get(record["task"], 0) + 1
    if counts != manifest["per_task"] or set(counts) != tasks:
        raise ValueError("Collection task coverage/counts differ")
    audit, cached_rows = None, None
    if target_audit is not None:
        if action_representation != "wrist_hand_keypoints":
            raise ValueError(
                "Hand target audit requires the hand keypoint representation"
            )
        audit, cached_rows = verified_hand_target_cache(
            target_audit, manifest_path, root, episodes
        )
    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w", zarr_format=2)
    data = group.create_group("data")
    total, ends, converted = 0, [], []
    for index in np.random.default_rng(seed).permutation(len(episodes)):
        record = episodes[int(index)]
        with h5py.File(root / record["path"], "r") as handle:
            demo = handle["data/demo_0"]
            metadata = json.loads(handle.attrs["metadata_json"])
            if (
                metadata["suite"] != suite
                or metadata["task"] != record["task"]
                or metadata["embodiment"] != "sharpa"
                or not handle.attrs["automatic_success"]
            ):
                raise ValueError("Actual episode metadata differs from upload manifest")
            if (
                handle["data"].attrs.get("rgb_row_transform")
                != "vertical_flip_from_raw_opengl_to_original_zarr_upright"
            ):
                raise ValueError(
                    "Recorded RGB orientation is unknown; no guessed/double flip"
                )
            if not str(demo.attrs.get("observation_alignment", "")).startswith(
                "after_action;"
            ):
                raise ValueError("Explicit post-action training view is required")
            source_start = 0
            if action_representation == "wrist_hand_keypoints":
                if audit is None:
                    target, source_start, sequence = aligned_hand_keypoint_actions(
                        handle, metadata
                    )
                else:
                    with np.load(audit["target_cache"], allow_pickle=False) as cache:
                        target, source_start, sequence = cached_hand_keypoint_actions(
                            cache, cached_rows[int(index)], handle
                        )
                values = causal_collection_episode_values(
                    handle,
                    metadata,
                    record["task"],
                    target,
                    source_start,
                    sequence,
                    resume_proof=None
                    if audit is None
                    else cached_rows[int(index)].get("_validated_resume_proof"),
                )
            else:
                # Preserve the historical native28 conversion API/view.
                values = native_episode_values(
                    demo,
                    record["task"],
                    action_dim=28,
                    hand_dim=22,
                    flip_rows=False,
                    native_quaternion=True,
                )
                for key in ("action_host_ns", "observation_host_ns"):
                    values[key] = np.asarray(demo[key], dtype=np.int64)
                    if len(values[key]) != len(values["action"]) or np.any(
                        np.diff(values[key]) <= 0
                    ):
                        raise ValueError("Native action/observation timeline differs")
                if np.any(values["action_host_ns"] > values["observation_host_ns"]):
                    raise ValueError(
                        "Post-action observation precedes its actual control"
                    )
            total = append_replay(
                data, values, total, image_chunk_frames=image_chunk_frames
            )
            ends.append(total)
            converted.append(
                {
                    **record,
                    "length": len(values["action"]),
                    "replay_episode_id": f"episode_{len(converted):06d}",
                    "source_start_step": source_start,
                    "source_start_proof": None
                    if audit is None
                    else cached_rows[int(index)].get("source_start_proof"),
                    "control_warnings": []
                    if audit is None
                    else cached_rows[int(index)].get("control_warnings", []),
                }
            )
    group.create_group("meta").create_array(
        "episode_ends", data=np.asarray(ends, dtype=np.int64)
    )
    result = {
        "complete": True,
        "suite": suite,
        "embodiment": "libero_panda_sharpa",
        "action_dim": 28 if action_representation == "native" else 69,
        "action_representation": action_representation,
        "keypoint_frame": None
        if action_representation == "native"
        else "wrist_relative_Quest_XYZ_m",
        "target_semantics": "robot_controls"
        if action_representation == "native"
        else "native_wrist_control6_plus_operator_handXYZ63",
        "hand_state_dim": 22,
        "seed": seed,
        "image_chunk_frames": image_chunk_frames,
        "target_audit_sha256": None
        if target_audit is None
        else hashlib.sha256(Path(target_audit).read_bytes()).hexdigest(),
        "target_cache_sha256": None if audit is None else audit["target_cache_sha256"],
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "observation_alignment": (
            "causal_pre_action_raw_lossless"
            if action_representation == "wrist_hand_keypoints"
            else "native_explicit_post_action_view"
        ),
        "camera_rows": (
            "raw_opengl_flipped_once_to_upright"
            if action_representation == "wrist_hand_keypoints"
            else "already_upright_no_second_flip"
        ),
        "quaternion": "native_XYZW",
        "raw_timestamps_preserved": True,
        "episodes": converted,
        "caveats": manifest.get("caveats", []),
        "split_assigned": False,
    }
    (output / "benchmark_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


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
            values = native_episode_values(demo, task)
            length = len(values["action"])
            prompt = json.loads(raw.attrs["problem_info"])["language_instruction"]
            total = append_replay(data, values, total)
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
    parser.add_argument(
        "--collection-manifest",
        help="Native Sharpa upload manifest; independently verify upload first",
    )
    parser.add_argument(
        "--action-representation",
        choices=("native", "wrist_hand_keypoints"),
        default="native",
    )
    parser.add_argument(
        "--target-audit", help="Complete source-bound hand target audit/cache to reuse"
    )
    parser.add_argument(
        "--image-chunk-frames",
        type=int,
        default=1,
        help="Lossless collection-image storage layout; historical default is one",
    )
    args = parser.parse_args()
    if args.collection_manifest:
        manifest = json.loads(Path(args.collection_manifest).read_text())
        if args.suite != manifest["suite"] or args.demos_per_task is not None:
            raise ValueError(
                "Collection suite must match; no implicit subset selection"
            )
        convert_collection_manifest(
            args.input_root,
            args.output,
            args.collection_manifest,
            seed=args.seed,
            action_representation=args.action_representation,
            target_audit=args.target_audit,
            image_chunk_frames=args.image_chunk_frames,
        )
        return
    if (
        args.action_representation != "native"
        or args.target_audit is not None
        or args.image_chunk_frames != 1
    ):
        raise ValueError(
            "Hand keypoint targets require an explicit collection manifest"
        )
    convert_suite(
        args.input_root,
        args.output,
        args.suite,
        seed=args.seed,
        demos_per_task=args.demos_per_task,
    )


if __name__ == "__main__":
    main()
