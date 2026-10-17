"""Aligned native-state RGB, measured-pose and persistent-track preparation.

This deliberately rerenders observations; original cached RGB/poses are never
paired with saved states by row or shifted heuristically. Outputs are raw
physical diagnostic data, not a normalized/DINO training cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import mujoco
import numpy as np

from egomimic.benchmarks.libero.replay import rebase_demo_xml

CONTRACT = "libero-same-recorded-state-v1"
RGB_POLICIES = ("exact", "uint8-roundoff-v1")


def compare_rgb(original, repeated, policy="exact"):
    """Separate bounded GPU raster rounding from exact physical-state replay.

    The opt-in policy was measured on repeated native GPU renders. It never
    changes cached pixels, geometry tolerances, or the historical exact default.
    """
    if policy not in RGB_POLICIES:
        raise ValueError("Unknown RGB repeatability policy")
    if original.shape != repeated.shape or original.dtype != np.uint8 or repeated.dtype != np.uint8:
        raise ValueError("RGB replay requires matching uint8 arrays")
    delta = np.abs(original.astype(np.int16) - repeated.astype(np.int16))
    result = {"policy": policy, "exact": bool(not np.any(delta)),
              "max_abs": int(delta.max(initial=0)),
              "changed_channels": int(np.count_nonzero(delta)),
              "total_channels": int(delta.size),
              "changed_fraction": float(np.count_nonzero(delta) / max(1, delta.size))}
    if policy == "exact":
        np.testing.assert_array_equal(original, repeated, err_msg="rgb")
    elif result["max_abs"] > 2 or result["changed_fraction"] > 0.00001:
        raise ValueError(f"RGB replay exceeds measured uint8 rounding policy: {result}")
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def restore_state(model, data, state):
    if model.na or model.nmocap:
        raise ValueError("Recorded state omits activation/mocap state; unsupported")
    if state.shape != (1 + model.nq + model.nv,) or not np.isfinite(state).all():
        raise ValueError("Invalid recorded state")
    mujoco.mj_resetData(model, data)
    data.time = state[0]
    data.qpos[:] = state[1 : 1 + model.nq]
    data.qvel[:] = state[1 + model.nq :]
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qpos, state[1 : 1 + model.nq])


def camera_geometry(model, data, camera, height, width):
    focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
    K = np.array(
        [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.0]]
    )
    transform = np.eye(4)
    # Camera convention: +X right, +Y down, +Z forward.
    transform[:3, :3] = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = data.cam_xpos[camera]
    return K, transform


def pose_joint_addresses(model, names):
    """Resolve explicitly named measured scalar joints; never infer policy actions."""
    if len(set(names)) != len(names):
        raise ValueError("Duplicate pose joint")
    result = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] not in (
            mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
        ):
            raise ValueError(f"Pose joint must be a known scalar joint: {name}")
        result.append(int(model.jnt_qposadr[joint]))
    return result


def build_arrays(model, states, cameras, pose_site, height, width, grid, pose_joints=()):
    if states.ndim != 2 or len(states) < 2 or not np.all(np.diff(states[:, 0]) > 0):
        raise ValueError(
            "Require at least two strictly increasing physical state times"
        )
    if min(height, width, *grid) < 2:
        raise ValueError("Invalid image/grid dimensions")
    data = mujoco.MjData(model)
    pose_addresses = pose_joint_addresses(model, pose_joints)
    camids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in cameras]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pose_site)
    if not cameras or len(set(cameras)) != len(cameras) or min([site, *camids]) < 0:
        raise ValueError("Unknown or duplicate camera / unknown pose site")
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    hit = np.zeros(1, np.int32)

    def ray(origin, direction):
        distance = mujoco.mj_ray(
            model, data, origin, direction, option.geomgroup, 1, -1, hit
        )
        return distance, int(hit[0])

    restore_state(model, data, states[0])
    K, transform = camera_geometry(model, data, camids[0], height, width)
    ys, xs = np.meshgrid(
        np.linspace(0, height - 1, grid[0]),
        np.linspace(0, width - 1, grid[1]),
        indexing="ij",
    )
    pixels = np.c_[xs.ravel(), ys.ravel()]
    local, bodies, geoms, valid_indices = [], [], [], []
    for index, pixel in enumerate(pixels):
        direction = transform[:3, :3] @ np.linalg.solve(K, [*pixel, 1.0])
        direction /= np.linalg.norm(direction)
        distance, geom = ray(transform[:3, 3], direction)
        if distance < 0 or geom < 0:
            continue
        body = int(model.geom_bodyid[geom])
        world = transform[:3, 3] + distance * direction
        local.append(data.xmat[body].reshape(3, 3).T @ (world - data.xpos[body]))
        bodies.append(body)
        geoms.append(geom)
        valid_indices.append(index)
    if not local:
        raise ValueError("No reference ray hits")
    local = np.asarray(local)
    bodies = np.asarray(bodies, dtype=np.int64)
    rows = {
        key: []
        for key in (
            "rgb",
            "pose",
            "xyz",
            "uv",
            "visibility",
            "in_view",
            "world_from_camera",
            "body_position",
            "body_rotation",
        )
    }
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for state in states:
            restore_state(model, data, state)
            rotations = data.xmat[bodies].reshape(-1, 3, 3)
            positions = data.xpos[bodies].copy()
            world = np.einsum("nij,nj->ni", rotations, local) + positions
            images, transforms, projections, in_views, visibles = [], [], [], [], []
            for camera in camids:
                K, transform = camera_geometry(model, data, camera, height, width)
                camera_xyz = (world - transform[:3, 3]) @ transform[:3, :3]
                z = camera_xyz[:, 2]
                uv = camera_xyz[:, :2] / np.where(z == 0, np.nan, z)[:, None]
                uv = uv * np.diag(K)[:2] + K[:2, 2]
                inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                inside &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
                visible = np.zeros(len(local), bool)
                for point in np.flatnonzero(inside):
                    delta = world[point] - transform[:3, 3]
                    distance = np.linalg.norm(delta)
                    hit_distance, geom = ray(transform[:3, 3], delta / distance)
                    visible[point] = (
                        geom >= 0
                        and model.geom_bodyid[geom] == bodies[point]
                        and abs(hit_distance - distance) < 1e-4
                    )
                renderer.update_scene(data, camera=camera, scene_option=option)
                images.append(renderer.render().copy())
                transforms.append(transform)
                projections.append(uv)
                in_views.append(inside)
                visibles.append(visible)
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[site])
            rows["pose"].append(np.r_[data.site_xpos[site], quaternion, data.qpos[pose_addresses]])
            for key, value in (
                ("rgb", images),
                ("xyz", world),
                ("uv", projections),
                ("world_from_camera", transforms),
                ("visibility", visibles),
                ("in_view", in_views),
                ("body_position", positions),
                ("body_rotation", rotations.copy()),
            ):
                rows[key].append(value)
    finally:
        renderer.close()
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays.update(
        time=states[:, 0],
        state=states,
        source_row=np.arange(len(states)),
        point_body_id=bodies,
        point_geom_id=np.asarray(geoms),
        body_local_xyz=local,
        reference_grid_xy=pixels,
        valid_grid_indices=np.asarray(valid_indices),
        coordinate_available=np.ones((len(states), len(local)), bool),
        camera_K=np.asarray(
            [camera_geometry(model, data, c, height, width)[0] for c in camids]
        ),
    )
    validate_arrays(arrays)
    return arrays


def validate_arrays(a):
    """Reject broken row identity, geometry, timestamps and visibility contracts."""
    count = len(a["time"])
    if not np.array_equal(a["source_row"], np.arange(count)):
        raise ValueError("Source row permutation/shift")
    if not np.array_equal(a["time"], a["state"][:, 0]) or not np.all(
        np.diff(a["time"]) > 0
    ):
        raise ValueError("State/time mismatch")
    for key in (
        "rgb",
        "pose",
        "xyz",
        "uv",
        "visibility",
        "in_view",
        "world_from_camera",
        "body_position",
        "body_rotation",
        "coordinate_available",
    ):
        if len(a[key]) != count:
            raise ValueError(f"Misaligned channel: {key}")
    if (
        a["rgb"].dtype != np.uint8
        or a["rgb"].ndim != 5
        or a["pose"].ndim != 2
        or a["pose"].shape[1] < 7
    ):
        raise ValueError("Invalid RGB/pose schema")
    if not np.isfinite(a["xyz"]).all() or not np.isfinite(a["pose"]).all():
        raise ValueError("Nonfinite geometry")
    if not np.all(a["coordinate_available"]):
        raise ValueError("Exact simulator coordinates must survive occlusion")
    if np.any(a["visibility"] & ~a["in_view"]):
        raise ValueError("Visible point outside camera")
    expected = (
        np.einsum("tpij,pj->tpi", a["body_rotation"], a["body_local_xyz"])
        + a["body_position"]
    )
    np.testing.assert_allclose(a["xyz"], expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(a["pose"][:, 3:7], axis=-1), 1, atol=1e-10)
    np.testing.assert_allclose(
        a["uv"][0, 0], a["reference_grid_xy"][a["valid_grid_indices"]], atol=1e-8
    )


def exclude_visualization_mocap(xml, names):
    """Explicitly remove nonphysical marker bodies whose poses were not saved.

    Refuse collision geometry, child bodies, joints or surviving references.
    Physical joint-state ordering must remain unchanged after removal.
    """
    if not names:
        return xml
    if len(names) != len(set(names)):
        raise ValueError("Duplicate marker body names")
    original = mujoco.MjModel.from_xml_string(xml)
    tree = ET.fromstring(xml)
    removed_names = set()
    for name in names:
        body_id = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, name)
        matches = [(p, e) for p in tree.iter() for e in p
                   if e.tag == "body" and e.get("name") == name]
        if body_id < 0 or original.body_mocapid[body_id] < 0 or len(matches) != 1:
            raise ValueError(f"Not a unique mocap visualization body: {name}")
        parent, element = matches[0]
        geoms = np.flatnonzero(original.geom_bodyid == body_id)
        if (any(e.tag in ("body", "joint", "freejoint") for e in list(element.iter())[1:])
                or np.any(original.geom_contype[geoms])
                or np.any(original.geom_conaffinity[geoms])):
            raise ValueError(f"Marker body has physical geometry or children: {name}")
        removed_names.update(e.get("name") for e in element.iter() if e.get("name"))
        parent.remove(element)
    if any(value in removed_names for e in tree.iter() for key, value in e.attrib.items()
           if key != "name"):
        raise ValueError("Excluded marker still referenced by model")
    result = ET.tostring(tree, encoding="unicode")
    filtered = mujoco.MjModel.from_xml_string(result)
    if (original.nq, original.nv, original.nu, original.na) != (
            filtered.nq, filtered.nv, filtered.nu, filtered.na):
        raise ValueError("Marker exclusion changed physical state dimensions")
    original_joints = [mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(original.njnt)]
    filtered_joints = [mujoco.mj_id2name(filtered, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(filtered.njnt)]
    if original_joints != filtered_joints:
        raise ValueError("Marker exclusion changed joint-state ordering")
    return result


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {output}")
    input_schema = getattr(args, "input_schema", "libero")
    with h5py.File(args.demo, "r") as handle:
        if input_schema == "quest":
            states = handle["state"][:]
            raw_xml = handle["destination_model_xml"][()]
            original_pose = handle["numeric_observations/robot0_eef_pos"][:]
        elif input_schema == "libero":
            demo = handle[f"data/{args.episode}"]
            states = demo["states"][:]
            raw_xml = demo.attrs["model_file"]
            original_pose = demo["obs/ee_pos"][:] if "obs/ee_pos" in demo else None
        else:
            raise ValueError("Unknown recorded input schema")
    remaps = getattr(args, "asset_remap", None) or ()
    excluded = getattr(args, "exclude_visualization_body", None) or ()
    xml = rebase_demo_xml(
        raw_xml, Path(args.robot_assets).resolve(), Path(args.scene_assets).resolve(), remaps
    )
    xml = exclude_visualization_mocap(xml, excluded)
    model = mujoco.MjModel.from_xml_string(xml)
    pose_joints = getattr(args, "pose_joints", None) or []
    arrays = build_arrays(
        model, states, args.cameras, args.pose_site, args.height, args.width, args.grid, pose_joints
    )
    assets = {
        str(Path(e.get("file")).resolve()): digest(e.get("file"))
        for e in ET.fromstring(xml).iter()
        if e.get("file")
    }
    output.mkdir(parents=True)
    np.savez_compressed(output / "aligned.npz", **arrays)
    (output / "model.xml").write_text(xml)
    mismatch = (
        None
        if original_pose is None
        else float(np.linalg.norm(arrays["pose"][:, :3] - original_pose, axis=-1).max())
    )
    manifest = dict(
        schema=CONTRACT,
        source_demo=str(Path(args.demo).resolve()),
        source_demo_sha256=digest(args.demo),
        episode=args.episode,
        input_schema=input_schema,
        asset_remaps=list(remaps),
        excluded_visualization_bodies=list(excluded),
        alignment="Every RGB, measured pose and track row is recomputed after restoring the exact same source state. No original cached observation is consumed as a target.",
        original_cache_max_eef_error_m=mismatch,
        original_cache_repaired=False,
        cameras=args.cameras,
        pose_site=args.pose_site,
        pose_joints=pose_joints,
        height=args.height,
        width=args.width,
        grid=args.grid,
        pose_fields=["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"] + [f"qpos:{name}" for name in pose_joints],
        reference_frame="recorded simulator world",
        time_unit="physical simulator seconds",
        rgb_origin="top_left",
        camera_axes="right_down_forward",
        visibility="ray first-hit surface at current state; separate from coordinate availability",
        track_method="reference visual-geometry ray-hit attached to persistent body-local XYZ",
        normalized=False,
        dino_features=False,
        training_ready=False,
        mujoco_version=mujoco.__version__,
        preparation_source_sha256=digest(__file__),
        asset_hashes=assets,
        artifacts={
            name: digest(output / name) for name in ("aligned.npz", "model.xml")
        },
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return validate_cache(output, replay=True)


def validate_cache(output, replay=False, rgb_policy="exact"):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["schema"] != CONTRACT:
        raise ValueError("Unsupported alignment contract")
    for name, expected in manifest["artifacts"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Cache hash mismatch: {name}")
    with np.load(output / "aligned.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    validate_arrays(arrays)
    if arrays["pose"].shape[1] != len(manifest["pose_fields"]):
        raise ValueError("Pose schema width disagrees with arrays")
    if replay:
        for name, expected in manifest["asset_hashes"].items():
            if digest(name) != expected:
                raise ValueError(f"Asset changed: {name}")
        model = mujoco.MjModel.from_xml_path(str((output / "model.xml").resolve()))
        repeated = build_arrays(
            model,
            arrays["state"],
            manifest["cameras"],
            manifest["pose_site"],
            manifest["height"],
            manifest["width"],
            manifest["grid"],
            manifest.get("pose_joints", ()),
        )
        # Geometry always remains bit-exact, independent of RGB rounding policy.
        for key in [key for key in arrays if key != "rgb"]:
            np.testing.assert_array_equal(arrays[key], repeated[key], err_msg=key)
        rgb_result = compare_rgb(arrays["rgb"], repeated["rgb"], rgb_policy)
    result = {
        "status": "PASS",
        "contract": CONTRACT,
        "frames": len(arrays["time"]),
        "points": len(arrays["point_body_id"]),
        "all_channels_native_replay_exact": bool(replay and rgb_result["exact"]),
        "native_geometry_replay_exact": bool(replay),
        "training_ready": False,
    }
    if replay:
        result["rgb_replay"] = rgb_result
    return result


def export_rgb(args):
    """Stream true high-resolution native RGB from verified saved states.

    Low-resolution persistent tracks are preserved; this does not regenerate
    the track grid or substitute upsampled images for high-resolution renders.
    """
    source = Path(args.output).resolve()
    target = Path(args.rgb_cache).resolve()
    receipt_path = target.with_suffix(target.suffix + ".json")
    if target.exists() or receipt_path.exists():
        raise FileExistsError("Preserve existing native RGB output and receipt")
    if min(args.height, args.width) < 2:
        raise ValueError("Invalid native RGB resolution")
    proof = validate_cache(source, replay=True, rgb_policy=args.rgb_policy)
    manifest = json.loads((source / "manifest.json").read_text())
    with np.load(source / "aligned.npz", allow_pickle=False) as archive:
        states = archive["state"]
        times = archive["time"]
        original_transforms = archive["world_from_camera"]
    model = mujoco.MjModel.from_xml_path(str(source / "model.xml"))
    # Increasing framebuffer capacity changes neither state nor visual settings.
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    data = mujoco.MjData(model)
    cameras = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
               for name in manifest["cameras"]]
    if min(cameras) < 0:
        raise ValueError("Missing recorded native camera")
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("Preserve partial native RGB output for diagnosis")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        with h5py.File(temporary, "x") as cache:
            cache.attrs.update(schema="libero-native-rgb-v1", native_render=True,
                               height=args.height, width=args.width)
            cache["time"] = times
            rgb = cache.create_dataset("rgb", (len(times), len(cameras), args.height,
                                               args.width, 3), dtype="u1",
                                       chunks=(1, 1, args.height, args.width, 3),
                                       compression="lzf")
            transforms = cache.create_dataset("camera_to_world", (len(times), len(cameras), 4, 4), dtype="f8")
            intrinsics = []
            for row, state in enumerate(states):
                restore_state(model, data, state)
                for view, camera in enumerate(cameras):
                    K, transform = camera_geometry(model, data, camera, args.height, args.width)
                    np.testing.assert_array_equal(transform, original_transforms[row, view],
                                                  err_msg="High-resolution camera/state alignment")
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    rgb[row, view] = renderer.render()
                    transforms[row, view] = transform
                    if row == 0:
                        intrinsics.append(K)
            cache["camera_K"] = np.asarray(intrinsics)
            cache.flush()
    finally:
        renderer.close()
    temporary.replace(target)
    receipt = {"schema": "libero-native-rgb-v1", "status": "PASS",
               "source_manifest_sha256": digest(source / "manifest.json"),
               "source_aligned_sha256": digest(source / "aligned.npz"),
               "rgb_cache_sha256": digest(target), "rgb_cache": str(target),
               "native_render": True, "height": args.height, "width": args.width,
               "cameras": manifest["cameras"], "frames": len(times),
               "native_geometry_replay_exact": True, "low_resolution_replay": proof,
               "source": str(source), "preparation_source_sha256": digest(__file__),
               "mujoco_version": mujoco.__version__, "training_ready": False}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _prepare_corpus_episode(options):
    """Reuse verified completed episodes; never accept a changed request as a hit."""
    deadline = time.monotonic() + options.get("wait_input_seconds", 0)
    while not Path(options["demo"]).is_file():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(options["demo"])
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    if options.get("expected_source_sha256") and digest(options["demo"]) != options["expected_source_sha256"]:
        raise ValueError("Source demonstration hash differs from corpus inventory")
    output = Path(options["output"])
    if output.exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["preparation_source_sha256"] != digest(__file__):
            raise ValueError("Existing cache preparation source mismatch")
        for path, expected in manifest["asset_hashes"].items():
            if digest(path) != expected:
                raise ValueError(f"Existing cache asset changed: {path}")
        for key in ("episode", "input_schema", "cameras", "pose_site", "height", "width", "grid", "pose_joints"):
            if manifest.get(key) != options.get(key):
                raise ValueError(f"Existing cache request mismatch: {key}")
        for recorded, requested in (("asset_remaps", "asset_remap"),
                                    ("excluded_visualization_bodies", "exclude_visualization_body")):
            if manifest.get(recorded, []) != options.get(requested, []):
                raise ValueError(f"Existing cache request mismatch: {recorded}")
        if manifest["source_demo"] != str(Path(options["demo"]).resolve()) or manifest["source_demo_sha256"] != digest(options["demo"]):
            raise ValueError("Existing cache source mismatch")
        # A manifest is written before native replay finishes. Its presence and
        # hashes therefore cannot prove that an earlier preparation passed.
        # Until replay success has a hash-bound receipt, every reuse must replay.
        result = validate_cache(output, replay=True)
        result["reused"] = True
        return result
    return prepare(argparse.Namespace(**options))


def prepare_corpus(args):
    """Run an explicit corpus inventory through the maintained native preparer."""
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text())
    output = Path(args.output).resolve()
    if args.workers < 1 or not spec.get("episodes"):
        raise ValueError("Need positive workers and an explicit episode inventory")
    options_list, identities, sources = [], set(), set()
    for episode in spec["episodes"]:
        embodiment, identity = episode["embodiment"], episode["episode_id"]
        if any(not name or Path(name).name != name or name in (".", "..") for name in (embodiment, identity)):
            raise ValueError("Unsafe corpus identity")
        key = (embodiment, identity)
        options = {**spec.get("common", {}), **spec["embodiments"][embodiment],
                   "demo": episode["demo"], "episode": episode["episode"],
                   "output": str(output / embodiment / identity),
                   "expected_source_sha256": episode.get("sha256"),
                   "wait_input_seconds": getattr(args, "wait_input_seconds", 0)}
        physical = (str(Path(options["demo"]).resolve()), options["episode"])
        if key in identities or physical in sources:
            raise ValueError("Duplicate corpus identity or physical episode")
        identities.add(key); sources.add(physical)
        options_list.append((key, options))
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "CORPUS_STATUS.json"
    if status_path.exists() and json.loads(status_path.read_text())["spec_sha256"] != digest(spec_path):
        raise ValueError("Corpus specification changed; preserve the original output")
    status = {"spec_sha256": digest(spec_path), "spec": str(spec_path),
              "status": "RUNNING", "expected": len(options_list), "episodes": [],
              "training_ready": False}

    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    save()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_prepare_corpus_episode, options): key for key, options in options_list}
        for future in as_completed(futures):
            key = futures[future]
            row = {"embodiment": key[0], "episode_id": key[1]}
            try:
                row.update(future.result())
            except Exception as error:
                row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            status["episodes"].append(row)
            save()
            print(json.dumps({"completed": len(status["episodes"]), **row}), flush=True)
    status["status"] = "PASS" if all(r["status"] == "PASS" for r in status["episodes"]) else "FAILED"
    save()
    if status["status"] != "PASS":
        raise RuntimeError("Native corpus preparation has failed episodes; see CORPUS_STATUS.json")
    return {key: value for key, value in status.items() if key != "episodes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("demo", "robot-assets", "scene-assets", "output", "pose-site"):
        prepare_parser.add_argument(f"--{name}", required=True)
    prepare_parser.add_argument("--episode", default="demo_0")
    prepare_parser.add_argument("--input-schema", choices=["libero", "quest"], default="libero")
    prepare_parser.add_argument("--asset-remap", nargs=2, action="append", metavar=("OLD_PREFIX", "NEW_ROOT"))
    prepare_parser.add_argument("--exclude-visualization-body", action="append")
    prepare_parser.add_argument("--cameras", nargs="+", required=True)
    prepare_parser.add_argument("--pose-joints", nargs="+", help="Explicit measured scalar joint names appended after EEF pose")
    prepare_parser.add_argument("--height", type=int, required=True)
    prepare_parser.add_argument("--width", type=int, required=True)
    prepare_parser.add_argument("--grid", nargs=2, type=int, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--output", required=True)
    check.add_argument("--replay", action="store_true")
    check.add_argument("--rgb-policy", choices=RGB_POLICIES, default="exact")
    rgb = commands.add_parser("export-rgb")
    rgb.add_argument("--output", required=True, help="Existing native aligned episode")
    rgb.add_argument("--rgb-cache", required=True)
    rgb.add_argument("--height", type=int, required=True)
    rgb.add_argument("--width", type=int, required=True)
    rgb.add_argument("--rgb-policy", choices=RGB_POLICIES, default="exact")
    corpus = commands.add_parser("prepare-corpus")
    corpus.add_argument("--spec", required=True)
    corpus.add_argument("--output", required=True)
    corpus.add_argument("--workers", type=int, default=1)
    corpus.add_argument("--wait-input-seconds", type=int, default=0,
                        help="Allow atomic dataset transfers to finish; never read temporary transfer files")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args)
            if args.command == "prepare"
            else prepare_corpus(args) if args.command == "prepare-corpus"
            else export_rgb(args) if args.command == "export-rgb"
            else validate_cache(args.output, args.replay, args.rgb_policy),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
