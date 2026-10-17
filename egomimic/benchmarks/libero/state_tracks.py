"""Aligned native-state RGB, measured-pose and persistent-track preparation.

This deliberately rerenders observations; original cached RGB/poses are never
paired with saved states by row or shifted heuristically. Outputs are raw
physical diagnostic data, not a normalized/DINO training cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import numpy as np

from egomimic.benchmarks.libero.replay import rebase_demo_xml

CONTRACT = "libero-same-recorded-state-v1"


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


def build_arrays(model, states, cameras, pose_site, height, width, grid):
    if states.ndim != 2 or len(states) < 2 or not np.all(np.diff(states[:, 0]) > 0):
        raise ValueError(
            "Require at least two strictly increasing physical state times"
        )
    if min(height, width, *grid) < 2:
        raise ValueError("Invalid image/grid dimensions")
    data = mujoco.MjData(model)
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
            rows["pose"].append(np.r_[data.site_xpos[site], quaternion])
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
        or a["pose"].shape != (count, 7)
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
    np.testing.assert_allclose(np.linalg.norm(a["pose"][:, 3:], axis=-1), 1, atol=1e-10)
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
    arrays = build_arrays(
        model, states, args.cameras, args.pose_site, args.height, args.width, args.grid
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
        height=args.height,
        width=args.width,
        grid=args.grid,
        pose_fields=["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"],
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


def validate_cache(output, replay=False):
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
        )
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], repeated[key], err_msg=key)
    return {
        "status": "PASS",
        "contract": CONTRACT,
        "frames": len(arrays["time"]),
        "points": len(arrays["point_body_id"]),
        "all_channels_native_replay_exact": replay,
        "training_ready": False,
    }


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
    prepare_parser.add_argument("--height", type=int, required=True)
    prepare_parser.add_argument("--width", type=int, required=True)
    prepare_parser.add_argument("--grid", nargs=2, type=int, required=True)
    check = commands.add_parser("validate")
    check.add_argument("--output", required=True)
    check.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args)
            if args.command == "prepare"
            else validate_cache(args.output, args.replay),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
