"""Convert ABC sim_224 teleop episodes (amazon-far/abc ``abc_sim``) to yam_bimanual zarrs.

Each task tar (``dataset/sim_224/<task>_v3.tar``) extracts to ``{train,val}/<episode>/``
holding ``states_actions.bin`` ((T, 28) float64: 14 recorded + 14 commanded dofs,
``[left j1..j6, left grip, right j1..j6, right grip]``, grippers in [0, 1]),
``combined_camera-images-rgb.mp4`` (top/left/right stacked vertically, 224x168, 30 fps)
and ``episode_metadata.json``.

Every episode becomes one ``yam_bimanual`` zarr with the real ABC keys
(``{left,right}.{obs,cmd}_{joints,gripper,ee_pose}``, ``images.{front_1,left_wrist,
right_wrist}``, ``annotations``), so the graph's Yam cartesian / ARC pipelines read sim
and real ABC alike. ``ee_pose`` (XYZ + WXYZ) is forward kinematics through abc_sim's own
bimanual MJCF at ``<side>_grasp_site`` in the ``left_arm`` base frame: on real ABC
episodes that reproduces the stored ``obs_ee_pose`` exactly for the left arm. The right
arm differs by a constant 10 mm in y, because the sim mounts the arms 0.62 m apart where
the real station data uses 0.61 m; this keeps the sim's own geometry, so IK through the
same MJCF inverts it exactly.

Prompts follow ABC's training rule (``abc_minimal.dataloader``): the episode's
``prompt_timeline`` if it has one, else its ``instruction`` when that differs from the
task name, else the task name with ``_`` -> `` ``. Episodes ABC's loader drops as
unlabelled are skipped.

    python -m egomimic.scripts.abc_sim.convert_sim_to_zarr --src <extracted tar dir> \
        --model <abc>/abc_sim/models/yam_bimanual_empty.xml --out <zarr root>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

log = logging.getLogger("abc_sim.convert")

STATE_DIM = 14
DATA_FPS = 30
CAMERAS = ("top", "left", "right")
CAMERA_TO_ZARR = {
    "top": "images.front_1",
    "left": "images.left_wrist",
    "right": "images.right_wrist",
}
CAMERA_FOVY_DEG = 58.0  # every abc_sim policy camera
SITE = "grasp_site"
BASE_BODY = "left_arm"
DONE_MARKER = ".abc_sim_done"


def task_name_to_prompt(task_name: str) -> str:
    """abc_minimal.dit.task_name_to_prompt."""
    return " ".join(task_name.replace("-", " ").replace("_", " ").split())


def prompt_spans(meta: dict, num_steps: int) -> list[tuple[str, int, int]]:
    """``(text, start, end)`` spans covering ``[0, num_steps)``, end exclusive."""
    task_name = str(meta.get("task_name") or "")
    timeline = meta.get("prompt_timeline")
    if timeline:
        entries = sorted((int(e["frame"]), str(e["prompt"])) for e in timeline)
    else:
        instruction = meta.get("instruction")
        if (
            isinstance(instruction, str)
            and instruction.strip()
            and task_name_to_prompt(instruction) != task_name_to_prompt(task_name)
        ):
            text = instruction.strip()
        else:
            text = task_name_to_prompt(task_name)
        entries = [(0, text)]
    spans = []
    for i, (start, text) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else num_steps
        start = 0 if i == 0 else max(0, min(start, num_steps))
        if end > start:
            spans.append((text, start, end))
    return spans


def pinhole_intrinsics(height: int, width: int, fovy_deg: float = CAMERA_FOVY_DEG):
    """3x4 K of a MuJoCo camera (square pixels, principal point centred)."""
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array(
        [[f, 0.0, width / 2.0, 0.0], [0.0, f, height / 2.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )


class BimanualFK:
    """abc_sim FK: (T, 14) dofs -> per-arm (T, 7) XYZ+WXYZ in the left_arm base frame."""

    def __init__(self, xml_path: str | Path):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.qadr = {
            side: [
                self.model.jnt_qposadr[self.model.joint(f"{side}_joint{i}").id]
                for i in range(1, 7)
            ]
            for side in ("left", "right")
        }
        self.site = {s: self.model.site(f"{s}_{SITE}").id for s in ("left", "right")}
        self.base = self.model.body(BASE_BODY).id

    def __call__(self, dofs: np.ndarray) -> dict[str, np.ndarray]:
        out = {s: np.empty((len(dofs), 7)) for s in ("left", "right")}
        quat = np.empty(4)
        for t, row in enumerate(dofs):
            self.data.qpos[self.qadr["left"]] = row[0:6]
            self.data.qpos[self.qadr["right"]] = row[7:13]
            self.mj.mj_kinematics(self.model, self.data)
            rb = self.data.xmat[self.base].reshape(3, 3)
            pb = self.data.xpos[self.base]
            for side, sid in self.site.items():
                rot = rb.T @ self.data.site_xmat[sid].reshape(3, 3)
                self.mj.mju_mat2Quat(quat, rot.reshape(-1))
                out[side][t, :3] = rb.T @ (self.data.site_xpos[sid] - pb)
                out[side][t, 3:] = quat if quat[0] >= 0 else -quat
        return out


def arm_arrays(dofs: np.ndarray, kind: str, fk: BimanualFK) -> dict[str, np.ndarray]:
    poses = fk(dofs)
    out = {}
    for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
        arm = dofs[:, sl].astype(np.float32)
        out[f"{side}.{kind}_joints"] = arm[:, :6]
        out[f"{side}.{kind}_gripper"] = arm[:, 6:7]
        out[f"{side}.{kind}_ee_pose"] = poses[side].astype(np.float32)
    return out


def decode_cameras(ep_dir: Path) -> dict[str, np.ndarray]:
    """``{camera: (T, H, W, 3) uint8}`` from the vertically stacked mp4."""
    import av

    with av.open(str(ep_dir / "combined_camera-images-rgb.mp4")) as container:
        stack = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    h = stack.shape[1] // len(CAMERAS)
    if h * len(CAMERAS) != stack.shape[1]:
        raise ValueError(f"video height {stack.shape[1]} is not {len(CAMERAS)} stacked cameras")
    return {cam: stack[:, i * h : (i + 1) * h] for i, cam in enumerate(CAMERAS)}


_FK: BimanualFK | None = None


def convert_episode(ep_dir: Path, split: str, out_dir: Path, model_xml: str) -> dict:
    """Write one zarr; returns a report row. Idempotent via a done marker."""
    global _FK
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    name = f"abcsim_{split}_{ep_dir.name}"
    dst = out_dir / f"{name}.zarr"
    if (dst / DONE_MARKER).exists():
        return {"episode": name, "status": "skipped"}
    meta = json.loads((ep_dir / "episode_metadata.json").read_text())
    if not meta.get("prompt_timeline") and (meta.get("prompt_source") or {}).get("status") == "excluded":
        return {"episode": name, "status": "skipped_unlabelled"}  # abc_minimal.dataloader drops these
    raw = np.fromfile(ep_dir / "states_actions.bin", dtype=np.float64)
    if raw.size == 0 or raw.size % (2 * STATE_DIM):
        raise ValueError(f"states_actions.bin holds {raw.size} float64s")
    table = raw.reshape(-1, 2 * STATE_DIM)
    num_steps = len(table)
    if meta.get("num_steps") is not None and int(meta["num_steps"]) != num_steps:
        raise ValueError(f"states_actions.bin has {num_steps} rows, metadata says {meta['num_steps']}")
    if tuple(meta.get("cameras") or CAMERAS) != CAMERAS:
        raise ValueError(f"camera order {meta.get('cameras')} != {CAMERAS}")
    images = decode_cameras(ep_dir)
    if min(len(v) for v in images.values()) != num_steps:
        raise ValueError(f"{len(images['top'])} video frames for {num_steps} state rows")
    if _FK is None:
        _FK = BimanualFK(model_xml)
    h, w = images["top"].shape[1:3]

    tmp = out_dir / f".{name}.zarr.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    spans = prompt_spans(meta, num_steps)
    ZarrWriter.create_and_write(
        tmp,
        numeric_data={
            **arm_arrays(table[:, :STATE_DIM], "obs", _FK),
            **arm_arrays(table[:, STATE_DIM:], "cmd", _FK),
        },
        image_data={CAMERA_TO_ZARR[c]: images[c] for c in CAMERAS},
        embodiment="yam_bimanual",
        fps=DATA_FPS,
        task_name=str(meta.get("task_name") or ""),
        task_description=spans[0][0] if spans else "",
        annotations=spans,
        intrinsics={"front_1": pinhole_intrinsics(h, w)},
        metadata_override={"lab": "abc_sim", "split": split, "abc_task": str(meta.get("task") or "")},
    )
    (tmp / DONE_MARKER).write_text("ok\n")
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    return {"episode": name, "status": "written", "frames": num_steps}


def find_episodes(src: Path) -> list[tuple[Path, str]]:
    found = []
    for split in ("train", "val"):
        root = next((src / d for d in (split, f"{split}_sim") if (src / d).is_dir()), None)
        if root is None:
            log.warning("no %s/ under %s", split, src)
            continue
        found += [(p.parent, split) for p in sorted(root.glob("*/states_actions.bin"))]
    return found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True, help="one extracted sim_224 task tar")
    p.add_argument("--model", required=True, help="abc_sim/models/yam_bimanual_empty.xml")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    episodes = find_episodes(args.src)
    if not episodes:
        log.error("no episodes under %s", args.src)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    log.info("converting %d episodes from %s -> %s", len(episodes), args.src, args.out)

    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_episode, ep, split, args.out, args.model): ep
            for ep, split in episodes
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ep = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:  # one bad episode must not sink the task
                failures.append({"episode": str(ep), "error": f"{type(e).__name__}: {e}"})
                log.warning("FAILED %s: %s", ep, e)
            if i % 200 == 0 or i == len(futures):
                log.info("%d/%d done (%d failed)", i, len(futures), len(failures))

    report = {
        "src": str(args.src),
        "written": sum(r["status"] == "written" for r in rows),
        "skipped": sum(r["status"] == "skipped" for r in rows),
        "skipped_unlabelled": sum(r["status"] == "skipped_unlabelled" for r in rows),
        "failed": failures,
        "frames_written": int(sum(r.get("frames", 0) for r in rows)),
    }
    (args.out / f"conversion_report_{args.src.name}.json").write_text(json.dumps(report, indent=2))
    log.info("report: %s", {k: v for k, v in report.items() if k != "failed"})
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
