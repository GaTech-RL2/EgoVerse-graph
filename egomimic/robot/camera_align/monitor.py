#!/usr/bin/env python3
"""Live front-camera alignment monitor for the YAM rollout station.

Receives the rollout dashboard's camera stream as a read-only websocket client
(``ws://127.0.0.1:8081/ws`` by default), compares the front camera against
(a) a saved reference frame via SIFT, (b) saved ArUco tag poses, and (c) the
desk's own edges, and serves a dashboard on ``127.0.0.1:8090``.  View it through
an SSH tunnel (``ssh -L 8090:127.0.0.1:8090 rl2-yam``) and open
http://localhost:8090.

Camera motion vs desk motion: SIFT features on the table top and in the
background (wall, floor, cabinets) are fitted separately.  Everything shifting
together is the camera; the table shifting while the background stays is the
desk moving relative to the camera; the table's left and right edges, located
in two rows above the robot bases, measure the same thing independently of
what is placed on the table.

The rollout dashboard server keeps a single websocket client, so connecting
supersedes its browser tab; when the operator reopens that tab this monitor is
superseded in turn (close code 4001) and stays disconnected until "Reconnect
stream" is pressed, so the two never fight.  The monitor never sends a command
to the rollout dashboard and never opens a camera or a robot.

All per-station state (reference frames, tag reference, snapshots) lives in
``--data-dir``; nothing is written inside the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import yaml
from aiohttp import web

SUPERSEDED_CLOSE_CODE = 4001
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_INTRINSICS = (
    Path(__file__).resolve().parents[2]
    / "hydra_configs"
    / "robot"
    / "yam_rl2_agentview_extrinsics.yaml"
)


@dataclass
class MonitorConfig:
    rollout_ws: str = "ws://127.0.0.1:8081/ws"
    camera: str = "front_img_1"
    host: str = "127.0.0.1"
    port: int = 8090
    data_dir: Path = field(default_factory=lambda: Path.home() / "camera_align")
    intrinsics_yaml: Path = DEFAULT_INTRINSICS
    marker_length_m: float = 0.200
    aruco_dict: str = "DICT_4X4_50"
    analysis_hz: float = 4.0
    history_s: float = 180.0
    good_deg: float = 0.15  # camera angles under this count as aligned
    good_px: float = 1.5  # scene shifts under this count as aligned
    desk_px: float = (
        2.0  # desk edge / table-vs-background shifts under this are "steady"
    )
    edge_rows: tuple[int, int] = (
        185,
        220,
    )  # rows between the far edge and the robot bases
    left_window: tuple[int, int] = (
        30,
        210,
    )  # where the side edges may be in a reference
    right_window: tuple[int, int] = (430, 610)
    track_halfwidth: int = 45  # live search window around the reference edge position

    @property
    def reference_pointer(self) -> Path:
        return self.data_dir / "reference.json"

    @property
    def tag_reference(self) -> Path:
        return self.data_dir / "tag_reference.json"

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots"


# ---------------------------------------------------------------- calibration
@dataclass(frozen=True)
class Calibration:
    """The D405's own intrinsics as recorded in the agent-view extrinsics yaml."""

    K: np.ndarray
    dist: np.ndarray
    serial: str
    camera_T_base: np.ndarray | None = None

    @classmethod
    def from_yaml(cls, path, channel="can_follower_r") -> "Calibration":
        data = yaml.safe_load(open(path))
        K = np.asarray(data["K"], float)
        dist = np.asarray(data["dist"], float)
        if K.shape != (3, 3) or dist.ndim != 1:
            raise ValueError(f"{path}: K must be 3x3 and dist a vector")
        T = data.get("channels", {}).get(channel, {}).get("camera_T_base")
        return cls(
            K,
            dist,
            str(data.get("serial")),
            None if T is None else np.asarray(T, float),
        )

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def table_range_m(self, u, v, fallback=1.2) -> float:
        """Range to the table plane (base z = 0) along the ray through pixel (u, v)."""
        if self.camera_T_base is None:
            return fallback
        T = self.camera_T_base
        n = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        r = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        denominator = float(n @ r)
        if abs(denominator) < 1e-9:
            return fallback
        s = float(n @ T[:3, 3]) / denominator
        return float(abs(s) * np.linalg.norm(r))


# ------------------------------------------------------------------ geometry
def far_edge(gray, y0=110, y1=240):
    """Longest near-horizontal Hough line in the band where the table meets the wall."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    mask = np.zeros_like(edges)
    mask[y0:y1, :] = 255
    edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=30, minLineLength=90, maxLineGap=15
    )
    best = None
    if lines is not None:
        for x1, y1_, x2, y2 in lines.reshape(-1, 4):
            angle = math.degrees(math.atan2(y2 - y1_, x2 - x1))
            length = math.hypot(x2 - x1, y2 - y1_)
            if abs(angle) < 12 and (best is None or length > best[0]):
                best = (length, int(x1), int(y1_), int(x2), int(y2), angle)
    if best is None:
        return None
    length, x1, y1_, x2, y2, angle = best
    y_mid = y1_ + (y2 - y1_) * (320 - x1) / (x2 - x1) if x2 != x1 else y1_
    return {
        "x1y1x2y2": [x1, y1_, x2, y2],
        "angle_deg": round(angle, 2),
        "y_at_x320": round(float(y_mid), 1),
        "length_px": round(length, 1),
    }


def side_edge_x(gray, y, window, half=5):
    """Sub-pixel x of the strongest horizontal gradient in rows y +- half within window.

    Returns (x, strength); x is the median over the rows of the parabola-refined
    argmax, so a small object touching the band cannot drag it far.
    """
    lo = int(max(1, window[0]))
    hi = int(min(gray.shape[1] - 1, window[1]))
    band = cv2.GaussianBlur(
        gray[y - half : y + half + 1, :].astype(np.float32), (9, 1), 0
    )
    gx = np.abs(band[:, 2:] - band[:, :-2])[:, lo - 1 : hi - 1]  # index i -> x = lo + i
    xs = []
    for r in range(gx.shape[0]):
        i = int(np.argmax(gx[r]))
        x = float(lo + i)
        if 0 < i < gx.shape[1] - 1:
            a, b, c = gx[r, i - 1], gx[r, i], gx[r, i + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-6:
                x += 0.5 * float(a - c) / float(den)
        xs.append(x)
    return float(np.median(xs)), float(np.median(gx.max(axis=1)))


def desk_geometry(gray, cfg: MonitorConfig, ref=None):
    """Far edge plus the left/right table edges at cfg.edge_rows.

    With ``ref`` (a previous result) the side edges are searched only near the
    reference positions, so tags and objects elsewhere cannot capture them.
    """
    out = {
        "far": far_edge(gray),
        "rows": list(cfg.edge_rows),
        "left": [],
        "right": [],
        "strength": [],
    }
    for i, y in enumerate(cfg.edge_rows):
        if ref and ref.get("left") and ref.get("right"):
            hw = cfg.track_halfwidth
            lw = (ref["left"][i] - hw, ref["left"][i] + hw)
            rw = (ref["right"][i] - hw, ref["right"][i] + hw)
        else:
            lw, rw = cfg.left_window, cfg.right_window
        xl, sl = side_edge_x(gray, y, lw)
        xr, sr = side_edge_x(gray, y, rw)
        out["left"].append(round(xl, 2))
        out["right"].append(round(xr, 2))
        out["strength"].append([round(sl, 1), round(sr, 1)])
    out["width_px"] = round(
        float(np.mean(np.array(out["right"]) - np.array(out["left"]))), 2
    )
    return out


def table_polygon(desk, height=480):
    """Table-top polygon from the two side edges (extrapolated) and the far edge."""
    y0, y1 = desk["rows"]
    far_y = desk["far"]["y_at_x320"] if desk.get("far") else 150

    def x_at(xs, y):
        return xs[0] + (xs[1] - xs[0]) * (y - y0) / (y1 - y0)

    points = [
        (x_at(desk["left"], far_y), far_y),
        (x_at(desk["right"], far_y), far_y),
        (x_at(desk["right"], height - 1), height - 1),
        (x_at(desk["left"], height - 1), height - 1),
    ]
    return np.array(
        [[int(round(max(-50, min(690, x)))), int(round(y))] for x, y in points],
        np.int32,
    )


def screen_motion(src, dst, cal: Calibration):
    """Similarity transform taking reference points to current points, in screen terms.

    ``roll_deg`` > 0: the scene turned clockwise on screen; ``dx_px`` > 0: the scene
    moved right; ``dy_px`` > 0: the scene moved down (both measured at the principal
    point).  ``pan_deg``/``tilt_deg`` are those shifts as angles through the focal
    length.  Returns None when there are too few points.
    """
    src = np.asarray(src, np.float32).reshape(-1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 2)
    if len(src) < 2:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
    )
    if M is None:
        return None
    theta = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    scale = math.hypot(M[0, 0], M[1, 0])
    p = M @ np.array([cal.cx, cal.cy, 1.0])
    dx, dy = float(p[0] - cal.cx), float(p[1] - cal.cy)
    return {
        "roll_deg": round(theta, 3),
        "scale": round(scale, 4),
        "dx_px": round(dx, 2),
        "dy_px": round(dy, 2),
        "pan_deg": round(math.degrees(math.atan(dx / cal.fx)), 3),
        "tilt_deg": round(math.degrees(math.atan(dy / cal.fy)), 3),
        "inliers": int(inliers.sum()) if inliers is not None else int(len(src)),
        "n": int(len(src)),
    }


def advice_for(m, good_deg=0.15, good_px=1.5):
    """What the image has to do to match the reference, one line per axis."""
    if m is None:
        return ["no measurement"]
    out = []
    if abs(m["roll_deg"]) >= good_deg:
        turn = "counter-clockwise" if m["roll_deg"] > 0 else "clockwise"
        out.append(f"ROLL: turn the image {turn} by {abs(m['roll_deg']):.2f} deg")
    if abs(m["tilt_deg"]) >= good_deg or abs(m["dy_px"]) >= good_px:
        out.append(
            f"TILT: camera {'down' if m['dy_px'] > 0 else 'up'} {abs(m['tilt_deg']):.2f} deg "
            f"(content must move {'up' if m['dy_px'] > 0 else 'down'} {abs(m['dy_px']):.1f} px)"
        )
    if abs(m["pan_deg"]) >= good_deg or abs(m["dx_px"]) >= good_px:
        out.append(
            f"PAN: camera {'right' if m['dx_px'] > 0 else 'left'} {abs(m['pan_deg']):.2f} deg "
            f"(content must move {'left' if m['dx_px'] > 0 else 'right'} {abs(m['dx_px']):.1f} px)"
        )
    return out or ["ALIGNED"]


# --------------------------------------------------------------------- ArUco
class TagDetector:
    """ArUco detection with a pose per tag from the calibrated intrinsics."""

    def __init__(self, cal: Calibration, marker_length_m=0.2, dictionary="DICT_4X4_50"):
        self.cal, self.L = cal, float(marker_length_m)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params
        )
        half = self.L / 2
        self.object_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            np.float32,
        )

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        corners, ids, _ = self.detector.detectMarkers(gray)
        tags = []
        if ids is not None:
            for i, c in zip(ids.ravel(), corners):
                c = c[0].astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    self.object_points,
                    c,
                    self.cal.K,
                    self.cal.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok:
                    continue
                undistorted = cv2.undistortPoints(
                    c.reshape(-1, 1, 2), self.cal.K, self.cal.dist, P=self.cal.K
                ).reshape(-1, 2)
                tags.append(
                    {
                        "id": int(i),
                        "centre": [float(c[:, 0].mean()), float(c[:, 1].mean())],
                        "side_px": float(
                            np.mean(
                                [
                                    np.linalg.norm(c[k] - c[(k + 1) % 4])
                                    for k in range(4)
                                ]
                            )
                        ),
                        "corners": c.tolist(),
                        "corners_undist": undistorted.tolist(),
                        "rvec": rvec.ravel().tolist(),
                        "tvec": tvec.ravel().tolist(),
                        "range_m": float(np.linalg.norm(tvec)),
                    }
                )
        tags.sort(key=lambda t: t["centre"][0])
        return tags


def match_tags(ref_tags, cur_tags, max_px=120):
    """Pair reference and current tags by id and nearest centre (duplicate ids allowed)."""
    pairs, used = [], set()
    for r in ref_tags:
        best = None
        for j, c in enumerate(cur_tags):
            if j in used or c["id"] != r["id"]:
                continue
            d = math.hypot(
                c["centre"][0] - r["centre"][0], c["centre"][1] - r["centre"][1]
            )
            if best is None or d < best[0]:
                best = (d, j)
        if best and best[0] < max_px:
            used.add(best[1])
            pairs.append((r, cur_tags[best[1]]))
    return pairs


def tag_compare(ref, cur_tags, cal: Calibration, moved_m=0.025):
    """Camera motion from the saved tag corners, plus per-tag pose deltas.

    With two or more matched tags the distance between them is compared with
    the reference; a change over ``moved_m`` means a tag moved, not the camera.
    """
    if not ref or not ref.get("tags"):
        return {"available": False}
    pairs = match_tags(ref["tags"], cur_tags)
    if not pairs:
        return {"available": True, "matched": 0, "of": len(ref["tags"])}
    src = np.concatenate([np.array(r["corners_undist"]) for r, _ in pairs])
    dst = np.concatenate([np.array(c["corners_undist"]) for _, c in pairs])
    per = []
    for r, c in pairs:
        Rr, _ = cv2.Rodrigues(np.array(r["rvec"]))
        Rc, _ = cv2.Rodrigues(np.array(c["rvec"]))
        cosine = (np.trace(Rr.T @ Rc) - 1) / 2
        per.append(
            {
                "id": c["id"],
                "shift_px": [
                    round(c["centre"][0] - r["centre"][0], 2),
                    round(c["centre"][1] - r["centre"][1], 2),
                ],
                "range_delta_mm": round((c["range_m"] - r["range_m"]) * 1000, 1),
                "pose_rot_delta_deg": round(
                    math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), 2
                ),
            }
        )
    out = {
        "available": True,
        "matched": len(pairs),
        "of": len(ref["tags"]),
        "motion": screen_motion(src, dst, cal),
        "per_tag": per,
    }
    if len(pairs) >= 2:
        baseline_ref = np.linalg.norm(
            np.array(pairs[0][0]["tvec"]) - np.array(pairs[1][0]["tvec"])
        )
        baseline_cur = np.linalg.norm(
            np.array(pairs[0][1]["tvec"]) - np.array(pairs[1][1]["tvec"])
        )
        out["baseline_mm"] = round(baseline_cur * 1000, 1)
        out["baseline_delta_mm"] = round((baseline_cur - baseline_ref) * 1000, 1)
        out["tags_moved_warning"] = bool(abs(baseline_cur - baseline_ref) > moved_m)
    return out


# ---------------------------------------------------------- reference frame
class Reference:
    """A reference frame with its SIFT features, edges, desk edges and table mask."""

    SIFT = None  # created lazily, shared

    def __init__(
        self, path, label, cal: Calibration, cfg: MonitorConfig, saved_at=None
    ):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        if Reference.SIFT is None:
            Reference.SIFT = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
        self.path, self.label, self.saved_at = str(path), label, saved_at
        self.raw = image
        self.u = cv2.undistort(image, cal.K, cal.dist)
        self.g = cv2.cvtColor(self.u, cv2.COLOR_BGR2GRAY)
        self.kp, self.des = Reference.SIFT.detectAndCompute(self.g, None)
        self.edges = cv2.Canny(self.g, 60, 150)
        self.desk = desk_geometry(self.g, cfg)
        self.poly = table_polygon(self.desk, self.g.shape[0])
        self.table_mask = np.zeros(self.g.shape, np.uint8)
        cv2.fillPoly(self.table_mask, [self.poly], 255)


def ref_compare(ref: Reference, gray_u, cal: Calibration):
    """SIFT similarity of the current frame vs the reference: whole frame, table, background."""
    kp, des = Reference.SIFT.detectAndCompute(gray_u, None)
    if des is None or ref.des is None or len(kp) < 8:
        return None, None, None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref.des, des, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return None, None, None
    src = np.float32([ref.kp[m.queryIdx].pt for m in good])
    dst = np.float32([kp[m.trainIdx].pt for m in good])
    whole = screen_motion(src, dst, cal)
    if whole:
        whole["matches"] = len(good)
    h, w = ref.table_mask.shape
    on_table = np.array(
        [
            ref.table_mask[min(h - 1, max(0, int(p[1]))), min(w - 1, max(0, int(p[0])))]
            > 0
            for p in src
        ]
    )
    table = (
        screen_motion(src[on_table], dst[on_table], cal)
        if on_table.sum() >= 12
        else None
    )
    background = (
        screen_motion(src[~on_table], dst[~on_table], cal)
        if (~on_table).sum() >= 12
        else None
    )
    return whole, table, background


def desk_compare(ref_desk, desk_live, table_m, bg_m, mm_per_px, threshold_px=2.0):
    """Desk motion from the edges and from table-vs-background features, with a verdict."""
    d = {
        "left_x": desk_live["left"],
        "right_x": desk_live["right"],
        "width_px": desk_live["width_px"],
        "ref_left_x": ref_desk["left"],
        "ref_right_x": ref_desk["right"],
        "ref_width_px": ref_desk["width_px"],
        "d_left_px": round(
            float(np.mean(np.array(desk_live["left"]) - np.array(ref_desk["left"]))), 2
        ),
        "d_right_px": round(
            float(np.mean(np.array(desk_live["right"]) - np.array(ref_desk["right"]))),
            2,
        ),
        "d_width_px": round(desk_live["width_px"] - ref_desk["width_px"], 2),
        "far_y": desk_live["far"]["y_at_x320"] if desk_live.get("far") else None,
        "far_angle": desk_live["far"]["angle_deg"] if desk_live.get("far") else None,
        "ref_far_y": ref_desk["far"]["y_at_x320"] if ref_desk.get("far") else None,
        "ref_far_angle": ref_desk["far"]["angle_deg"] if ref_desk.get("far") else None,
        "mm_per_px": round(mm_per_px, 2),
        "table_motion": table_m,
        "background_motion": bg_m,
    }
    d["d_far_y"] = (
        round(d["far_y"] - d["ref_far_y"], 1)
        if None not in (d["far_y"], d["ref_far_y"])
        else None
    )
    d["d_far_angle"] = (
        round(d["far_angle"] - d["ref_far_angle"], 2)
        if None not in (d["far_angle"], d["ref_far_angle"])
        else None
    )
    edges_dx = (d["d_left_px"] + d["d_right_px"]) / 2
    d["desk_vs_camera_px"] = round(edges_dx, 2)
    d["desk_vs_camera_mm"] = round(edges_dx * mm_per_px, 0)
    cam = bg_m["dx_px"] if bg_m else None
    tab = table_m["dx_px"] if table_m else None
    d["desk_vs_room_px"] = round(tab - cam, 2) if None not in (cam, tab) else None
    d["desk_vs_room_mm"] = (
        round((tab - cam) * mm_per_px, 0) if d["desk_vs_room_px"] is not None else None
    )
    lines = []
    th = threshold_px
    if cam is not None and abs(cam) >= th and tab is not None and abs(tab - cam) < th:
        lines.append(
            f"CAMERA panned: table and background shifted together {'right' if cam > 0 else 'left'} {abs(cam):.1f} px"
        )
    if abs(edges_dx) >= th:
        lines.append(
            f"DESK edges {'right' if edges_dx > 0 else 'left'} {abs(edges_dx):.1f} px = "
            f"{abs(edges_dx) * mm_per_px:.0f} mm relative to the camera"
        )
    if d["desk_vs_room_px"] is not None and abs(d["desk_vs_room_px"]) >= th:
        lines.append(
            f"DESK moved {'right' if d['desk_vs_room_px'] > 0 else 'left'} {abs(d['desk_vs_room_px']):.1f} px = "
            f"{abs(d['desk_vs_room_mm']):.0f} mm relative to the room (table vs background)"
        )
    if cam is not None and abs(cam) >= th and (tab is None or abs(tab) < th):
        lines.append(
            f"background shifted {abs(cam):.1f} px but the table did not: camera moved with the desk, or the background changed"
        )
    if abs(d["d_width_px"]) >= 3:
        lines.append(
            f"table width changed {d['d_width_px']:+.1f} px: desk closer/farther or yawed"
        )
    if (
        d["d_far_angle"] is not None
        and abs(d["d_far_angle"]) >= 0.3
        and (bg_m is None or abs(bg_m["roll_deg"]) < 0.15)
    ):
        lines.append(
            f"far edge tilted {d['d_far_angle']:+.2f} deg while the background did not roll: desk yawed"
        )
    d["verdict"] = lines or ["desk steady"]
    return d


# ------------------------------------------------------------------ rendering
def render(
    mode,
    ref: Reference,
    cur_u,
    gray_u,
    desk_live,
    tags,
    tag_ref,
    header,
    cal: Calibration,
):
    if mode == "blend":
        out = cv2.addWeighted(ref.u, 0.5, cur_u, 0.5, 0)
    elif mode == "raw":
        out = cur_u.copy()
    else:
        out = (cur_u * 0.45).astype(np.uint8)
        edges = cv2.Canny(gray_u, 60, 150)
        out[ref.edges > 0] = (0, 255, 0)
        out[edges > 0] = (0, 0, 255)
        out[(ref.edges > 0) & (edges > 0)] = (0, 255, 255)
    cv2.polylines(out, [ref.poly.reshape(-1, 1, 2)], True, (0, 120, 0), 1)
    if ref.desk.get("far"):
        x1, y1, x2, y2 = ref.desk["far"]["x1y1x2y2"]
        cv2.line(out, (x1, y1), (x2, y2), (0, 200, 0), 1)
    if desk_live.get("far"):
        x1, y1, x2, y2 = desk_live["far"]["x1y1x2y2"]
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
    for i, y in enumerate(
        desk_live["rows"]
    ):  # side-edge ticks: green reference, magenta live
        for xs, colour, half in (
            (ref.desk["left"], (0, 200, 0), 10),
            (ref.desk["right"], (0, 200, 0), 10),
            (desk_live["left"], (255, 0, 255), 14),
            (desk_live["right"], (255, 0, 255), 14),
        ):
            x = int(round(xs[i]))
            cv2.line(out, (x, y - half), (x, y + half), colour, 2 if half > 10 else 1)
    if tag_ref:
        for t in tag_ref.get("tags", []):
            cv2.polylines(
                out,
                [np.array(t["corners_undist"], np.int32).reshape(-1, 1, 2)],
                True,
                (0, 200, 0),
                1,
            )
    for t in tags:
        cv2.polylines(
            out,
            [np.array(t["corners_undist"], np.int32).reshape(-1, 1, 2)],
            True,
            (255, 255, 0),
            2,
        )
        c = t["corners_undist"][0]
        cv2.putText(
            out,
            f"id{t['id']} {t['range_m']:.2f}m",
            (int(c[0]), max(12, int(c[1]) - 5)),
            FONT,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.drawMarker(
        out,
        (int(round(cal.cx)), int(round(cal.cy))),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        16,
        1,
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, header, (4, 13), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


# -------------------------------------------------------------------- monitor
class Monitor:
    """Shared state between the websocket receiver, the analysis thread and HTTP."""

    def __init__(self, cfg: MonitorConfig, cal: Calibration):
        self.cfg, self.cal = cfg, cal
        self.lock = threading.Lock()
        self.detector = TagDetector(cal, cfg.marker_length_m, cfg.aruco_dict)
        self.mm_per_px = (
            1000.0 * cal.table_range_m(cal.cx, sum(cfg.edge_rows) / 2) / cal.fx
        )
        self.latest_jpeg = None
        self.latest_bgr = None
        self.frame_id = self.frames = 0
        self.frame_ts = 0.0
        self.connected = self.superseded = False
        self.close_code = self.last_error = None
        self.status_text, self.age_ms, self.connect_ts = "", None, None
        self.metrics = {"ts": None}
        self.overlays = {}
        self.analysis_hz = 0.0
        self.history = deque(maxlen=int(cfg.history_s * cfg.analysis_hz) + 10)
        self.stop = False
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.ref = self._load_reference()
        self.tag_reference = None
        if cfg.tag_reference.exists():
            try:
                self.tag_reference = json.load(open(cfg.tag_reference))
            except (OSError, ValueError):
                self.tag_reference = None

    def _load_reference(self):
        pointer = self.cfg.reference_pointer
        if pointer.exists():
            try:
                d = json.load(open(pointer))
                path = self.cfg.data_dir / d["image"]
                if path.exists():
                    return Reference(
                        path,
                        d.get("label", d["image"]),
                        self.cal,
                        self.cfg,
                        d.get("saved_at"),
                    )
            except (OSError, ValueError, KeyError):
                traceback.print_exc()
        return None

    # ---- analysis thread
    def analysis_loop(self):
        period = 1.0 / self.cfg.analysis_hz
        last_id = -1
        t_prev = time.time()
        while not self.stop:
            t0 = time.time()
            with self.lock:
                fid, jpeg, tag_ref, ref = (
                    self.frame_id,
                    self.latest_jpeg,
                    self.tag_reference,
                    self.ref,
                )
            if jpeg is not None and fid != last_id:
                last_id = fid
                try:
                    self._analyse(jpeg, tag_ref, ref)
                    now = time.time()
                    with self.lock:
                        self.analysis_hz = 1.0 / max(1e-3, now - t_prev)
                    t_prev = now
                except Exception:
                    with self.lock:
                        self.last_error = traceback.format_exc()[-600:]
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _analyse(self, jpeg, tag_ref, ref):
        cfg, cal = self.cfg, self.cal
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        cur_u = cv2.undistort(bgr, cal.K, cal.dist)
        gray_u = cv2.cvtColor(cur_u, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(bgr)
        m_tag = tag_compare(tag_ref, tags, cal)
        use_tag = bool(m_tag.get("available") and m_tag.get("motion"))
        now = time.time()
        if ref is None:
            metrics = {
                "ts": now,
                "basis": "tag reference" if use_tag else None,
                "reference": None,
                "advice": (
                    advice_for(m_tag["motion"], cfg.good_deg, cfg.good_px)
                    if use_tag
                    else [
                        "no frame reference: press Save frame reference with an empty table"
                    ]
                ),
                "chosen": m_tag["motion"] if use_tag else None,
                "vs_reference": None,
                "desk": None,
                "tags": [
                    {k: t[k] for k in ("id", "centre", "side_px", "range_m", "tvec")}
                    for t in tags
                ],
                "vs_tag_reference": m_tag,
                "tag_reference_saved_at": tag_ref.get("saved_at") if tag_ref else None,
                "calibration": self._calibration_summary(),
            }
            header = (
                f"{time.strftime('%H:%M:%S')} no frame reference | tags {len(tags)}"
            )
            plain = cur_u.copy()
            cv2.rectangle(plain, (0, 0), (plain.shape[1], 18), (0, 0, 0), -1)
            cv2.putText(
                plain, header, (4, 13), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )
            overlays = dict.fromkeys(
                ("edges", "blend", "raw"),
                cv2.imencode(".jpg", plain, [cv2.IMWRITE_JPEG_QUALITY, 85])[
                    1
                ].tobytes(),
            )
            desk_row = None
        else:
            whole, table_m, bg_m = ref_compare(ref, gray_u, cal)
            desk_live = desk_geometry(gray_u, cfg, ref.desk)
            desk = desk_compare(
                ref.desk, desk_live, table_m, bg_m, self.mm_per_px, cfg.desk_px
            )
            basis = "tag reference" if use_tag else ref.label
            chosen = m_tag["motion"] if use_tag else whole
            metrics = {
                "ts": now,
                "basis": basis,
                "advice": advice_for(chosen, cfg.good_deg, cfg.good_px),
                "chosen": chosen,
                "reference": {
                    "label": ref.label,
                    "image": os.path.basename(ref.path),
                    "saved_at": ref.saved_at,
                },
                "vs_reference": whole,
                "desk": desk,
                "tags": [
                    {k: t[k] for k in ("id", "centre", "side_px", "range_m", "tvec")}
                    for t in tags
                ],
                "vs_tag_reference": m_tag,
                "tag_reference_saved_at": tag_ref.get("saved_at") if tag_ref else None,
                "calibration": self._calibration_summary(),
            }
            header = (
                f"{time.strftime('%H:%M:%S')} vs {basis}: "
                + (
                    f"roll {chosen['roll_deg']:+.2f} tilt {chosen['tilt_deg']:+.2f} pan {chosen['pan_deg']:+.2f}"
                    if chosen
                    else "no match"
                )
                + f" | desk edges {desk['desk_vs_camera_px']:+.1f}px"
                + (
                    f" vs room {desk['desk_vs_room_px']:+.1f}px"
                    if desk["desk_vs_room_px"] is not None
                    else ""
                )
                + f" | tags {len(tags)}"
            )
            overlays = {
                mode: render(
                    mode, ref, cur_u, gray_u, desk_live, tags, tag_ref, header, cal
                )
                for mode in ("edges", "blend", "raw")
            }
            desk_row = [
                desk["desk_vs_camera_px"],
                desk["desk_vs_room_px"],
                bg_m["dx_px"] if bg_m else None,
            ]
            metrics["_history"] = {
                "ref": (
                    [whole["roll_deg"], whole["tilt_deg"], whole["pan_deg"]]
                    if whole
                    else None
                )
            }
        with self.lock:
            self.metrics = {k: v for k, v in metrics.items() if k != "_history"}
            self.overlays = overlays
            self.latest_bgr = bgr
            self.history.append(
                {
                    "t": now,
                    "ref": metrics.get("_history", {}).get("ref"),
                    "tag": (
                        [
                            m_tag["motion"]["roll_deg"],
                            m_tag["motion"]["tilt_deg"],
                            m_tag["motion"]["pan_deg"],
                        ]
                        if use_tag
                        else None
                    ),
                    "desk": desk_row,
                }
            )

    def _calibration_summary(self):
        cal, cfg = self.cal, self.cfg
        return {
            "serial": cal.serial,
            "fx": cal.fx,
            "fy": cal.fy,
            "cx": cal.cx,
            "cy": cal.cy,
            "marker_length_m": cfg.marker_length_m,
            "aruco_dict": cfg.aruco_dict,
            "mm_per_px_at_edge_rows": self.mm_per_px,
        }

    # ---- rollout dashboard stream (receive only)
    async def stream_task(self):
        cfg = self.cfg
        while not self.stop:
            if self.superseded:
                await asyncio.sleep(0.5)
                continue
            pending = []
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        cfg.rollout_ws, heartbeat=10, max_msg_size=8 * 1024 * 1024
                    ) as ws:
                        with self.lock:
                            self.connected, self.close_code, self.connect_ts = (
                                True,
                                None,
                                time.time(),
                            )
                        async for message in ws:
                            if message.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(message.data)
                                except json.JSONDecodeError:
                                    continue
                                if data.get("type") == "frame":
                                    pending = list(data.get("images", []))
                                    with self.lock:
                                        self.status_text, self.age_ms = data.get(
                                            "status", ""
                                        ), data.get("age_ms")
                            elif message.type == aiohttp.WSMsgType.BINARY:
                                if pending and pending.pop(0) == cfg.camera:
                                    with self.lock:
                                        self.latest_jpeg, self.frame_ts = (
                                            message.data,
                                            time.time(),
                                        )
                                        self.frame_id += 1
                                        self.frames += 1
                            else:
                                break
                        with self.lock:
                            self.close_code = ws.close_code
            except Exception as error:  # connection refused, reset, ...: retried below
                with self.lock:
                    self.last_error = repr(error)
            with self.lock:
                self.connected = False
                if self.close_code == SUPERSEDED_CLOSE_CODE:
                    self.superseded = True
            await asyncio.sleep(2.0)

    # ---- HTTP
    def make_app(self):
        cfg = self.cfg

        async def index(_request):
            return web.Response(text=HTML, content_type="text/html")

        async def frame(request):
            mode = request.query.get("mode", "edges")
            with self.lock:
                data = self.overlays.get(mode) or self.overlays.get("edges")
            if data is None:
                return web.Response(status=503, text="no frame yet")
            return web.Response(
                body=data,
                content_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

        async def metrics(_request):
            with self.lock:
                m = dict(self.metrics)
                m["stream"] = {
                    "connected": self.connected,
                    "superseded": self.superseded,
                    "close_code": self.close_code,
                    "frames": self.frames,
                    "fps": (
                        (self.frames / max(1e-3, time.time() - self.connect_ts))
                        if (self.connected and self.connect_ts)
                        else 0.0
                    ),
                    "analysis_hz": self.analysis_hz,
                    "status_text": self.status_text,
                    "age_ms": self.age_ms,
                    "last_error": self.last_error,
                    "frame_age_s": (
                        (time.time() - self.frame_ts) if self.frame_ts else None
                    ),
                }
            return web.json_response(m)

        async def history(_request):
            with self.lock:
                items = list(self.history)
            return web.json_response({"now": time.time(), "items": items})

        def latest_frame():
            with self.lock:
                return None if self.latest_bgr is None else self.latest_bgr.copy()

        def retire(path: Path, stamp):
            if path.exists():
                os.replace(path, path.with_name(f"{path.name}.{stamp}.bak"))

        async def save_frame_reference(_request):
            bgr = latest_frame()
            if bgr is None:
                return web.json_response({"error": "no frame yet"})
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"reference_{stamp}.png"
            cv2.imwrite(str(cfg.data_dir / name), bgr)
            pointer = {
                "image": name,
                "label": f"frame reference {stamp}",
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            retire(cfg.reference_pointer, stamp)
            json.dump(pointer, open(cfg.reference_pointer, "w"), indent=1)
            ref = Reference(
                cfg.data_dir / name,
                pointer["label"],
                self.cal,
                cfg,
                pointer["saved_at"],
            )
            with self.lock:
                self.ref = ref
                self.history.clear()
            return web.json_response({"ok": True, "image": name, "desk": ref.desk})

        async def save_tag_reference(_request):
            bgr = latest_frame()
            if bgr is None:
                return web.json_response({"error": "no frame yet"})
            tags = self.detector.detect(bgr)
            if not tags:
                return web.json_response(
                    {"error": "no tag detected in the current frame"}
                )
            stamp = time.strftime("%Y%m%dT%H%M%S")
            ref = {
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "camera": cfg.camera,
                "serial": self.cal.serial,
                "marker_length_m": cfg.marker_length_m,
                "aruco_dict": cfg.aruco_dict,
                "tags": tags,
            }
            retire(cfg.tag_reference, stamp)
            json.dump(ref, open(cfg.tag_reference, "w"), indent=1)
            cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cfg.snapshot_dir / "tag_reference_frame.png"), bgr)
            with self.lock:
                self.tag_reference = ref
            return web.json_response(
                {"ok": True, "tags": len(tags), "saved_at": ref["saved_at"]}
            )

        async def clear_tag_reference(_request):
            retire(cfg.tag_reference, time.strftime("%Y%m%dT%H%M%S"))
            with self.lock:
                self.tag_reference = None
            return web.json_response({"ok": True})

        async def snapshot(_request):
            with self.lock:
                bgr = None if self.latest_bgr is None else self.latest_bgr.copy()
                overlay, m = self.overlays.get("edges"), dict(self.metrics)
            if bgr is None:
                return web.json_response({"error": "no frame yet"})
            cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%S")
            cv2.imwrite(str(cfg.snapshot_dir / f"{stamp}_raw.png"), bgr)
            if overlay:
                (cfg.snapshot_dir / f"{stamp}_edges.jpg").write_bytes(overlay)
            json.dump(
                m,
                open(cfg.snapshot_dir / f"{stamp}_metrics.json", "w"),
                indent=1,
                default=str,
            )
            return web.json_response({"ok": True, "stamp": stamp})

        async def reconnect(_request):
            with self.lock:
                self.superseded, self.close_code = False, None
            return web.json_response({"ok": True})

        @web.middleware
        async def loopback_only(request, handler):
            if request.remote not in {"127.0.0.1", "::1"}:
                raise web.HTTPForbidden()
            return await handler(request)

        app = web.Application(middlewares=[loopback_only])
        app.router.add_get("/", index)
        app.router.add_get("/frame.jpg", frame)
        app.router.add_get("/metrics.json", metrics)
        app.router.add_get("/history.json", history)
        app.router.add_post("/api/frame_reference", save_frame_reference)
        app.router.add_post("/api/reference", save_tag_reference)
        app.router.add_post("/api/reference/clear", clear_tag_reference)
        app.router.add_post("/api/snapshot", snapshot)
        app.router.add_post("/api/reconnect", reconnect)

        async def on_startup(app):
            app["stream"] = asyncio.create_task(self.stream_task())

        async def on_cleanup(app):
            self.stop = True
            app["stream"].cancel()

        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)
        return app


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Camera align</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}#wrap{display:flex;gap:16px;padding:12px;flex-wrap:wrap}
#img{width:800px;height:600px;background:#000;display:block}table{border-collapse:collapse;margin-bottom:10px}td{padding:2px 8px;vertical-align:top}td.k{color:#9ab}
.good{color:#6f6}.bad{color:#f66}.warn{color:#fc6}button{margin:2px}canvas{background:#000;display:block;margin-top:8px}#advice{font-size:17px;margin:8px 0;line-height:1.4}
h3{margin:10px 0 4px;color:#9ab;font-size:13px;text-transform:uppercase}#side{min-width:440px;max-width:560px}</style></head>
<body><div id="wrap"><div><img id="img">
<div style="margin-top:6px"><label><input type=radio name=mode value=edges checked> edges (green=reference, red=live)</label>
<label><input type=radio name=mode value=blend> blend</label> <label><input type=radio name=mode value=raw> raw</label>
&nbsp; <button onclick="post('/api/frame_reference')" title="Take the current frame as the reference image (empty table, arms at home)">Save frame reference</button>
<button onclick="post('/api/reference')">Save tag reference</button><button onclick="post('/api/reference/clear')">Clear tag reference</button>
<button onclick="post('/api/snapshot')">Snapshot</button><button onclick="post('/api/reconnect')">Reconnect stream</button></div>
<canvas id="spark" width=800 height=170></canvas><div style="color:#9ab;font-size:12px">camera: roll (yellow), tilt (cyan), pan (magenta) in degrees, solid = tag reference, dotted = frame reference; grid at 0 and ±0.5°</div>
<canvas id="spark2" width=800 height=120></canvas><div style="color:#9ab;font-size:12px">desk: edges vs camera (orange), table vs background = desk vs room (white), background alone = camera vs room (grey), in px; grid at 0 and ±5 px</div></div>
<div id="side"><div id="status"></div><div id="advice"></div><div id="tables"></div></div></div>
<script>
let mode='edges';document.querySelectorAll('input[name=mode]').forEach(r=>r.onchange=()=>mode=r.value);
async function post(u){try{const r=await fetch(u,{method:'POST'});const j=await r.json();if(j.error)alert(j.error);}catch(e){alert(e)}}
let lastUrl=null;
async function frame(){try{const r=await fetch('/frame.jpg?mode='+mode+'&t='+Date.now());if(r.ok){const b=await r.blob();const u=URL.createObjectURL(b);document.getElementById('img').src=u;if(lastUrl)URL.revokeObjectURL(lastUrl);lastUrl=u;}}catch(e){}setTimeout(frame,150);}
const f=(v,d=2)=>v==null?'–':(typeof v==='number'?v.toFixed(d):v);
const cls=(v,lim)=>v==null?'':(Math.abs(v)<lim?'good':'bad');
const row=(k,v,c)=>`<tr><td class=k>${k}</td><td class="${c||''}">${v}</td></tr>`;
function motionTable(m,title){if(!m)return `<h3>${title}</h3><div class=warn>no match</div>`;
 return `<h3>${title}</h3><table>${row('roll (scene turned clockwise +)',f(m.roll_deg)+'°',cls(m.roll_deg,0.15))}${row('tilt (scene down +)',f(m.tilt_deg)+'°  ('+f(m.dy_px,1)+' px)',cls(m.tilt_deg,0.15))}${row('pan (scene right +)',f(m.pan_deg)+'°  ('+f(m.dx_px,1)+' px)',cls(m.pan_deg,0.15))}${row('scale',f(m.scale,4),cls(m.scale-1,0.005))}${row('points / inliers',(m.matches||m.n)+' / '+m.inliers)}</table>`;}
function render(m){const s=m.stream||{};let st=`<b>stream:</b> ${s.connected?'<span class=good>connected</span>':(s.superseded?'<span class=bad>superseded by the rollout dashboard tab — press Reconnect (this closes that tab)</span>':'<span class=warn>disconnected</span>')} · frames ${s.frames} · ${f(s.fps,1)} fps · analysis ${f(s.analysis_hz,1)} Hz · rollout: ${s.status_text||''}`;
 const r=m.reference;st+=`<div style="color:#9ab;font-size:12px">frame reference: ${r?r.label+' ('+r.image+(r.saved_at?', saved '+r.saved_at:'')+')':'<span class=warn>none saved</span>'}</div>`;
 if(s.last_error)st+=`<div class=warn style="font-size:11px">${s.last_error}</div>`;document.getElementById('status').innerHTML=st;
 const d=m.desk||{};const adv=(m.advice||[]).map(a=>`<div class="${a==='ALIGNED'?'good':'bad'}">${a}</div>`).join('');
 const ver=(d.verdict||[]).map(a=>`<div class="${a==='desk steady'?'good':'warn'}">${a}</div>`).join('');
 document.getElementById('advice').innerHTML=`<div style="color:#9ab;font-size:12px">camera, vs ${m.basis||'–'}</div>${adv}<div style="color:#9ab;font-size:12px;margin-top:6px">desk</div>${ver}`;
 let t='';
 if(m.desk){t+=`<h3>desk (edges at rows 185/220, ${f(d.mm_per_px,2)} mm/px there)</h3><table>${row('left edge x (live / ref)',f(d.left_x[0],1)+', '+f(d.left_x[1],1)+'  /  '+f(d.ref_left_x[0],1)+', '+f(d.ref_left_x[1],1)+'  → Δ '+f(d.d_left_px,1)+' px',cls(d.d_left_px,2))}${row('right edge x (live / ref)',f(d.right_x[0],1)+', '+f(d.right_x[1],1)+'  /  '+f(d.ref_right_x[0],1)+', '+f(d.ref_right_x[1],1)+'  → Δ '+f(d.d_right_px,1)+' px',cls(d.d_right_px,2))}${row('table width',f(d.width_px,1)+' px (ref '+f(d.ref_width_px,1)+', Δ '+f(d.d_width_px,1)+')',cls(d.d_width_px,3))}${row('far edge y / angle',f(d.far_y,1)+' px / '+f(d.far_angle)+'°  (ref '+f(d.ref_far_y,1)+' / '+f(d.ref_far_angle)+'°, Δ '+f(d.d_far_y,1)+' px, '+f(d.d_far_angle)+'°)',cls(d.d_far_y,2))}${row('desk vs camera (edges)',f(d.desk_vs_camera_px,1)+' px = '+f(d.desk_vs_camera_mm,0)+' mm',cls(d.desk_vs_camera_px,2))}${row('desk vs room (table − background)',d.desk_vs_room_px==null?'need background features':f(d.desk_vs_room_px,1)+' px = '+f(d.desk_vs_room_mm,0)+' mm',cls(d.desk_vs_room_px,2))}${row('table features dx/dy/roll',d.table_motion?f(d.table_motion.dx_px,1)+' / '+f(d.table_motion.dy_px,1)+' px / '+f(d.table_motion.roll_deg)+'° ('+d.table_motion.inliers+' inl)':'–')}${row('background features dx/dy/roll',d.background_motion?f(d.background_motion.dx_px,1)+' / '+f(d.background_motion.dy_px,1)+' px / '+f(d.background_motion.roll_deg)+'° ('+d.background_motion.inliers+' inl)':'too few')}</table>`;}
 const tr=m.vs_tag_reference||{};
 if(tr.available){t+=motionTable(tr.motion,'camera vs saved tag reference ('+tr.matched+'/'+tr.of+' tags, saved '+(m.tag_reference_saved_at||'')+')');
  if(tr.per_tag){t+='<table>'+tr.per_tag.map(p=>row('tag '+p.id,'shift '+f(p.shift_px[0],1)+', '+f(p.shift_px[1],1)+' px · range Δ '+f(p.range_delta_mm,0)+' mm · pose rot Δ '+f(p.pose_rot_delta_deg)+'°')).join('')+(tr.baseline_mm?row('tag baseline',f(tr.baseline_mm,0)+' mm (Δ '+f(tr.baseline_delta_mm,0)+' mm)',tr.tags_moved_warning?'bad':'good'):'')+'</table>';}}
 else t+='<h3>tag reference</h3><div class=warn>none saved — align, then press "Save tag reference"</div>';
 t+=motionTable(m.vs_reference,'camera vs frame reference (SIFT, whole frame)');
 t+='<h3>tags in view</h3>'+(m.tags&&m.tags.length?'<table>'+m.tags.map(x=>row('id '+x.id,'centre '+f(x.centre[0],1)+', '+f(x.centre[1],1)+' · side '+f(x.side_px,0)+' px · range '+f(x.range_m,3)+' m · cam xyz '+x.tvec.map(v=>f(v,3)).join(', '))).join('')+'</table>':'<div class=warn>none detected</div>');
 const c=m.calibration||{};t+=`<div style="color:#789;font-size:11px">D405 ${c.serial} fx ${f(c.fx,1)} fy ${f(c.fy,1)} cx ${f(c.cx,1)} cy ${f(c.cy,1)} · ${c.aruco_dict} · marker ${c.marker_length_m} m · overlay is undistorted</div>`;
 document.getElementById('tables').innerHTML=t;}
async function metrics(){try{render(await (await fetch('/metrics.json')).json());}catch(e){}setTimeout(metrics,350);}
function series(cv,h,R,grid,specs){const g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);const now=h.now,span=180;
 const y=v=>H/2-v/R*(H/2-8),x=t=>W-(now-t)/span*W;g.setLineDash([]);g.lineWidth=1;g.strokeStyle='#333';grid.forEach(v=>{g.beginPath();g.moveTo(0,y(v));g.lineTo(W,y(v));g.stroke();});g.strokeStyle='#666';g.beginPath();g.moveTo(0,y(0));g.lineTo(W,y(0));g.stroke();
 for(const sp of specs){g.strokeStyle=sp.col;g.setLineDash(sp.dash||[]);g.lineWidth=sp.w||2;g.beginPath();let on=false;for(const p of h.items){const v=p[sp.key];const val=v?v[sp.i]:null;if(val==null){on=false;continue;}const px=x(p.t),py=y(Math.max(-R,Math.min(R,val)));if(!on){g.moveTo(px,py);on=true;}else g.lineTo(px,py);}g.stroke();}g.setLineDash([]);}
function draw(h){series(document.getElementById('spark'),h,1.5,[-1,-0.5,0.5,1],[{key:'ref',i:0,col:'#ff6',dash:[2,3],w:1},{key:'ref',i:1,col:'#6ff',dash:[2,3],w:1},{key:'ref',i:2,col:'#f6f',dash:[2,3],w:1},{key:'tag',i:0,col:'#ff6'},{key:'tag',i:1,col:'#6ff'},{key:'tag',i:2,col:'#f6f'}]);
 series(document.getElementById('spark2'),h,15,[-10,-5,5,10],[{key:'desk',i:2,col:'#888',w:1},{key:'desk',i:0,col:'#f93'},{key:'desk',i:1,col:'#fff'}]);}
async function spark(){try{draw(await (await fetch('/history.json')).json());}catch(e){}setTimeout(spark,1000);}
frame();metrics();spark();
</script></body></html>"""


def parse_args(argv=None) -> MonitorConfig:
    defaults = MonitorConfig()
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=defaults.data_dir,
        help="reference frames, tag reference and snapshots (default: %(default)s)",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=defaults.intrinsics_yaml,
        help="agent-view extrinsics yaml carrying the camera's K and dist",
    )
    parser.add_argument(
        "--rollout-ws",
        default=defaults.rollout_ws,
        help="rollout dashboard websocket (default: %(default)s)",
    )
    parser.add_argument(
        "--camera",
        default=defaults.camera,
        help="camera name in the rollout stream (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=defaults.port,
        help="loopback port to serve on (default: %(default)s)",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=defaults.marker_length_m,
        help="ArUco marker side in metres (default: %(default)s)",
    )
    parser.add_argument(
        "--aruco-dict",
        default=defaults.aruco_dict,
        help="cv2.aruco predefined dictionary name (default: %(default)s)",
    )
    parser.add_argument("--analysis-hz", type=float, default=defaults.analysis_hz)
    args = parser.parse_args(argv)
    return MonitorConfig(
        rollout_ws=args.rollout_ws,
        camera=args.camera,
        port=args.port,
        data_dir=args.data_dir.expanduser(),
        intrinsics_yaml=args.intrinsics,
        marker_length_m=args.marker_length,
        aruco_dict=args.aruco_dict,
        analysis_hz=args.analysis_hz,
    )


def main(argv=None):
    cfg = parse_args(argv)
    cal = Calibration.from_yaml(cfg.intrinsics_yaml)
    monitor = Monitor(cfg, cal)
    threading.Thread(target=monitor.analysis_loop, daemon=True).start()
    print(
        f"camera_align_monitor: D405 {cal.serial} fx={cal.fx:.2f} fy={cal.fy:.2f} cx={cal.cx:.2f} cy={cal.cy:.2f}; "
        f"{monitor.mm_per_px:.2f} mm/px at the edge rows; data dir {cfg.data_dir}; "
        f"reference {monitor.ref.label + ' (' + os.path.basename(monitor.ref.path) + ')' if monitor.ref else 'none'}; "
        f"tag reference {'loaded' if monitor.tag_reference else 'none'}; serving http://{cfg.host}:{cfg.port}",
        flush=True,
    )
    web.run_app(monitor.make_app(), host=cfg.host, port=cfg.port, print=None)


if __name__ == "__main__":
    main()
