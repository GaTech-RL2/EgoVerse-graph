"""Pure-function tests for the camera alignment monitor: no camera, no network."""

import json
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

from egomimic.robot.camera_align import monitor as m  # noqa: E402

K = np.array([[391.5, 0.0, 317.8], [0.0, 391.0, 238.4], [0.0, 0.0, 1.0]])
NO_DIST = np.zeros(5)
CAL = m.Calibration(K, NO_DIST, "test")


def rotate_about_centre(points, degrees, dx=0.0, dy=0.0):
    """Rotate image points clockwise-on-screen by ``degrees`` about the principal point."""
    theta = math.radians(degrees)
    R = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    centre = np.array([CAL.cx, CAL.cy])
    return (points - centre) @ R.T + centre + np.array([dx, dy])


def test_screen_motion_reads_roll_and_shift_in_screen_terms():
    rng = np.random.default_rng(0)
    src = rng.uniform([20, 20], [620, 460], size=(200, 2))
    dst = rotate_about_centre(src, 0.5, dx=3.0, dy=-2.0)
    motion = m.screen_motion(src, dst, CAL)
    assert motion["roll_deg"] == pytest.approx(0.5, abs=0.02)
    assert motion["dx_px"] == pytest.approx(3.0, abs=0.1)
    assert motion["dy_px"] == pytest.approx(-2.0, abs=0.1)
    assert motion["pan_deg"] == pytest.approx(
        math.degrees(math.atan(3.0 / CAL.fx)), abs=0.01
    )
    assert motion["tilt_deg"] == pytest.approx(
        math.degrees(math.atan(-2.0 / CAL.fy)), abs=0.01
    )
    assert motion["scale"] == pytest.approx(1.0, abs=1e-3)
    assert motion["inliers"] >= 190


def test_screen_motion_ignores_outliers_and_needs_two_points():
    rng = np.random.default_rng(1)
    src = rng.uniform([20, 20], [620, 460], size=(100, 2))
    dst = src + np.array([4.0, 0.0])
    dst[:10] += rng.uniform(-80, 80, size=(10, 2))  # moved objects on the table
    motion = m.screen_motion(src, dst, CAL)
    assert motion["dx_px"] == pytest.approx(4.0, abs=0.2)
    assert motion["inliers"] <= 92
    assert m.screen_motion(src[:1], dst[:1], CAL) is None


def test_advice_names_the_direction_the_image_must_move():
    assert m.advice_for(None) == ["no measurement"]
    aligned = {
        "roll_deg": 0.05,
        "tilt_deg": 0.02,
        "pan_deg": -0.03,
        "dx_px": -0.2,
        "dy_px": 0.1,
    }
    assert m.advice_for(aligned) == ["ALIGNED"]
    off = {
        "roll_deg": 0.5,
        "tilt_deg": 0.3,
        "pan_deg": -0.4,
        "dx_px": -2.7,
        "dy_px": 2.0,
    }
    lines = m.advice_for(off)
    assert lines[0].startswith("ROLL: turn the image counter-clockwise by 0.50 deg")
    assert (
        lines[1].startswith("TILT: camera down 0.30 deg")
        and "move up 2.0 px" in lines[1]
    )
    assert (
        lines[2].startswith("PAN: camera left 0.40 deg")
        and "move right 2.7 px" in lines[2]
    )


def synthetic_table(left=112.3, right=531.6, far_y=157, width=640, height=480):
    """Dark room, light trapezoid table whose side edges pass through (left, 185)/(right, 185)
    with a slope of 0.3 px per row, and a wall above the far edge."""
    img = np.full((height, width), 70, np.uint8)
    img[:far_y, :] = 120  # wall
    rows = np.arange(far_y, height)
    xl = left + 0.3 * (185 - rows)
    xr = right - 0.3 * (185 - rows)
    for y, a, b in zip(rows, xl, xr):
        img[y, int(round(a)) : int(round(b))] = 190
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_side_edge_is_located_to_a_fraction_of_a_pixel():
    # The synthetic table is drawn on whole pixels, so an edge at a.b sits between
    # pixels round(a.b) - 1 and round(a.b); the detector reports that boundary.
    gray = synthetic_table()
    x, strength = m.side_edge_x(gray, 185, (30, 210))
    assert x == pytest.approx(round(112.3) - 0.5, abs=0.6)
    assert strength > 20
    x, _ = m.side_edge_x(gray, 220, (430, 610))
    assert x == pytest.approx(round(531.6 + 0.3 * 35) - 0.5, abs=0.6)


def test_desk_geometry_tracks_a_shifted_table_and_builds_a_mask():
    cfg = m.MonitorConfig()
    ref = m.desk_geometry(synthetic_table(), cfg)
    assert ref["far"] is not None and ref["far"]["y_at_x320"] == pytest.approx(
        157, abs=2
    )
    assert abs(ref["far"]["angle_deg"]) < 0.5
    assert ref["left"][0] == pytest.approx(round(112.3) - 0.5, abs=0.6)
    live = m.desk_geometry(synthetic_table(left=117.3, right=536.6), cfg, ref)
    assert live["left"][0] - ref["left"][0] == pytest.approx(5.0, abs=0.6)
    assert live["right"][1] - ref["right"][1] == pytest.approx(5.0, abs=0.6)
    assert live["width_px"] == pytest.approx(ref["width_px"], abs=0.8)
    poly = m.table_polygon(ref)
    assert poly.shape == (4, 2) and poly[0, 1] == poly[1, 1] and poly[2, 1] == 479
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    assert mask[300, 320] == 255 and mask[300, 20] == 0 and mask[100, 320] == 0


def test_desk_compare_separates_camera_pan_from_desk_motion():
    ref = {
        "left": [110.0, 99.0],
        "right": [518.0, 530.0],
        "width_px": 419.5,
        "far": {"y_at_x320": 157.0, "angle_deg": -0.4},
        "rows": [185, 220],
    }
    live = {
        "left": [116.0, 105.0],
        "right": [524.0, 536.0],
        "width_px": 419.5,
        "far": {"y_at_x320": 157.0, "angle_deg": -0.4},
        "rows": [185, 220],
    }
    shift = {
        "roll_deg": 0.0,
        "scale": 1.0,
        "dx_px": 6.0,
        "dy_px": 0.0,
        "pan_deg": 0.9,
        "tilt_deg": 0.0,
        "inliers": 50,
        "n": 50,
    }
    still = {**shift, "dx_px": 0.0, "pan_deg": 0.0}
    # Everything shifted together: the camera panned.
    camera = m.desk_compare(ref, live, shift, shift, 3.0)
    assert camera["desk_vs_camera_px"] == 6.0 and camera["desk_vs_room_px"] == 0.0
    assert camera["verdict"][0].startswith("CAMERA panned")
    # The table and its edges shifted while the background stayed: the desk moved.
    desk = m.desk_compare(ref, live, shift, still, 3.0)
    assert desk["desk_vs_room_px"] == 6.0 and desk["desk_vs_room_mm"] == 18
    assert any(
        line.startswith("DESK moved right 6.0 px = 18 mm relative to the room")
        for line in desk["verdict"]
    )
    assert any(line.startswith("DESK edges right 6.0 px") for line in desk["verdict"])
    # Nothing moved.
    steady = m.desk_compare(ref, ref, still, still, 3.0)
    assert steady["verdict"] == ["desk steady"]
    # No background features: desk vs room is unknown, edges still report.
    partial = m.desk_compare(ref, live, shift, None, 3.0)
    assert partial["desk_vs_room_px"] is None and partial["desk_vs_camera_mm"] == 18


def project(points_m, rvec, tvec):
    image, _ = cv2.projectPoints(
        np.asarray(points_m, np.float32), rvec, tvec, K, NO_DIST
    )
    return image.reshape(-1, 2)


def render_marker(canvas, tag_id, side_m, rvec, tvec, dictionary="DICT_4X4_50"):
    """Draw a marker of ``side_m`` metres at pose (rvec, tvec) into ``canvas`` and return its centre."""
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), tag_id, 200
    )
    # The detector's corner order: top-left, top-right, bottom-right, bottom-left of the marker.
    half = side_m / 2
    corners_m = [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]]
    dst = project(corners_m, rvec, tvec).astype(np.float32)
    src = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        marker,
        H,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
    coverage = cv2.warpPerspective(
        np.full_like(marker, 255),
        H,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    canvas[coverage > 0] = warped[coverage > 0]
    return dst.mean(axis=0)


def test_tag_detector_recovers_id_range_and_centre():
    canvas = np.full((480, 640), 200, np.uint8)
    rvec = np.array([math.pi, 0.0, 0.0])  # tag faces the camera
    centre = render_marker(canvas, 0, 0.2, rvec, np.array([0.1, 0.05, 1.0]))
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    tags = m.TagDetector(CAL, 0.2).detect(bgr)
    assert [t["id"] for t in tags] == [0]
    assert tags[0]["range_m"] == pytest.approx(
        math.sqrt(0.1**2 + 0.05**2 + 1.0**2), rel=0.03
    )
    assert np.allclose(tags[0]["centre"], centre, atol=1.5)
    assert tags[0]["side_px"] == pytest.approx(0.2 * K[0, 0] / 1.0, rel=0.05)


def test_tag_compare_measures_camera_motion_and_flags_a_moved_tag():
    rvec = np.array([math.pi, 0.0, 0.0])
    detector = m.TagDetector(CAL, 0.2)

    def scene(offset_left=(0.0, 0.0, 0.0)):
        canvas = np.full((480, 640), 200, np.uint8)
        render_marker(
            canvas, 0, 0.2, rvec, np.array([-0.35, 0.0, 1.3]) + np.asarray(offset_left)
        )
        render_marker(canvas, 0, 0.2, rvec, np.array([0.3, 0.1, 1.0]))
        return detector.detect(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    ref = {"tags": scene()}
    assert len(ref["tags"]) == 2
    # Same scene, shifted 4 px right as an image: a camera pan.
    moved = []
    for tag in ref["tags"]:
        t = json.loads(json.dumps(tag))
        t["centre"] = [t["centre"][0] + 4.0, t["centre"][1]]
        t["corners"] = [[x + 4.0, y] for x, y in t["corners"]]
        t["corners_undist"] = [[x + 4.0, y] for x, y in t["corners_undist"]]
        moved.append(t)
    result = m.tag_compare(ref, moved, CAL)
    assert result["matched"] == 2
    assert result["motion"]["dx_px"] == pytest.approx(4.0, abs=0.2)
    assert result["tags_moved_warning"] is False
    # One tag physically moved 6 cm: the baseline changes and the warning fires.
    result = m.tag_compare(ref, scene(offset_left=(0.06, 0.0, 0.0)), CAL)
    assert result["matched"] == 2 and result["tags_moved_warning"] is True
    assert abs(result["baseline_delta_mm"]) > 40
    assert m.tag_compare(None, moved, CAL) == {"available": False}
    assert m.tag_compare(ref, [], CAL)["matched"] == 0


def test_reference_from_a_png_has_features_edges_and_a_table_mask(tmp_path):
    rng = np.random.default_rng(2)
    gray = synthetic_table()
    textured = np.clip(
        gray.astype(int) + rng.integers(-25, 25, gray.shape), 0, 255
    ).astype(np.uint8)
    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), cv2.cvtColor(textured, cv2.COLOR_GRAY2BGR))
    cfg = m.MonitorConfig(data_dir=tmp_path)
    ref = m.Reference(path, "synthetic", CAL, cfg)
    assert ref.des is not None and len(ref.kp) > 100
    assert ref.table_mask[300, 320] == 255 and ref.table_mask[100, 320] == 0
    whole, table, background = m.ref_compare(ref, ref.g, CAL)
    assert whole["dx_px"] == pytest.approx(0.0, abs=0.05) and whole[
        "roll_deg"
    ] == pytest.approx(0.0, abs=0.01)
    assert table is not None and table["inliers"] >= 12


def test_calibration_loads_the_station_yaml_and_ranges_the_table():
    if not m.DEFAULT_INTRINSICS.exists():
        pytest.skip("agent-view extrinsics yaml not in this checkout")
    cal = m.Calibration.from_yaml(m.DEFAULT_INTRINSICS)
    assert cal.K.shape == (3, 3) and cal.dist.shape[0] == 5 and cal.serial
    assert 300 < cal.fx < 500 and 250 < cal.cx < 400
    rng = cal.table_range_m(cal.cx, 200)
    assert 0.5 < rng < 3.0
    assert m.Calibration(K, NO_DIST, "x").table_range_m(320, 200, fallback=1.25) == 1.25


def _fit(dy=0.0, scale=1.0):
    return {
        "roll_deg": 0.0,
        "tilt_deg": 0.0,
        "pan_deg": 0.0,
        "dx_px": 0.0,
        "dy_px": dy,
        "scale": scale,
        "inliers": 40,
        "n": 40,
    }


def test_advice_names_a_camera_moved_along_its_viewing_axis():
    assert m.advice_for(_fit(scale=1.001)) == ["ALIGNED"]
    assert "closer to" in m.advice_for(_fit(scale=1.016))[0]
    assert "farther from" in m.advice_for(_fit(scale=0.985))[0]


def test_desk_compare_names_a_desk_slid_along_the_camera_axis():
    desk = {
        "left": [110.0, 99.0],
        "right": [518.0, 530.0],
        "width_px": 419.5,
        "far": {"y_at_x320": 157.0, "angle_deg": -0.4},
        "rows": [185, 220],
    }
    assert m.desk_compare(desk, desk, _fit(0.5), _fit(), 3.0)["verdict"] == [
        "desk steady"
    ]
    away = m.desk_compare(desk, desk, _fit(-6.0), _fit(), 3.0)["verdict"]
    assert len(away) == 1 and "away from the camera" in away[0]
    toward = m.desk_compare(desk, desk, _fit(6.0), _fit(), 3.0)["verdict"]
    assert len(toward) == 1 and "toward the camera" in toward[0]


def test_stale_verdict_is_withdrawn_and_a_failed_save_keeps_the_reference(
    tmp_path, monkeypatch
):
    import asyncio
    import time

    from aiohttp.test_utils import TestClient, TestServer

    # Nothing listens on the discard port, so the stream never connects.
    cfg = m.MonitorConfig(data_dir=tmp_path, rollout_ws="ws://127.0.0.1:9/ws")
    monitor = m.Monitor(cfg, CAL)
    monitor.metrics = {
        "ts": time.time(),
        "advice": ["ALIGNED"],
        "desk": {"verdict": ["desk steady"]},
    }
    monitor.latest_bgr = np.zeros((480, 640, 3), np.uint8)
    cfg.reference_pointer.write_text('{"image": "old.png"}')

    async def scenario():
        async with TestClient(TestServer(monitor.make_app())) as client:

            async def get_metrics():
                return await (await client.get("/metrics.json")).json()

            async def save():
                return await (await client.post("/api/frame_reference")).json()

            disconnected = await get_metrics(), await save()
            monitor.connected = True
            live = await get_metrics()
            monitor.age_ms = 6000  # the dashboard re-sending an old frame
            resent = await get_metrics()
            monitor.age_ms = 0
            monkeypatch.setattr(m.cv2, "imwrite", lambda *_: False)
            return disconnected, live, resent, await save()

    (stale, refused), live, resent, failed = asyncio.run(scenario())
    assert stale["advice"] == stale["desk"]["verdict"] == ["STALE: no live measurement"]
    assert stale["stream"]["stale"] and "error" in refused
    assert live["advice"] == ["ALIGNED"] and live["desk"]["verdict"] == ["desk steady"]
    assert resent["advice"] == ["STALE: no live measurement"]
    assert "error" in failed
    assert json.loads(cfg.reference_pointer.read_text()) == {"image": "old.png"}
