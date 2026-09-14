from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def test_arc_overlay_uses_rollout_shared_token_inference():
    rollout = (ROOT / "egomimic/eval/core/ckpt_loading.py").read_text()
    overlay = (ROOT / "egomimic/eval/arc_teacher_overlay.py").read_text()
    assert "def predict_tokens" in rollout
    assert 'self.predict_tokens(obs_zarr, namespace="rollout")' in rollout
    assert "graph.predict_tokens" in overlay


def test_arc_overlay_keeps_timing_row_out_of_xy_geometry():
    overlay = (ROOT / "egomimic/eval/arc_teacher_overlay.py").read_text()
    assert "target[:16, :2]" in overlay
    assert "pred[:16, :2]" in overlay
    assert '"timing_row_drawn": False' in overlay


def test_arc_overlay_uses_small_start_marker_and_supports_artifact_rerender():
    overlay = (ROOT / "egomimic/eval/arc_teacher_overlay.py").read_text()
    assert "def render_prediction_artifact" in overlay
    assert "tuple(points[0]), 1" in overlay
    assert "for index, point in enumerate(points)" not in overlay


def test_arc_silhouette_uses_exact_sim_v2_geometry_and_pose_decode():
    overlay = (ROOT / "egomimic/eval/arc_teacher_overlay.py").read_text()
    assert "from Tsimulation.pushshapes.shapes import U_SOCKET_RECTS" in overlay
    assert 'Tsimulation.ACTIVE != "sim_v2"' in overlay
    assert "math.atan2(float(token[3]), float(token[2]))" in overlay
    assert 'view not in {"xy", "silhouette"}' in overlay


def test_arc_silhouette_axis_aligned_geometry_matches_sim_v2(monkeypatch):
    monkeypatch.setenv("TSIM_VERSION", "sim_v2")
    from egomimic.eval.arc_teacher_overlay import _usocket_polygons

    polygons = _usocket_polygons(
        np.asarray([256.0, 256.0, 1.0, 0.0, 0.0]), width=512, height=512
    )
    points = np.concatenate([polygon.reshape(-1, 2) for polygon in polygons], axis=0)
    assert len(polygons) == 3
    assert points.min(axis=0).tolist() == [236, 230]
    assert points.max(axis=0).tolist() == [276, 282]


def test_arc_overlay_is_validation_split_and_artifact_bound():
    overlay = (ROOT / "egomimic/eval/arc_teacher_overlay.py").read_text()
    assert "episode_id not in valid_ids" in overlay
    assert "refusing to overwrite immutable output" in overlay
    assert '"not_a_policy_score": True' in overlay
