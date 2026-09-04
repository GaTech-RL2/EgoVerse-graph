from pathlib import Path


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


def test_arc_overlay_is_validation_split_and_artifact_bound():
    overlay = (ROOT / "egomimic/eval/arc_teacher_overlay.py").read_text()
    assert "episode_id not in valid_ids" in overlay
    assert "refusing to overwrite immutable output" in overlay
    assert '"not_a_policy_score": True' in overlay
