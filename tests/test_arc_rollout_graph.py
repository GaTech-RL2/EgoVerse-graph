from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_arc_rollout_is_data_bound_outside_route_agnostic_pipeline():
    algo = (ROOT / "egomimic/pipeline/algo.py").read_text().lower()
    rollout = (ROOT / "egomimic/eval/core/ckpt_loading.py").read_text()
    assert "def inference_step" not in algo
    assert "class PipelineRolloutGraph" in rollout
    assert 'mode="inference"' in rollout
    assert "PlanarArcWaypointZeroNativeDecoder" not in rollout


def test_arc_rollout_requires_one_step_native_decoder():
    rollout = (ROOT / "egomimic/eval/core/ckpt_loading.py").read_text()
    assert "arc decoder must return one replanned action" in rollout
    assert "waypoint-zero arc rollout replans every step" in rollout
