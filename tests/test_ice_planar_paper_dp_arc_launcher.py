import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "ice" / "launch_planar_paper_dp_arc.sbatch"


def test_paper_dp_arc_launcher_is_valid_and_ice_t_bound():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text()
    assert "#SBATCH --account=ece" in text
    assert "#SBATCH --partition=coe-gpu" in text
    assert "#SBATCH --qos=coe-ice" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "#SBATCH --constraint=H100|H200" in text
    assert "#SBATCH --requeue" in text
    assert "#SBATCH --signal=B:USR1@600" in text
    assert "usocket-paper-dp-arc-d40-m16-r24" in text
    assert "c3kb" not in text
    assert "skynet]" not in text


def test_paper_dp_arc_launcher_guards_both_fair_rows():
    text = LAUNCHER.read_text()
    uniform = "pusht/planar_v2_usocket_arc_paper_uniform_D40_M16_R24deg"
    curvature = "pusht/planar_v2_usocket_arc_paper_curvature_D40_M16_R24deg"
    assert text.count(uniform) >= 2
    assert text.count(curvature) >= 2
    assert 'expected_sampling = "curvature" if "curvature" in experiment else "uniform"' in text
    assert 'assert int(cfg.planar.action_horizon) == 17' in text
    assert 'assert int(transform.raw_action_horizon) == 40' in text
    assert 'assert int(transform.action_target_offset) == 1' in text
    assert 'assert int(action_contract.replan_every) == 1' in text


def test_paper_dp_arc_launcher_keeps_smoke_and_checkpoint_gates():
    text = LAUNCHER.read_text()
    assert 'assert payload["scheduled_validation"] == "passed"' in text
    assert 'assert payload["strict_checkpoint_reload"] == "passed"' in text
    assert 'assert payload["wandb_run_visible"] is True' in text
    assert "callbacks.model_checkpoint.save_top_k=-1" in text
    assert '"callbacks.model_checkpoint.filename=\'epoch-{epoch}-step-{step}\'"' in text
    assert "--checkpoint-validator" in text
    assert "--completion-sentinel" in text
    assert "Paper DP run mode requires hash-pinned precomputed normalization" in text
    assert "precomputed norm_stats hash mismatch" in text
    assert '"++run_provenance.source_commit=$ICE_EXPECTED_HEAD"' in text
    assert 'OUTPUT_NORM_STATS=$ICE_OUTPUT_DIR/norm_stats/norm_stats.json' in text
    assert 'die "copied norm_stats hash mismatch"' in text
