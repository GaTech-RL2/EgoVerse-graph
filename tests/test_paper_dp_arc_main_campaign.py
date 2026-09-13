from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).parents[1]


def _compose(experiment: str):
    config_dir = ROOT / "egomimic" / "hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}"],
        )


def test_arc_campaign_encodes_validation_targets_with_training_codec():
    cfg = _compose(
        "pusht/planar_v2_cotrain_obstacle_arc_duration_D80_M16_R26deg_paper"
    )
    encoder = OmegaConf.to_container(cfg.evaluator.target_encoder, resolve=True)
    stage = OmegaConf.to_container(cfg.model.pipeline.stages[1], resolve=True)
    assert encoder == stage


def test_campaign_smoke_verifier_and_launcher_preserve_gate_artifacts():
    verifier = (ROOT / "scripts/train/verify_planar_paper_dp_smoke.py").read_text()
    launcher = (
        ROOT / "scripts/ice/launch_planar_v2_paper_dp_arc_cotrain.sbatch"
    ).read_text()
    assert 'artifact["seed_bank"]' in verifier
    assert 'mkdir -p "$RUN_DIR/norm_stats"' in launcher
    assert 'cp --reflink=auto "$PRECOMPUTED_NORM_DIR/norm_stats.json"' in launcher
    assert 'sha256sum "$RUN_DIR/norm_stats/norm_stats.json"' in launcher
