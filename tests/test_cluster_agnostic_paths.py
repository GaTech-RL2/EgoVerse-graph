"""Weight and cache paths come from the environment, not from tracked files."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIGS = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"


def test_abc_dit_weights_follow_env(monkeypatch):
    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/elsewhere/arc_abc")
    monkeypatch.setenv("EGOVERSE_DINOV3_CHECKPOINT", "/elsewhere/dinov3.pth")
    monkeypatch.setenv("EGOVERSE_CLIP_CHECKPOINT", "/elsewhere/clip")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=abc_arc/robot_bc/abc_rl2_towels_abcdit_multistream_visual_openloop",
                "++paths.root_dir=.",
            ],
        )
    assert cfg.abc_dit.backbone_checkpoint == "/elsewhere/dinov3.pth"
    assert cfg.abc_dit.text_checkpoint == "/elsewhere/clip"


def test_visualization_cache_roots_follow_env(monkeypatch):
    monkeypatch.setenv("EGOVERSE_ABC_DATASET_DIR", "/elsewhere/arc_abc")
    monkeypatch.delenv("EGOVERSE_ZARR_CACHE_DIR", raising=False)
    for name in ("rl2_organize_yam_bimanual", "rl2_organize_human_bimanual"):
        path = CONFIGS / "visualization" / f"{name}.yaml"
        roots = OmegaConf.to_container(OmegaConf.load(path), resolve=True)["cache_roots"]
        assert roots == ["/elsewhere/arc_abc", "/elsewhere/arc_abc"]
    monkeypatch.setenv("EGOVERSE_ZARR_CACHE_DIR", "/elsewhere/zarr")
    roots = OmegaConf.to_container(
        OmegaConf.load(CONFIGS / "visualization/rl2_organize_yam_bimanual.yaml"), resolve=True
    )["cache_roots"]
    assert roots == ["/elsewhere/arc_abc", "/elsewhere/zarr"]
