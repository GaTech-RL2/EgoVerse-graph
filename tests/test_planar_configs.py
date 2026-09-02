from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROWS = {
    "usocket_arc_bc": ("pushshapes_sim_u_socket", 101, 1, "PlanarFlowSampler"),
    "chain_arc_bc": ("pushshapes_sim_chain_gripper", 101, 1, "PlanarFlowSampler"),
    "usocket_direct_bc": ("pushshapes_sim_u_socket", 16, 1, "PlanarFlowSampler"),
    "chain_direct_bc": ("pushshapes_sim_chain_gripper", 16, 1, "PlanarFlowSampler"),
    "usocket_dp_standard": ("pushshapes_sim_u_socket", 16, 1, "ConditionalUnet1D"),
    "chain_dp_standard": ("pushshapes_sim_chain_gripper", 16, 1, "ConditionalUnet1D"),
    "usocket_dp_paper": ("pushshapes_sim_u_socket", 16, 2, "PaperConditionalUnet1D"),
    "chain_dp_paper": ("pushshapes_sim_chain_gripper", 16, 2, "PaperConditionalUnet1D"),
}


@pytest.mark.parametrize("row,expected", ROWS.items())
def test_planar_row_composes_to_one_pinned_domain(row, expected):
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/planar_v2_{row}", "++paths.root_dir=."],
        )
    for section in (cfg.planar, cfg.model, cfg.data):
        OmegaConf.resolve(section)
    domain, horizon, observation_horizon, implementation = expected
    assert cfg.planar.domain == domain
    assert cfg.model.robomimic_model.action_horizon == horizon
    assert list(cfg.data.train_datasets) == [domain]
    dataset = cfg.data.train_datasets[domain]
    assert dataset.valid_ratio == cfg.data.valid_datasets[domain].valid_ratio == 0.01
    assert dataset.split_seed == cfg.seed == 42
    assert dataset.expected_train_episode_count == 990
    assert dataset.expected_valid_episode_count == 10
    assert dataset.resolver.expected_episode_count == 1000
    assert cfg.planar.observation_horizon == observation_horizon
    assert implementation in OmegaConf.to_yaml(cfg.model)
    if "dp_paper" in row:
        assert cfg.callbacks.ema.use_warmup is True
        assert "DDPMScheduler" in OmegaConf.to_yaml(cfg.model)
    elif "dp_standard" in row:
        assert "DDIMScheduler" in OmegaConf.to_yaml(cfg.model)
        assert "ema" not in cfg.callbacks


def test_runtime_targets_are_declared_dependencies():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert '"diffusers==0.30.3"' in pyproject
