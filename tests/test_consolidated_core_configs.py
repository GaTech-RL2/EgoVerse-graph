from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"


def _compose(row, *extra):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{row}", "++paths.root_dir=.", *extra],
        )


@pytest.mark.parametrize(
    "row",
    (
        "action_flow_bc_usocket_latent_fm_sg_recon1_200m_adamw_lr1e5_s42",
        "action_flow_bc_usocket_latent_fm_sg_recon1_200m_muon_lr1e5_s42",
        "action_flow_usocket_latent_fm_sg_unite_h384_s42",
        "action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42",
        "action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_scale1_s42",
        "action_flow_chain_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42",
    ),
)
def test_action_flow_core_rows_compose_with_portable_dataset_root(
    row, monkeypatch
):
    root = "/verified/cluster/datasets/Tsim_v2"
    monkeypatch.setenv("PUSHSHAPES_DATA_ROOT", root)

    cfg = _compose(row)

    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        for dataset in split.values():
            assert str(dataset.resolver.folder_path).startswith(f"{root}/")
    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.model.training_behavior._target_.endswith(
        "ActionFlowTrainingBehavior"
    )
    assert cfg.model.action_horizon == 16
    assert cfg.model.flow_samples_per_content == 14


def test_released_unite_cotrain_core_row_composes(monkeypatch):
    root = "/verified/cluster/datasets/Tsim_v2"
    monkeypatch.setenv("PUSHSHAPES_DATA_ROOT", root)

    cfg = _compose(
        "unite_cotrain_usocket_chain_val01_h16",
        "model=bf/ct_unite_register_separate_nt8_h384_s42",
    )

    assert tuple(cfg.data.train_datasets) == (
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    )
    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        for dataset in split.values():
            assert str(dataset.resolver.folder_path).startswith(f"{root}/")
    assert cfg.model.num_latent_tokens == 8
    assert cfg.model.hidden_dim == 384
    assert cfg.model.action_horizon == 16
    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.model.training_behavior._target_.endswith(
        "ReleasedUniteTrainingBehavior"
    )
