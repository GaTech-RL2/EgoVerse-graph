"""Guard the corrected manual standard + obstacle-gen ChainGripper recipes."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.pipeline.pushshapes import ChainGripperPointsNativeDecoder


CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
ROWS = (
    "action_flow_chain_manual4919_avln_80k_s42",
    "action_flow_cotrain_uc_manual4919_avln_80k_s42",
)


@pytest.mark.parametrize("row", ROWS)
def test_manual4919_recipe_uses_effective_split_and_cadence(monkeypatch, row):
    monkeypatch.setenv("PUSHSHAPES_USOCKET_ROOT", "/verified/usocket-clean2999")
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/verified/manual4919-view")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{row}"],
        )
    chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper
    valid = cfg.data.valid_datasets.pushshapes_sim_chain_gripper
    assert "data" not in cfg.data
    assert set(cfg.data.train_datasets) == (
        {"pushshapes_sim_chain_gripper"}
        if row.startswith("action_flow_chain_")
        else {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
    )
    for dataset in (chain, valid):
        assert dataset.resolver.folder_path == "/verified/manual4919-view"
        assert dataset.resolver.expected_episode_count == 4919
        assert dataset.expected_train_episode_count == 4870
        assert dataset.expected_valid_episode_count == 49
        assert dataset.valid_ratio == 0.01
        assert dataset.split_seed == 42
    assert cfg.run_provenance.chain_collection.standard_manual_episodes == 3000
    assert cfg.run_provenance.chain_collection.obstacle_gen_manual_episodes == 1919
    assert cfg.run_provenance.chain_collection.excluded_frozen_mixed_view
    assert cfg.trainer.max_steps == 80000
    assert cfg.trainer.val_check_interval == 15000
    assert cfg.val_at_end
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 5000
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert cfg.evaluator.action_flow_diagnostics.native_error is None


@pytest.mark.parametrize("row", ROWS)
def test_manual4919_native_evaluator_decoder_instantiates(monkeypatch, row):
    """Catch inherited Hydra kwargs before an allocated GPU smoke starts."""
    monkeypatch.setenv("PUSHSHAPES_USOCKET_ROOT", "/verified/usocket-clean2999")
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/verified/manual4919-view")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{row}"],
        )
    decoder_cfg = (
        cfg.evaluator.native_decoder
        if row.startswith("action_flow_chain_")
        else cfg.evaluator.native_decoders.pushshapes_sim_chain_gripper
    )
    decoder = instantiate(decoder_cfg)
    assert isinstance(decoder, ChainGripperPointsNativeDecoder)
    if row.startswith("action_flow_chain_"):
        assert decoder_cfg.action_horizon == 16
        assert decoder_cfg.native_action_dim == 4


def test_chain_native_decoder_rejects_wrong_native_shape():
    with pytest.raises(ValueError, match="native_action_dim=4"):
        ChainGripperPointsNativeDecoder(native_action_dim=6)
