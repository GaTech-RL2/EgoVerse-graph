from omegaconf import OmegaConf

from egomimic.trainHydra import _build_model_config_tree


def test_model_config_tree_preserves_checkpoint_identity_provenance():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "example.ModelWrapper", "latent_dim": 8},
            "logger": {"wandb": {"id": "action-flow-smoke-v1"}},
            "run_provenance": {
                "source_commit": "a" * 40,
                "split_manifest_sha256": "b" * 64,
                "normalization_sha256": "c" * 64,
            },
        }
    )

    config_tree = _build_model_config_tree(cfg)

    assert config_tree.model.latent_dim == 8
    assert config_tree.run_provenance.source_commit == "a" * 40
    assert config_tree.run_provenance.split_manifest_sha256 == "b" * 64
    assert config_tree.run_provenance.normalization_sha256 == "c" * 64
    assert config_tree.run_provenance.run_id == "action-flow-smoke-v1"


def test_model_config_tree_keeps_provenance_optional():
    cfg = OmegaConf.create({"model": {"_target_": "example.ModelWrapper"}})

    config_tree = _build_model_config_tree(cfg)

    assert set(config_tree) == {"model"}
