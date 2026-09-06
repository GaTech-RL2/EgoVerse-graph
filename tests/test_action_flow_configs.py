from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.pushshapes import (
    get_usocket_rotvec_action_transform_list,
)

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"


def _compose(row: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{row}", "++paths.root_dir=."],
        )
    return cfg


def test_action_only_adapter_preserves_native_state_and_encodes_target_angle():
    state = np.array([1.0, -2.0, 0.7], dtype=np.float32)
    actions = np.array([[3.0, 4.0, 0.0], [5.0, 6.0, np.pi / 2]], dtype=np.float32)
    sample = {"state_agent_obj": state.copy(), "actions": actions.copy()}

    for transform in get_usocket_rotvec_action_transform_list():
        sample = transform.transform(sample)

    np.testing.assert_array_equal(sample["state_agent_obj"], state)
    assert sample["actions"].shape == (2, 4)
    np.testing.assert_allclose(sample["actions"][:, :2], actions[:, :2])
    np.testing.assert_allclose(
        sample["actions"][:, 2:], [[1.0, 0.0], [0.0, 1.0]], atol=1e-6
    )


@pytest.mark.parametrize(
    ("row", "reconstruction_weight"),
    [
        ("action_flow_bc_usocket_recon1_s42", 1.0),
        ("action_flow_bc_usocket_recon10_s42", 10.0),
    ],
)
def test_action_flow_pair_composes_to_exact_shared_contract(
    row: str, reconstruction_weight: float
):
    cfg = _compose(row)
    stages = cfg.model.pipeline.stages
    names = [stage._target_.rsplit(".", 1)[-1] for stage in stages]

    assert names == [
        "FusedObsEncoder",
        "GaussianLatentNoise",
        "ActionTargetBuilder",
        "ContentEncoderStage",
        "LatentBridgeStage",
        "ConditionalVelocityStage",
        "ContentDecoderStage",
        "ActionFlowObjectiveStage",
    ]
    assert cfg.model._target_.endswith("ActionFlowModelWrapper")
    assert cfg.model.action_horizon == 16
    assert cfg.model.action_dim == 4
    assert cfg.model.latent_dim == 8
    assert cfg.model.condition_dim == 67
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.condition_dropout_probability == pytest.approx(0.3)
    assert cfg.model.num_inference_steps == 16
    assert cfg.model.reconstruction_weight == reconstruction_weight
    assert stages[-1].flow_weight == 1.0
    assert stages[-1].reconstruction_weight == reconstruction_weight
    assert stages[-1].action_velocity_weight == 1.0

    field = stages[5].field
    assert field.horizon == 16
    assert field.input_dim == field.output_dim == 8
    assert field.condition_dim == 67
    assert field.hidden_dim == 512
    assert field.depth == 12
    assert field.num_heads == 8
    assert field.feedforward_dim == 2048
    assert field.time_scale == pytest.approx(1000.0)
    assert field.condition_dropout_probability == pytest.approx(0.3)

    encoder = stages[3].encoder
    decoder = stages[6].decoder
    assert encoder.horizon == decoder.horizon == 16
    assert encoder.input_dim == decoder.output_dim == 4
    assert encoder.latent_dim == decoder.latent_dim == 8
    assert encoder.hidden_dim == decoder.hidden_dim == 20
    assert encoder.depth == decoder.depth == 2

    dataset = cfg.data.train_datasets.pushshapes_sim_u_socket
    valid = cfg.data.valid_datasets.pushshapes_sim_u_socket
    assert dataset.resolver.key_map._target_.endswith("get_planar_keymap")
    assert dataset.resolver.transform_list._target_.endswith(
        "get_usocket_rotvec_action_transform_list"
    )
    assert dataset.valid_ratio == valid.valid_ratio == 0.01
    assert dataset.split_seed == valid.split_seed == 42
    assert dataset.expected_train_episode_count == 2970
    assert dataset.expected_valid_episode_count == 29
    assert cfg.run_provenance.split_manifest_sha256 == (
        "3683e3461596eef8df2432fa865779b3c77b2a2057dabd0fea125595729cf313"
    )
    assert cfg.run_provenance.content_manifest_path == (
        "egomimic/hydra_configs/data/pusht/manifests/"
        "usocket_3000_v2_clean_content_v1.json"
    )
    assert cfg.run_provenance.content_manifest_sha256 == (
        "a1c81fb0ce8967aba795383a293180f9ba08a0ecfdd6f4a878afb20b39733761"
    )
    assert cfg.run_provenance.dataset_content_aggregate_sha256 == (
        "80f835ad37c3d5c5b7b2d5c3e1656c307ee567a1f63f51081165bf404b8ceb52"
    )
    assert cfg.evaluator.energy_score_distance.type == (
        "usocket_normalized_xy_wrapped_theta_v1"
    )
    assert list(
        cfg.evaluator.energy_score_distance.complete_normalized_chunk_shape
    ) == [16, 4]
    assert cfg.evaluator.energy_score_distance.rotation_scale_radians == pytest.approx(
        math.pi
    )
    assert dict(cfg.evaluator.energy_score_distance.semantic_weights) == {
        "translation": 0.5,
        "rotation": 0.5,
    }
    assert OmegaConf.to_container(
        cfg.run_provenance.energy_score_contract.distance, resolve=True
    ) == OmegaConf.to_container(cfg.evaluator.energy_score_distance, resolve=True)
    assert OmegaConf.to_container(
        cfg.evaluator.energy_score_provenance.dataset_content, resolve=True
    ) == {
        "manifest_path": cfg.run_provenance.content_manifest_path,
        "manifest_sha256": cfg.run_provenance.content_manifest_sha256,
        "aggregate_sha256": cfg.run_provenance.dataset_content_aggregate_sha256,
    }
    assert cfg.evaluator.action_flow_diagnostics.native_error.type == (
        "usocket_native_xy_wrapped_theta_mse_v1"
    )
    diagnostic_provenance = cfg.evaluator.action_flow_diagnostics.provenance
    assert diagnostic_provenance.source_commit == cfg.run_provenance.source_commit
    assert diagnostic_provenance.normalization_sha256 == (
        cfg.run_provenance.normalization_sha256
    )
    assert diagnostic_provenance.split_manifest_sha256 == (
        cfg.run_provenance.split_manifest_sha256
    )
    assert diagnostic_provenance.dataset_content.manifest_sha256 == (
        cfg.run_provenance.content_manifest_sha256
    )
    assert diagnostic_provenance.dataset_content.aggregate_sha256 == (
        cfg.run_provenance.dataset_content_aggregate_sha256
    )

    assert cfg.launch_params.gpus_per_node == 1
    assert cfg.launch_params.nodes == 1
    assert cfg.trainer.max_steps == 240000
    assert cfg.trainer.val_check_interval == 10000
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 40000
    assert cfg.model.optimizer.lr == pytest.approx(3e-5)
    assert list(cfg.model.optimizer.betas) == [0.9, 0.999]
    assert cfg.model.optimizer.eps == pytest.approx(1e-8)
    assert cfg.model.optimizer.weight_decay == pytest.approx(1e-4)
    assert cfg.model.scheduler.warmup_steps == 8000
    assert cfg.model.scheduler.warmup_start_factor == pytest.approx(0.1)
    assert cfg.model.scheduler.eta_min == pytest.approx(3e-6)

    yaml = OmegaConf.to_yaml(cfg.model)
    forbidden = (
        "Unite",
        "UNITE",
        "CrossTransformer",
        "DiffusionNoisingStage",
        "embodiment",
        "domains:",
        "ac_keys",
        "monotonic",
        "scale_weight",
        "cfg_scale",
    )
    assert not any(token in yaml for token in forbidden)


def test_only_reconstruction_weight_differs_between_initial_pair():
    left = _compose("action_flow_bc_usocket_recon1_s42")
    right = _compose("action_flow_bc_usocket_recon10_s42")
    left_model = OmegaConf.to_container(left.model, resolve=True)
    right_model = OmegaConf.to_container(right.model, resolve=True)

    assert left_model["reconstruction_weight"] == 1.0
    assert right_model["reconstruction_weight"] == 10.0
    left_model["reconstruction_weight"] = right_model["reconstruction_weight"]
    left_model["pipeline"]["stages"][-1]["reconstruction_weight"] = right_model[
        "pipeline"
    ]["stages"][-1]["reconstruction_weight"]
    assert left_model == right_model
