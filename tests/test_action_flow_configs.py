from __future__ import annotations

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
        ("action_flow_bc_usocket_recon100_s42", 100.0),
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
    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.model.training_behavior._target_ == (
        "egomimic.pl_utils.training_behavior_action_flow.ActionFlowTrainingBehavior"
    )
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


def test_only_reconstruction_weight_differs_across_sweep():
    reference = OmegaConf.to_container(
        _compose("action_flow_bc_usocket_recon1_s42").model, resolve=True
    )
    for suffix, expected_weight in (("10", 10.0), ("100", 100.0)):
        candidate = OmegaConf.to_container(
            _compose(f"action_flow_bc_usocket_recon{suffix}_s42").model,
            resolve=True,
        )
        assert candidate["reconstruction_weight"] == expected_weight
        candidate["reconstruction_weight"] = reference["reconstruction_weight"]
        candidate["pipeline"]["stages"][-1]["reconstruction_weight"] = reference[
            "pipeline"
        ]["stages"][-1]["reconstruction_weight"]
        assert candidate == reference


def test_recon10_warmup_row_changes_only_the_objective_schedule():
    reference_cfg = _compose("action_flow_bc_usocket_recon10_s42")
    candidate_cfg = _compose("action_flow_bc_usocket_recon10_warmup10k_s42")
    reference = OmegaConf.to_container(reference_cfg, resolve=False)
    candidate = OmegaConf.to_container(candidate_cfg, resolve=False)

    assert candidate["model"]["reconstruction_only_warmup_steps"] == 10000
    assert (
        candidate_cfg.run_provenance.objective.reconstruction_only_warmup_steps == 10000
    )
    del candidate["model"]["reconstruction_only_warmup_steps"]
    assert (
        candidate["run_provenance"]["objective"].pop("reconstruction_only_warmup_steps")
        == "${model.reconstruction_only_warmup_steps}"
    )
    candidate["name"] = reference["name"]
    candidate["description"] = reference["description"]
    assert candidate == reference
