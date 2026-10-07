"""The robot decoder must use the same hybrid codec as the training target."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import (
    build_inference_config,
    validate_inference_config,
)
from egomimic.rldb.zarr import e1_arc_tokenizer
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1
from egomimic.robot.arc_decoder import BimanualArcDecoder


def _actions():
    rows = np.arange(100)
    actions = np.zeros((100, 14))
    actions[:, 0] = 0.2 * np.clip(rows / 30, 0, 1)
    actions[:, 3] = np.pi / 3 * np.clip((rows - 30) / 40, 0, 1)
    actions[:, 6] = 1 - np.clip((rows - 30) / 40, 0, 1)
    actions[:, 7:10] = [0.1, -0.3, 0.2]
    actions[:, 13] = np.clip((rows - 20) / 40, 0, 1)
    return actions


@pytest.mark.parametrize(
    "layout,mode", [("e1_durhyb", "durhyb"), ("e1_profhyb", "profhyb")]
)
@pytest.mark.parametrize("rotation_distance", [2 * np.pi, np.pi / 6])
def test_robot_hybrid_decode_matches_training_codec(layout, mode, rotation_distance):
    codec = TokenizeBimanualArcLengthE1(
        action_key="action",
        output_action_key="action",
        min_distance_unit=0.4,
        resampled_vector_length=100,
        dt=1 / 30,
        velocity_norm="path",
        velocity_mode=mode,
        rotation_distance_unit=rotation_distance,
    )
    token = codec.transform({"action": _actions()})["action"]
    decoder = BimanualArcDecoder(
        token_layout=layout,
        rotation_distance_unit=rotation_distance,
    )
    assert token.shape == decoder.shape == (100, 18)
    assert decoder.codec.rotation_distance_unit == rotation_distance
    np.testing.assert_allclose(
        decoder(token[None])[0], codec.detokenize(token, action_horizon=100), atol=1e-12
    )


@pytest.mark.parametrize("layout", ["e1_durhyb", "e1_profhyb"])
def test_hybrid_decoder_rejects_plain_timing_shape(layout):
    with pytest.raises(ValueError, match="Expected native tokens"):
        BimanualArcDecoder(token_layout=layout)(np.zeros((1, 100, 16)))


@pytest.mark.parametrize("rotation_distance", [0, -1, np.inf, np.nan])
def test_hybrid_decoder_rejects_invalid_rotation_budget(rotation_distance):
    with pytest.raises(ValueError, match="finite and positive"):
        BimanualArcDecoder(
            token_layout="e1_durhyb", rotation_distance_unit=rotation_distance
        )


def _training(variant):
    contract_path = (
        Path(e1_arc_tokenizer.__file__).parents[2]
        / "hydra_configs/model_contract/e1_wrists.yaml"
    )
    contract = OmegaConf.load(contract_path)
    return OmegaConf.merge(
        {"model": contract},
        {
            "e1": {
                "variant": variant,
                "policy_domain": "yam",
                "M": 100,
                "D": 0.4,
                "time_rows": 100,
                "chunk_length": 100,
                "velocity_norm": "path",
                "fixed_spacing": False,
                "progress_smooth_hz": None,
            },
            "model": {
                "action_horizon": 100,
                "action_token_dim": 18,
                "rotation_distance_unit": 2 * np.pi,
                "sampling_steps": 50,
                "sampling_attribute": "num_inference_steps",
                "pipeline": {
                    "stages": [{"_target_": "test.Sampler"}],
                    "stage_ids": {"sampler": 0},
                },
            },
        },
    )


@pytest.mark.parametrize(
    "variant,layout", [("arcdurhyb", "e1_durhyb"), ("arcvelhyb", "e1_profhyb")]
)
def test_hybrid_model_owned_contract_exports_native_and_canonical_shapes(
    variant, layout
):
    training = _training(variant)
    artifact = build_inference_config(training)
    assert artifact["status"] == "ready"
    graph = validate_inference_config(artifact, training)
    assert graph["native_output"]["shape"] == [100, 18]
    assert graph["output"]["shape"] == [100, 14]
    assert graph["compatibility"]["normalizer_schema"]["native_shape"] == [100, 18]
    decoder = graph["profiles"]["default"]["adapter"]["decoder"]
    assert decoder["token_layout"] == layout
    assert decoder["rotation_distance_unit"] == 2 * np.pi


def test_inference_artifact_binds_hybrid_rotation_budget():
    training = _training("arcdurhyb")
    artifact = build_inference_config(training)
    training.model.rotation_distance_unit = np.pi / 6
    with pytest.raises(ValueError, match="codec and inference defaults"):
        validate_inference_config(artifact, training)
