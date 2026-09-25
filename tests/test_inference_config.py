"""Model-owned inference artifact generation and compatibility checks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import (
    DIFFUSION_DENOISER,
    FLOW_DENOISER,
    build_inference_config,
    checkpoint_run_prefix,
    find_inference_config,
    load_inference_config,
    validate_inference_config,
    write_inference_config,
)


def training_config(
    *,
    family="flow",
    variant=None,
    horizon=100,
    action_dim=14,
    history_length=1,
):
    denoiser = {
        "_target_": FLOW_DENOISER,
        "action_horizon": horizon,
        "action_dim": action_dim,
        "num_inference_steps": 50,
    }
    if family == "diffusion":
        denoiser = {
            "_target_": DIFFUSION_DENOISER,
            "action_horizon": horizon,
            "action_dim": action_dim,
            "policy": {
                "num_inference_steps": 100,
                "noise_scheduler": {"num_train_timesteps": 100},
            },
        }
    payload = {
        "model": {
            "pipeline": {
                "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                "stages": [
                    {
                        "_target_": "egomimic.pipeline.stages_sampler.FusedObsEncoder",
                        "n_obs_steps": history_length,
                    },
                    {
                        "_target_": "egomimic.pipeline.stages_io.ActionTargetBuilder",
                        "action_key": "actions_cartesian",
                    },
                    denoiser,
                ],
            }
        },
        "evaluator": {"dt": 1 / 30},
        "inference_config": {
            "flow_inference_steps": 10,
            "diffusion_inference_steps": None,
            "max_flow_inference_steps": 100,
            "replan_every": 30,
            "action_dt": None,
        },
    }
    if variant is not None:
        payload["e1"] = {
            "variant": variant,
            "D": 0.4,
            "M": 100,
            "time_rows": 100,
            # The source window is intentionally independent of model output.
            "chunk_length": 400,
        }
    return OmegaConf.create(payload)


def test_time_flow_artifact_owns_history_output_and_runtime_controls():
    artifact = build_inference_config(training_config(history_length=2))

    assert artifact["status"] == "ready"
    graph = artifact["inference_graph"]
    assert graph["input"] == {"history_length": 2}
    assert graph["output"] == {
        "representation": "cartesian",
        "shape": [100, 14],
    }
    profile = graph["profiles"]["flow_time"]
    assert profile["native_shape"] == [100, 14]
    assert profile["adapter"] == {"decoder": None}
    assert profile["overrides"]["inference_steps"]["default"] == 10
    assert profile["overrides"]["replan_every"]["default"] == 30


@pytest.mark.parametrize(
    ("variant", "layout"),
    [
        ("arcvel", "e1_profile"),
        ("arcdur", "e1_dur"),
        ("arclogdur", "e1_logdur"),
    ],
)
def test_arc_artifact_decodes_native_tokens_to_fixed_cartesian_output(variant, layout):
    artifact = build_inference_config(training_config(variant=variant, action_dim=16))

    assert artifact["status"] == "ready"
    graph = artifact["inference_graph"]
    assert graph["output"]["shape"] == [100, 14]
    profile = graph["profiles"][f"flow_{variant}"]
    assert profile["native_shape"] == [100, 16]
    assert profile["adapter"]["decoder"] == {
        "_target_": "egomimic.robot.arc_decoder.BimanualArcDecoder",
        "token_layout": layout,
        "min_distance_unit": 0.4,
        "resampled_vector_length": 100,
        "dt": 1 / 30,
        "action_horizon": 100,
    }


def test_diffusion_artifact_targets_the_diffusion_policy_not_flow():
    artifact = build_inference_config(training_config(family="diffusion"))

    assert artifact["status"] == "ready"
    profile = artifact["inference_graph"]["profiles"]["diffusion_time"]
    control = profile["overrides"]["inference_steps"]
    assert control["label"] == "Diffusion denoising steps"
    assert control["default"] == 100
    assert control["target"] == {
        "kind": "stage_attribute",
        "attribute_path": "policy.num_inference_steps",
    }


def test_legacy_mean_timing_and_noncartesian_models_fail_closed():
    mean = build_inference_config(
        training_config(variant="arcmean", horizon=101, action_dim=14)
    )
    assert mean["status"] == "unsupported"
    assert "token-wide mean timing" in mean["reason"]

    planar = training_config()
    planar.model.pipeline.stages[1].action_key = "actions"
    artifact = build_inference_config(planar)
    assert artifact["status"] == "unsupported"
    assert "actions_cartesian" in artifact["reason"]


@pytest.mark.parametrize("mode", ["race", "multistream", "joint_distance"])
@pytest.mark.parametrize("saved_contract", [False, True])
def test_abc_arc_tokens_are_never_exported_as_cartesian_frames(mode, saved_contract):
    training = training_config(horizon=200)
    if saved_contract:
        training.run_provenance = {"action_contract": {
            "representation": "hybrid_arc_tokenizer_cartesian",
            "arc_chunking_mode": mode,
        }}
    else:
        training.abc = {
            "action_mode": "hybrid_arc_tokenizer_cartesian",
            "arc_chunking_mode": mode,
            "arc_velocity_mode": "per_waypoint",
        }
    artifact = build_inference_config(training)
    assert artifact["status"] == "unsupported"
    assert "ABC ARC" in artifact["reason"]
    assert "inference_graph" not in artifact


def test_same_shape_abc_codec_change_invalidates_cartesian_artifact():
    training = training_config(horizon=100)
    training.abc = {"action_mode": "cartesian", "arc_waypoints": 50}
    artifact = build_inference_config(training)
    assert artifact["status"] == "ready"
    changed = deepcopy(training)
    changed.abc.action_mode = "hybrid_arc_tokenizer_cartesian"
    with pytest.raises(ValueError, match="codec and inference defaults"):
        validate_inference_config(artifact, changed)


@pytest.mark.parametrize("mode", [None, "cartesian"])
def test_cartesian_abc_metadata_preserves_existing_sidecars_and_resume(tmp_path, mode):
    legacy = training_config()
    path = tmp_path / "inference-config.yaml"
    _, artifact = write_inference_config(legacy, path)
    current = deepcopy(legacy)
    current.abc = {"action_mode": mode, "arc_waypoints": 100}
    current.run_provenance = {"action_contract": {"representation": "cartesian"}}
    assert validate_inference_config(artifact, current)["output"]["shape"] == [100, 14]
    # Resuming a baseline must not trip the immutable-sidecar writer.
    assert write_inference_config(current, path)[1] == artifact


def test_artifact_is_bound_to_exact_pipeline_and_graph_content():
    training = training_config()
    artifact = build_inference_config(training)
    assert validate_inference_config(artifact, training)["output"]["shape"] == [
        100,
        14,
    ]

    changed_training = deepcopy(training)
    changed_training.model.pipeline.stages[-1].action_horizon = 99
    with pytest.raises(ValueError, match="does not match"):
        validate_inference_config(artifact, changed_training)

    arc_training = training_config(variant="arcdur", action_dim=16)
    arc_artifact = build_inference_config(arc_training)
    changed_codec = deepcopy(arc_training)
    changed_codec.e1.D = 0.75
    with pytest.raises(ValueError, match="codec and inference defaults"):
        validate_inference_config(arc_artifact, changed_codec)

    changed_artifact = deepcopy(artifact)
    changed_artifact["inference_graph"]["output"]["shape"] = [99, 14]
    with pytest.raises(ValueError, match="content hash"):
        validate_inference_config(changed_artifact, training)


def test_writer_is_idempotent_but_never_overwrites_another_model(tmp_path):
    path = tmp_path / "inference-config.yaml"
    training = training_config()

    write_inference_config(training, path)
    write_inference_config(training, path)
    assert load_inference_config(path, training)["output"]["shape"] == [100, 14]

    changed = training_config(horizon=50)
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        write_inference_config(changed, path)


def test_checkpoint_prefixed_artifact_takes_precedence_over_generic(tmp_path):
    checkpoint = tmp_path / "run_name__epoch-2399-step-240000__sha256-abcd.ckpt"
    checkpoint.touch()
    generic = tmp_path / "inference-config.yaml"
    generic.touch()
    prefixed = tmp_path / "run_name.inference-config.yaml"
    prefixed.touch()

    assert find_inference_config(checkpoint) == prefixed.resolve()
    assert checkpoint_run_prefix(checkpoint) == "run_name"


def test_train_defaults_emit_artifact_beside_checkpoints():
    path = (
        Path(__file__).parents[1] / "egomimic/hydra_configs/train_zarr_cartesian.yaml"
    )
    config = OmegaConf.load(path)
    settings = OmegaConf.to_container(config.inference_config, resolve=False)

    assert settings["enabled"] is True
    assert settings["output_path"].endswith("/checkpoints/inference-config.yaml")
