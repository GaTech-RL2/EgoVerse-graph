"""A deployment bundle is inseparable from its model and complete data state."""

import json
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.pipeline.construction import checkpoint_construction, restoring_parameters
from egomimic.pipeline.inference_config import write_inference_config
from egomimic.pipeline.inference_session import InferenceSession, load_bound_graph
from egomimic.pl_utils.data_context import DataContext, serializable_state
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.data_module import _digest, load_data_context
from tests.test_robot_graph_policy import (
    ACTION,
    PROPRIO,
    declare_test_model,
    normalizer,
)


def bundle(tmp_path, *, initialization=False):
    training = declare_test_model(
        OmegaConf.create(
            {
                "model": {
                    "pipeline": {
                        "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                        "device": "cpu",
                        "stages": [
                            {"_target_": "tests.test_robot_graph_policy.EchoStage"}
                        ],
                    }
                },
                "data": {"_target_": "must.never.be.instantiated"},
            }
        )
    )
    if initialization:
        training.model.pipeline.initialization = [
            {
                "stage_id": "sampler",
                "source": str(tmp_path / "unavailable.pt"),
                "sha256": "0" * 64,
            }
        ]
    norm = normalizer()
    state = {
        "kind": "zarr-normalizer-v1",
        "normalizer_state": norm.to_state(),
        "sha256": _digest(norm.to_state()),
        "preprocessing": {"7": {"frame": "fixture_frame", "source_fps": 30}},
    }
    context = DataContext(norm, norm.shapes, (), state)
    with checkpoint_construction():
        wrapper = ModelWrapper(config_tree=training)
        context.bind(wrapper.model)
    wrapper.data_context = context
    checkpoint = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    context_path = tmp_path / "data-context.json"
    context_path.write_text(json.dumps(serializable_state(state)))
    artifact_path = tmp_path / "inference-config.yaml"
    write_inference_config(training, artifact_path, data_context=context)
    paths = dict(
        checkpoint_path=path, context_path=context_path, artifact_path=artifact_path
    )
    return training, context, checkpoint, paths


def test_bound_sequence_inference_and_robot_share_checkpoint_without_hardware(tmp_path):
    training, context, checkpoint, paths = bundle(tmp_path, initialization=True)
    session = InferenceSession.load(training, identity=7, **paths)
    observations = {
        PROPRIO: torch.full((1, 14), 4.0),
        "front": torch.zeros(1, 3, 2, 3),
        "embodiment": torch.tensor([7]),
    }
    prediction = session.predict(observations)
    torch.testing.assert_close(
        prediction,
        context.normalizer.unnormalize({ACTION: torch.zeros(1, 2, 14)}, 7)[ACTION],
    )
    torch.testing.assert_close(
        session.graph.pipeline.stages[0].seen, torch.ones(1, 14), atol=1e-6, rtol=1e-6
    )
    assert session.execution_plan(prediction).shape == (1, 1, 14)
    session.apply_inference_overrides({"replan_every": 2})
    assert session.execution_plan(prediction).shape == (1, 2, 14)
    with pytest.raises((ValueError, TypeError)):
        session.apply_inference_overrides({"replan_every": True})
    assert session.replan_every == 2
    assert not restoring_parameters()
    from egomimic.robot.graph_policy import load_graph_policy
    from tests.test_robot_runtime import FakeRobot

    config_path = tmp_path / "training.yaml"
    OmegaConf.save(training, config_path)
    robot_policy = load_graph_policy(
        {
            "training_config": str(config_path),
            "checkpoint": str(paths["checkpoint_path"]),
            "normalizer_path": str(paths["context_path"]),
            "device": "cpu",
            "adapter": {
                "_target_": "egomimic.robot.graph_policy.CartesianGraphAdapter",
                "base_T_model": {
                    arm: torch.eye(4).tolist() for arm in ("left", "right")
                },
                "camera_keys": {"front_img_1": "front"},
                "embodiment_id": 7,
                "rotation_mode": "euler",
                "action_frame": "model_frame",
                "image_hw": [2, 3],
            },
        }
    )
    torch.testing.assert_close(
        torch.from_numpy(robot_policy.predict(FakeRobot().get_obs())).float(),
        prediction[0],
    )


@pytest.mark.parametrize(
    "damage",
    [
        "weights_model",
        "stats",
        "preprocessing",
        "missing_binding",
        "unbound_artifact",
        "checkpoint_context",
    ],
)
def test_corrupt_or_mismatched_bundle_fails_before_model_construction(
    tmp_path, monkeypatch, damage
):
    training, context, checkpoint, paths = bundle(tmp_path)
    if damage == "weights_model":
        checkpoint["inference_binding"]["model_pipeline_sha256"] = "0" * 64
    elif damage == "missing_binding":
        del checkpoint["inference_binding"]
    elif damage == "checkpoint_context":
        checkpoint["data_context"]["preprocessing"]["7"]["frame"] = "other"
    elif damage == "unbound_artifact":
        paths["artifact_path"] = tmp_path / "unbound.yaml"
        write_inference_config(training, paths["artifact_path"])
    else:
        changed = serializable_state(context.snapshot())
        if damage == "stats":
            changed["normalizer_state"]["norm_stats"]["7"][ACTION]["mean"][0] += 1
            changed["sha256"] = _digest(changed["normalizer_state"])
        else:
            changed["preprocessing"]["7"]["frame"] = "other"
        paths["context_path"].write_text(json.dumps(changed))
    torch.save(checkpoint, paths["checkpoint_path"])
    import egomimic.pipeline.inference_session as module

    original = module.instantiate

    def guarded(config, **kwargs):
        assert (
            config.get("_target_") == "egomimic.rldb.zarr.data_module.load_data_context"
        ), "Constructed a model before validating its binding"
        return original(config, **kwargs)

    monkeypatch.setattr(module, "instantiate", guarded)
    with pytest.raises(ValueError, match="binding|bound|verified"):
        load_bound_graph(training, **paths)


def test_resume_verifies_full_context_and_context_serialization_is_exact(tmp_path):
    training, context, checkpoint, paths = bundle(tmp_path)
    assert (
        context.fingerprint() == load_data_context(paths["context_path"]).fingerprint()
    )
    wrapper = ModelWrapper(config_tree=training)
    wrapper.data_context = context
    wrapper.on_load_checkpoint(checkpoint)
    changed = deepcopy(checkpoint)
    changed["data_context"]["preprocessing"]["7"]["frame"] = "other"
    with pytest.raises(ValueError, match="data context"):
        wrapper.on_load_checkpoint(changed)


def test_restore_construction_is_nested_and_resets_after_failure():
    assert not restoring_parameters()
    with pytest.raises(RuntimeError), checkpoint_construction():
        assert restoring_parameters()
        with checkpoint_construction(enabled=False):
            assert not restoring_parameters()
        assert restoring_parameters()
        raise RuntimeError("deliberate")
    assert not restoring_parameters()
