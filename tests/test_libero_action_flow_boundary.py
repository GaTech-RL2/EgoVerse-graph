import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from egomimic.eval.libero_action_flow_eval import LiberoActionFlowEvaluator
from egomimic.pipeline.libero_action_flow import LiberoActionFlowObservationAdapter
from egomimic.pl_utils.training_behavior_action_flow import ActionFlowTrainingBehavior
from egomimic.rldb.zarr.libero_action_flow import LiberoActionFlowDataset


def _native_profile(seed_file, tmp_path):
    """Explicit synthetic constructor inputs, not a native run receipt."""
    from egomimic.benchmarks.libero.native_diagnostic_config import (
        build_native_diagnostic_config,
    )

    return build_native_diagnostic_config(
        {
            "energy_seed_bank_sha256": hashlib.sha256(
                seed_file.read_bytes()
            ).hexdigest(),
            "split_sha256": "synthetic-boundary-test",
        },
        str(tmp_path / "diagnostics"),
        str(seed_file),
        {
            "num_latent_tokens": 8,
            "latent_dim": 16,
            "num_inference_steps": 50,
            "pipeline": {
                "stages": [
                    {
                        "_target_": "fixture.ContentEncoderStage",
                        "encoder": {"backbone": {"depth": 12}},
                    },
                    {
                        "_target_": "fixture.ConditionalVelocityStage",
                        "field": {"backbone": {"depth": 12}},
                        "inference_method": "euler",
                    },
                ]
            },
        },
    )


def test_libero_action_flow_recipe_selects_effective_train_and_validation_data(
    monkeypatch,
):
    monkeypatch.setenv("LIBERO10_REPLAY_ROOT", "/tmp/libero10-replay-for-compose")
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=["+experiment=libero_historical/action_flow_libero10_avln_80k_s42"],
        )
    assert "data" not in cfg.data
    assert cfg.data.train_datasets.libero_panda._target_.endswith(
        "libero_action_flow.LiberoActionFlowDataset._from_resolver"
    )
    assert cfg.data.valid_datasets.libero_panda._target_.endswith(
        "libero_action_flow.LiberoActionFlowDataset._from_resolver"
    )
    assert cfg.data.train_datasets.libero_panda.valid_ratio == 0.01
    assert cfg.data.valid_datasets.libero_panda.valid_ratio == 0.01
    assert cfg.data.train_datasets.libero_panda.split_seed == 42
    assert cfg.data.valid_datasets.libero_panda.split_seed == 42


def test_logical_replay_keys_are_normalized_and_invertible():
    dataset = LiberoActionFlowDataset.__new__(LiberoActionFlowDataset)
    dataset.norm_stats = {
        21: {
            "actions": {
                "scale": [2.0] * 7,
                "offset": [-1.0] * 7,
            }
        }
    }
    raw = torch.full((2, 16, 7), 0.25)
    task = torch.tensor([[2.0], [5.0]])
    normalized = dataset.normalize({"actions": raw, "task_uid": task}, 21)
    assert torch.allclose(normalized["actions"], torch.full_like(raw, -0.5))
    assert torch.equal(normalized["task_uid"], task)
    recovered = dataset.unnormalize(normalized, 21)
    assert torch.allclose(recovered["actions"], raw)


def test_libero_observation_adapter_shapes_and_integer_task_identity():
    adapter = LiberoActionFlowObservationAdapter()
    batch = {
        "agentview_rgb": torch.zeros(2, 1, 128, 128, 3),
        "robot0_eye_in_hand_rgb": torch.zeros(2, 1, 128, 128, 3),
        "robot0_eef_pos": torch.zeros(2, 1, 3),
        "robot0_eef_quat": torch.zeros(2, 1, 4),
        "robot0_gripper_qpos": torch.zeros(2, 1, 2),
        "task_uid": torch.tensor([[1.0], [2.0]]),
    }
    result = adapter(batch)
    assert result["front_img_1"].shape == (2, 3, 96, 96)
    assert result["front_img_2"].shape == (2, 3, 96, 96)
    assert result["proprio_condition"].shape == (2, 64)
    batch["task_uid"] = torch.tensor([[1.5], [2.0]])
    try:
        adapter(batch)
    except ValueError as error:
        assert "unnormalized integer IDs" in str(error)
    else:
        raise AssertionError("fractional task IDs must fail closed")


def test_libero_validation_dispatches_generic_action_flow_diagnostic(
    tmp_path, monkeypatch
):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({"seeds": list(range(32))}))

    class IdentityNormalizer:
        def unnormalize(self, values, embodiment):
            del embodiment
            return values

    class GenericModel:
        global_rank = current_epoch = global_step = 0
        trainer = SimpleNamespace(precision="32-true")

        def __init__(self):
            self.logged = []
            self.diagnostic_calls = []

        def forward_eval(self, batch):
            return {
                source: {"pred_action": values["actions"]}
                for source, values in batch.items()
            }

        def run_diagnostic(self, capability, batch, **kwargs):
            self.diagnostic_calls.append((capability, kwargs))
            return {
                source: {
                    "schema": "action-flow-validation-diagnostics/v1",
                    "latent/clean": torch.ones(2, 8, 16),
                    "decoded/reconstruction": torch.ones(2, 16, 7),
                }
                for source in batch
            }

        def log(self, name, value, **kwargs):
            self.logged.append((name, value, kwargs))

    evaluator = LiberoActionFlowEvaluator(
        energy_seed_bank_path=seed_file,
        energy_seed_bank_sha256=hashlib.sha256(seed_file.read_bytes()).hexdigest(),
        native_diagnostic_config=_native_profile(seed_file, tmp_path),
    )
    # This test isolates wrapper/evaluator dispatch; it does not certify the
    # native model's owner topology. Numeric analyzer tests are separate.
    owner_calls = []
    monkeypatch.setattr(
        "egomimic.benchmarks.libero.native_diagnostic_config.assert_native_diagnostic_owners",
        owner_calls.append,
    )
    analyses = []

    def analyze(**kwargs):
        analyses.append(kwargs)
        return {"metrics": {}, "artifact": {}}

    monkeypatch.setattr(evaluator.shared_diagnostics, "analyze_precomputed", analyze)
    model = GenericModel()
    evaluator.model = model
    evaluator.bind_data_context(normalizer=IdentityNormalizer())
    evaluator.on_validation_step({"libero": {"actions": torch.zeros(2, 16, 7)}}, 0)
    assert len(model.diagnostic_calls) == 1
    assert model.diagnostic_calls[0][0] == "action_flow"
    assert owner_calls == [model]
    assert len(analyses) == 1
    assert analyses[0]["native_error_fns"]["libero"](
        torch.zeros(2, 16, 7), torch.ones(2, 16, 7)
    ).tolist() == [1.0, 1.0]
    assert any(
        name == "Valid/energy_score32_native_equal_components"
        for name, _, _ in model.logged
    )
    assert any(
        name == "Valid/diagnostic_clean_latent_rms" for name, _, _ in model.logged
    )
    assert any(
        name == "Valid/diagnostic_clean_decoded_action_normalized_rms"
        for name, _, _ in model.logged
    )


def test_libero_diagnostic_binds_wrapper_not_inner_pipeline(tmp_path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({"seeds": list(range(32))}))
    evaluator = LiberoActionFlowEvaluator(
        energy_seed_bank_path=seed_file,
        energy_seed_bank_sha256=hashlib.sha256(seed_file.read_bytes()).hexdigest(),
        native_diagnostic_config=_native_profile(seed_file, tmp_path),
    )
    inner_pipeline = SimpleNamespace(device=None)
    evaluator.model = inner_pipeline

    class Wrapper:
        def __init__(self):
            self.model = inner_pipeline
            self.device = torch.device("cpu")
            self.evaluator = evaluator

    wrapper = Wrapper()
    behavior = ActionFlowTrainingBehavior()
    behavior._context_ref = weakref.ref(wrapper)
    behavior._validation_metrics = SimpleNamespace(reset=lambda: None)
    behavior.on_validation_start()
    assert evaluator.model is wrapper
    assert inner_pipeline.device == wrapper.device


def test_libero_diagnostic_profile_is_not_implicitly_waived(tmp_path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({"seeds": list(range(32))}))
    kwargs = {
        "energy_seed_bank_path": seed_file,
        "energy_seed_bank_sha256": hashlib.sha256(seed_file.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="explicit native same-pass"):
        LiberoActionFlowEvaluator(**kwargs)
    profile = _native_profile(seed_file, tmp_path)
    profile["max_samples"] = 7
    with pytest.raises(ValueError, match="native8/k2/all12"):
        LiberoActionFlowEvaluator(**kwargs, native_diagnostic_config=profile)
