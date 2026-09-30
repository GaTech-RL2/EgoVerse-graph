import torch
import hashlib
import json

from egomimic.eval.libero_action_flow_eval import LiberoActionFlowEvaluator
from egomimic.pipeline.libero_action_flow import LiberoActionFlowObservationAdapter
from egomimic.rldb.zarr.libero_action_flow import LiberoActionFlowDataset


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


def test_libero_validation_uses_generic_action_flow_diagnostic(tmp_path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(json.dumps({"seeds": list(range(32))}))

    class IdentityNormalizer:
        def unnormalize(self, values, embodiment):
            del embodiment
            return values

    class GenericModel:
        def __init__(self):
            self.logged = []
            self.diagnostic_calls = []

        def forward_eval(self, batch):
            return {source: {"pred_action": values["actions"]} for source, values in batch.items()}

        def run_diagnostic(self, capability, batch, **kwargs):
            self.diagnostic_calls.append((capability, kwargs))
            return {
                source: {
                    "clean_latent": torch.ones(1),
                    "clean_decoded_action_normalized": torch.ones(1),
                }
                for source in batch
            }

        def log(self, name, value, **kwargs):
            self.logged.append((name, value, kwargs))

    evaluator = LiberoActionFlowEvaluator(
        energy_seed_bank_path=seed_file,
        energy_seed_bank_sha256=hashlib.sha256(seed_file.read_bytes()).hexdigest(),
    )
    model = GenericModel()
    evaluator.model = model
    evaluator.bind_data_context(normalizer=IdentityNormalizer())
    evaluator.on_validation_step({"libero": {"actions": torch.zeros(2, 16, 7)}}, 0)
    assert len(model.diagnostic_calls) == 1
    assert model.diagnostic_calls[0][0] == "action_flow"
    assert any(name == "Valid/energy_score32_native_equal_components" for name, _, _ in model.logged)
