import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.eval.human_robot_overlay_eval import HumanRobotOverlayEval
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import (
    build_fold_keypoint_wristframe_revert_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class IdentityNorm:
    def unnormalize(self, batch, _emb_id):
        return batch


SEED_BANK = (
    Path(__file__).resolve().parents[1]
    / "egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json"
)


def _evaluator(emb_id, action_key, prediction, **kwargs):
    evaluator = HumanRobotOverlayEval(**kwargs)
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: action_key},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: {f"emb{emb_id}_{action_key}": prediction},
    )
    return evaluator


def test_metrics_score_the_full_unnormalized_denoised_chunk():
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    target = torch.zeros(2, 4, 3)
    prediction = target.clone()
    prediction[:, -1] = 2.0
    evaluator = _evaluator(
        emb_id, action_key, prediction, viz_func=None, frame_stride=1
    )

    metrics, images = evaluator.compute_metrics_and_viz({emb_id: {action_key: target}})

    prefix = f"Valid/emb{emb_id}_{action_key}_action"
    assert metrics[f"{prefix}_mse"].item() == 1.0
    assert metrics[f"{prefix}_squared_error_median"].item() == 0.0
    assert metrics[f"{prefix}_squared_error_max"].item() == 4.0
    assert metrics[f"Valid/emb{emb_id}_{action_key}_copybaseline_mse"] == 0.0
    assert images == {}


def test_metrics_include_per_domain_and_cotrain_mse_aliases():
    emb_ids = [
        get_embodiment_id("eva_bimanual"),
        get_embodiment_id("human_bimanual"),
    ]
    action_key = "actions_cartesian"
    targets = {
        emb_ids[0]: torch.zeros(2, 4, 3),
        emb_ids[1]: torch.zeros(2, 4, 3),
    }
    predictions = {
        f"emb{emb_ids[0]}_{action_key}": torch.ones(2, 4, 3),
        f"emb{emb_ids[1]}_{action_key}": torch.full((2, 4, 3), 2.0),
    }
    evaluator = HumanRobotOverlayEval(viz_func=None)
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: action_key for emb_id in emb_ids},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: predictions,
    )

    metrics, _ = evaluator.compute_metrics_and_viz(
        {emb_id: {action_key: targets[emb_id]} for emb_id in emb_ids}
    )

    assert metrics["Valid/MSE/eva_bimanual"].item() == 1.0
    assert metrics["Valid/MSE/human_bimanual"].item() == 4.0
    assert metrics["Valid/MSE"].item() == 2.5
    assert metrics["Valid/Native_MSE"].item() == 2.5


def test_explicit_paper_metrics_decode_native_actions_and_weight_elements():
    domains = [
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    ]
    emb_ids = [get_embodiment_id(domain) for domain in domains]
    target = torch.zeros(1, 1, 5)
    predictions = {
        f"emb{emb_ids[0]}_actions": target.clone(),
        f"emb{emb_ids[1]}_actions": target.clone(),
    }
    predictions[f"emb{emb_ids[0]}_actions"][..., 0] = 1.0
    predictions[f"emb{emb_ids[1]}_actions"][..., :4] = 1.0

    class SliceAdapter:
        def __init__(self, width):
            self.width = int(width)

        def decode(self, actions, context=None):
            del context
            return actions[..., : self.width]

    adapters = {
        emb_ids[0]: SliceAdapter(3),
        emb_ids[1]: SliceAdapter(4),
    }
    evaluator = HumanRobotOverlayEval(
        viz_func=None,
        log_normalized_mse=True,
        log_native_mse=True,
    )
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: "actions" for emb_id in emb_ids},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: predictions,
        rollout_adapter_for=lambda emb_id: adapters[emb_id],
    )

    metrics, _ = evaluator.compute_metrics_and_viz(
        {emb_id: {"actions": target} for emb_id in emb_ids}
    )

    assert metrics[f"Valid/Native_MSE/{domains[0]}"] == pytest.approx(1.0 / 3.0)
    assert metrics[f"Valid/Native_MSE/{domains[1]}"] == pytest.approx(1.0)
    assert metrics["Valid/Native_MSE"] == pytest.approx(5.0 / 7.0)
    assert metrics["Valid/MSE"] == pytest.approx(0.5)


def test_energy_score_accepts_shared_and_per_domain_distance_schemas(tmp_path):
    seed_sha = hashlib.sha256(SEED_BANK.read_bytes()).hexdigest()
    common = {
        "enabled": True,
        "sample_count": 32,
        "seed_bank_path": str(SEED_BANK),
        "seed_bank_sha256": seed_sha,
        "max_batches_per_rank": 1,
        "artifact_root": str(tmp_path),
    }
    shared = HumanRobotOverlayEval(
        viz_func=None,
        energy_score={
            **common,
            "action_dim": 5,
            "distance_blocks": {
                "translation": {"indices": [0, 1], "weight": 1.0},
                "rotation": {"indices": [2, 3], "weight": 1.0},
                "grip": {"indices": [4], "weight": 1.0},
            },
        },
    )
    per_domain = HumanRobotOverlayEval(
        viz_func=None,
        energy_score={
            **common,
            "action_dims": {"pushshapes_sim_u_socket": 4, "eva_bimanual": 3},
            "distance_blocks": {
                "pushshapes_sim_u_socket": {
                    "translation": {"indices": [0, 1], "weight": 1.0},
                    "rotation": {"indices": [2, 3], "weight": 1.0},
                },
                "eva_bimanual": {
                    "position": {"indices": [0, 1, 2], "weight": 1.0},
                },
            },
        },
    )

    shared_distance = shared._energy_distance(
        torch.zeros(2, 3, 5), torch.ones(2, 3, 5), "legacy-domain"
    )
    mapped_distance = per_domain._energy_distance(
        torch.zeros(2, 3, 4),
        torch.ones(2, 3, 4),
        "pushshapes_sim_u_socket",
    )

    assert shared._energy_schema == "shared"
    assert per_domain._energy_schema == "per_domain"
    assert torch.equal(shared_distance, torch.ones_like(shared_distance))
    assert torch.equal(mapped_distance, torch.ones_like(mapped_distance))
    with pytest.raises(ValueError, match="No Energy Score distance contract"):
        per_domain._energy_distance(
            torch.zeros(2, 3, 4), torch.ones(2, 3, 4), "unconfigured-domain"
        )


def test_training_log_info_includes_per_domain_and_cotrain_mse_aliases():
    algo = SimpleNamespace(domain_by_id={3: "u_socket", 7: "chain_grabber"})
    info = {
        "losses": {
            "action_loss": torch.tensor(2.5),
            "3_loss_native_action": torch.tensor(1.0),
            "7_loss_native_action": torch.tensor(4.0),
        }
    }

    logged = PipelineAlgo.log_info(algo, info)

    assert logged["MSE/u_socket"] == 1.0
    assert logged["MSE/chain_grabber"] == 4.0
    assert logged["MSE"] == 2.5


def test_prediction_unnormalize_preserves_slotwise_arc_token_stats():
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    horizon, action_dim = 4, 3
    stats = {
        "quantile_1": np.zeros((horizon, action_dim), dtype=np.float32),
        "quantile_99": np.arange(1, horizon * action_dim + 1, dtype=np.float32).reshape(
            horizon, action_dim
        ),
    }
    norm_stats = MultiDataset.from_state(
        {
            "norm_mode": "quantile",
            "embodiments": [emb_id],
            "key_types": {emb_id: {action_key: "action_keys"}},
            "zarr_keys": {emb_id: {action_key: action_key}},
            "shapes": {emb_id: {action_key: (horizon, action_dim)}},
            "norm_stats": {emb_id: {action_key: stats}},
        }
    )
    evaluator = HumanRobotOverlayEval(viz_func=None)
    evaluator.model = SimpleNamespace(norm_stats=norm_stats)

    actual = evaluator._unnormalize_prediction(
        torch.zeros(2, horizon, action_dim), emb_id, action_key
    )

    expected = torch.from_numpy(stats["quantile_99"]) * 0.5 + 0.5e-6
    assert actual.shape == (2, horizon, action_dim)
    assert torch.allclose(actual[0], expected)
    assert torch.allclose(actual[1], expected)


def test_frame_limit_is_cumulative_across_validation_batches():
    emb_id = get_embodiment_id("eva_bimanual")
    action_key = "actions_cartesian"
    actions = torch.zeros(8, 4, 3)

    def viz(*, batch, **_kwargs):
        return np.zeros((batch[action_key].shape[0], 8, 8, 3), dtype=np.uint8)

    evaluator = _evaluator(
        emb_id,
        action_key,
        actions.clone(),
        frame_stride=1,
        max_frames=11,
        viz_func={"eva_bimanual": viz},
    )
    batch = {
        emb_id: {
            action_key: actions,
            "front_img_1": torch.zeros(8, 2, 3, 8, 8),
        }
    }

    _, first = evaluator.compute_metrics_and_viz(batch)
    _, second = evaluator.compute_metrics_and_viz(batch)
    _, third = evaluator.compute_metrics_and_viz(batch)

    assert first[emb_id].shape[0] == 8
    assert second[emb_id].shape[0] == 3
    assert third == {}


def test_canonical_126d_overlay_reverts_keypoints_to_head_frame():
    emb_id = get_embodiment_id("human_bimanual")
    action_key = "actions_keypoints"
    actions = torch.zeros(2, 3, 126)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    wrist_poses = torch.cat((identity, identity)).repeat(2, 1)
    seen = {}

    def viz(*, predictions, batch):
        seen["target"] = tuple(batch[action_key].shape)
        seen["prediction"] = tuple(predictions[f"human_bimanual_{action_key}"].shape)
        return np.zeros((2, 8, 8, 3), dtype=np.uint8)

    evaluator = _evaluator(
        emb_id,
        action_key,
        actions.clone(),
        frame_stride=1,
        max_frames=2,
        viz_func={"human_bimanual": viz},
        transform_lists={
            "human_bimanual": build_fold_keypoint_wristframe_revert_transform_list()
        },
    )

    metrics, images = evaluator.compute_metrics_and_viz(
        {
            emb_id: {
                action_key: actions,
                "viz_current_wrist_poses": wrist_poses,
                "front_img_1": torch.zeros(2, 2, 3, 8, 8),
            }
        }
    )

    assert seen == {"target": (2, 3, 126), "prediction": (2, 3, 126)}
    assert f"Valid/human_bimanual_{action_key}_camera_action_mse" in metrics
    assert images[emb_id].shape == (2, 8, 8, 3)


def test_deterministic_validation_noise_is_repeatable_and_rng_isolated(monkeypatch):
    seen = []

    def fake_step(_self, _batch, _batch_idx, _dataloader_idx=0):
        seen.append(torch.randn(4))

    monkeypatch.setattr(EvalVideo, "on_validation_step", fake_step)
    evaluator = HumanRobotOverlayEval(deterministic_seed=42)
    evaluator.trainer = SimpleNamespace(
        global_rank=0,
        lightning_module=SimpleNamespace(device=torch.device("cpu")),
    )

    torch.manual_seed(123)
    expected_next = torch.randn(4)
    torch.manual_seed(123)
    evaluator.on_validation_step({}, 7)
    actual_next = torch.randn(4)
    evaluator.on_validation_step({}, 7)

    assert torch.equal(seen[0], seen[1])
    assert torch.equal(actual_next, expected_next)


def test_exact_epoch_metrics_weight_every_action_element_once():
    emb_ids = [
        get_embodiment_id("eva_bimanual"),
        get_embodiment_id("human_bimanual"),
    ]
    logged = {}

    def log_dict(metrics, **kwargs):
        logged.update(metrics)

    evaluator = HumanRobotOverlayEval(exact_epoch_metrics=True)
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu"), log_dict=log_dict)
    )
    evaluator._accumulate_exact(
        emb_ids[0], torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0])
    )
    evaluator._accumulate_exact(emb_ids[0], torch.tensor([5.0]), torch.tensor([6.0]))
    evaluator._accumulate_exact(
        emb_ids[1], torch.tensor([4.0, 4.0]), torch.tensor([8.0, 8.0])
    )

    evaluator.on_validation_end()

    assert logged["Valid/MSE/eva_bimanual"].item() == 3.0
    assert logged["Valid/MSE/human_bimanual"].item() == 4.0
    assert logged["Valid/MSE"].item() == 3.5
    assert logged["Valid/Native_MSE/eva_bimanual"].item() == 4.0
    assert logged["Valid/Native_MSE/human_bimanual"].item() == 8.0


def test_exact_epoch_metrics_reduce_one_global_embodiment_order(monkeypatch):
    emb_ids = sorted(
        [
            get_embodiment_id("eva_bimanual"),
            get_embodiment_id("human_bimanual"),
        ]
    )
    logged = {}
    collective_calls = []

    def log_dict(metrics, **kwargs):
        logged.update(metrics)

    def all_gather_object(output, local_ids):
        assert local_ids == [emb_ids[0]]
        output[:] = [[emb_ids[0]], [emb_ids[1]]]

    remote_totals = [
        torch.zeros(4, dtype=torch.float64),
        torch.tensor([8.0, 2.0, 16.0, 2.0], dtype=torch.float64),
    ]

    def all_reduce(totals, op):
        assert op == torch.distributed.ReduceOp.SUM
        index = len(collective_calls)
        collective_calls.append(totals.clone())
        totals.add_(remote_totals[index].to(totals))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    evaluator = HumanRobotOverlayEval(exact_epoch_metrics=True)
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu"), log_dict=log_dict)
    )
    evaluator._accumulate_exact(
        emb_ids[0], torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0])
    )

    evaluator.on_validation_end()

    if get_embodiment_id("eva_bimanual") == emb_ids[0]:
        first_domain, second_domain = "eva_bimanual", "human_bimanual"
    else:
        first_domain, second_domain = "human_bimanual", "eva_bimanual"
    assert len(collective_calls) == 2
    assert collective_calls[0].tolist() == [4.0, 2.0, 6.0, 2.0]
    assert collective_calls[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert logged[f"Valid/MSE/{first_domain}"].item() == pytest.approx(2.0)
    assert logged[f"Valid/MSE/{second_domain}"].item() == pytest.approx(4.0)
    assert logged["Valid/MSE"].item() == pytest.approx(3.0)
    assert logged[f"Valid/Native_MSE/{first_domain}"].item() == pytest.approx(3.0)
    assert logged[f"Valid/Native_MSE/{second_domain}"].item() == pytest.approx(8.0)
    assert logged["Valid/Native_MSE"].item() == pytest.approx(5.5)
