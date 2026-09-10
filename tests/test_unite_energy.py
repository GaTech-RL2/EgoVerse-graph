from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from egomimic.eval.energy_score import (
    USOCKET_ENERGY_DISTANCE_CONFIG,
    normalize_usocket_energy_distance_config,
    usocket_energy_distance_metadata,
)
from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.pipeline.pushshapes import (
    PlanarCommon5NativeDecoder,
    USocketRotVecNativeDecoder,
)


class _IdentityNormalizer:
    @staticmethod
    def unnormalize(values, embodiment_id):
        assert embodiment_id == 19
        return values


class _AnyIdentityNormalizer:
    @staticmethod
    def unnormalize(values, embodiment_id):
        assert embodiment_id in (19, 20)
        return values


class _AffineActionNormalizer:
    def __init__(self, scale, bias):
        self.scale = torch.as_tensor(scale)
        self.bias = torch.as_tensor(bias)

    def unnormalize(self, values, embodiment_id):
        assert embodiment_id == 19
        return {
            key: value * self.scale.to(value) + self.bias.to(value)
            for key, value in values.items()
        }

    def normalize_tensor(self, value):
        return (value - self.bias.to(value)) / self.scale.to(value)


class _NestedRandomModel:
    @staticmethod
    def forward_eval(batch):
        return {
            source_id: {"pred_action": torch.rand_like(source_batch["actions"])}
            for source_id, source_batch in batch.items()
        }


def _write_seed_bank(tmp_path, seeds=range(32), filename="energy-seeds.json"):
    payload = json.dumps({"seeds": list(seeds)}, sort_keys=True).encode()
    path = tmp_path / filename
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _evaluator(tmp_path, **kwargs):
    path, digest = _write_seed_bank(tmp_path)
    return PlanarActionEval(
        seed_bank_path=str(path),
        seed_bank_sha256=digest,
        artifact_root=str(tmp_path / "artifacts"),
        semantic_blocks=((0, 2), (2, 4)),
        deterministic_seed=0,
        **kwargs,
    )


def _typed_evaluator(tmp_path):
    path, digest = _write_seed_bank(tmp_path)
    run_dir = tmp_path / "typed-run"
    config_path = run_dir / ".hydra" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("name: typed-usocket-test\n")
    content_manifest_path = run_dir / "content-manifest.json"
    content_manifest_payload = {
        "schema_version": 1,
        "status": "ZARR_CONTENT_MANIFEST",
        "aggregate_sha256": "d" * 64,
    }
    content_manifest_path.write_text(
        json.dumps(content_manifest_payload, sort_keys=True, separators=(",", ":"))
    )
    split_hash = "c" * 64
    provenance = {
        "source_commit": "a" * 40,
        "normalization_sha256": "b" * 64,
        "split_manifest_sha256": split_hash,
        "resolved_config_path": str(config_path),
        "wandb": {
            "entity": "rl2-group",
            "project": "pushshapes-action-flow",
            "run_id": "typed-energy-test",
        },
        "distance_contract": USOCKET_ENERGY_DISTANCE_CONFIG,
        "dataset_content": {
            "manifest_path": str(content_manifest_path),
            "manifest_sha256": hashlib.sha256(
                content_manifest_path.read_bytes()
            ).hexdigest(),
            "aggregate_sha256": content_manifest_payload["aggregate_sha256"],
        },
        "action_representation": "x_y_cos_theta_sin_theta",
    }
    evaluator = PlanarActionEval(
        seed_bank_path=str(path),
        seed_bank_sha256=digest,
        artifact_root=str(run_dir / "energy"),
        semantic_blocks=((0, 2), (2, 4)),
        deterministic_seed=0,
        native_decoder=USocketRotVecNativeDecoder(),
        energy_score_distance=USOCKET_ENERGY_DISTANCE_CONFIG,
        energy_score_validation_view={
            "definition": "first_deterministic_validation_batch_per_rank",
            "split_manifest_sha256": split_hash,
            "per_rank_batch_size": 2,
            "world_size": 1,
        },
        energy_score_provenance=provenance,
    )
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    return evaluator, run_dir, config_path, content_manifest_path


def test_energy_score_supports_source_specific_action_partitions(tmp_path):
    evaluator = _evaluator(
        tmp_path,
        semantic_blocks_by_source={
            "pushshapes_sim_u_socket": ((0, 2), (2, 4)),
            "pushshapes_sim_chain_gripper": ((0, 2), (2, 4), (4, 5)),
        },
    )
    for embodiment_id, width in ((19, 4), (20, 5)):
        target = torch.zeros(2, 16, width)
        samples = target.unsqueeze(0).repeat(32, 1, 1, 1)
        values = evaluator._energy_values(samples, target, embodiment_id)
        assert values["score"] == pytest.approx(0.0)


def test_generic_energy_identity_accepts_list_provenance_for_tuple_blocks(tmp_path):
    evaluator, _, _, _ = _typed_evaluator(tmp_path)
    evaluator.energy_score_distance = None
    evaluator.energy_score_distance_metadata = {
        "space": "normalized_action_chunk",
        "formula": "mean_equal_weight_semantic_block_rms",
        "semantic_blocks": ((0, 2), (2, 4), (4, 5)),
    }
    evaluator.energy_score_provenance["distance_contract"] = {
        "space": "normalized_action_chunk",
        "formula": "mean_equal_weight_semantic_block_rms",
        "semantic_blocks": [[0, 2], [2, 4], [4, 5]],
    }

    identity = evaluator._typed_artifact_identity(
        domains={"pushshapes_sim_chain_gripper": {"condition_ids": []}},
        global_step=2,
    )

    assert identity["distance_contract"]["semantic_blocks"] == [
        [0, 2],
        [2, 4],
        [4, 5],
    ]


def test_chain_native_decoder_supports_seed_and_batch_leading_dimensions(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.bind_data_context(normalizer=_AnyIdentityNormalizer())
    common_five = torch.zeros(32, 2, 16, 5)
    common_five[..., 2] = 1.0
    decoded = evaluator._native(
        common_five,
        20,
        PlanarCommon5NativeDecoder(action_horizon=16, native_action_dim=4),
    )
    assert decoded.shape == (32, 2, 16, 4)


def test_energy_artifacts_preserve_validation_across_slurm_attempts(tmp_path, monkeypatch):
    target = torch.zeros(2, 2, 4)
    samples = target.unsqueeze(0).repeat(32, 1, 1, 1)
    batch = _batch(target)
    legacy = tmp_path / "artifacts/epoch-0-step-20001/rank-0-batch-0.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"historical validation artifact")
    saved_paths = []
    for job_id, restart in (("5691426", 0), ("5691426", 1), ("5692000", 0)):
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
        monkeypatch.setenv("SLURM_RESTART_COUNT", str(restart))
        evaluator = _evaluator(tmp_path)
        evaluator.trainer = SimpleNamespace(current_epoch=0, global_step=20001, global_rank=0)
        scores = {"validation/usocket": evaluator._energy_values(samples, target, 19)}
        evaluator._save_artifact(0, {"validation/usocket": samples}, scores, batch)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            evaluator._save_artifact(0, {"validation/usocket": samples}, scores, batch)
        path = tmp_path / "artifacts" / f"job-{job_id}-restart-{restart}" / "epoch-0-step-20001/rank-0-batch-0.pt"
        saved_paths.append(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["execution"] == {"slurm_job_id": job_id, "slurm_restart_count": restart}
        torch.testing.assert_close(payload["domains"]["pushshapes_sim_u_socket"]["predictions"], samples)
    assert legacy.read_bytes() == b"historical validation artifact"
    assert len(set(saved_paths)) == 3


def _rotvec(theta, *, batch_size=2, horizon=16):
    theta = torch.full((batch_size, horizon), float(theta))
    return torch.stack(
        (
            torch.zeros_like(theta),
            torch.zeros_like(theta),
            torch.cos(theta),
            torch.sin(theta),
        ),
        dim=-1,
    )


def _batch(target):
    return {
        "validation/usocket": {
            "actions": target,
            "embodiment": torch.full((target.shape[0],), 19),
        }
    }


def test_energy_seed_bank_hash_and_seed_identity_are_strict(tmp_path):
    path, digest = _write_seed_bank(tmp_path)
    with pytest.raises(ValueError, match="seed-bank identity mismatch"):
        PlanarActionEval(
            seed_bank_path=str(path),
            seed_bank_sha256="0" * 64,
            artifact_root=str(tmp_path / "bad-hash"),
        )

    duplicate_path, duplicate_digest = _write_seed_bank(
        tmp_path, [7] * 32, "duplicate-energy-seeds.json"
    )
    with pytest.raises(ValueError, match="32 unique seeds"):
        PlanarActionEval(
            seed_bank_path=str(duplicate_path),
            seed_bank_sha256=duplicate_digest,
            artifact_root=str(tmp_path / "duplicate-seeds"),
        )

    evaluator = PlanarActionEval(
        seed_bank_path=str(path),
        seed_bank_sha256=digest,
        artifact_root=str(tmp_path / "artifacts"),
        semantic_blocks=((0, 2), (2, 4)),
        deterministic_seed=0,
    )
    evaluator.model = _NestedRandomModel()
    target = torch.zeros(2, 2, 4)
    batch = _batch(target)

    torch.manual_seed(8675309)
    original_rng_state = torch.random.get_rng_state()
    sampled, first_result = evaluator._seeded_predictions(batch)
    sampled = sampled["validation/usocket"]
    actual_next_random = torch.rand(4)
    torch.random.set_rng_state(original_rng_state)
    expected_next_random = torch.rand(4)

    with torch.random.fork_rng():
        torch.manual_seed(evaluator.seeds[0])
        expected_first_sample = torch.rand_like(target)
    assert sampled.shape == (32, 2, 2, 4)
    torch.testing.assert_close(sampled[0], expected_first_sample)
    torch.testing.assert_close(
        first_result["validation/usocket"]["pred_action"], expected_first_sample
    )
    torch.testing.assert_close(actual_next_random, expected_next_random)


def test_typed_usocket_energy_uses_wrapped_native_theta_over_complete_h16(tmp_path):
    evaluator, _, _, _ = _typed_evaluator(tmp_path)
    normalizer = _AffineActionNormalizer(
        scale=[5.0, 7.0, 2.0, 4.0],
        bias=[1.0, -2.0, 0.25, -0.75],
    )
    evaluator.bind_data_context(normalizer=normalizer)
    target = normalizer.normalize_tensor(_rotvec(math.pi - 0.01))
    prediction = normalizer.normalize_tensor(_rotvec(-math.pi + 0.01))
    samples = prediction.unsqueeze(0).repeat(32, 1, 1, 1)

    values = evaluator._energy_values(samples, target, embodiment_id=19)

    expected = torch.tensor(0.01 / math.pi)
    torch.testing.assert_close(values["accuracy"], expected, atol=1.0e-7, rtol=0.0)
    torch.testing.assert_close(values["diversity"], torch.zeros(()))
    torch.testing.assert_close(values["score"], expected, atol=1.0e-7, rtol=0.0)


def test_typed_usocket_energy_weights_xy_and_circular_theta_equally(tmp_path):
    evaluator, _, _, _ = _typed_evaluator(tmp_path)
    target = _rotvec(0.0)
    translated = target.clone()
    translated[..., :2] = 1.0
    rotated = _rotvec(math.pi)

    translation_score = evaluator._energy_values(
        translated.unsqueeze(0).repeat(32, 1, 1, 1),
        target,
        embodiment_id=19,
    )["score"]
    rotation_score = evaluator._energy_values(
        rotated.unsqueeze(0).repeat(32, 1, 1, 1),
        target,
        embodiment_id=19,
    )["score"]

    torch.testing.assert_close(translation_score, torch.tensor(0.5))
    torch.testing.assert_close(rotation_score, torch.tensor(0.5))

    short_target = target[:, :-1]
    with pytest.raises(ValueError, match="complete normalized chunks"):
        evaluator._energy_values(
            short_target.unsqueeze(0).repeat(32, 1, 1, 1),
            short_target,
            embodiment_id=19,
        )


def test_typed_usocket_energy_contract_rejects_undeclared_weight_knobs():
    contract = {
        **USOCKET_ENERGY_DISTANCE_CONFIG,
        "semantic_weights": {
            **USOCKET_ENERGY_DISTANCE_CONFIG["semantic_weights"],
            "undeclared": 0.0,
        },
    }

    with pytest.raises(ValueError, match="semantic weight keys differ"):
        normalize_usocket_energy_distance_config(contract)


def test_typed_usocket_energy_artifact_binds_full_metric_identity(tmp_path):
    evaluator, run_dir, config_path, content_manifest_path = _typed_evaluator(tmp_path)
    target = _rotvec(math.pi - 0.01)
    prediction = _rotvec(-math.pi + 0.01)
    samples = prediction.unsqueeze(0).repeat(32, 1, 1, 1)
    batch = _batch(target)
    batch["validation/usocket"]["episode_hash"] = ["episode-a", "episode-b"]
    batch["validation/usocket"]["frame_index"] = torch.tensor([3, 9])
    evaluator.model = SimpleNamespace()
    evaluator.trainer = SimpleNamespace(
        current_epoch=0,
        global_step=2,
        global_rank=0,
        precision="bf16-mixed",
        lightning_module=SimpleNamespace(log_dict=lambda *_args, **_kwargs: None),
    )
    evaluator._seeded_predictions = lambda _batch: (
        {"validation/usocket": samples},
        {"validation/usocket": {"pred_action": samples[0]}},
    )

    evaluator.on_validation_step(batch, batch_idx=0)

    artifact_path = run_dir / "energy/epoch-0-step-2/rank-0-batch-0.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert artifact["schema_version"] == 2
    assert artifact["distance"] == usocket_energy_distance_metadata(
        USOCKET_ENERGY_DISTANCE_CONFIG
    )
    assert artifact["identity"]["metric"] == {
        "name": "EnergyScore@32",
        "sample_count": 32,
        "deterministic_seed": 0,
    }
    assert artifact["identity"]["source_commit"] == "a" * 40
    assert artifact["identity"]["normalization_sha256"] == "b" * 64
    assert artifact["identity"]["split_manifest_sha256"] == "c" * 64
    assert (
        artifact["identity"]["resolved_config_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )
    assert artifact["identity"]["wandb"] == {
        "entity": "rl2-group",
        "project": "pushshapes-action-flow",
        "run_id": "typed-energy-test",
    }
    assert artifact["identity"]["dataset_content"] == {
        "manifest_path": str(content_manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(
            content_manifest_path.read_bytes()
        ).hexdigest(),
        "aggregate_sha256": "d" * 64,
    }
    assert artifact["identity"]["validation_view"] == {
        "definition": "first_deterministic_validation_batch_per_rank",
        "split_manifest_sha256": "c" * 64,
        "per_rank_batch_size": 2,
        "world_size": 1,
    }
    assert artifact["identity"]["checkpoint_binding"] == {
        "global_step": 2,
        "checkpoint_sha256": None,
        "sha256_status": (
            "unavailable_during_validation_artifact_write; post-run_smoke_"
            "verifier_binds_checkpoint_and_artifact_by_global_step_and_records_"
            "both_file_hashes"
        ),
    }
    encoded_identity = json.dumps(
        artifact["identity"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    assert artifact["identity_sha256"] == hashlib.sha256(encoded_identity).hexdigest()
    domain = artifact["domains"]["pushshapes_sim_u_socket"]
    assert domain["native_predictions"].shape == (32, 2, 16, 3)
    assert domain["native_targets"].shape == (2, 16, 3)
    assert [item["episode_hash"] for item in domain["condition_ids"]] == [
        "episode-a",
        "episode-b",
    ]
    assert [item["frame_index"] for item in domain["condition_ids"]] == [3, 9]
    assert artifact["identity"]["validation_conditions"] == {
        "pushshapes_sim_u_socket": domain["condition_ids"]
    }


def test_typed_usocket_energy_identity_rejects_mutated_content_manifest(tmp_path):
    evaluator, _, _, content_manifest_path = _typed_evaluator(tmp_path)
    content_manifest_path.write_text('{"aggregate_sha256":"e"}')

    with pytest.raises(ValueError, match="manifest hash differs"):
        evaluator._typed_artifact_identity(
            domains={"pushshapes_sim_u_socket": {"condition_ids": []}},
            global_step=2,
        )


def test_energy_artifact_records_provenance_and_per_condition_outputs(tmp_path):
    validation_view = {
        "definition": "first_energy_score_batch_per_ddp_rank",
        "split_manifest_sha256": "split-sha",
        "per_rank_batch_size": 2,
        "world_size": 4,
    }
    provenance = {
        "action_representations": {
            "pushshapes_sim_u_socket": "x_y_cos_theta_sin_theta"
        },
        "prediction_horizon": 2,
        "sampler": "dopri5",
        "sampler_evaluation_points": 50,
        "sampler_atol": 1.0e-6,
        "sampler_rtol": 1.0e-3,
        "model_autocast_precision": "bf16",
        "dopri5_state_precision": "fp32",
        "dopri5_derivative_precision": "fp32",
        "dopri5_error_control_precision": "fp32",
    }
    evaluator = _evaluator(
        tmp_path,
        energy_score_max_batches_per_rank=1,
        energy_score_validation_view=validation_view,
        energy_score_provenance=provenance,
    )
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    target = torch.zeros(2, 2, 4)
    batch = _batch(target)
    evaluator.model = SimpleNamespace(
        forward_eval=lambda grouped: {
            source_id: {"pred_action": source_batch["actions"]}
            for source_id, source_batch in grouped.items()
        }
    )
    logged = {}
    evaluator.trainer = SimpleNamespace(
        current_epoch=3,
        global_step=41,
        global_rank=2,
        precision="bf16-mixed",
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        ),
    )

    samples = target.unsqueeze(0).repeat(32, 1, 1, 1)
    condition_values = torch.linspace(0.0, 1.0, 32)
    samples[:, 1] = condition_values[:, None, None]
    evaluator._seeded_predictions = lambda _batch: (
        {"validation/usocket": samples},
        {"validation/usocket": {"pred_action": samples[0]}},
    )

    evaluator.on_validation_step(batch, batch_idx=0)

    artifact_path = tmp_path / "artifacts" / "epoch-3-step-41" / "rank-2-batch-0.pt"
    assert artifact_path.is_file()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert artifact["schema_version"] == 1
    assert artifact["metric"] == "EnergyScore@32"
    assert artifact["sample_count"] == 32
    assert artifact["seed_bank"] == list(range(32))
    assert artifact["seed_bank_sha256"] == evaluator.seed_bank_sha256
    assert artifact["deterministic_seed"] == 0
    assert artifact["distance"] == {
        "space": "normalized_action_chunk",
        "formula": "mean_equal_weight_semantic_block_rms",
        "semantic_blocks": ((0, 2), (2, 4)),
    }
    assert artifact["aggregation"] == ("condition_mean_then_equal_domain_macro_mean")
    assert artifact["global_step"] == 41
    assert artifact["epoch"] == 3
    assert artifact["rank"] == 2
    assert artifact["batch_idx"] == 0
    assert artifact["precision"] == "bf16-mixed"
    assert artifact["validation_view"] == validation_view
    assert artifact["provenance"] == provenance

    domain = artifact["domains"]["pushshapes_sim_u_socket"]
    assert domain["source_id"] == "validation/usocket"
    assert domain["embodiment_id"] == 19
    assert domain["action_key"] == "actions"
    assert domain["predictions"].shape == (32, 2, 2, 4)
    assert domain["targets"].shape == (2, 2, 4)
    for name in (
        "accuracy_by_condition",
        "diversity_by_condition",
        "score_by_condition",
    ):
        assert domain[name].shape == (2,)
        assert bool(torch.isfinite(domain[name]).all())

    off_diagonal = ~torch.eye(32, dtype=torch.bool)
    expected_diversity = (
        (condition_values[:, None] - condition_values[None, :])
        .abs()[off_diagonal]
        .mean()
    )
    expected_accuracy = torch.tensor([0.0, condition_values.mean()])
    expected_diversity_by_condition = torch.tensor([0.0, expected_diversity])
    expected_score = expected_accuracy - 0.5 * expected_diversity_by_condition
    torch.testing.assert_close(domain["accuracy_by_condition"], expected_accuracy)
    torch.testing.assert_close(
        domain["diversity_by_condition"], expected_diversity_by_condition
    )
    torch.testing.assert_close(domain["score_by_condition"], expected_score)
    torch.testing.assert_close(logged["Valid/EnergyScore@32"], expected_score.mean())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.on_validation_step(batch, batch_idx=0)


def test_energy_batch_cap_does_not_cap_normal_mse_validation(tmp_path):
    evaluator = _evaluator(tmp_path, energy_score_max_batches_per_rank=1)
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    target = torch.zeros(2, 2, 4)
    batch = _batch(target)
    evaluator.model = SimpleNamespace(
        forward_eval=lambda grouped: {
            source_id: {"pred_action": source_batch["actions"]}
            for source_id, source_batch in grouped.items()
        }
    )
    logged = []
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.append(metrics)
        )
    )
    energy_calls = []
    evaluator._seeded_predictions = lambda _batch: (
        energy_calls.append(True)
        or (
            {"validation/usocket": target.unsqueeze(0).repeat(32, 1, 1, 1)},
            {"validation/usocket": {"pred_action": target}},
        )
    )
    evaluator._save_artifact = lambda *_args: None

    evaluator.on_validation_step(batch, batch_idx=1)
    evaluator.on_validation_step(batch, batch_idx=0)

    assert energy_calls == [True]
    assert "Valid/MSE" in logged[0] and "Valid/EnergyScore@32" not in logged[0]
    assert "Valid/MSE" in logged[1] and "Valid/EnergyScore@32" in logged[1]
