from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.eval.action_flow_diagnostics import ActionFlowDiagnostics
from egomimic.eval.planar_action_eval import (
    USOCKET_NATIVE_ERROR_CONFIG,
    PlanarActionEval,
)
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.pushshapes import USocketRotVecNativeDecoder
from egomimic.pipeline.stages_action_flow import (
    ConditionalVelocityStage,
    ContentDecoderStage,
    ContentEncoderStage,
)
from egomimic.pl_utils.pl_model_action_flow import ActionFlowModelWrapper

_CONFIG_DIR = Path(__file__).parents[1] / "egomimic/hydra_configs"


class _IdentityNormalizer:
    @staticmethod
    def unnormalize(values, _selector):
        return values


def _seed_bank(tmp_path, seeds=(101, 202, 303, 404)):
    path = tmp_path / "diagnostic-seeds.json"
    payload = json.dumps({"seeds": list(seeds)}, sort_keys=True).encode()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _config(tmp_path, **overrides):
    path, digest = _seed_bank(tmp_path)
    config = {
        "noise_seed_bank_path": str(path),
        "noise_seed_bank_sha256": digest,
        "raw_noise_levels": [0.0, 0.5, 1.0],
        "max_batches_per_rank": 1,
        "max_samples": None,
        "jacobian_samples": 2,
        "capture_activations": True,
        "activation_layer_map": {0: 0, 1: 2},
        "cknna_k": 2,
        "artifact_root": str(tmp_path / "action-flow-artifacts"),
        "validation_view": {
            "definition": "first_fixed_validation_batch_per_rank",
            "per_rank_batch_size": 6,
            "world_size": 1,
            "split_manifest_sha256": "split-sha",
        },
        "provenance": {
            "source_commit": "0123456789abcdef",
            "resolved_config_sha256": "config-sha",
            "normalization_sha256": "normalization-sha",
            "sampler": "reverse_euler",
            "sampler_steps": 2,
        },
    }
    config.update(overrides)
    return config


def _decode(latent, action_dim=2):
    if action_dim == 4:
        return torch.cat(
            (
                latent[..., :2],
                torch.cos(latent[..., 2:3]),
                torch.sin(latent[..., 2:3]),
            ),
            dim=-1,
        )
    return latent[..., :action_dim]


def _diagnostic(
    seed=101,
    *,
    nonfinite=False,
    include_activations=True,
    action_dim=2,
):
    batch_size, horizon, latent_dim = 6, 2, 3
    levels = torch.tensor([0.0, 0.5, 1.0])
    clean = (
        torch.arange(batch_size * horizon * latent_dim, dtype=torch.float32)
        .reshape(batch_size, horizon, latent_dim)
        .div(10.0)
        .add(0.25)
    )
    noise = torch.flip(clean, dims=(0,)).neg().sub(0.5)
    fixed_states = torch.stack(
        [(1.0 - level) * clean + level * noise for level in levels]
    )
    predicted_clean = torch.stack([clean + float(level) * 0.03 for level in levels])
    fixed_final = torch.stack([clean + float(level) * 0.01 for level in levels])
    generated = fixed_final[-1]
    middle = 0.5 * (noise + generated)
    trajectory = torch.stack((noise, middle, generated))
    target = _decode(clean, action_dim)
    diagnostic = {
        "schema": "action-flow-validation-diagnostics/v1",
        "source": "validation/usocket",
        "noise_seed": seed,
        "returned_samples": batch_size,
        "jacobian_samples": 2,
        "integration_semantics": (
            "reverse_euler_partial_to_lower_canonical_grid_then_uniform_to_zero"
        ),
        "noise_levels": levels,
        "num_inference_steps": 2,
        "target": target,
        "condition": torch.arange(batch_size * 4, dtype=torch.float32).reshape(
            batch_size, 4
        ),
        "latent/clean": clean,
        "latent/noise": noise,
        "latent/generated": generated,
        "latent/fixed_states": fixed_states,
        "latent/predicted_clean": predicted_clean,
        "latent/fixed_final": fixed_final,
        "field/predicted_velocity": noise.unsqueeze(0).repeat(3, 1, 1, 1),
        "field/velocity_residual": torch.full_like(fixed_states, 0.125),
        "latent/trajectory": trajectory,
        "decoded/reconstruction": target,
        "decoded/noise": _decode(noise, action_dim),
        "decoded/generated": _decode(generated, action_dim),
        "decoded/fixed_states": _decode(fixed_states, action_dim),
        "decoded/predicted_clean": _decode(predicted_clean, action_dim),
        "decoded/fixed_final": _decode(fixed_final, action_dim),
        "decoded/trajectory": _decode(trajectory, action_dim),
        "fixed_level_field_evaluations": torch.tensor([0, 1, 2]),
        "decoder_jacobian/clean_singular_values": torch.tensor(
            [[2.0, 1.0, 0.5, 0.25], [1.8, 0.9, 0.4, 0.2]]
        ),
        "decoder_jacobian/noise_singular_values": torch.tensor(
            [[1.5, 0.75, 0.3, 0.1], [1.4, 0.7, 0.25, 0.08]]
        ),
        "decoder_jacobian/fixed_singular_values": torch.tensor(
            [
                [[2.0, 1.0, 0.5, 0.25], [1.8, 0.9, 0.4, 0.2]],
                [[1.7, 0.8, 0.4, 0.2], [1.6, 0.7, 0.3, 0.1]],
                [[1.5, 0.7, 0.3, 0.1], [1.4, 0.6, 0.2, 0.08]],
            ]
        ),
    }
    if include_activations:
        row = torch.arange(batch_size, dtype=torch.float32)
        base = torch.stack((row, row.square(), row.sin(), row.cos()), dim=-1)
        base = base[:, None, :].repeat(1, horizon, 1)
        encoder = torch.stack((base, 2.0 * base + 0.1))
        field = torch.empty(3, 3, batch_size, horizon, 4)
        for level in range(3):
            field[level, 0] = encoder[0] * (1.0 + level)
            field[level, 1] = base.roll(1, dims=0)
            field[level, 2] = encoder[1] * (1.0 + 0.5 * level)
        diagnostic.update(
            {
                "activation/encoder_blocks": encoder,
                "activation/encoder_block_indices": torch.tensor([0, 1]),
                "activation/field_blocks": field,
                "activation/field_block_indices": torch.tensor([0, 1, 2]),
            }
        )
    if nonfinite:
        diagnostic["latent/generated"] = generated.clone()
        diagnostic["latent/generated"][0, 0, 0] = float("nan")
    return diagnostic


class _DiagnosticModel:
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        self.calls = []

    @staticmethod
    def forward_eval(batch):
        return {
            source: {"pred_action": source_batch["actions"]}
            for source, source_batch in batch.items()
        }

    def forward_action_flow_diagnostics(self, batch, **kwargs):
        assert not torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        return {source: self.diagnostic for source in batch}


class _TwoBlockCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    def forward(self, value):
        for block in self.blocks:
            value = torch.tanh(block(value))
        return value


class _TwoBlockField(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    def forward(self, value, time, condition, *, condition_drop_mask=None):
        if condition_drop_mask is None:
            raise AssertionError("diagnostics must supply an explicit drop mask")
        effective = torch.where(
            condition_drop_mask[:, None], torch.zeros_like(condition), condition
        )
        value = value + time[:, None, None] + effective[:, None, :]
        for block in self.blocks:
            value = torch.tanh(block(value))
        return value


def _batch(target):
    return {
        "validation/usocket": {
            "actions": target,
            "embodiment": torch.full((target.shape[0],), 19),
        }
    }


def test_planar_evaluator_logs_and_hashes_action_flow_diagnostics(tmp_path):
    diagnostic = _diagnostic()
    config = _config(tmp_path)
    evaluator = PlanarActionEval(
        energy_score_enabled=False,
        action_flow_diagnostics={"enabled": True, **config},
    )
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    evaluator.model = _DiagnosticModel(diagnostic)
    logged = {}
    evaluator.trainer = SimpleNamespace(
        current_epoch=3,
        global_step=41,
        global_rank=0,
        precision="bf16-mixed",
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        ),
    )
    evaluator.on_validation_start()
    torch.manual_seed(8675309)
    rng_state = torch.random.get_rng_state().clone()

    evaluator.on_validation_step(_batch(diagnostic["target"]), batch_idx=7)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert evaluator.model.calls == [
        {
            "raw_noise_levels": [0.0, 0.5, 1.0],
            "noise_seed": 101,
            "max_samples": None,
            "jacobian_samples": 2,
            "capture_activations": True,
        }
    ]
    required_metrics = (
        "Valid/ActionFlow/Latent/clean/RMS",
        "Valid/ActionFlow/Latent/generated/EffectiveRank",
        "Valid/ActionFlow/Latent/clean/CovarianceEigenvalue/eig_00",
        "Valid/ActionFlow/DecoderJacobian/clean/spectral_norm_mean",
        "Valid/ActionFlow/DecodedFullNoise/token_radius_mean",
        "Valid/ActionFlow/Compute/InferenceFieldEvaluations",
        "Valid/ActionFlow/DenoisingTrajectory/LatentMSE/t1000",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedMSE/t0000",
        "Valid/ActionFlow/FixedLevel/final_latent_mse/t0500",
        "Valid/ActionFlow/Alignment/FinalLatentCosine/t1000",
        "Valid/ActionFlow/Alignment/CKA/encoder_00__field_00/t0500",
        "Valid/ActionFlow/Alignment/CKNNA/encoder_01__field_02/t1000",
    )
    for name in required_metrics:
        assert name in logged
        assert logged[name].ndim == 0
        assert bool(torch.isfinite(logged[name]))
        assert f"{name}/pushshapes_sim_u_socket" in logged
    assert float(
        logged["Valid/ActionFlow/Alignment/CKA/encoder_00__field_00/t0500"]
    ) == pytest.approx(1.0, abs=1.0e-5)

    artifact = tmp_path / "action-flow-artifacts/epoch-3-step-41/rank-0-batch-7.pt"
    sidecar = artifact.with_name(f"{artifact.name}.sha256")
    assert artifact.is_file() and sidecar.is_file()
    sidecar_payload = json.loads(sidecar.read_text())
    assert (
        sidecar_payload["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["metric"] == "ActionFlowValidationDiagnostics"
    assert (
        payload["identity_sha256"] == evaluator._action_flow_diagnostics.identity_sha256
    )
    assert payload["noise_seed"] == 101
    assert payload["sampler"]["name"] == "reverse_euler"
    assert payload["statistics"]["trajectory_error"] == (
        "paired_diagnostic_not_distributional_score"
    )
    source = payload["sources"]["pushshapes_sim_u_socket"]
    assert source["computed"]["final_latent_cosine_by_condition"].shape == (3, 6)
    assert source["computed"]["latent_statistics"]["clean"][
        "covariance_eigenvalues"
    ].shape == (3,)

    evaluator.on_validation_start()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.on_validation_step(_batch(diagnostic["target"]), batch_idx=7)


def test_action_flow_artifacts_preserve_slurm_attempts_and_same_attempt_refusal(tmp_path, monkeypatch):
    diagnostic = _diagnostic()
    legacy = tmp_path / "action-flow-artifacts/epoch-3-step-41/rank-0-batch-7.pt"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"historical diagnostic")
    for job_id, restart in (("5714540", 0), ("5714540", 1), ("5715500", 0)):
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
        monkeypatch.setenv("SLURM_RESTART_COUNT", str(restart))
        evaluator = PlanarActionEval(
            energy_score_enabled=False,
            action_flow_diagnostics={"enabled": True, **_config(tmp_path)},
        )
        evaluator.bind_data_context(normalizer=_IdentityNormalizer())
        evaluator.model = _DiagnosticModel(diagnostic)
        evaluator.trainer = SimpleNamespace(
            current_epoch=3, global_step=41, global_rank=0, precision="32-true",
            lightning_module=SimpleNamespace(log_dict=lambda *_args, **_kwargs: None),
        )
        evaluator.on_validation_start()
        evaluator.on_validation_step(_batch(diagnostic["target"]), batch_idx=7)
        artifact = legacy.parent.parent / f"job-{job_id}-restart-{restart}" / legacy.parent.name / legacy.name
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        assert payload["execution"] == {"slurm_job_id": job_id, "slurm_restart_count": restart}
        assert payload["identity_sha256"] == evaluator._action_flow_diagnostics.identity_sha256
        sidecar = json.loads(Path(f"{artifact}.sha256").read_text())
        assert sidecar["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        evaluator.on_validation_start()
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            evaluator.on_validation_step(_batch(diagnostic["target"]), batch_idx=7)
    assert legacy.read_bytes() == b"historical diagnostic"


def test_planar_action_flow_diagnostics_emit_native_circular_errors(tmp_path):
    diagnostic = _diagnostic(action_dim=4)
    config = _config(tmp_path, native_error=USOCKET_NATIVE_ERROR_CONFIG)
    evaluator = PlanarActionEval(
        energy_score_enabled=False,
        native_decoder=USocketRotVecNativeDecoder(),
        action_flow_diagnostics={"enabled": True, **config},
    )
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    evaluator.model = _DiagnosticModel(diagnostic)
    logged = {}
    evaluator.trainer = SimpleNamespace(
        current_epoch=0,
        global_step=2,
        global_rank=0,
        precision="bf16-mixed",
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        ),
    )
    evaluator.on_validation_start()

    evaluator.on_validation_step(_batch(diagnostic["target"]), batch_idx=0)

    required = (
        "Valid/ActionFlow/CleanReconstructionNativeMSE",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedNativeMSE/t1000",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedNativeMSE/t0500",
        "Valid/ActionFlow/DenoisingTrajectory/DecodedNativeMSE/t0000",
        "Valid/ActionFlow/FixedLevel/final_decoded_native_mse/t0500",
    )
    for name in required:
        assert name in logged
        assert f"{name}/pushshapes_sim_u_socket" in logged
        assert bool(torch.isfinite(logged[name]))
    assert float(logged["Valid/ActionFlow/CleanReconstructionNativeMSE"]) == 0.0

    artifact = tmp_path / "action-flow-artifacts/epoch-0-step-2/rank-0-batch-0.pt"
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["identity"]["native_error"] == USOCKET_NATIVE_ERROR_CONFIG
    assert payload["statistics"]["native_action_error"] == (USOCKET_NATIVE_ERROR_CONFIG)
    computed = payload["sources"]["pushshapes_sim_u_socket"]["computed"]
    assert computed["clean_reconstruction_native_mse_by_condition"].shape == (6,)
    assert computed["trajectory_decoded_native_mse_by_condition"].shape == (3, 6)


def test_native_action_flow_error_wraps_theta_at_pi_boundary():
    target_theta = torch.full((2, 3), torch.pi - 0.01)
    predicted_theta = torch.full((2, 3), -torch.pi + 0.01)

    def rotvec(theta):
        return torch.stack(
            (
                torch.zeros_like(theta),
                torch.zeros_like(theta),
                torch.cos(theta),
                torch.sin(theta),
            ),
            dim=-1,
        )

    decoder = USocketRotVecNativeDecoder()
    values = PlanarActionEval._native_mse_by_condition(
        decoder.decode(rotvec(predicted_theta)),
        decoder.decode(rotvec(target_theta)),
        decoder,
    )

    expected = torch.full((2,), (0.02**2) / 3.0)
    torch.testing.assert_close(values, expected, atol=1.0e-7, rtol=0.0)


def test_action_flow_consumer_matches_real_wrapper_schema(tmp_path):
    torch.manual_seed(91)
    wrapper = ActionFlowModelWrapper(
        pipeline=PipelineAlgo(
            stages=[
                ContentEncoderStage(_TwoBlockCodec()),
                ConditionalVelocityStage(_TwoBlockField(), num_inference_steps=2),
                ContentDecoderStage(_TwoBlockCodec()),
            ],
            device="cpu",
        ),
        gradient_telemetry_cadence=0,
    )
    wrapper.eval()
    runner = ActionFlowDiagnostics(_config(tmp_path, activation_layer_map={0: 0, 1: 1}))
    batch = {
        "opaque_source": {
            "target": torch.randn(6, 2, 2) + 0.25,
            "condition": torch.randn(6, 2),
        }
    }

    metrics = runner.run(
        model=wrapper,
        batch=batch,
        batch_idx=0,
        rank=0,
        epoch=0,
        global_step=0,
        precision="32-true",
        source_labels={"opaque_source": "opaque"},
    )

    assert "Valid/ActionFlow/Latent/clean/RMS" in metrics
    assert "Valid/ActionFlow/Alignment/CKA/encoder_01__field_01/t1000" in metrics
    assert all(bool(torch.isfinite(value)) for value in metrics.values())


@pytest.mark.parametrize(
    "experiment",
    (
        "action_flow_bc_usocket_recon1_s42",
        "action_flow_bc_usocket_recon10_s42",
    ),
)
def test_real_action_flow_config_constructs_strict_diagnostics(tmp_path, experiment):
    with initialize_config_dir(
        version_base=None, config_dir=str(_CONFIG_DIR.resolve())
    ):
        config = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{experiment}", "++paths.root_dir=."],
        )
    diagnostics = config.evaluator.action_flow_diagnostics
    diagnostics.noise_seed_bank_path = str(
        _CONFIG_DIR / "evaluator/energy_score_seed_bank_k32_v1.json"
    )
    diagnostics.artifact_root = str(tmp_path / experiment)
    diagnostics = OmegaConf.to_container(diagnostics, resolve=True)

    evaluator = PlanarActionEval(
        energy_score_enabled=False,
        action_flow_diagnostics=diagnostics,
    )

    runner = evaluator._action_flow_diagnostics
    assert evaluator.action_flow_diagnostics_enabled
    assert runner.noise_levels == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert runner.max_samples == 16
    assert runner.jacobian_samples == 2
    assert runner.activation_layer_map == ((0, 0), (1, 11))
    assert runner.cknna_k == 10


def test_action_flow_diagnostics_fail_closed_without_requested_activations(tmp_path):
    runner = ActionFlowDiagnostics(_config(tmp_path))
    diagnostic = _diagnostic(include_activations=False)
    model = _DiagnosticModel(diagnostic)
    batch = _batch(diagnostic["target"])

    with pytest.raises(TypeError, match="activation/encoder_blocks"):
        runner.run(
            model=model,
            batch=batch,
            batch_idx=0,
            rank=0,
            epoch=0,
            global_step=0,
            precision="32-true",
            source_labels={"validation/usocket": "usocket"},
        )
    assert not list((tmp_path / "action-flow-artifacts").rglob("*.pt"))


def test_action_flow_diagnostics_reject_nonfinite_model_outputs(tmp_path):
    runner = ActionFlowDiagnostics(_config(tmp_path))
    diagnostic = _diagnostic(nonfinite=True)

    with pytest.raises(ValueError, match="non-finite.*latent/generated"):
        runner.run(
            model=_DiagnosticModel(diagnostic),
            batch=_batch(diagnostic["target"]),
            batch_idx=0,
            rank=0,
            epoch=0,
            global_step=0,
            precision=None,
            source_labels={"validation/usocket": "usocket"},
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"jacobian_samples": 0}, "jacobian_samples must be positive"),
        ({"max_samples": 2, "jacobian_samples": 3}, "cannot exceed max_samples"),
        ({"capture_activations": True, "activation_layer_map": {}}, "non-empty"),
        ({"cknna_k": 6}, "k < diagnostic batch size"),
        ({"raw_noise_levels": [0.5, 0.5]}, "strictly increase"),
    ),
)
def test_action_flow_diagnostic_config_fails_closed(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        ActionFlowDiagnostics(_config(tmp_path, **overrides))


@pytest.mark.parametrize("restart", [None, 0, 1])
def test_existing_latent_diagnostic_native_conversion_uses_decoder_argument(
    tmp_path, monkeypatch, restart,
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if restart is not None:
        monkeypatch.setenv("SLURM_JOB_ID", "5714540")
        monkeypatch.setenv("SLURM_RESTART_COUNT", str(restart))
    evaluator = PlanarActionEval(energy_score_enabled=False)
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    evaluator.unite_diagnostics = {"artifact_root": str(tmp_path / "legacy-diagnostic")}
    evaluator._unite_noise_seeds = [101]
    evaluator._unite_noise_seed_sha256 = "seed-sha"
    evaluator._unite_noise_levels = [0.0, 1.0]
    evaluator._unite_noise_labels = ["t000", "t1000"]
    evaluator._unite_cknna_k = 2
    target = _diagnostic()["target"]
    clean = torch.cat((target, torch.ones_like(target[..., :1])), dim=-1)
    feature = torch.arange(1, 7, dtype=torch.float32)[:, None, None].repeat(1, 2, 3)
    evaluator.model = SimpleNamespace(
        forward_unite_diagnostics=lambda _batch, raw_noise_levels: {
            "validation/usocket": {
                "clean_latent": clean,
                "sampler_latents": torch.stack((clean + 1.0, clean)),
                "decoded_actions_normalized": torch.stack((target + 1.0, target)),
                "noise_level_final_predictions": torch.stack((clean, clean + 0.1)),
                "tokenization_activations": {"layer": feature},
                "denoising_activations": {
                    "layer": torch.stack((feature, feature * 2.0))
                },
            }
        }
    )
    evaluator.trainer = SimpleNamespace(
        current_epoch=1,
        global_step=2,
        global_rank=0,
    )
    batch = _batch(target)

    metrics = evaluator._unite_metrics_and_artifact(batch, batch_idx=0)

    assert "Valid/DenoisingTrajectory/DecodedNativeMSE/step_0" in metrics
    root = tmp_path / "legacy-diagnostic"
    if restart is not None:
        root = root / f"job-5714540-restart-{restart}"
    artifact = root / "epoch-1-step-2/rank-0-batch-0.pt"
    assert artifact.is_file()
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["execution"] == (
        None if restart is None else {"slurm_job_id": "5714540", "slurm_restart_count": restart}
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator._unite_metrics_and_artifact(batch, batch_idx=0)
