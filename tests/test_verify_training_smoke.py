import json
from pathlib import Path

import pytest
import torch
import wandb
from omegaconf import OmegaConf

from egomimic.scripts import verify_training_smoke as verifier


def _wandb_record(**items):
    record = verifier.wandb_internal_pb2.Record()
    for key, value in items.items():
        item = record.history.item.add()
        item.key = key
        item.value_json = json.dumps(value)
    return record.SerializeToString()


def _wandb_exit(exit_code: int = 0):
    record = verifier.wandb_internal_pb2.Record()
    record.exit.exit_code = exit_code
    return record.SerializeToString()


_RELEASED_TRAIN_METRICS = {
    "Train/UNITE/TotalLoss",
    "Train/UNITE/ReconstructionLoss",
    "Train/UNITE/FlowLoss",
    "Train/UNITE/ReconstructionL1",
    "Train/MSE",
    "Train/MSE/pushshapes_sim_u_socket",
}
_RELEASED_OPTIMIZER_METRICS = {"Optimizer/LR/AdamW", "Optimizer/LR/Muon"}
_RELEASED_VALID_METRICS = {
    "Valid/UNITE/TotalLoss",
    "Valid/UNITE/ReconstructionLoss",
    "Valid/UNITE/FlowLoss",
    "Valid/UNITE/ReconstructionL1",
    "Valid/UNITE/ReconstructionNativeMSE",
    "Valid/UNITE/ReconstructionNativeL1",
    "Valid/MSE",
    "Valid/MSE/pushshapes_sim_u_socket",
    "Valid/Native_MSE",
    "Valid/Native_MSE/pushshapes_sim_u_socket",
    "Valid/EnergyScore@32",
    "Valid/EnergyScore@32/pushshapes_sim_u_socket",
    "Valid/EnergyScoreAccuracy@32",
    "Valid/EnergyScoreAccuracy@32/pushshapes_sim_u_socket",
    "Valid/EnergyScoreDiversity@32",
    "Valid/EnergyScoreDiversity@32/pushshapes_sim_u_socket",
}


def test_read_wandb_history_collects_all_supported_metric_namespaces(
    monkeypatch,
) -> None:
    payloads = [
        _wandb_record(
            **{
                "trainer/global_step": 1,
                "Train/Loss": 1.0,
                "Train/MSE/pushshapes_sim_u_socket": 0.9,
                "Timing/Custom_Sec": 0.2,
                "Optimizer/custom_lr": 3.0e-6,
                "Valid/MSE": 0.8,
                "Valid/Native_MSE/pushshapes_sim_u_socket": 0.7,
                "Ignored/Metric": 123.0,
            }
        ),
        _wandb_exit(),
        None,
    ]

    class FakeDataStore:
        def __init__(self) -> None:
            self.closed = False

        def open_for_scan(self, path: str) -> None:
            assert path.endswith("run.wandb")

        def scan_data(self):
            return payloads.pop(0)

        def close(self) -> None:
            self.closed = True

    store = FakeDataStore()
    monkeypatch.setattr(verifier, "DataStore", lambda: store)
    training, validation, exit_code = verifier.read_wandb_history(
        Path("/tmp/run.wandb")
    )

    assert exit_code == 0
    assert store.closed is True
    assert training == [
        {
            "trainer_global_step": 1,
            "train_metrics": {
                "Train/Loss": 1.0,
                "Train/MSE/pushshapes_sim_u_socket": 0.9,
            },
            "timing_metrics": {"Timing/Custom_Sec": 0.2},
            "optimizer_metrics": {"Optimizer/custom_lr": 3.0e-6},
        }
    ]
    assert validation == [
        {
            "trainer_global_step": 1,
            "validation_metrics": {
                "Valid/MSE": 0.8,
                "Valid/Native_MSE/pushshapes_sim_u_socket": 0.7,
            },
        }
    ]


def test_configured_metrics_are_required_on_finite_appropriate_rows() -> None:
    configured_metrics = [
        "Train/MSE",
        "Train/MSE/pushshapes_sim_u_socket",
        "Timing/Custom_Sec",
        "Optimizer/custom_lr",
        "Valid/MSE",
        "Valid/Native_MSE/pushshapes_sim_u_socket",
    ]
    cfg = OmegaConf.create(
        {
            "run_provenance": {
                "required_wandb_metrics": configured_metrics,
            }
        }
    )
    configured, required_step, required_validation = verifier._required_metric_contract(
        cfg
    )

    assert configured == configured_metrics
    assert required_step == {
        "train_metrics": {
            "Train/Loss",
            "Train/MSE",
            "Train/MSE/pushshapes_sim_u_socket",
        },
        "timing_metrics": {
            "Timing/Process_Batch_Sec",
            "Timing/Forward_Pass_Sec",
            "Timing/Compute_Losses_Sec",
            "Timing/Custom_Sec",
        },
        "optimizer_metrics": {
            "Optimizer/param_group_0_lr",
            "Optimizer/custom_lr",
        },
    }
    assert required_validation == {
        "Valid/MSE",
        "Valid/Native_MSE/pushshapes_sim_u_socket",
    }

    row = {
        "trainer_global_step": 0,
        "train_metrics": {metric: 1.0 for metric in required_step["train_metrics"]},
        "timing_metrics": {metric: 1.0 for metric in required_step["timing_metrics"]},
        "optimizer_metrics": {
            metric: 3.0e-6 for metric in required_step["optimizer_metrics"]
        },
    }
    history = [row, {**row, "trainer_global_step": 1}]
    _, steps = verifier._select_dense_training_history(history, required_step)
    assert steps == [0, 1]

    missing_train_metric = [
        row,
        {
            **history[1],
            "train_metrics": {
                "Train/Loss": 1.0,
                "Train/MSE": 1.0,
            },
        },
    ]
    with pytest.raises(AssertionError):
        verifier._select_dense_training_history(missing_train_metric, required_step)

    nonfinite_training = [
        row,
        {
            **history[1],
            "train_metrics": {
                **history[1]["train_metrics"],
                "Train/MSE": float("nan"),
            },
        },
    ]
    with pytest.raises(AssertionError):
        verifier._select_dense_training_history(nonfinite_training, required_step)

    validation_history = [
        {
            "trainer_global_step": 1,
            "validation_metrics": {metric: 0.5 for metric in required_validation},
        }
    ]
    selected = verifier._select_scheduled_validation(
        validation_history,
        [19, 20],
        required_validation,
    )
    assert selected == validation_history[0]

    with pytest.raises(AssertionError):
        verifier._select_scheduled_validation(
            [
                {
                    "trainer_global_step": 1,
                    "validation_metrics": {"Valid/MSE": 0.5},
                }
            ],
            [19, 20],
            required_validation,
        )
    with pytest.raises(AssertionError):
        verifier._select_scheduled_validation(
            [
                {
                    "trainer_global_step": 1,
                    "validation_metrics": {
                        "Valid/MSE": 0.5,
                        "Valid/Native_MSE/pushshapes_sim_u_socket": float("inf"),
                    },
                }
            ],
            [19, 20],
            required_validation,
        )


def test_sparse_log_metrics_are_routed_without_weakening_dense_steps() -> None:
    cfg = OmegaConf.create(
        {
            "run_provenance": {
                "required_wandb_metrics": [
                    "Train/MSE",
                    "log/unite_gradient_cosine",
                    "Valid/MSE",
                ]
            }
        }
    )
    _, required_step, required_validation = verifier._required_metric_contract(cfg)
    assert required_step["telemetry_metrics"] == {"log/unite_gradient_cosine"}
    assert required_validation == {"Valid/MSE"}

    history = []
    for step in range(3):
        row = {
            "trainer_global_step": step,
            "train_metrics": {"Train/Loss": 1.0, "Train/MSE": 0.5},
            "timing_metrics": {
                "Timing/Process_Batch_Sec": 0.1,
                "Timing/Forward_Pass_Sec": 0.2,
                "Timing/Compute_Losses_Sec": 0.3,
            },
            "optimizer_metrics": {"Optimizer/param_group_0_lr": 1.0e-4},
        }
        if step == 0:
            row["telemetry_metrics"] = {"log/unite_gradient_cosine": 0.25}
        history.append(row)

    _, steps = verifier._select_dense_training_history(
        history, required_step, expected_steps=3
    )
    assert steps == [0, 1, 2]
    telemetry = verifier._validate_required_telemetry(history, required_step)
    assert telemetry == [
        {
            "metric": "log/unite_gradient_cosine",
            "occurrences": [{"trainer_global_step": 0, "value": 0.25}],
        }
    ]


def test_validation_step_index_is_converted_to_completed_optimizer_steps() -> None:
    history = [
        {
            "trainer_global_step": 0,
            "validation_metrics": {"Valid/MSE": 0.5},
        }
    ]
    selected = verifier._select_scheduled_validation(
        history,
        [19],
        {"Valid/MSE"},
        minimum_validation_step=1,
    )
    assert selected == history[0]


def test_ema_backends_are_mutually_exclusive() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"ema": {"enabled": True}},
            "callbacks": {
                "ema": {
                    "_target_": "egomimic.utils.ema_callback.EMACallback",
                    "decay": 0.9978,
                    "validate_with_ema": True,
                }
            },
        }
    )
    with pytest.raises(AssertionError, match="cannot both be enabled"):
        verifier._configured_ema_backends(cfg)


def test_released_register_rows_remain_launch_blocked() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "unite_flow_updates_per_reconstruction": 0,
                "unite_gradient_telemetry_every_n_steps": 0,
                "robomimic_model": {
                    "stages": [
                        {
                            "_target_": (
                                "egomimic.pipeline.stages_unite_released."
                                "ReleasedRecipeUniteLatentPolicy"
                            )
                        }
                    ]
                },
            }
        }
    )
    with pytest.raises(AssertionError, match="remain launch-blocked"):
        verifier._validate_unite_alternating_contract(
            cfg,
            training_history=[],
            expected_steps=2,
            external_ema_enabled=False,
        )


def test_legacy_validation_contract_still_requires_finite_action_mse() -> None:
    history = [
        {
            "trainer_global_step": 1,
            "validation_metrics": {
                "Valid/emb19_actions_action_mse": 0.1,
                "Valid/emb20_actions_action_mse": 0.2,
            },
        }
    ]
    assert verifier._select_scheduled_validation(history, [19, 20], set()) == history[0]

    history[0]["validation_metrics"]["Valid/emb20_actions_action_mse"] = float("nan")
    with pytest.raises(AssertionError):
        verifier._select_scheduled_validation(history, [19, 20], set())


def test_checkpoint_model_wrapper_is_reconstructed_strictly_on_cpu(
    monkeypatch,
) -> None:
    captured = {}

    class FakeModelWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)
            self.register_buffer("scale", torch.ones(1))

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, **kwargs):
            captured["checkpoint_path"] = checkpoint_path
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(verifier.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(verifier, "ModelWrapper", FakeModelWrapper)
    result = verifier._strict_load_model_wrapper(Path("/tmp/smoke.ckpt"))

    assert captured == {
        "checkpoint_path": "/tmp/smoke.ckpt",
        "map_location": "cpu",
        "weights_only": False,
        "strict": True,
    }
    assert result == {
        "status": "passed",
        "model_class": (
            f"{FakeModelWrapper.__module__}.{FakeModelWrapper.__qualname__}"
        ),
        "map_location": "cpu",
        "strict": True,
        "state_dict_key_count": 3,
        "parameter_count": 9,
        "buffer_count": 1,
    }


def test_checkpoint_reload_uses_the_configured_wrapper_class(monkeypatch) -> None:
    captured = {}

    class FakeModelWrapper(torch.nn.Module):
        pass

    class ConfiguredWrapper(FakeModelWrapper):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, **kwargs):
            captured.update(checkpoint_path=checkpoint_path, **kwargs)
            return cls()

    monkeypatch.setattr(verifier.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(verifier, "ModelWrapper", FakeModelWrapper)
    monkeypatch.setattr(
        verifier.hydra.utils,
        "get_class",
        lambda target: ConfiguredWrapper,
    )
    config = OmegaConf.create({"model": {"_target_": "tests.ConfiguredWrapper"}})

    result = verifier._strict_load_model_wrapper(
        Path("/tmp/released.ckpt"),
        config=config,
    )

    assert captured == {
        "checkpoint_path": "/tmp/released.ckpt",
        "map_location": "cpu",
        "weights_only": False,
        "strict": True,
    }
    assert result["model_class"].endswith(".ConfiguredWrapper")
    assert result["parameter_count"] == 2


def test_checkpoint_reload_requires_cuda_to_be_hidden(monkeypatch) -> None:
    monkeypatch.setattr(verifier.torch.cuda, "is_available", lambda: True)
    with pytest.raises(AssertionError, match="CUDA hidden"):
        verifier._strict_load_model_wrapper(Path("/tmp/smoke.ckpt"))


@pytest.mark.parametrize(
    "energy_shape",
    [
        {"action_dim": 4},
        {"action_dims": {"pushshapes_sim_u_socket": 4}},
    ],
    ids=("paper_shared_action_dim", "unite_per_domain_action_dims"),
)
def test_energy_score_artifact_validation_accepts_both_schemas(
    tmp_path,
    energy_shape,
) -> None:
    output_dir = tmp_path / "run"
    artifact_root = output_dir / "validation_predictions" / "energy_score"
    artifact_dir = artifact_root / "epoch-0-step-2"
    artifact_dir.mkdir(parents=True)
    seed_bank = tmp_path / "energy_seeds.json"
    seed_bank.write_text(json.dumps({"seeds": list(range(32))}))
    predictions = torch.zeros(32, 2, 16, 4)
    targets = torch.zeros(2, 16, 4)
    payload = {
        "schema_version": 1,
        "metric": "EnergyScore@32",
        "sample_count": 32,
        "seed_bank": list(range(32)),
        "seed_bank_sha256": verifier._sha256(seed_bank),
        "global_step": 2,
        "rank": 0,
        "domains": {
            "pushshapes_sim_u_socket": {
                "embodiment_id": 19,
                "predictions": predictions,
                "targets": targets,
                "accuracy_by_condition": torch.zeros(2),
                "diversity_by_condition": torch.zeros(2),
                "score_by_condition": torch.zeros(2),
            }
        },
    }
    torch.save(payload, artifact_dir / "rank-0-batch-0.pt")
    config = OmegaConf.create(
        {
            "evaluator": {
                "energy_score": {
                    "enabled": True,
                    "sample_count": 32,
                    "max_batches_per_rank": 1,
                    "seed_bank_path": str(seed_bank),
                    "seed_bank_sha256": verifier._sha256(seed_bank),
                    "artifact_root": str(artifact_root),
                    **energy_shape,
                }
            }
        }
    )

    records = verifier._validate_energy_score_artifacts(
        output_dir,
        config,
        global_step=2,
        expected_world_size=1,
        required_embodiments=[19],
    )

    assert len(records) == 1
    assert records[0]["rank"] == 0
    assert Path(records[0]["path"]).name == "rank-0-batch-0.pt"


def test_exact_wandb_visibility_requires_every_finite_metric(monkeypatch) -> None:
    required = {"Train/MSE", "Valid/MSE"}
    requested = []

    class FakeRun:
        path = ["entity", "project", "run-id"]

        def scan_history(self, *, keys, page_size):
            requested.append((keys, page_size))
            return iter([{"Train/MSE": 1.0}, {"Valid/MSE": 0.5}])

    class FakeApi:
        def run(self, run_path):
            assert run_path == "entity/project/run-id"
            return FakeRun()

    monkeypatch.setattr(wandb, "Api", lambda timeout: FakeApi())
    verifier._verify_wandb_visibility("entity/project/run-id", required)

    assert requested == [(["Train/MSE", "Valid/MSE"], 1000)]


@pytest.mark.parametrize(
    ("topology", "ema_backend"),
    (("shared", "callback"), ("separate", "internal")),
)
def test_released_sweep_gate_unites_reload_ema_metrics_and_telemetry(
    tmp_path,
    monkeypatch,
    topology,
    ema_backend,
) -> None:
    output_dir = tmp_path / "run"
    config_path = output_dir / ".hydra" / "config.yaml"
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    stream_path = output_dir / "wandb" / "run-test" / "run-test.wandb"
    for path in (config_path, checkpoint_path, stream_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    parameter_manifest = tmp_path / "parameter_manifest.json"
    parameter_manifest.write_text(
        json.dumps(
            {
                "topology": topology,
                "num_latent_tokens": 4,
                "latent_dim": 16,
                "action_horizon": 16,
            }
        )
    )
    split_manifest = tmp_path / "split.json"
    normalization_artifact = tmp_path / "normalization.json"
    split_manifest.write_text("{}\n")
    normalization_artifact.write_text("{}\n")

    model_config = {
        "_target_": "tests.ConfiguredReleasedWrapper",
        "share_encoder_denoiser": topology == "shared",
        "latent_dim": 16,
        "num_latent_tokens": 4,
        "unite_flow_updates_per_reconstruction": 0,
        "unite_gradient_telemetry_every_n_steps": 3,
    }
    callback_config = None
    if ema_backend == "internal":
        model_config["ema"] = {
            "enabled": True,
            "power": 0.75,
            "max_value": 0.9999,
            "use_for_validation": True,
        }
    else:
        callback_config = {
            "ema": {
                "_target_": "egomimic.utils.ema_callback.EMACallback",
                "decay": 0.9978,
                "validate_with_ema": True,
            }
        }
    config = OmegaConf.create(
        {
            "trainer": {
                "max_steps": 3,
                "limit_train_batches": 3,
                "val_check_interval": 3,
                "limit_val_batches": 1,
                "num_sanity_val_steps": 0,
                "precision": "bf16",
                "strategy": "ddp_find_unused_parameters_true",
                "devices": 2,
                "num_nodes": 1,
                "accumulate_grad_batches": 1,
            },
            "model": model_config,
            "callbacks": callback_config,
            "data": {
                "train_datasets": {"pushshapes_sim_u_socket": {}},
                "valid_datasets": {"pushshapes_sim_u_socket": {}},
            },
            "evaluator": {
                "energy_score": {
                    "enabled": True,
                    "sample_count": 32,
                    "seed_bank_sha256": (
                        "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
                    ),
                    "action_dims": {"pushshapes_sim_u_socket": 4},
                }
            },
            "run_provenance": {
                "sweep_task_id": "us_unite_register_test",
                "topology": topology,
                "num_latent_tokens": 4,
                "latent_dim": 16,
            },
        }
    )
    checkpoint = {
        "global_step": 3,
        "optimizer_states": [
            {
                "adamw": {"param_groups": [{"lr": 1.0e-4}]},
                "muon": {"param_groups": [{"lr": 2.0e-4}]},
                "group_manifest": {},
            }
        ],
        "lr_schedulers": [{"last_epoch": 3}],
    }
    if ema_backend == "callback":
        checkpoint.update(
            ema_state_dict={"model.weight": torch.ones(2)},
            ema_decay=0.9978,
            ema_num_updates=3,
            ema_validate_with_ema=True,
        )
    monkeypatch.setattr(verifier.torch, "load", lambda *args, **kwargs: checkpoint)

    strict_calls = []
    strict_record = {
        "status": "passed",
        "model_class": "tests.ConfiguredReleasedWrapper",
        "map_location": "cpu",
        "strict": True,
    }
    if ema_backend == "internal":
        strict_record["ema"] = {
            "backend": "model_wrapper",
            "enabled": True,
            "optimization_step": 3,
            "decay": 0.5,
            "power": 0.75,
            "max_value": 0.9999,
            "use_for_validation": True,
            "parameter_tree_exact": True,
            "parameters_finite": True,
        }

    def strict_reload(path, *, config, expected_steps):
        strict_calls.append((path, config, expected_steps))
        return strict_record

    monkeypatch.setattr(verifier, "_strict_load_model_wrapper", strict_reload)
    topology_metrics = (
        {
            "log/unite_gradient_cosine": 0.25,
            "log/unite_recon_grad_norm": 1.0,
            "log/unite_denoise_grad_norm": 2.0,
        }
        if topology == "shared"
        else {
            "log/unite_tokenizer_recon_grad_norm": 1.0,
            "log/unite_denoiser_flow_grad_norm": 2.0,
        }
    )
    training_history = [
        {
            "trainer_global_step": 0,
            "train_metrics": {key: 1.0 for key in _RELEASED_TRAIN_METRICS},
            "timing_metrics": {},
            "optimizer_metrics": {key: 1.0e-4 for key in _RELEASED_OPTIMIZER_METRICS},
            # The generic row contract omits this optional key when no sparse
            # telemetry was logged. The released verifier must accept that row.
        },
        {
            "trainer_global_step": 1,
            "train_metrics": {key: 0.5 for key in _RELEASED_TRAIN_METRICS},
            "timing_metrics": {},
            "optimizer_metrics": {key: 1.0e-4 for key in _RELEASED_OPTIMIZER_METRICS},
        },
        {
            "trainer_global_step": 2,
            "train_metrics": {key: 0.25 for key in _RELEASED_TRAIN_METRICS},
            "timing_metrics": {},
            "optimizer_metrics": {key: 1.0e-4 for key in _RELEASED_OPTIMIZER_METRICS},
            "telemetry_metrics": topology_metrics,
        },
    ]
    validation_history = [
        {
            "trainer_global_step": 2,
            "validation_metrics": {key: 0.25 for key in _RELEASED_VALID_METRICS},
        }
    ]
    monkeypatch.setattr(
        verifier,
        "read_wandb_history",
        lambda path: (training_history, validation_history, 0),
    )
    energy_calls = []

    def validate_energy(path, resolved_config, **kwargs):
        energy_calls.append((path, resolved_config, kwargs))
        return [{"path": "energy.pt", "rank": 0, "sha256": "abc"}]

    monkeypatch.setattr(
        verifier,
        "_validate_energy_score_artifacts",
        validate_energy,
    )
    visibility_calls = []
    monkeypatch.setattr(
        verifier,
        "_verify_wandb_visibility",
        lambda run_path, required: visibility_calls.append((run_path, required)),
    )

    record = verifier._verify_released_sweep_smoke(
        output_dir,
        config,
        [19],
        "a" * 40,
        "ddp_find_unused_parameters_true",
        2,
        3,
        3,
        3,
        topology,
        "us_unite_register_test",
        16,
        4,
        "entity/project/run-id",
        parameter_manifest,
        split_manifest,
        normalization_artifact,
    )

    assert strict_calls == [(checkpoint_path, config, 3)]
    assert energy_calls == [
        (
            output_dir,
            config,
            {
                "global_step": 3,
                "expected_world_size": 2,
                "required_embodiments": [19],
            },
        )
    ]
    expected_required = (
        _RELEASED_TRAIN_METRICS
        | _RELEASED_OPTIMIZER_METRICS
        | _RELEASED_VALID_METRICS
        | set(topology_metrics)
    )
    assert visibility_calls == [("entity/project/run-id", expected_required)]
    assert record["model_wrapper_load"] is strict_record
    assert record["ema"]["backend"] == (
        "callback" if ema_backend == "callback" else "model_wrapper"
    )
    assert record["dense_training_steps"] == [0, 1, 2]
    assert record["required_wandb_metrics"] == sorted(expected_required)
