"""Offline prepared-data preflight retains real graph construction and guards."""

from pathlib import Path

import pytest
import torch

from egomimic.pipeline.construction import restoring_parameters
from egomimic.rldb.dino_proprio_datamodule import DinoProprioDataModule
from egomimic.rldb.pixel_proprio_datamodule import PixelProprioDataModule
from egomimic.rldb.world_model_datamodule import WorldModelDataModule
from scripts import audit_components as components
from scripts.audit_hydra_configs import CONFIGS


@pytest.mark.parametrize(
    "module, options",
    [
        (WorldModelDataModule, {}),
        (DinoProprioDataModule, {}),
        (
            DinoProprioDataModule,
            {
                "task": "direct_action",
                "action_manifest_path": "/not-opened/actions.json",
            },
        ),
        (PixelProprioDataModule, {"image_only": True}),
        (
            PixelProprioDataModule,
            {
                "mode": "discrete",
                "sampling_strategy": "discrete_window",
                "discrete_frames": 16,
                "stride": 4,
            },
        ),
        (
            PixelProprioDataModule,
            {
                "task": "visual_state_action",
                "distributed_world_size": 4,
                "checkpointable_sampling": True,
                "validation_sampling_strategy": "discrete_window",
                "validation_discrete_frames": 16,
                "validation_stride": 4,
            },
        ),
    ],
)
def test_configuration_preflight_does_not_open_or_bind_data(
    monkeypatch, module, options
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Configuration preflight must not access data or construct a loader"
        )

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(module, "setup", forbidden)
    dm = module("/not-opened/manifest.json", **options)
    before = dict(dm.settings)
    result = dm.preflight_configuration()
    assert result["scope"] == "configuration_only"
    assert result["artifacts_opened"] is False
    assert result["sampling"] == before == dm.settings
    assert dm.datasets == {}
    assert dm.train_sampler is None
    assert dm.consumed_batches == 0


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"num_workers": -1},
        {"seed": -1},
        {"min_frames": 1},
        {"min_frames": 8, "max_frames": 4},
        {"discrete_frames": True},
        {"stride": 0},
        {"mode": "unknown"},
        {"forecasting_probability": float("nan")},
        {"forecasting_probability": 1.1},
        {"max_past_seconds": 1.0},
        {"mode": "discrete", "max_past_seconds": 1.0, "max_future_seconds": 1.0},
    ],
)
def test_prepared_preflight_rejects_invalid_configuration_without_setup(options):
    dm = WorldModelDataModule("/not-opened/manifest.json", **options)
    with pytest.raises(ValueError):
        dm.preflight_configuration()
    assert dm.datasets == {}


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "discrete", "sampling_strategy": "full_episode_uniform"},
        {"sampling_strategy": "unknown"},
        {"max_past_seconds": 1.0, "max_future_seconds": 1.0},
        {"validation_sampling_strategy": "unknown"},
        {"validation_discrete_frames": 1},
        {"validation_stride": 0},
        {"task": "direct_action", "validation_sampling_strategy": "discrete_window"},
    ],
)
def test_pixel_preflight_validates_validation_sampling_too(options):
    dm = PixelProprioDataModule("/not-opened/manifest.json", **options)
    with pytest.raises(ValueError):
        dm.preflight_configuration()
    assert dm.datasets == {}


def test_direct_action_still_requires_explicit_action_manifest():
    with pytest.raises(ValueError, match="verified action alignment manifest"):
        DinoProprioDataModule("/not-opened/manifest.json", task="direct_action")


def test_family_component_audit_builds_real_meta_graphs_without_data_or_weights(
    monkeypatch,
):
    families = ("world_model", "dino_proprio", "pixel_proprio")
    paths = [
        path
        for group in ("data", "model", "experiment")
        for family in families
        for path in (CONFIGS / group / family).glob("*.yaml")
    ]
    assert len(paths) == 17
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert "/tmp/egoverse-config-audit/" not in str(path), path
        return original_open(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Offline component audit accessed data or external weights")

    for module in (WorldModelDataModule, DinoProprioDataModule, PixelProprioDataModule):
        monkeypatch.setattr(module, "setup", forbidden)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(torch, "load", forbidden)
    original_construct = components._instantiate_model_wrapper
    constructed = []

    def construct(cfg):
        wrapper = original_construct(cfg)
        parameters = list(wrapper.parameters())
        assert parameters and all(p.device.type == "meta" for p in parameters)
        assert wrapper.model.initialization_receipts == []
        constructed.append(
            {
                "stages": len(wrapper.model.pipeline.stages),
                "initialization": bool(cfg.model.pipeline.get("initialization")),
                "frozen": any(not p.requires_grad for p in parameters),
                "trainable": any(p.requires_grad for p in parameters),
            }
        )
        return wrapper

    monkeypatch.setattr(components, "_instantiate_model_wrapper", construct)
    assert restoring_parameters() is False
    records = components.audit_components(paths)
    assert restoring_parameters() is False
    assert not [row for row in records if row["status"] != "passed"], records
    for record in records:
        assert set(record["components"]) == {"data", "evaluator", "model"}
        assert record["components"]["data"]["scope"] == "configuration_only"
        assert record["components"]["model"]["optimizer_scheduler"] == "passed"
        assert (
            record["components"]["model"]["late_bound_parameter_placeholder"] is False
        )
    assert any(item["stages"] == 4 for item in constructed)
    readouts = [item for item in constructed if item["initialization"]]
    assert readouts and all(item["stages"] == 10 for item in readouts)
    assert all(item["frozen"] and item["trainable"] for item in readouts)
