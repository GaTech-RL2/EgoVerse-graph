"""Structural fixtures cannot become learned initialization or run evidence."""

import json

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.audit_components import offline_static_buffer_construction
from scripts.audit_constructor_inputs import evaluator_inputs, model_inputs
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit


def test_native_fixture_is_explicit_and_does_not_read_runtime_snapshot(monkeypatch):
    monkeypatch.setattr(
        "egomimic.benchmarks.libero.native_artifact_identity._snapshot",
        lambda *args: pytest.fail("offline constructor read a runtime snapshot"),
    )
    path = (
        CONFIGS
        / "experiment/libero/action_flow_libero10_h240_euler50_dithalf_80k_s42.yaml"
    )
    with compose_for_audit(path) as cfg:
        before = OmegaConf.to_container(cfg.evaluator, resolve=True)
        inputs, evidence = evaluator_inputs(cfg.evaluator)
        evaluator = instantiate(inputs)
        assert evaluator.shared_diagnostics.max_samples == 8
        assert evaluator.shared_diagnostics.activation_layer_map == tuple(
            (i, i) for i in range(12)
        )
        assert evidence["external_binding"].startswith("UNVERIFIED")
        assert OmegaConf.to_container(cfg.evaluator, resolve=True) == before
        invalid = OmegaConf.create(before)
        invalid.artifact_identity._target_ = "not.a.canonical.factory"
        with pytest.raises(ValueError, match="canonical descriptors"):
            evaluator_inputs(invalid)


@pytest.mark.parametrize(
    ("recipe", "dimension", "horizon"),
    [
        ("libero_oatpolicy", 7, 32),
        # Use the declared ARC recipe geometry, not the native-action horizon.
        ("libero_arc_oatpolicy", 12, 32),
    ],
)
def test_meta_tokenizer_comes_from_declared_template_and_scope_restores(
    recipe, dimension, horizon
):
    from egomimic.models.oat import factory

    original = factory.load_tokenizer
    with compose_for_audit(CONFIGS / f"experiment/oat/{recipe}.yaml") as cfg:
        before = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
        with (
            torch.device("meta"),
            offline_static_buffer_construction(),
            model_inputs(cfg, CONFIGS) as evidence,
        ):
            tokenizer = factory.load_tokenizer(None)
            assert tokenizer.decoder.sample_dim == dimension
            assert tokenizer.decoder.sample_horizon == horizon
            assert all(
                p.device.type == "meta" and not p.requires_grad
                for p in tokenizer.parameters()
            )
            assert evidence["external_binding"].startswith("UNVERIFIED")
            assert not hasattr(tokenizer, "_training_data_context")
            if recipe == "libero_arc_oatpolicy":
                from egomimic.pipeline.stages_libero_arc import LiberoArcStage

                codec = instantiate(cfg.model.pipeline.stages[0])
                assert isinstance(codec, LiberoArcStage)
                assert (
                    tokenizer._training_input_representation
                    == codec.representation_context()
                )
            else:
                assert tokenizer._training_input_representation is None
        assert factory.load_tokenizer is original

        assert (
            json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
            == before
        )
    with pytest.raises(ValueError, match="trained OAT tokenizer"):
        factory.load_tokenizer(None)


def test_initializer_rejects_cpu_use_and_explicit_checkpoint_and_restores():
    from egomimic.models.oat import factory

    original = factory.load_tokenizer
    with compose_for_audit(CONFIGS / "experiment/oat/libero_oatpolicy.yaml") as cfg:
        with pytest.raises(ValueError, match="restricted to meta"):
            with model_inputs(cfg, CONFIGS):
                factory.load_tokenizer(None)
        assert factory.load_tokenizer is original
        cfg.benchmark.tokenizer_checkpoint = "/real/run/tokenizer.ckpt"
        with pytest.raises(ValueError, match="explicit learned checkpoint"):
            with model_inputs(cfg, CONFIGS):
                pytest.fail("explicit checkpoint replaced")
        assert factory.load_tokenizer is original


def test_arc_structural_metadata_preserves_real_values_rng_and_method():
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    original = LiberoArcStage.representation_context
    with torch.device("cpu"):
        expected = LiberoArcStage(num_waypoints=32, horizon=32).representation_context()
    rng = torch.get_rng_state().clone()
    with torch.device("meta"), offline_static_buffer_construction():
        codec = LiberoArcStage(num_waypoints=32, horizon=32)
        assert codec.action_scale.device.type == "meta"
        assert codec.representation_context() == expected
    assert LiberoArcStage.representation_context is original
    assert torch.equal(torch.get_rng_state(), rng)
