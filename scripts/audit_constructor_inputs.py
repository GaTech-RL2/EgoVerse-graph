"""Explicit offline constructor inputs, never run or checkpoint certificates.

Reusable recipes require persisted run identities or learned tokenizers at
launch. This audit supplies structural inputs only, on a cloned configuration;
the production factories, strict reloads and launch guards are not changed.
"""

import hashlib
import inspect
from contextlib import contextmanager
from unittest.mock import patch

import torch
from hydra.utils import get_method, instantiate
from omegaconf import OmegaConf


def evaluator_inputs(config):
    cfg = OmegaConf.to_container(config, resolve=True)
    if cfg.get("_target_") != (
        "egomimic.eval.libero_action_flow_eval.LiberoActionFlowEvaluator"
    ):
        return cfg, {}
    from egomimic.benchmarks.libero.native_diagnostic_config import (
        build_native_diagnostic_config,
    )

    prefix = "egomimic.benchmarks.libero.native_artifact_identity."
    for key, factory in (
        ("artifact_identity", "build_native_artifact_identity_from_config"),
        ("native_diagnostic_config", "build_native_diagnostic_config_from_config"),
    ):
        descriptor = cfg.get(key)
        if descriptor is not None:
            if descriptor.get("_target_") != prefix + factory:
                raise ValueError(
                    "Offline native constructor needs canonical descriptors"
                )
            kwargs = {k: v for k, v in descriptor.items() if k != "_target_"}
            inspect.signature(get_method(prefix + factory)).bind(**kwargs)
    identity = {
        "suite": "libero10",
        "seed": 42,
        "source": "libero_panda",
        "action_dim": 7,
        "action_horizon": 16,
        "sample_count": 32,
        "energy_seed_bank_sha256": cfg["energy_seed_bank_sha256"],
        "inference_method": "euler",
        "inference_steps": 50,
        "normalization_scope": "training_episodes_only",
        "effective_batch_size": 32,
        "homogeneous": "not_applicable_single_source",
        "source_commit": "0" * 40,
        **{
            key: "0" * 64
            for key in (
                "resolved_config_sha256",
                "split_sha256",
                "normalizer_state_sha256",
                "dataset_logical_sha256",
            )
        },
    }
    cfg["artifact_identity"] = identity
    cfg["native_diagnostic_config"] = build_native_diagnostic_config(
        identity,
        "/tmp/egoverse-constructor-audit/diagnostics",
        cfg["energy_seed_bank_path"],
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
    return cfg, {
        "external_binding": "UNVERIFIED: synthetic profile/identity; runtime factories not executed",
        "scope": "constructor only, not model-profile pairing or native run validation",
    }


@contextmanager
def model_inputs(cfg, config_root):
    stages = cfg.model.pipeline.get("stages", [])
    policies = [
        s.get("policy")
        for s in stages
        if s.get("policy", {}).get("_target_")
        == "egomimic.models.oat.factory.make_policy"
    ]
    if not policies:
        yield {}
        return
    if any(p.get("tokenizer_checkpoint") is not None for p in policies):
        raise ValueError(
            "Offline audit cannot substitute an explicit learned checkpoint"
        )
    representation = cfg.model.get("benchmark_protocol", {}).get(
        "action_representation"
    )
    template = (
        config_root
        / "model/oat"
        / ("arc_tokenizer.yaml" if representation == "arc_oat" else "tokenizer.yaml")
    )
    context = OmegaConf.create(
        {
            "benchmark": OmegaConf.to_container(cfg.benchmark, resolve=True),
            "model": OmegaConf.load(template),
        }
    )
    candidates = [
        s.tokenizer
        for s in context.model.pipeline.stages
        if s.get("_target_") == "egomimic.pipeline.stages_oat.OATTokenizerStage"
    ]
    if len(candidates) != 1:
        raise ValueError("Offline tokenizer template must declare exactly one owner")
    tokenizer_config = OmegaConf.to_container(candidates[0], resolve=True)

    def construct(checkpoint, *, use_ema=True):
        del use_ema
        if checkpoint is not None or torch.empty(0).device.type != "meta":
            raise ValueError(
                "Synthetic tokenizer input is restricted to meta construction"
            )
        # Build the declared tokenizer graph, including its real ARC codecs.
        # Shape alone does not certify the codec's units or representation.
        from egomimic.models.oat.checkpoint import validate_input_representation
        from egomimic.pipeline.stages_oat import OATTokenizerStage

        graph = instantiate(context.model.pipeline, device="meta")
        representation_context = validate_input_representation(graph.pipeline.stages)
        tokenizer = next(
            stage.tokenizer
            for stage in graph.pipeline.stages
            if isinstance(stage, OATTokenizerStage)
        )
        tokenizer._training_input_representation = representation_context
        return tokenizer.eval().requires_grad_(False)

    from egomimic.models.oat import factory

    with patch.object(factory, "load_tokenizer", side_effect=construct):
        yield {
            "external_binding": "UNVERIFIED: untrained meta tokenizer, no checkpoint loaded",
            "tokenizer_template": str(template.relative_to(config_root)),
            "tokenizer_template_sha256": hashlib.sha256(
                template.read_bytes()
            ).hexdigest(),
            "tokenizer_config": tokenizer_config,
        }
