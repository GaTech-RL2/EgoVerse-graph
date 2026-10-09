"""Maintained-launcher integration proposal, never a training entry point.

Pure metadata only. This module emits phase bindings for integration into the
single maintained launcher. It cannot execute trainHydra or submit a job.
Resolved configs must still be composed and constructor/smoke tested through
scheduled pinned compute after a reviewed source/profile implementation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

LAUNCHER_SHA = "a4b1f0f54767d8eb723158b108afa9c682b05f945a3923c0d2a2951abe074bc4"
REFERENCE_SHA = "720e77b2f7c0e528441170a8a7838a1c8644ccfa083ba234c434ae29de2e1ba3"
CPU_SHA = "6137e056e3c901499eced1a38d0db090c7ddb0077dc90c41e90c0f8ea39089c6"
SEED_BANK_SHA = "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"

COMMON = {
    "seed": 42, "trainer.precision": "bf16-mixed",
    "trainer.devices": 1, "trainer.num_nodes": 1,
    "trainer.accumulate_grad_batches": 1,
    "model.action_dim": 7, "model.action_horizon": 16,
    "model.hidden_dim": 240, "model.num_latent_tokens": 8,
    "model.latent_dim": 16, "model.condition_dim": 128,
    "model.flow_samples_per_content": 14, "model.flow_mini_batch": 14,
    "model.pipeline.stages.6.inference_method": "euler",
    "model.pipeline.stages.6.num_inference_steps": 50,
    "model.pipeline.stages.8.action_velocity_weight": 1.0,
    "model.pipeline.stages.4.encoder.backbone.num_heads": 8,
    "model.pipeline.stages.6.field.backbone.num_heads": 8,
    "model.pipeline.stages.7.decoder.num_heads": 8,
    "model.pipeline.stages.1.encoder.img_encoders.front_img_1.resnet_model": "resnet18_half",
    "model.pipeline.stages.1.encoder.img_encoders.front_img_2.resnet_model": "resnet18_half",
    "data.train_dataloader_params.libero_panda.batch_size": 32,
    "data.valid_dataloader_params.libero_panda.batch_size": 32,
    "evaluator._target_": "egomimic.eval.libero_action_flow_eval.LiberoActionFlowEvaluator",
    "evaluator.energy_sample_count": 32,
    "evaluator.energy_seed_bank_sha256": SEED_BANK_SHA,
    "callbacks.model_checkpoint.save_top_k": -1,
    "callbacks.model_checkpoint.filename": "epoch-{epoch}-step-{step}",
}
PHASES = {
    "preflight": {"trainer.max_steps": 80000, "trainer.val_check_interval": 15000,
                  "trainer.limit_val_batches": 8, "model.gradient_telemetry_cadence": 100,
                  "callbacks.model_checkpoint.every_n_train_steps": 5000},
    "smoke": {"trainer.max_steps": 2, "trainer.val_check_interval": 2,
              "trainer.limit_train_batches": 2, "trainer.limit_val_batches": 1,
              "model.gradient_telemetry_cadence": 2,
              "callbacks.model_checkpoint.every_n_train_steps": 1},
    "full": {"trainer.max_steps": 80000, "trainer.val_check_interval": 15000,
             "trainer.limit_val_batches": 8, "model.gradient_telemetry_cadence": 100,
             "callbacks.model_checkpoint.every_n_train_steps": 5000},
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def phase_bindings(phase):
    if phase not in PHASES:
        raise ValueError("unknown maintained phase")
    return {**COMMON, **PHASES[phase]}

def validate_flat_config(config, phase, *, training_seed=42, action_velocity_weight=1.0):
    """Check flattened resolved values, not guessed Hydra argument reachability."""
    if action_velocity_weight not in (0.0, 1.0):
        raise ValueError("unsupported native action velocity weight")
    wanted_bindings = {**phase_bindings(phase), "seed": training_seed, "model.pipeline.stages.8.action_velocity_weight": action_velocity_weight}
    for key, wanted in wanted_bindings.items():
        if key not in config or config[key] != wanted:
            raise ValueError(("resolved profile mismatch", key, wanted, config.get(key)))
    for stage in (4, 6):
        stem = f"model.pipeline.stages.{stage}.{'encoder' if stage == 4 else 'field'}.backbone"
        for field, wanted in (("hidden_dim", 240), ("depth", 12), ("num_heads", 8)):
            if config.get(f"{stem}.{field}") != wanted:
                raise ValueError(("DiT constructor mismatch", stem, field))
    if config.get("model.pipeline.stages.7.decoder.num_heads") != 8:
        raise ValueError("decoder head mismatch")
    if config.get("data.train_source_names") != ["libero_panda"] or config.get("data.valid_source_names") != ["libero_panda"]:
        raise ValueError("exact single logical source required")
    if config.get("typed_profile.velocity_augmentation") is not False:
        raise ValueError("noaug row must not retime or augment")
    if config.get("typed_profile.homogeneous") != "not_applicable_single_source":
        raise ValueError("homogeneous grouping claim prohibited")
    if config.get("typed_profile.checkpoint_policy") != "dit-half":
        raise ValueError("typed DiT-half option not resolved")
    if any(k.startswith("evaluator.energy_score_validation_view") for k in config):
        raise ValueError("U-Socket-only evaluator overrides prohibited")
    return True

def validate_native_reference_paths(config):
    """Historical native paths must exist before desired overlay, never aliases."""
    for key in COMMON:
        if key == "trainer.num_nodes":
            if key in config and config[key] != 1:
                raise ValueError("native operational node count must be one")
            continue  # Canonical BASE_OVERRIDES explicitly adds trainer.num_nodes=1.
        if key not in config:
            raise ValueError(("native reference key absent; do not fabricate via overlay", key))
    forbidden = {"model.latent_tokens", "model.flow_samples", "model.flow_minibatch_samples"}
    if forbidden.intersection(config):
        raise ValueError("non-native metadata aliases prohibited")
    return True

def fixture(phase):
    config = phase_bindings(phase)
    for stage in (4, 6):
        stem = f"model.pipeline.stages.{stage}.{'encoder' if stage == 4 else 'field'}.backbone"
        config.update({f"{stem}.hidden_dim": 240, f"{stem}.depth": 12, f"{stem}.num_heads": 8})
    config.update({"model.pipeline.stages.7.decoder.num_heads": 8,
                   "data.train_source_names": ["libero_panda"], "data.valid_source_names": ["libero_panda"],
                   "typed_profile.velocity_augmentation": False,
                   "typed_profile.homogeneous": "not_applicable_single_source",
                   "typed_profile.checkpoint_policy": "dit-half"})
    return config

def regression():
    for phase in PHASES:
        validate_flat_config(fixture(phase), phase)
    mutations = {
        "batch64": ("data.train_dataloader_params.libero_panda.batch_size", 64),
        "wrong_precision": ("trainer.precision", "32-true"),
        "old_sampler": ("model.pipeline.stages.6.inference_method", "dopri5"),
        "wrong_budget": ("trainer.max_steps", 240000),
        "wrong_k": ("evaluator.energy_sample_count", 8),
        "changed_objective": ("model.pipeline.stages.8.action_velocity_weight", 0),
        "wrong_heads": ("model.pipeline.stages.6.field.backbone.num_heads", 6),
        "extra_source": ("data.train_source_names", ["libero_panda", "invented"]),
        "augmentation": ("typed_profile.velocity_augmentation", True),
        "false_homogeneous": ("typed_profile.homogeneous", "grouped64"),
        "missing_policy": ("typed_profile.checkpoint_policy", "unchanged"),
        "usocket_evaluator": ("evaluator.energy_score_validation_view.world_size", 1),
    }
    for name, (key, value) in mutations.items():
        config = fixture("full")
        config[key] = value
        try:
            validate_flat_config(config, "full")
        except ValueError:
            continue
        raise AssertionError(("negative regression unexpectedly accepted", name))
    missing = fixture("full")
    del missing["model.num_latent_tokens"]
    try: validate_native_reference_paths(missing)
    except ValueError: pass
    else: raise AssertionError("missing native key accepted")
    aliases = fixture("full")
    aliases["model.latent_tokens"] = 8
    try: validate_native_reference_paths(aliases)
    except ValueError: pass
    else: raise AssertionError("historical metadata alias accepted")
    return {"status": "TYPED_LIBERO_NATIVE_PATH_REGRESSIONS_V2_PASS_ONLY", "positive_phases": 3,
            "negative_cases": list(mutations), "actual_hydra_composition": False,
            "native_path_negative_cases": 2, "model_constructed": False, "gpu_ready": False}

def proposal(launcher, reference, cpu):
    for path, expected in ((launcher, LAUNCHER_SHA), (reference, REFERENCE_SHA), (cpu, CPU_SHA)):
        if digest(path) != expected:
            raise ValueError(("immutable proposal input hash mismatch", str(path)))
    text = Path(launcher).read_text()
    for anchor in ('BASE_OVERRIDES=(', 'AF_EXPERIMENT must select an explicitly approved Action Flow config',
                   'evaluator.energy_score_validation_view.world_size=1', 'validate_planar_dataset.py',
                   'verify_action_flow_training_smoke.py'):
        if anchor not in text:
            raise ValueError(("maintained integration anchor changed", anchor))
    return {"schema_version": 1, "status": "PROPOSAL_NOT_INSTALLED_NOT_LAUNCHABLE",
            "launcher_sha256": LAUNCHER_SHA, "reference_sha256": REFERENCE_SHA, "cpu_reference_sha256": CPU_SHA,
            "bindings": {phase: phase_bindings(phase) for phase in PHASES},
            "single_source": "libero_panda", "homogeneous": "not_applicable_single_source",
            "scheduled_checkpoint_count": 16, "historical_parameter_counts_not_constructed_here": [26858231, 39750391],
            "maintained_edit_sites": ["explicit experiment/profile case and audited source allowlist",
              "typed LIBERO dataset/split/normalizer validator dispatch; never planar fallback",
              "shared BASE_OVERRIDES: profile-select evaluator fields; omit U-Socket-only validation view",
              "official all-phase resolved composition and constructor validator dispatch",
              "DiT-half-only maintained callback, no two-source homogeneous wrapper",
              "native LIBERO finite-metric/ES32/alignment/EMA/optimizer strict-reload smoke verifier dispatch"],
            "must_retain": ["existing scheduler/health/storage/signal/requeue/checkpoint runner", "stable unique W&B/output/provenance", "full-model finite optimizer plus Euler50 validation", "root allocation lease and cap8 gates"],
            "blockers": ["new source not yet audited/allowlisted for LIBERO", "profile not installed in maintained launcher", "DiT-half actual activation unresolved", "official resolved model constructor and real smoke incomplete"],
            "metadata_regression": regression()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = regression()
    else:
        if not all((args.launcher, args.reference, args.cpu)):
            parser.error("all three immutable inputs are required")
        result = proposal(args.launcher, args.reference, args.cpu)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()


def regression_native_reference(config):
    """Actual immutable flattened reference, not a synthesized all-key fixture."""
    validate_native_reference_paths(config)
    if set(COMMON).difference(config) != {"trainer.num_nodes"}:
        raise ValueError("unexpected historical missing keys")
    for key in COMMON:
        bad = dict(config)
        if key == "trainer.num_nodes":
            bad[key] = 2
        else:
            del bad[key]
        try:
            validate_native_reference_paths(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(("reference negative accepted", key))
    for alias in ("model.latent_tokens", "model.flow_samples", "model.flow_minibatch_samples"):
        bad = dict(config); bad[alias] = 1
        try: validate_native_reference_paths(bad)
        except ValueError: pass
        else: raise AssertionError(("alias accepted", alias))
    return {"reference_positive": 1, "native_missing_or_operational_negatives": len(COMMON), "alias_negatives": 3}
