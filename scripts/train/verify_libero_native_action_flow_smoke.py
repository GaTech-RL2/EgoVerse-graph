"""Typed native LIBERO noaug verifier, dispatched by maintained shared verifier.

All heavyweight work runs in a scheduled allocation. This module supplies no
training loop and cannot accept preparation-only or tiny-model evidence.
"""

import hashlib
import json
from pathlib import Path

PROFILE = "libero/action_flow_libero10_h240_euler50_dithalf_80k_s42"
PARAMETER_COUNT = 39750391
SOURCE = "libero_panda"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_ref(ref):
    require(set(ref) == {"path", "sha256"}, "evidence reference schema mismatch")
    path = Path(ref["path"]).resolve(strict=True)
    require(sha(path) == ref["sha256"], "evidence reference hash mismatch")
    return path


def checked_json(ref):
    return json.loads(checked_ref(ref).read_text())


def validate_preflight(preflight, identities, expected_profile=PROFILE):
    require(
        preflight["schema"] == "libero-native-launch-preflight/v1"
        and preflight["status"] == "PASS",
        "native preflight did not pass",
    )
    require(preflight["profile"] == expected_profile, "native profile mismatch")
    for key, value in identities.items():
        if key != "resolved_config_sha256":
            require(
                preflight[key] == value, "native preflight identity mismatch: " + key
            )
    require(
        sha(preflight["resolved_config_path"]) == preflight["resolved_config_sha256"],
        "immutable preflight snapshot changed",
    )
    physical = checked_json(preflight["physical_proof"])
    require(
        physical["status"] == "PASS"
        and physical["zero_frame_overlap"] is True
        and physical["complete_frame_coverage"] is True,
        "native physical proof incomplete",
    )
    require(
        physical["split_manifest_sha256"] == identities["split_manifest_sha256"],
        "physical split identity mismatch",
    )
    cpu = checked_json(preflight["CPU_constructor"])
    require(
        cpu["status"] == "PASS"
        and cpu["full_model_constructor"] is True
        and cpu["parameter_count"] == PARAMETER_COUNT,
        "genuine full native constructor proof missing",
    )
    require(
        cpu["source_commit"] == identities["source_commit"]
        and cpu["resolved_config_sha256"] == preflight["resolved_config_sha256"],
        "CPU constructor identity mismatch",
    )
    require(
        cpu["model_contract_sha256"] == identities["model_contract_sha256"],
        "constructor scientific model identity mismatch",
    )
    require(
        cpu["native_saved_state_binding_validated"] is True
        and cpu["physical_proof_sha256"] == preflight["physical_proof"]["sha256"],
        "CPU native normalization/physical proof mismatch",
    )
    return cpu


def validate_native_checkpoint_context(payload, cfg, binding):
    """Native ModelWrapper checkpoints record model/provenance, not norm state."""
    tree = payload.get("hyper_parameters", {}).get("config_tree")
    require(isinstance(tree, dict), "native checkpoint config_tree mapping missing")
    require(tree.get("model") == cfg["model"], "native checkpoint model config differs")
    provenance = tree.get("run_provenance", {})
    expected = {
        "source_commit": binding["source_commit"],
        "normalization_sha256": binding["file_sha256"],
        "split_manifest_sha256": binding["split_receipt_sha256"],
        "dataset_sha256": cfg["run_provenance"]["dataset_sha256"],
    }
    for key, value in expected.items():
        require(
            provenance.get(key) == value and cfg["run_provenance"].get(key) == value,
            "native checkpoint provenance differs: " + key,
        )
    return expected


def validate_dit_counts(rows, shared):
    names = tuple(
        "Train/Execution/DiTHalf/" + owner + "/" + mode
        for owner in ("encoder", "velocity")
        for mode in ("direct", "checkpoint")
    )
    step, row = shared._complete_row(
        rows, names, minimum_step=1, label="native actual DiT-half activation"
    )
    require(all(row[k] > 0 for k in names), "native DiT-half path inactive")
    return {"step": step, "counts": row}


def verify_native_smoke(
    *,
    run_dir,
    expected_head,
    expected_config_sha256,
    expected_split_sha256,
    expected_normalization_sha256,
    expected_preflight_sha256,
    shared,
    expected_profile=PROFILE,
):
    import hydra
    import torch
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.action_flow_artifacts import (
        REQUIRED,
        validate_payload,
        verify_tensor_payload,
    )
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.zarr.libero_action_flow import LiberoActionFlowNormalizer
    from egomimic.rldb.zarr.libero_saved_state import bind_saved_native_state

    run_dir = Path(run_dir).resolve(strict=True)
    configpath = run_dir / ".hydra/config.yaml"
    config = None
    from egomimic.benchmarks.libero.saved_hydra_context import load_saved_native_config

    config = load_saved_native_config(run_dir, shared.REPOSITORY_ROOT)
    from egomimic.benchmarks.libero.native_launch_profiles import profile_for_config

    selected_profile = profile_for_config(config)
    profile = selected_profile.experiment
    require(profile == expected_profile, "native requested/configured profile mismatch")
    require(
        all(
            v is not None
            for v in (
                expected_config_sha256,
                expected_split_sha256,
                expected_normalization_sha256,
                expected_preflight_sha256,
            )
        ),
        "native smoke needs every explicit expected hash",
    )
    require(shared._git_head() == expected_head, "native verifier source HEAD mismatch")
    import subprocess

    require(
        not subprocess.check_output(
            [
                "git",
                "-C",
                str(shared.REPOSITORY_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
        ),
        "native verifier source must be clean",
    )
    require(
        sha(configpath) == expected_config_sha256,
        "native resolved config byte hash mismatch",
    )

    def select(key):
        return OmegaConf.select(config, key)

    for key, value in (
        ("model._target_", "egomimic.pl_utils.pl_model.ModelWrapper"),
        (
            "normalizer._target_",
            "egomimic.rldb.zarr.libero_action_flow.LiberoActionFlowNormalizer",
        ),
        ("model.action_flow_method", shared.STOPGRAD_UNITE_METHOD),
        ("model.action_dim", 7),
        ("model.action_horizon", 16),
        ("model.hidden_dim", 240),
        ("model.num_inference_steps", 50),
        ("model.flow_samples_per_content", 14),
        ("model.flow_loss_aggregation", "sum_samples"),
        ("trainer.precision", "bf16-mixed"),
        ("trainer.max_steps", 2),
        ("trainer.devices", 1),
        ("trainer.num_nodes", 1),
        ("model.gradient_telemetry_cadence", 2),
        ("data.train_dataloader_params.libero_panda.batch_size", 32),
        (
            "model.optimizer._target_",
            "egomimic.utils.unite_optim.ReleasedUniteCompositeOptimizer",
        ),
    ):
        require(select(key) == value, "native smoke config mismatch: " + key)
    require(
        set(config.data.train_datasets) == {SOURCE}
        and set(config.data.valid_datasets) == {SOURCE},
        "native smoke source inventory mismatch",
    )
    require(len(config.model.pipeline.stages) == 9, "native noaug topology mismatch")
    import sys

    if str(shared.REPOSITORY_ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(shared.REPOSITORY_ROOT / "tools"))
    from tools.libero_maintained_dispatch_v1 import validate_resolved

    validate_resolved(OmegaConf.to_container(config, resolve=True), "smoke")
    binding = OmegaConf.to_container(
        config.norm_stats.native_saved_state_binding, resolve=True
    )
    require(binding["source_commit"] == expected_head, "native binding source mismatch")
    require(
        binding["file_sha256"] == expected_normalization_sha256
        and sha(binding["path"]) == expected_normalization_sha256,
        "native cached state hash mismatch",
    )
    require(
        binding["split_receipt_sha256"] == expected_split_sha256
        and sha(binding["split_receipt_path"]) == expected_split_sha256,
        "native split hash mismatch",
    )
    require(
        sha(binding["dataset_receipt_path"]) == binding["dataset_receipt_sha256"]
        and sha(binding["physical_proof_path"]) == binding["physical_proof_sha256"],
        "native receipt byte identity mismatch",
    )
    identities = dict(
        source_commit=expected_head,
        resolved_config_sha256=expected_config_sha256,
        split_manifest_sha256=expected_split_sha256,
        normalization_sha256=expected_normalization_sha256,
        dataset_sha256=select("run_provenance.dataset_sha256"),
    )
    for key, value in identities.items():
        if key != "resolved_config_sha256":
            require(
                select("run_provenance." + key) == value,
                "native config provenance mismatch: " + key,
            )
    from scripts.train.libero_native_launch_contract import contract_sha

    fingerprint = contract_sha(OmegaConf.to_container(config, resolve=True))
    evidence = json.loads(
        (run_dir / "provenance/NATIVE_SMOKE_EVIDENCE.json").read_text()
    )
    require(
        evidence["schema"] == "libero-native-postsmoke-evidence/v1",
        "post-smoke envelope schema mismatch",
    )
    require(
        evidence["resolved_config_sha256"] == expected_config_sha256
        and evidence["model_contract_sha256"] == fingerprint,
        "post-smoke snapshot/model identity mismatch",
    )
    require(
        evidence["preflight"]["sha256"] == expected_preflight_sha256,
        "native immutable preflight hash mismatch",
    )
    preflightpath = checked_ref(evidence["preflight"])
    preflight = json.loads(preflightpath.read_text())
    require(
        contract_sha(
            OmegaConf.to_container(
                OmegaConf.load(preflight["resolved_config_path"]), resolve=True
            )
        )
        == fingerprint,
        "preflight and smoke scientific contracts differ",
    )
    cpu = validate_preflight(
        preflight, {**identities, "model_contract_sha256": fingerprint}, profile
    )
    require(
        preflight["physical_proof"]
        == {
            "path": binding["physical_proof_path"],
            "sha256": binding["physical_proof_sha256"],
        },
        "native physical proof binding differs",
    )
    gpu = shared._validate_gpu_probes(run_dir)
    checkpoint = shared._validate_checkpoint(
        run_dir,
        reconstruction_weight=1.0,
        flow_weight=1.0,
        method=shared.STOPGRAD_UNITE_METHOD,
        config=config,
        expected_parameter_count_override=PARAMETER_COUNT,
    )
    payload = torch.load(
        checkpoint["immutable_checkpoint_path"],
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    require(
        payload.get("ema_num_updates") == 2, "native checkpoint EMA updates missing"
    )

    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    saved = json.loads(Path(binding["path"]).read_text())["normalizer_state"]
    checkpoint_tree = payload.get("hyper_parameters", {}).get("config_tree")
    if OmegaConf.is_config(checkpoint_tree):
        checkpoint_tree = OmegaConf.to_container(checkpoint_tree, resolve=True)
    checkpoint_context = validate_native_checkpoint_context(
        {"hyper_parameters": {"config_tree": checkpoint_tree}},
        OmegaConf.to_container(config, resolve=True),
        binding,
    )
    # The native checkpoint preserves an external cached-state identity. Use
    # the same actual dataset metadata/physical binding as trainHydra; never fit.
    from libero_native_metadata_probe import instantiate_metadata_dataset

    dataset = instantiate_metadata_dataset(
        config.data.train_datasets.libero_panda,
        omega_conf=OmegaConf,
        instantiate=hydra.utils.instantiate,
    )
    normalizer = hydra.utils.instantiate(
        config.normalizer, state={}, norm_mode=config.norm_stats.norm_mode
    )
    normalizer.populate_from_datasets({SOURCE: dataset})
    normalizer.infer_shapes_from_batch(dataset[0])
    normalizer = bind_saved_native_state(
        norm_stats=normalizer,
        dataset=dataset,
        dataset_name=SOURCE,
        binding=binding,
        run_provenance=OmegaConf.to_container(config.run_provenance, resolve=True),
    )
    require(
        type(normalizer) is LiberoActionFlowNormalizer,
        "native reload normalizer class differs",
    )
    require(
        plain(normalizer.to_state()) == saved,
        "native reloaded external cached normalizer state differs",
    )
    dataset.set_norm_stats_from(normalizer)
    require(
        dataset.norm_stats is normalizer.norm_stats
        and dataset.key_types is normalizer.key_types
        and dataset.zarr_keys is normalizer.zarr_keys
        and dataset.shapes is normalizer.shapes,
        "native dataset cached normalization binding differs",
    )
    require(
        payload.get("action_flow_loss_schedule") is not None,
        "native joint checkpoint loss schedule missing",
    )
    ema = payload.get("ema_state_dict")
    require(isinstance(ema, dict) and ema, "native EMA state missing")
    shared._finite_tree(ema, "native EMA")
    shared._finite_tree(payload["loops"], "native loops")
    restored = ModelWrapper.load_from_checkpoint(
        checkpoint["immutable_checkpoint_path"],
        map_location="cpu",
        strict=True,
        weights_only=False,
    )
    restored.model.bind_data_context(normalizer=normalizer)
    parameters = dict(restored.named_parameters())
    require(set(ema) == set(parameters), "native EMA parameter inventory differs")
    state = restored.state_dict()
    online = restored.nets.state_dict()
    restored.nets.load_state_dict(online, strict=True)
    canonical_by_id = {id(value): key for key, value in parameters.items()}
    for key, value in ema.items():
        require(
            key in state and tuple(value.shape) == tuple(state[key].shape),
            "native EMA tensor contract differs",
        )
    # Wrapper registers shared model/nets aliases; update every state alias so
    # strict reload cannot overwrite EMA values later with an online alias.
    for alias, parameter in restored.named_parameters(remove_duplicate=False):
        state[alias] = ema[canonical_by_id[id(parameter)]]
    restored.load_state_dict(state, strict=True)
    for key, parameter in restored.named_parameters():
        torch.testing.assert_close(
            parameter.detach().cpu(), ema[key].detach().cpu(), rtol=0, atol=0
        )
    restored.nets.load_state_dict(restored.nets.state_dict(), strict=True)
    del restored, payload
    runid = str(select("logger.wandb.id"))
    require(
        runid and select("logger.wandb.mode") != "offline",
        "native smoke needs actual online W&B",
    )
    streams = list(run_dir.glob("wandb/run-*/run-" + runid + ".wandb"))
    require(len(streams) == 1, "native W&B stream ambiguous/missing")
    offline, exitcode = shared._wandb_history(streams[0])
    import wandb

    api = wandb.Api(timeout=20)
    run = api.run(
        str(select("logger.wandb.entity"))
        + "/"
        + str(select("logger.wandb.project"))
        + "/"
        + runid
    )
    require(run.state == "finished", "native online W&B run not finished")
    require(
        run.config["run_provenance"]["source_commit"] == expected_head,
        "native online W&B source binding mismatch",
    )
    online = {}
    for row in run.scan_history(page_size=100):
        step = row.get("trainer/global_step")
        if step is not None:
            online.setdefault(int(step), {}).update(
                {k: v for k, v in row.items() if isinstance(v, (float, int))}
            )
    nativevalid = tuple("Valid/" + key for key in REQUIRED)
    history = {}
    for label, rows in (("offline", offline), ("online", online)):
        history[label] = shared._validate_history(
            rows,
            reconstruction_weight=1.0,
            flow_weight=1.0,
            action_velocity_weight=selected_profile.action_velocity_weight,
            method=shared.STOPGRAD_UNITE_METHOD,
            source_label=SOURCE,
            validation_metric_names=nativevalid,
            diagnostic_metric_names=(),
        )
        history[label]["dit_half"] = validate_dit_counts(rows, shared)
    require(
        history["offline"]["valid_step"] == history["online"]["valid_step"],
        "native online/offline validation step mismatch",
    )
    native = checked_json(evidence["native_metrics_artifact"])
    validate_payload(native)
    require(
        native["identity"]["source_commit"] == expected_head
        and native["identity"]["resolved_config_sha256"] == expected_config_sha256
        and native["identity"]["split_sha256"] == expected_split_sha256
        and native["identity"]["normalizer_state_sha256"]
        == expected_normalization_sha256,
        "native metric artifact identity mismatch",
    )
    require(native["global_step"] == 2, "native artifact not final smoke step")
    # V5 tensor/diagnostic contract must be supplied by maintained source.
    require(
        native["identity"]["dataset_logical_sha256"] == identities["dataset_sha256"],
        "native artifact dataset identity mismatch",
    )
    require(
        Path(native["identity"]["resolved_config_path"]).resolve()
        == configpath.resolve(),
        "native artifact snapshot path mismatch",
    )
    require(
        native["identity"]["source_normalizer_module_sha256"]
        == binding["normalizer_module_sha256"]
        == sha(shared.REPOSITORY_ROOT / "egomimic/rldb/zarr/libero_action_flow.py"),
        "native artifact normalizer module identity mismatch",
    )
    require(
        native["identity"]["dataset_receipt_sha256"]
        == binding["dataset_receipt_sha256"]
        and native["identity"]["physical_split_proof_sha256"]
        == binding["physical_proof_sha256"],
        "native artifact receipt identity mismatch",
    )
    require(
        native["identity"]["historical_split_sha256"]
        == select("run_provenance.historical_split_manifest_sha256"),
        "native artifact lost historical split provenance",
    )
    require(
        native["identity"]["precision"] == "bf16-mixed"
        and native["identity"]["native_objective_weights"]
        == {
            "reconstruction": 1.0,
            "flow": 1.0,
            "action_velocity": selected_profile.action_velocity_weight,
        },
        "native artifact precision/objective mismatch",
    )
    require(
        native["group"] == "valid", "native smoke metrics must use scheduled validation"
    )
    artifactroot = Path(evidence["native_metrics_artifact"]["path"]).resolve().parent
    for batch in native["batches"]:
        require(
            Path(batch["tensor_payload"]["filename"]).name
            == batch["tensor_payload"]["filename"],
            "native tensor filename escapes artifact root",
        )
        verify_tensor_payload(artifactroot, batch["tensor_payload"])
    first = native["batches"][0]["tensor_payload"]
    reference = first["shared_analysis"]
    require(reference["global_step"] == 2, "native shared diagnostic step mismatch")
    diagnostic_path = Path(reference["path"]).resolve(strict=True)
    diagnostic = torch.load(diagnostic_path, map_location="cpu", weights_only=True)
    require(
        diagnostic["schema_version"] == 1
        and diagnostic["metric"] == "ActionFlowValidationDiagnostics",
        "shared diagnostic schema mismatch",
    )
    require(
        diagnostic["global_step"] == 2
        and diagnostic["batch_idx"] == 0
        and diagnostic["rank"] == 0,
        "shared diagnostic capture identity mismatch",
    )
    require(set(diagnostic["sources"]) == {SOURCE}, "shared diagnostic sources differ")
    require(
        diagnostic["raw_noise_levels"] == [0.0, 0.25, 0.5, 0.75, 1.0],
        "shared diagnostic noise contract mismatch",
    )
    require(
        diagnostic["provenance"]["source_commit"] == expected_head,
        "shared diagnostic source identity mismatch",
    )
    require(
        diagnostic["provenance"]["normalization_sha256"]
        == expected_normalization_sha256
        and diagnostic["provenance"]["split_manifest_sha256"] == expected_split_sha256,
        "shared diagnostic cache identity mismatch",
    )
    require(
        diagnostic["provenance"]["sampler_steps"] == 50,
        "shared diagnostic sampler mismatch",
    )
    tensors, numbers = shared._finite_tree(
        diagnostic["sources"][SOURCE]["computed"], "shared computed native diagnostic"
    )
    require(tensors + numbers > 0, "shared computed diagnostic empty")
    tensorproof = dict(
        native_metrics_sha256=evidence["native_metrics_artifact"]["sha256"],
        batch_count=len(native["batches"]),
        tensor_sha256=first["sha256"],
        shared_diagnostic_sha256=reference["sha256"],
        same_pass_verified=True,
    )
    result = dict(
        schema="libero-native-action-flow-smoke/v1",
        status="PASS",
        run_dir=str(run_dir),
        profile=profile,
        identities=identities,
        model_contract_sha256=fingerprint,
        preflight={"path": str(preflightpath), "sha256": expected_preflight_sha256},
        checkpoint=checkpoint,
        gpu_probes=gpu,
        cpu_constructor=cpu,
        history=history,
        native_artifacts=tensorproof,
        wandb=dict(run_id=runid, stream_sha256=sha(streams[0]), exit_code=exitcode),
        strict_online_and_ema_reload=True,
        normalization=dict(
            storage="external_cached_native_state",
            embedded_in_checkpoint=False,
            checkpoint_context=checkpoint_context,
            actual_native_class_and_state_verified=True,
            metadata_resolver_decoded_cache=False,
        ),
    )
    destination = run_dir / "SMOKE_RESULT.json"
    rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        require(
            destination.read_text() == rendered,
            "native smoke result overwrite prohibited",
        )
    else:
        with destination.open("x") as f:
            f.write(rendered)
    return result
