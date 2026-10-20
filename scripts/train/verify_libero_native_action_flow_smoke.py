"""Typed native LIBERO noaug verifier, dispatched by maintained shared verifier.

All heavyweight work runs in a scheduled allocation. This module supplies no
training loop and cannot accept preparation-only or tiny-model evidence.
"""

import hashlib
import json
from pathlib import Path

PROFILE = "sim/libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42"
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


def validate_joint_checkpoint_context(payload, cfg):
    """Bind a joint checkpoint to its complete precomputed train-only context."""
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from egomimic.pipeline.checkpoint_binding import validate_checkpoint_binding
    from egomimic.pl_utils.data_context import state_fingerprint
    from egomimic.rldb.zarr.data_module import load_data_context

    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    joint_profile_for_config(config)
    provenance = config["run_provenance"]
    path = Path(config["norm_stats"].get("precomputed_norm_path") or "")
    require(
        path.is_absolute() and path.is_file(), "joint complete context path missing"
    )
    require(
        sha(path) == provenance["normalization_sha256"],
        "joint context file SHA mismatch",
    )
    context = load_data_context(path)  # Restores statistics only; never opens a replay.
    state = context.snapshot()
    require(
        bool(state.get("normalizer_config"))
        and set(state.get("preprocessing", {})) == {"21", "22"},
        "joint full normalizer/preprocessing binding missing",
    )
    split_path = Path(provenance["split_manifest_path"])
    require(
        sha(split_path) == provenance["split_manifest_sha256"],
        "joint split file SHA mismatch",
    )
    split = json.loads(split_path.read_text())
    require(
        split.get("status") == "PASS"
        and set(split.get("sources", {})) == {"libero_panda", "libero_panda_sharpa"},
        "joint split source coverage differs",
    )
    for proof in split["sources"].values():
        require(
            proof["episodes"] == 500
            and len(proof["train"]) == 450
            and len(proof["valid"]) == 50
            and proof["split_seed"] == 42
            and proof["valid_ratio"] == 0.1
            and all(
                proof[key]
                for key in (
                    "zero_id_overlap",
                    "zero_physical_row_overlap",
                    "complete_episode_and_frame_coverage",
                )
            ),
            "joint checkpoint requires verified450/50 physical split",
        )
    tree = payload.get("hyper_parameters", {}).get("config_tree")
    tree = (
        OmegaConf.to_container(tree, resolve=True)
        if OmegaConf.is_config(tree)
        else tree
    )
    require(
        isinstance(tree, dict) and tree.get("model") == config["model"],
        "joint checkpoint configured model differs",
    )
    for key in ("source_commit", "normalization_sha256", "split_manifest_sha256"):
        require(
            tree.get("run_provenance", {}).get(key) == provenance[key],
            "joint checkpoint provenance differs: " + key,
        )
    require(
        isinstance(payload.get("data_context"), dict)
        and state_fingerprint(payload["data_context"]) == state_fingerprint(state),
        "joint checkpoint full data context differs",
    )
    binding = validate_checkpoint_binding(payload, config, context)
    return {
        "context_file_sha256": sha(path),
        "split_file_sha256": sha(split_path),
        "checkpoint_binding": binding,
    }


def validate_joint_energy_artifact(payload, cfg):
    """Verify the actual generic 7/69 EnergyScore artifact, without inference."""
    import torch
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from egomimic.eval.energy_score import energy_score

    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    joint_profile_for_config(config)
    evaluator = config["evaluator"]
    require(
        payload.get("schema_version") == 1
        and payload.get("metric") == "EnergyScore@32"
        and payload.get("global_step") == 8
        and payload.get("sample_count") == 32,
        "joint EnergyScore schema/step/sample contract differs",
    )
    require(
        payload.get("precision") == "bf16-mixed"
        and payload.get("rank") == 0
        and payload.get("batch_idx") == 0,
        "joint EnergyScore precision/worker differs",
    )
    require(
        payload.get("aggregation") == "condition_mean_then_equal_domain_macro_mean",
        "joint EnergyScore aggregation differs",
    )
    require(
        payload.get("provenance") == evaluator["energy_score_provenance"]
        and payload.get("validation_view") == evaluator["energy_score_validation_view"],
        "joint EnergyScore provenance/view differs",
    )
    seed_path = Path(evaluator["seed_bank_path"])
    require(
        sha(seed_path)
        == evaluator["seed_bank_sha256"]
        == payload.get("seed_bank_sha256"),
        "joint EnergyScore seed bank SHA differs",
    )
    seeds = json.loads(seed_path.read_text())["seeds"]
    require(
        payload.get("seed_bank") == seeds
        and len(seeds) == len(set(seeds)) == 32
        and payload.get("deterministic_seed") == evaluator["deterministic_seed"],
        "joint EnergyScore random seed contract differs",
    )

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    distance = dict(
        space="normalized_action_chunk",
        formula="mean_equal_weight_semantic_block_rms",
        fallback_semantic_blocks=evaluator["semantic_blocks"],
        semantic_blocks_by_embodiment=evaluator["semantic_blocks_by_embodiment"],
    )
    require(
        canonical(payload.get("distance")) == canonical(distance),
        "joint EnergyScore semantic distance differs",
    )
    require(
        set(payload.get("domains", {})) == {"libero_panda", "libero_panda_sharpa"},
        "joint EnergyScore source coverage differs",
    )
    results = {}
    for name, identity, width in (
        ("libero_panda", 21, 7),
        ("libero_panda_sharpa", 22, 69),
    ):
        domain = payload["domains"][name]
        samples, target = domain.get("predictions"), domain.get("targets")
        require(
            domain.get("embodiment_id") == identity
            and domain.get("action_key") == "actions",
            "joint EnergyScore embodiment/action key differs",
        )
        require(
            torch.is_tensor(samples)
            and tuple(samples.shape) == (32, 16, 32, width)
            and torch.is_tensor(target)
            and tuple(target.shape) == (16, 32, width),
            "joint EnergyScore tensor shape differs",
        )
        require(
            bool(torch.isfinite(samples).all() and torch.isfinite(target).all()),
            "joint EnergyScore tensors are nonfinite",
        )
        scores = energy_score(
            samples.float(),
            target.float(),
            evaluator["semantic_blocks_by_embodiment"][name],
        )
        for key in (
            "accuracy_by_condition",
            "diversity_by_condition",
            "score_by_condition",
        ):
            saved = domain.get(key)
            require(
                torch.is_tensor(saved)
                and tuple(saved.shape) == (16,)
                and bool(torch.isfinite(saved).all())
                and torch.allclose(saved.float(), scores[key], rtol=1e-5, atol=1e-6),
                "joint saved EnergyScore public math differs: " + name + "/" + key,
            )
        results[name] = {
            key: float(scores[key]) for key in ("score", "accuracy", "diversity")
        }
    return {
        "sources": results,
        "macro_score": sum(row["score"] for row in results.values()) / 2,
    }


def validate_joint_diagnostic_artifact(path, payload, cfg):
    """Check the immutable saved diagnostic envelope and routed tensor contract."""
    import torch
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from egomimic.eval.action_flow_diagnostics import ActionFlowDiagnostics

    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    joint_profile_for_config(config)
    options = config["evaluator"]["action_flow_diagnostics"]
    inspector = ActionFlowDiagnostics(options)
    require(
        payload.get("schema_version") == 1
        and payload.get("metric") == "ActionFlowValidationDiagnostics"
        and payload.get("global_step") == 8
        and payload.get("rank") == 0
        and payload.get("batch_idx") == 0
        and payload.get("precision") == "bf16-mixed",
        "joint diagnostic execution contract differs",
    )
    require(
        payload.get("identity") == inspector.identity
        and payload.get("identity_sha256") == inspector.identity_sha256,
        "joint diagnostic identity differs",
    )
    require(
        payload.get("provenance") == options["provenance"]
        and payload.get("validation_view") == options["validation_view"]
        and payload.get("noise_seed_bank_sha256") == inspector.noise_seed_bank_sha256
        and payload.get("noise_seed") == inspector.noise_seeds[0]
        and payload.get("raw_noise_levels") == inspector.noise_levels,
        "joint diagnostic provenance/view/seed differs",
    )
    require(
        payload.get("sampler", {}).get("name") == "reverse_euler",
        "joint diagnostic sampler differs",
    )
    sidecar = json.loads(Path(str(path) + ".sha256").read_text())
    require(
        sidecar
        == dict(
            algorithm="sha256",
            artifact=Path(path).name,
            sha256=sha(path),
            identity_sha256=inspector.identity_sha256,
        ),
        "joint diagnostic sidecar differs",
    )
    require(
        set(payload.get("sources", {})) == {"libero_panda", "libero_panda_sharpa"},
        "joint diagnostic source coverage differs",
    )
    for name, width in (("libero_panda", 7), ("libero_panda_sharpa", 69)):
        source = payload["sources"][name]
        raw = source.get("diagnostic", {})
        require(
            raw.get("source") == source.get("source_id")
            and raw.get("schema") == inspector.schema
            and raw.get("noise_seed") == inspector.noise_seeds[0]
            and raw.get("num_inference_steps") == 50
            and raw.get("returned_samples") == 8,
            "joint diagnostic raw source/sampler contract differs",
        )
        for key, shape in (
            ("target", (8, 32, width)),
            ("condition", (8, 276)),
            ("latent/clean", (8, 16, 16)),
            ("latent/noise", (8, 16, 16)),
            ("latent/generated", (8, 16, 16)),
            ("decoded/reconstruction", (8, 32, width)),
            ("decoded/generated", (8, 32, width)),
            ("latent/trajectory", (51, 8, 16, 16)),
            ("decoded/trajectory", (51, 8, 32, width)),
        ):
            value = raw.get(key)
            require(
                torch.is_tensor(value)
                and tuple(value.shape) == shape
                and bool(torch.isfinite(value).all()),
                "joint diagnostic tensor differs: " + name + "/" + key,
            )
        require(
            isinstance(source.get("computed"), dict) and source["computed"],
            "joint diagnostic computed statistics missing",
        )
        ActionFlowDiagnostics._validate_tensor_tree(source)
    return dict(
        path=str(path), sha256=sha(path), identity_sha256=inspector.identity_sha256
    )


def validate_joint_ema_reload(restored, payload, cfg):
    """Prove the eight-update EMA inventory survives every shared state alias."""
    import torch
    from omegaconf import OmegaConf

    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    expected = {
        key: config["callbacks"]["ema"][key]
        for key in (
            "decay",
            "use_warmup",
            "update_after_step",
            "inv_gamma",
            "power",
            "min_decay",
        )
    }
    require(
        payload.get("global_step") == payload.get("ema_num_updates") == 8,
        "joint EMA/global update count differs",
    )
    require(payload.get("ema_config") == expected, "joint EMA configuration differs")
    ema = payload.get("ema_state_dict")
    parameters = dict(restored.named_parameters())
    require(
        isinstance(ema, dict) and set(ema) == set(parameters),
        "joint EMA parameter inventory differs",
    )
    state = restored.state_dict()
    canonical_by_id = {id(parameter): name for name, parameter in parameters.items()}
    for name, parameter in parameters.items():
        value = ema[name]
        require(
            torch.is_tensor(value)
            and tuple(value.shape) == tuple(parameter.shape)
            and bool(torch.isfinite(value).all()),
            "joint EMA tensor differs: " + name,
        )
    aliases = list(restored.named_parameters(remove_duplicate=False))
    canonical_by_alias = {
        alias: canonical_by_id[id(parameter)] for alias, parameter in aliases
    }
    for alias, canonical in canonical_by_alias.items():
        state[alias] = ema[canonical]

    def assert_reloaded_ema():
        # Native normalizers recreate ParameterDict entries on load. Names and
        # alias groups remain the contract; Python object IDs need not survive.
        current = dict(restored.named_parameters(remove_duplicate=False))
        require(set(current) == set(canonical_by_alias), "joint EMA aliases changed")
        require(
            set(dict(restored.named_parameters())) == set(parameters),
            "joint EMA canonical inventory changed",
        )
        for alias, parameter in current.items():
            canonical = canonical_by_alias[alias]
            require(
                parameter is current[canonical], "joint EMA alias ownership changed"
            )
            torch.testing.assert_close(
                parameter.detach().cpu(), ema[canonical].detach().cpu(), rtol=0, atol=0
            )

    restored.load_state_dict(state, strict=True)
    assert_reloaded_ema()
    restored.nets.load_state_dict(restored.nets.state_dict(), strict=True)
    assert_reloaded_ema()
    return dict(
        global_step=8,
        ema_num_updates=8,
        canonical_parameters=len(parameters),
        state_parameter_aliases=len(aliases),
        alias_safe_strict_reload=True,
    )


def validate_joint_smoke_config(cfg, run_dir, expected_head):
    """Reject preparation/full-run configs before any heavyweight smoke checks."""
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )

    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    profile = joint_profile_for_config(config)
    root = str(Path(run_dir).resolve())
    expected = {
        "trainer.accelerator": "gpu",
        "trainer.precision": "bf16-mixed",
        "trainer.max_steps": 8,
        "trainer.val_check_interval": 8,
        "trainer.devices": 1,
        "trainer.num_nodes": 1,
        "trainer.accumulate_grad_batches": 1,
        "trainer.num_sanity_val_steps": 0,
        "trainer.limit_train_batches": 1.0,
        "trainer.limit_val_batches": 1.0,
        "trainer.default_root_dir": root,
        "paths.output_dir": root,
        "callbacks.model_checkpoint.every_n_train_steps": 8,
        "callbacks.model_checkpoint.save_last": False,
        "callbacks.model_checkpoint.save_top_k": -1,
        "callbacks.model_checkpoint.dirpath": root + "/checkpoints",
        "runtime.slurm_requeue_owner": "runner",
        "runtime.slurm_signal_checkpoint_dir": root + "/checkpoints",
        "run_provenance.run_kind": "smoke",
        "run_provenance.source_commit": expected_head,
        "norm_stats_only": False,
        "norm_stats.save_cache_dir": None,
        "ckpt_path": None,
        "logger.wandb.mode": "online",
        "logger.wandb.offline": False,
        "logger.wandb.resume": "never",
        "logger.wandb.save_dir": root,
    }
    resolved = OmegaConf.create(config)
    for key, value in expected.items():
        require(
            OmegaConf.select(resolved, key) == value,
            "joint smoke config mismatch: " + key,
        )
    require(
        bool(OmegaConf.select(resolved, "norm_stats.precomputed_norm_path")),
        "joint smoke requires the complete precomputed context",
    )
    require(
        OmegaConf.select(
            resolved, "run_provenance.data_preparation_source_snapshot_sha256"
        )
        is None,
        "joint smoke cannot use a data-only snapshot",
    )
    return {"profile": profile.experiment, "optimizer_updates": 8, "run_dir": root}


def validate_joint_wandb(run_dir, cfg, shared):
    """Read the real smoke stream and online run only after training has finished."""
    import wandb
    from omegaconf import OmegaConf

    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    logger = config["logger"]["wandb"]
    require(
        logger.get("mode") == "online" and logger.get("offline") is False,
        "joint smoke requires explicit online W&B",
    )
    run_id = logger["id"]
    streams = list(Path(run_dir).glob("wandb/run-*/run-" + run_id + ".wandb"))
    require(len(streams) == 1, "joint W&B stream missing or ambiguous")
    offline, exitcode = shared._wandb_history(streams[0])
    require(exitcode == 0, "joint local W&B stream exit differs")
    run = wandb.Api(timeout=20).run(
        logger["entity"] + "/" + logger["project"] + "/" + run_id
    )
    require(run.state == "finished", "joint online W&B publication is not finished")
    online = {}
    for count, row in enumerate(run.scan_history(page_size=100)):
        require(
            count < 1000, "joint eight-update W&B history exceeds bounded inspection"
        )
        step = row.get("trainer/global_step")
        if step is not None:
            online.setdefault(int(step), {}).update(
                {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (float, int)) and not isinstance(value, bool)
                }
            )
    return validate_joint_wandb_records(
        offline=offline,
        online=online,
        online_config=dict(run.config),
        online_state=run.state,
        cfg=config,
        shared=shared,
    )


def validate_joint_wandb_records(
    *, offline, online, online_config, online_state, cfg, shared
):
    """Validate two actual history channels; this pure layer does not query W&B."""
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )

    config = (
        OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    )
    joint_profile_for_config(config)
    logger = config["logger"]["wandb"]
    require(
        logger.get("mode") == "online"
        and logger.get("id")
        and online_state == "finished",
        "joint smoke needs finished online W&B",
    )
    for key in ("source_commit", "normalization_sha256", "split_manifest_sha256"):
        require(
            online_config.get("run_provenance", {}).get(key)
            == config["run_provenance"][key],
            "joint online W&B identity differs: " + key,
        )
    # Lightning logs nested DictConfig values without resolving Hydra references.
    # Resolve the logged model only in the exact saved configuration context;
    # equality still covers every resulting model field.
    logged_context = OmegaConf.create(config)
    logged_context.model = online_config.get("model")
    logged_model = OmegaConf.to_container(logged_context.model, resolve=True)
    require(
        json.loads(json.dumps(logged_model)) == json.loads(json.dumps(config["model"])),
        "joint online W&B model differs",
    )
    # Generic diagnostics have no U-Socket-specific native-error adapter.
    diagnostics = tuple(
        "Valid/ActionFlow/" + suffix
        for suffix in (
            "CleanReconstructionMSE",
            "Latent/clean/RMS",
            "Latent/generated/EffectiveRank",
            "DecoderJacobian/clean/spectral_norm_mean",
            "DecodedFullNoise/token_radius_mean",
            "DenoisingTrajectory/LatentMSE/t1000",
            "DenoisingTrajectory/DecodedMSE/t0000",
            "Alignment/FinalLatentCosine/t1000",
            "Alignment/CKA/encoder_00__field_00/t0500",
            "Alignment/CKNNA/encoder_00__field_00/t0500",
        )
    )
    execution_names = tuple(
        "Train/Execution/DiTHalf/" + owner + "/" + mode
        for owner in ("encoder/libero_panda", "encoder/libero_panda_sharpa", "velocity")
        for mode in ("direct", "checkpoint")
    )
    result = {}
    for channel, rows in (("offline", offline), ("online", online)):
        result[channel] = shared._validate_history(
            rows,
            method=shared.STOPGRAD_UNITE_METHOD,
            source_labels=("libero_panda", "libero_panda_sharpa"),
            minimum_step=7,
            reconstruction_weight=1.0,
            flow_weight=1.0,
            action_velocity_weight=1.0,
            diagnostic_metric_names=diagnostics,
        )
        step, counts = shared._complete_row(
            rows,
            execution_names,
            minimum_step=7,
            label="joint actual DiT-half activation",
        )
        require(
            all(value > 0 for value in counts.values()),
            "joint DiT checkpoint path inactive",
        )
        result[channel]["dit_half"] = dict(step=step, counts=counts)
    require(
        result["offline"]["valid_step"] == result["online"]["valid_step"],
        "joint local/online validation step differs",
    )
    return result


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


def verify_joint_smoke(
    *,
    run_dir,
    expected_head,
    expected_config_sha256,
    expected_split_sha256,
    expected_normalization_sha256,
    expected_preflight_sha256,
    shared,
):
    """Complete explicit joint route; preparation receipts alone cannot pass."""
    import subprocess

    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from egomimic.benchmarks.libero.saved_hydra_context import load_saved_native_config
    from scripts.train.libero_native_launch_contract import contract_sha

    require(
        all(
            isinstance(value, str) and len(value) == 64
            for value in (
                expected_config_sha256,
                expected_split_sha256,
                expected_normalization_sha256,
                expected_preflight_sha256,
            )
        ),
        "joint smoke needs every explicit expected SHA",
    )
    run_dir = Path(run_dir).resolve(strict=True)
    require(shared._git_head() == expected_head, "joint verifier source HEAD mismatch")
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
        "joint verifier source must be clean",
    )
    require(
        sha(run_dir / ".hydra/config.yaml") == expected_config_sha256,
        "joint resolved config byte hash mismatch",
    )
    cfg = load_saved_native_config(run_dir, shared.REPOSITORY_ROOT)
    operational = validate_joint_smoke_config(cfg, run_dir, expected_head)
    config = OmegaConf.to_container(cfg, resolve=True)
    identities = dict(
        source_commit=expected_head,
        normalization_sha256=expected_normalization_sha256,
        split_manifest_sha256=expected_split_sha256,
    )
    require(
        all(
            config["run_provenance"].get(key) == value
            for key, value in identities.items()
        ),
        "joint smoke expected provenance differs",
    )
    fingerprint = contract_sha(config)
    evidence = json.loads(
        (run_dir / "provenance/JOINT_SMOKE_EVIDENCE.json").read_text()
    )
    require(
        evidence.get("schema") == "libero-joint-postsmoke-evidence/v1"
        and evidence.get("resolved_config_sha256") == expected_config_sha256
        and evidence.get("model_contract_sha256") == fingerprint,
        "joint post-smoke envelope identity differs",
    )
    require(
        evidence["preflight"]["sha256"] == expected_preflight_sha256,
        "joint expected preflight SHA differs",
    )
    preflight = checked_json(evidence["preflight"])
    require(
        preflight.get("schema") == "libero-joint-launch-preflight/v1"
        and preflight.get("status") == "PASS"
        and preflight.get("profile") == operational["profile"]
        and preflight.get("model_contract_sha256") == fingerprint,
        "joint preflight scientific contract differs",
    )
    require(
        all(preflight.get(key) == value for key, value in identities.items()),
        "joint preflight provenance differs",
    )
    preflight_config = OmegaConf.load(
        checked_ref(
            {
                "path": preflight["resolved_config_path"],
                "sha256": preflight["resolved_config_sha256"],
            }
        )
    )
    require(
        contract_sha(OmegaConf.to_container(preflight_config, resolve=True))
        == fingerprint,
        "joint preflight/smoke scientific configurations differ",
    )
    from scripts.train.libero_native_launch_contract import validate_joint_constructor

    validate_joint_constructor(
        preflight["CPU_constructor"],
        source_head=expected_head,
        resolved_sha=preflight["resolved_config_sha256"],
        fingerprint=fingerprint,
        parameter_count=joint_profile_for_config(config).parameter_count,
    )
    from scripts.train.libero_native_launch_contract import runtime_equivalent

    runtime_equivalent(
        checked_json(preflight["runtime_lock"]), checked_json(evidence["runtime_lock"])
    )
    gpu = shared._validate_gpu_probes(run_dir)
    checkpoint = shared._validate_checkpoint(
        run_dir,
        reconstruction_weight=1.0,
        flow_weight=1.0,
        method=shared.STOPGRAD_UNITE_METHOD,
        config=cfg,
        expected_parameter_count_override=joint_profile_for_config(
            config
        ).parameter_count,
        expected_optimizer_steps=8,
    )
    artifact_epoch = shared._joint_smoke_artifact_epoch(run_dir, cfg)
    energy_path, energy = shared._step_artifact(
        Path(config["evaluator"]["artifact_root"]),
        label="joint EnergyScore",
        expected_step=8,
        expected_epoch=artifact_epoch,
    )
    diagnostic_path, diagnostic = shared._step_artifact(
        Path(config["evaluator"]["action_flow_diagnostics"]["artifact_root"]),
        label="joint diagnostics",
        expected_step=8,
        expected_epoch=artifact_epoch,
    )
    require(
        checked_ref(evidence["energy_artifact"]) == energy_path.resolve()
        and checked_ref(evidence["diagnostic_artifact"]) == diagnostic_path.resolve(),
        "joint envelope artifact paths differ",
    )
    energy_proof = validate_joint_energy_artifact(energy, cfg)
    diagnostic_proof = validate_joint_diagnostic_artifact(
        diagnostic_path, diagnostic, cfg
    )
    history = validate_joint_wandb(run_dir, cfg, shared)
    result = dict(
        schema="libero-joint-action-flow-smoke/v1",
        status="PASS",
        run_dir=str(run_dir),
        profile=operational["profile"],
        identities=identities,
        resolved_config_sha256=expected_config_sha256,
        model_contract_sha256=fingerprint,
        preflight=evidence["preflight"],
        checkpoint=checkpoint,
        gpu_probes=gpu,
        energy=energy_proof,
        diagnostics=diagnostic_proof,
        history=history,
        strict_online_and_alias_safe_ema_reload=True,
    )
    if getattr(shared, "VERIFIER_PROVENANCE", None) is not None:
        result["verifier_provenance"] = shared.VERIFIER_PROVENANCE
    rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    destination = run_dir / "SMOKE_RESULT.json"
    if destination.exists():
        require(
            destination.read_text() == rendered,
            "joint smoke result overwrite prohibited",
        )
    else:
        with destination.open("x") as stream:
            stream.write(rendered)
    return result


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
