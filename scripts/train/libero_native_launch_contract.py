"""Native phase proof adapter around canonical exact argv and shared validators."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROFILE = "libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def model_contract(cfg):
    import copy

    model = copy.deepcopy(cfg["model"])
    model.pop("gradient_telemetry_cadence", None)
    evaluator = copy.deepcopy(cfg["evaluator"])
    evaluator.pop("artifact_root", None)
    for name in ("artifact_identity", "native_diagnostic_config"):
        if isinstance(evaluator.get(name), dict):
            evaluator[name].pop("config_path", None)
            if name == "native_diagnostic_config":
                evaluator[name].pop("artifact_root", None)
    if len(cfg["data"]["train_datasets"]) > 1:
        from egomimic.benchmarks.libero.native_launch_profiles import (
            joint_profile_for_config,
        )

        profile = joint_profile_for_config(cfg)
        # Phase destinations are independently bound by the launch envelope.
        # Keep all scientific EMA/diagnostic options in the shared fingerprint.
        ema = copy.deepcopy(cfg["callbacks"]["ema"])
        ema.pop("final_checkpoint_path", None)
        if isinstance(evaluator.get("action_flow_diagnostics"), dict):
            evaluator["action_flow_diagnostics"].pop("artifact_root", None)
        return dict(
            evaluator=evaluator,
            schema="libero-joint-scientific-contract/v1",
            profile=profile.experiment,
            model=model,
            normalizer=cfg["normalizer"],
            normalization=cfg["norm_stats"],
            train_datasets=cfg["data"]["train_datasets"],
            valid_datasets=cfg["data"]["valid_datasets"],
            train_batch={
                name: cfg["data"]["train_dataloader_params"][name]["batch_size"]
                for name, *_ in profile.sources
            },
            valid_batch={
                name: cfg["data"]["valid_dataloader_params"][name]["batch_size"]
                for name, *_ in profile.sources
            },
            precision=cfg["trainer"]["precision"],
            accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
            seed=cfg["seed"],
            ema=ema,
            dit_half=cfg["callbacks"]["dit_half"],
        )
    return dict(
        evaluator=evaluator,
        schema="libero-native-scientific-contract/v1",
        model=model,
        normalizer=cfg["normalizer"],
        binding=cfg["norm_stats"]["native_saved_state_binding"],
        train_datasets=cfg["data"]["train_datasets"],
        valid_datasets=cfg["data"]["valid_datasets"],
        train_batch=cfg["data"]["train_dataloader_params"]["libero_panda"][
            "batch_size"
        ],
        valid_batch=cfg["data"]["valid_dataloader_params"]["libero_panda"][
            "batch_size"
        ],
        precision=cfg["trainer"]["precision"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        seed=cfg["seed"],
        ema=cfg["callbacks"]["ema"],
        dit_half=cfg["callbacks"]["dit_half"],
    )


def contract_sha(cfg):
    return hashlib.sha256(
        json.dumps(
            model_contract(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def checked(path, expected):
    assert sha(path) == expected, "native evidence SHA mismatch"
    return json.loads(Path(path).read_text())


def runtime_equivalent(a, b):
    for key in (
        "python",
        "installed_distributions",
        "core_distribution_versions",
        "dependency_inputs",
        "source",
    ):
        assert a[key] == b[key], ("native runtime differs", key)
    ta = dict(a["torch_runtime"])
    tb = dict(b["torch_runtime"])
    ta.pop("cuda_available_on_capture_node", None)
    tb.pop("cuda_available_on_capture_node", None)
    assert ta == tb, "native torch runtime differs"


def joint_split_contract(cfg, dataset_manifests, *, expected_episodes):
    """Materialize physical split evidence through the configured replay views.

    Data-only preparation: no model, normalizer fitting, or launch-ready claim.
    Manifest hashes come from the accepted conversion receipts, not discovery.
    """
    import hydra

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from egomimic.rldb.zarr.libero_saved_state import make_replay_split_proof

    profile = joint_profile_for_config(cfg)
    names = {name for name, *_ in profile.sources}
    if set(dataset_manifests) != names or expected_episodes <= 0:
        raise ValueError(
            "joint split requires exact source manifest bindings and episode count"
        )
    bindings, roots = {}, set()
    # Verify every cheap content binding before opening any replay views.
    for name in sorted(names):
        root = Path(
            cfg["data"]["train_datasets"][name]["resolver"]["folder_path"]
        ).resolve(strict=True)
        valid_root = Path(
            cfg["data"]["valid_datasets"][name]["resolver"]["folder_path"]
        ).resolve(strict=True)
        if root != valid_root or root in roots:
            raise ValueError(
                "joint source physical roots must match their views and be distinct"
            )
        roots.add(root)
        manifest = root / "benchmark_manifest.json"
        if sha(manifest) != dataset_manifests[name]:
            raise ValueError("joint dataset manifest SHA mismatch: " + name)
        bindings[name] = reference(manifest)
    proofs = {}
    for name in sorted(names):
        train = hydra.utils.instantiate(cfg["data"]["train_datasets"][name])
        valid = hydra.utils.instantiate(cfg["data"]["valid_datasets"][name])
        proofs[name] = make_replay_split_proof(
            train, valid, expected_episodes=expected_episodes, valid_ratio=0.1, seed=42
        )
    return dict(
        schema="libero-joint-physical-split/v1",
        status="PASS",
        profile=profile.experiment,
        dataset_manifests=bindings,
        sources=proofs,
        normalization_scope="training_episodes_only",
        launch_ready=False,
    )


def normalize_joint_context(cfg):
    """Fit once through the configured DataModule after exact physical split proof."""
    import hydra
    import lightning as L
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )

    profile = joint_profile_for_config(cfg)
    options = cfg["norm_stats"]
    export = Path(options.get("export_context_path") or "")
    if (
        not cfg.get("norm_stats_only")
        or not export.is_absolute()
        or export.exists()
        or options.get("precomputed_norm_path")
    ):
        raise ValueError(
            "joint normalization requires a fresh absolute complete-context export"
        )
    provenance = cfg["run_provenance"]
    split = checked(
        provenance["split_manifest_path"], provenance["split_manifest_sha256"]
    )
    if (
        split.get("schema") != "libero-joint-physical-split/v1"
        or split.get("status") != "PASS"
        or split.get("profile") != profile.experiment
    ):
        raise ValueError(
            "joint normalization requires passing typed physical split proof"
        )
    original_path = split["resolved_config"]["path"]
    if sha(original_path) != split["resolved_config"]["sha256"]:
        raise ValueError("joint split configuration SHA mismatch")
    original = OmegaConf.to_container(OmegaConf.load(original_path), resolve=True)
    # The split SHA cannot appear in its own bootstrap config. DataModule
    # provenance is filled in afterwards; it is not a dataset selection field.
    import copy

    actual_data, original_data = (
        copy.deepcopy(cfg["data"]),
        copy.deepcopy(original["data"]),
    )
    actual_data.pop("run_provenance", None)
    original_data.pop("run_provenance", None)
    if actual_data != original_data or cfg["normalizer"] != original["normalizer"]:
        raise ValueError(
            "joint normalization data/normalizer differs from physical split config"
        )
    if set(split["sources"]) != {name for name, *_ in profile.sources}:
        raise ValueError("joint physical split source coverage differs")
    for name, *_ in profile.sources:
        proof = split["sources"][name]
        manifest = split["dataset_manifests"][name]
        root = Path(
            cfg["data"]["train_datasets"][name]["resolver"]["folder_path"]
        ).resolve(strict=True)
        if (
            Path(proof["data_root"]) != root
            or Path(manifest["path"]) != root / "benchmark_manifest.json"
            or sha(manifest["path"]) != manifest["sha256"]
            or proof["split_seed"] != 42
            or proof["valid_ratio"] != 0.1
            or proof["episodes"] != 500
            or len(proof["train"]) != 450
            or len(proof["valid"]) != 50
            or not all(
                proof[key]
                for key in (
                    "zero_id_overlap",
                    "zero_physical_row_overlap",
                    "complete_episode_and_frame_coverage",
                )
            )
        ):
            raise ValueError("joint normalization split/content binding differs")
    L.seed_everything(cfg["seed"], workers=True)
    module = hydra.utils.instantiate(OmegaConf.create(cfg["data"]), _recursive_=False)
    context = module.prepare_context(
        mode="normalization",
        normalization=options,
        normalizer=OmegaConf.create(cfg["normalizer"]),
    )
    return dict(
        schema="libero-joint-normalization/v1",
        status="PASS",
        profile=profile.experiment,
        split=reference(provenance["split_manifest_path"]),
        complete_context=reference(export),
        context_fingerprint=context.fingerprint(),
        normalization_scope="training_episodes_only",
        launch_ready=False,
    )


def compose_joint_phase(inputs, *, phase, output, repo):
    """Materialize the one maintained argv used by the shell child and composer."""
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import joint_phase_argv

    if phase not in {"preflight", "smoke", "full"} or inputs.get("source_root") != str(
        Path(repo).resolve()
    ):
        raise ValueError("joint operational phase/source root differs")
    argv = joint_phase_argv(phase=phase, **inputs)
    output = Path(output)
    # Immutable bytes are consumed directly into BASE_OVERRIDES by the caller.
    with (output / "exact-phase.argv0").open("xb") as stream:
        stream.write(b"\0".join(value.encode() for value in argv) + b"\0")
    rendered = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "egomimic.trainHydra",
            *argv,
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=repo,
        text=True,
    )
    config = OmegaConf.to_container(OmegaConf.create(rendered), resolve=True)
    (output / "resolved.yaml").write_text(rendered)
    if config["run_provenance"]["run_kind"] != phase:
        raise ValueError("joint composed phase identity differs")
    return dict(
        schema="libero-joint-phase-composition/v1",
        phase=phase,
        argv=reference(output / "exact-phase.argv0"),
        resolved_config=reference(output / "resolved.yaml"),
        model_contract_sha256=contract_sha(config),
        launch_ready=False,
    )


def joint_preflight_evidence(
    cfg, resolved_path, *, constructor_ref, runtime_ref, source_head
):
    """Consume exact constructor/runtime proofs without constructing a model."""
    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from egomimic.rldb.zarr.data_module import load_data_context
    from scripts.train.verify_libero_native_action_flow_smoke import checked_json

    profile = joint_profile_for_config(cfg)
    provenance = cfg["run_provenance"]
    if (
        provenance.get("run_kind") != "preflight"
        or provenance.get("source_commit") != source_head
    ):
        raise ValueError("joint preflight phase/source differs")
    if (
        cfg["trainer"]["accelerator"] != "cpu"
        or cfg.get("norm_stats_only") is not False
    ):
        raise ValueError("joint preflight must be CPU with a precomputed context")
    fingerprint = contract_sha(cfg)
    resolved_sha = sha(resolved_path)
    constructor = checked_json(constructor_ref)
    required = dict(
        status="PASS",
        full_model_constructor=True,
        parameter_count=78182614,
        source_commit=source_head,
        resolved_config_sha256=resolved_sha,
        model_contract_sha256=fingerprint,
    )
    if any(constructor.get(key) != value for key, value in required.items()):
        raise ValueError(
            "joint preflight needs an exact source/config-bound constructor proof; never relabel cached evidence"
        )
    runtime = checked_json(runtime_ref)
    runtime_equivalent(runtime, runtime)
    if (
        runtime["source"].get("head") != source_head
        or runtime["source"].get("dirty") is not False
    ):
        raise ValueError("joint preflight runtime source binding differs")
    context_path = cfg["norm_stats"].get("precomputed_norm_path")
    if not context_path or sha(context_path) != provenance["normalization_sha256"]:
        raise ValueError("joint preflight complete context SHA differs")
    context = load_data_context(Path(context_path))
    state = context.snapshot()
    if not state.get("normalizer_config") or set(state.get("preprocessing", {})) != {
        "21",
        "22",
    }:
        raise ValueError("joint preflight full context preprocessing missing")
    split_path = provenance["split_manifest_path"]
    split = checked(split_path, provenance["split_manifest_sha256"])
    if (
        split.get("schema") != "libero-joint-physical-split/v1"
        or split.get("status") != "PASS"
        or set(split.get("sources", {})) != {name for name, *_ in profile.sources}
    ):
        raise ValueError("joint preflight physical split contract differs")
    for proof in split["sources"].values():
        if not (
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
            )
        ):
            raise ValueError("joint preflight requires physical450/50 split")
    return dict(
        schema="libero-joint-launch-preflight/v1",
        status="PASS",
        profile=profile.experiment,
        source_commit=source_head,
        resolved_config_path=str(Path(resolved_path).resolve()),
        resolved_config_sha256=resolved_sha,
        model_contract_sha256=fingerprint,
        normalization_sha256=provenance["normalization_sha256"],
        split_manifest_sha256=provenance["split_manifest_sha256"],
        complete_context=reference(context_path),
        physical_proof=reference(split_path),
        CPU_constructor=constructor_ref,
        runtime_lock=runtime_ref,
        scope="joint_preexecution_cached_proofs",
        launch_ready=False,
    )


def joint_prepare_evidence(cfg, *, phase, preflight_ref, runtime_ref, smoke_ref=None):
    """Check immutable prerequisites before the common training child starts."""
    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from scripts.train.verify_libero_native_action_flow_smoke import checked_json

    profile = joint_profile_for_config(cfg)
    if phase not in {"smoke", "full"} or cfg["run_provenance"].get("run_kind") != phase:
        raise ValueError("joint prepare phase differs")
    expected_steps, cadence = (8, 8) if phase == "smoke" else (120000, 15000)
    if (
        cfg["trainer"]["max_steps"] != expected_steps
        or cfg["trainer"]["val_check_interval"] != cadence
        or cfg["callbacks"]["model_checkpoint"]["every_n_train_steps"] != cadence
    ):
        raise ValueError("joint prepare stop/validation/checkpoint cadence differs")
    fingerprint = contract_sha(cfg)
    preflight = checked_json(preflight_ref)
    identities = {
        key: cfg["run_provenance"][key]
        for key in ("source_commit", "normalization_sha256", "split_manifest_sha256")
    }
    if (
        preflight.get("schema") != "libero-joint-launch-preflight/v1"
        or preflight.get("status") != "PASS"
        or preflight.get("profile") != profile.experiment
        or preflight.get("model_contract_sha256") != fingerprint
        or any(preflight.get(key) != value for key, value in identities.items())
    ):
        raise ValueError("joint prepare preflight identity differs")
    runtime_equivalent(
        checked_json(preflight["runtime_lock"]), checked_json(runtime_ref)
    )
    if phase == "full":
        if smoke_ref is None:
            raise ValueError("joint full requires a genuine eight-update smoke proof")
        smoke = checked_json(smoke_ref)
        if not (
            smoke.get("schema") == "libero-joint-action-flow-smoke/v1"
            and smoke.get("status") == "PASS"
            and smoke.get("profile") == profile.experiment
            and smoke.get("model_contract_sha256") == fingerprint
            and smoke.get("identities") == identities
            and smoke.get("preflight") == preflight_ref
            and smoke.get("strict_online_and_alias_safe_ema_reload") is True
            and smoke.get("checkpoint", {}).get("parameter_count") == 78182614
            and smoke["checkpoint"].get("global_step") == 8
            and smoke["checkpoint"].get("joint_ema", {}).get("alias_safe_strict_reload")
            is True
        ):
            raise ValueError("joint full smoke proof differs")
    return dict(
        schema="libero-joint-run-prepared/v1",
        phase=phase,
        model_contract_sha256=fingerprint,
        preflight=preflight_ref,
        runtime_lock=runtime_ref,
        smoke=smoke_ref,
        launch_ready=False,
    )


def joint_smoke_evidence(cfg, root, *, preflight_ref, runtime_ref, shared):
    """Bind real saved outputs; this envelope is not a smoke PASS receipt."""
    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )
    from scripts.train.verify_libero_native_action_flow_smoke import (
        checked_json,
        checked_ref,
    )

    profile = joint_profile_for_config(cfg)
    if (
        cfg["run_provenance"].get("run_kind") != "smoke"
        or cfg["trainer"]["max_steps"] != 8
    ):
        raise ValueError("joint finalize requires the eight-update smoke phase")
    preflight = checked_json(preflight_ref)
    fingerprint = contract_sha(cfg)
    if (
        preflight.get("schema") != "libero-joint-launch-preflight/v1"
        or preflight.get("status") != "PASS"
        or preflight.get("profile") != profile.experiment
        or preflight.get("model_contract_sha256") != fingerprint
    ):
        raise ValueError("joint finalize preflight scientific identity differs")
    for key in ("source_commit", "split_manifest_sha256", "normalization_sha256"):
        if preflight.get(key) != cfg["run_provenance"][key]:
            raise ValueError("joint finalize provenance differs: " + key)
    runtime_equivalent(
        checked_json(preflight["runtime_lock"]), checked_json(runtime_ref)
    )
    energy_path, _ = shared._step_artifact(
        Path(cfg["evaluator"]["artifact_root"]),
        label="joint EnergyScore",
        expected_step=8,
    )
    diagnostic_path, _ = shared._step_artifact(
        Path(cfg["evaluator"]["action_flow_diagnostics"]["artifact_root"]),
        label="joint diagnostics",
        expected_step=8,
    )
    checked_ref(preflight_ref)
    return dict(
        schema="libero-joint-postsmoke-evidence/v1",
        preflight=preflight_ref,
        runtime_lock=runtime_ref,
        energy_artifact=reference(energy_path),
        diagnostic_artifact=reference(diagnostic_path),
        resolved_config_sha256=sha(Path(root) / ".hydra/config.yaml"),
        model_contract_sha256=fingerprint,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--action",
        choices=[
            "preflight",
            "prepare",
            "finalize",
            "joint-split",
            "joint-normalize",
            "joint-preflight",
            "joint-compose",
            "joint-prepare",
        ],
        required=True,
    )
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--argv", type=Path)
    p.add_argument("--phase", choices=["preflight", "smoke", "full"])
    p.add_argument("--resolved-config", type=Path)
    p.add_argument(
        "--dataset-manifests",
        type=Path,
        help="JSON source-name to accepted benchmark_manifest SHA256 mapping",
    )
    p.add_argument("--preflight-proof", type=Path)
    p.add_argument("--expected-preflight-sha256")
    p.add_argument("--smoke-proof", type=Path)
    p.add_argument("--expected-smoke-sha256")
    p.add_argument(
        "--phase-inputs",
        type=Path,
        help="JSON exact keyword inputs to joint_phase_argv",
    )
    p.add_argument("--expected-phase-inputs-sha256")
    p.add_argument(
        "--constructor-proof",
        type=Path,
        help="Immutable exact source/config-bound constructor JSON",
    )
    p.add_argument("--expected-episodes", type=int)
    p.add_argument(
        "--runtime-lock",
        type=Path,
        help="Actual phase runtime-lock JSON, required for joint finalize",
    )
    a = p.parse_args()
    assert os.environ.get("SLURM_STEP_ID"), "scheduled srun only"
    if a.action in {
        "joint-split",
        "joint-normalize",
        "joint-preflight",
        "joint-compose",
        "joint-prepare",
    }:
        if not all(
            os.environ.get(key, "").isdigit()
            for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")
        ):
            raise ValueError(
                "joint data preparation requires a scheduled numeric srun step"
            )
    sys.path[:0] = [str(a.repo), str(a.repo / "tools")]
    from omegaconf import OmegaConf

    if a.action == "joint-prepare":
        if not (
            a.resolved_config
            and a.phase
            and a.preflight_proof
            and a.expected_preflight_sha256
            and a.runtime_lock
        ):
            p.error(
                "joint-prepare needs resolved config/phase/preflight SHA/runtime lock"
            )
        checked(a.preflight_proof, a.expected_preflight_sha256)
        if a.smoke_proof:
            if not a.expected_smoke_sha256:
                p.error("joint smoke proof requires its expected SHA")
            checked(a.smoke_proof, a.expected_smoke_sha256)
        cfg = OmegaConf.to_container(OmegaConf.load(a.resolved_config), resolve=True)
        result = joint_prepare_evidence(
            cfg,
            phase=a.phase,
            preflight_ref=reference(a.preflight_proof),
            runtime_ref=reference(a.runtime_lock),
            smoke_ref=reference(a.smoke_proof) if a.smoke_proof else None,
        )
        write(a.output / "JOINT_RUN_PREPARED.json", result)
        return

    if a.action == "joint-compose":
        if not (a.phase_inputs and a.expected_phase_inputs_sha256 and a.phase):
            p.error(
                "joint-compose requires --phase-inputs, --expected-phase-inputs-sha256 and --phase"
            )
        inputs = checked(a.phase_inputs, a.expected_phase_inputs_sha256)
        for key, envkey in {
            "source_root": "AF_REPO",
            "source_commit": "AF_EXPECTED_HEAD",
            "original_root": "AF_DATASET_DIR",
            "sharpa_root": "AF_SECOND_DATASET_DIR",
            "output_root": "AF_OUTPUT_DIR",
            "context_path": "AF_NORM_STATS_PATH",
            "split_path": "AF_SPLIT_MANIFEST",
            "split_sha256": "AF_EXPECTED_SPLIT_MANIFEST_SHA256",
            "normalization_sha256": "AF_EXPECTED_NORM_SHA256",
            "wandb_entity": "AF_WANDB_ENTITY",
            "wandb_project": "AF_WANDB_PROJECT",
            "wandb_run_id": "AF_WANDB_RUN_ID",
        }.items():
            if not os.environ.get(envkey) or inputs.get(key) != os.environ[envkey]:
                raise ValueError(
                    "joint exact factory input/environment mismatch: " + key
                )
        result = compose_joint_phase(
            inputs, phase=a.phase, output=a.output, repo=a.repo
        )
        write(a.output / "JOINT_PHASE_COMPOSITION.json", result)
        return

    if a.action == "joint-preflight":
        if not (a.resolved_config and a.constructor_proof and a.runtime_lock):
            p.error(
                "joint-preflight requires --resolved-config, --constructor-proof and --runtime-lock"
            )
        head = subprocess.check_output(
            ["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if subprocess.check_output(
            [
                "git",
                "-C",
                str(a.repo),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
        ):
            raise ValueError("joint preflight source must be clean")
        cfg = OmegaConf.to_container(OmegaConf.load(a.resolved_config), resolve=True)
        result = joint_preflight_evidence(
            cfg,
            a.resolved_config,
            constructor_ref=reference(a.constructor_proof),
            runtime_ref=reference(a.runtime_lock),
            source_head=head,
        )
        a.output.mkdir(parents=True, exist_ok=True)
        write(a.output / "PREFLIGHT_RESULT.json", result)
        return

    if a.action == "joint-normalize":
        if not a.resolved_config:
            p.error("joint-normalize requires --resolved-config")
        cfg = OmegaConf.to_container(OmegaConf.load(a.resolved_config), resolve=True)
        result = normalize_joint_context(cfg)
        result["resolved_config"] = reference(a.resolved_config)
        a.output.mkdir(parents=True, exist_ok=True)
        write(a.output / "JOINT_NORMALIZATION.json", result)
        return

    if a.action == "joint-split":
        if not (a.resolved_config and a.dataset_manifests and a.expected_episodes):
            p.error(
                "joint-split requires --resolved-config, --dataset-manifests and --expected-episodes"
            )
        cfg = OmegaConf.to_container(OmegaConf.load(a.resolved_config), resolve=True)
        result = joint_split_contract(
            cfg,
            json.loads(a.dataset_manifests.read_text()),
            expected_episodes=a.expected_episodes,
        )
        result["resolved_config"] = reference(a.resolved_config)
        result["dataset_manifest_bindings"] = reference(a.dataset_manifests)
        a.output.mkdir(parents=True, exist_ok=True)
        write(a.output / "JOINT_PHYSICAL_SPLIT.json", result)
        return

    if a.action in ("preflight", "prepare"):
        assert a.argv and a.phase
        env = dict(os.environ)
        env["AF_NATIVE_CONSTRUCT"] = "true" if a.action == "preflight" else "false"
        env["AF_NATIVE_OPERATIONAL_VERIFIER_PENDING"] = "false"
        env["AF_NATIVE_PENDING_PROOFS"] = ""
        subprocess.run(
            [
                sys.executable,
                str(a.repo / "scripts/train/libero_native_schema_contract.py"),
                "--repo",
                str(a.repo),
                "--phase",
                a.phase,
                "--output",
                str(a.output),
                "--argv",
                str(a.argv),
                "--expected-argv-sha256",
                sha(a.argv),
            ],
            env=env,
            check=True,
            cwd=a.repo,
        )
        from egomimic.benchmarks.libero.native_launch_profiles import profile_for_config

        cfg = OmegaConf.to_container(
            OmegaConf.load(a.output / "resolved.yaml"), resolve=True
        )
        PROFILE = profile_for_config(cfg).experiment
        binding = cfg["norm_stats"]["native_saved_state_binding"]
        fingerprint = contract_sha(cfg)
        runtime = json.loads((a.output / "runtime-lock.json").read_text())
        if a.action == "preflight":
            result = json.loads((a.output / "RESULT.json").read_text())
            constructor = checked(
                result["CPU_constructor"]["path"], result["CPU_constructor"]["sha256"]
            )
            constructor["model_contract_sha256"] = fingerprint
            # A new immutable receipt references genuine constructor output; never relabel its config hash.
            constructor["constructor_original"] = result["CPU_constructor"]
            cp = a.output / "CPU_CONSTRUCTOR_CONTRACT.json"
            write(cp, constructor)
            preflight = dict(
                schema="libero-native-launch-preflight/v1",
                status="PASS",
                profile=PROFILE,
                source_commit=binding["source_commit"],
                resolved_config_sha256=sha(a.output / "resolved.yaml"),
                resolved_config_path=str((a.output / "resolved.yaml").resolve()),
                model_contract_sha256=fingerprint,
                normalization_sha256=binding["file_sha256"],
                split_manifest_sha256=binding["split_receipt_sha256"],
                dataset_sha256=cfg["run_provenance"]["dataset_sha256"],
                CPU_constructor=reference(cp),
                physical_proof=dict(
                    path=binding["physical_proof_path"],
                    sha256=binding["physical_proof_sha256"],
                ),
                runtime_lock=reference(a.output / "runtime-lock.json"),
                scope="native_preexecution_cpu_constructor",
                launch_ready=False,
            )
            write(a.output / "PREFLIGHT_RESULT.json", preflight)
        else:
            preflight = checked(
                os.environ["AF_PREFLIGHT_RESULT"],
                os.environ["AF_EXPECTED_PREFLIGHT_SHA256"],
            )
            assert (
                preflight["status"] == "PASS"
                and preflight["profile"] == PROFILE
                and preflight["model_contract_sha256"] == fingerprint
            ), "native preflight scientific identity mismatch"
            assert (
                preflight["source_commit"] == binding["source_commit"]
                and preflight["normalization_sha256"] == binding["file_sha256"]
                and preflight["split_manifest_sha256"]
                == binding["split_receipt_sha256"]
            )
            runtime_equivalent(
                checked(
                    preflight["runtime_lock"]["path"],
                    preflight["runtime_lock"]["sha256"],
                ),
                runtime,
            )
            if a.phase == "full":
                smoke = checked(
                    os.environ["AF_SMOKE_RESULT"],
                    os.environ["AF_EXPECTED_SMOKE_SHA256"],
                )
                assert (
                    smoke["status"] == "PASS"
                    and smoke["profile"] == PROFILE
                    and smoke["model_contract_sha256"] == fingerprint
                ), "native full run lacks exact genuine smoke"
                assert (
                    smoke["preflight"]["sha256"]
                    == os.environ["AF_EXPECTED_PREFLIGHT_SHA256"]
                )
                assert (
                    smoke["checkpoint"]["parameter_count"] == 39750391
                    and smoke["strict_online_and_ema_reload"] is True
                    and smoke["native_artifacts"]["same_pass_verified"] is True
                )
            write(
                a.output / "NATIVE_RUN_PREPARED.json",
                dict(
                    status="PASS",
                    phase=a.phase,
                    model_contract_sha256=fingerprint,
                    resolved_config_sha256=sha(a.output / "resolved.yaml"),
                    preflight=reference(os.environ["AF_PREFLIGHT_RESULT"]),
                ),
            )
    else:
        root = a.output
        from egomimic.benchmarks.libero.saved_hydra_context import (
            load_saved_native_config,
        )

        cfg = OmegaConf.to_container(
            load_saved_native_config(root, a.repo), resolve=True
        )
        preflight = checked(
            os.environ["AF_PREFLIGHT_RESULT"],
            os.environ["AF_EXPECTED_PREFLIGHT_SHA256"],
        )
        assert contract_sha(cfg) == preflight["model_contract_sha256"]
        if len(cfg["data"]["train_datasets"]) > 1:
            if not all(
                os.environ.get(key, "").isdigit()
                for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")
            ):
                raise ValueError(
                    "joint finalize requires a scheduled numeric srun step"
                )
            if not a.runtime_lock:
                p.error("joint finalize requires --runtime-lock")
            from scripts.train import verify_action_flow_training_smoke as shared

            evidence = joint_smoke_evidence(
                cfg,
                root,
                preflight_ref=reference(os.environ["AF_PREFLIGHT_RESULT"]),
                runtime_ref=reference(a.runtime_lock),
                shared=shared,
            )
            write(root / "provenance/JOINT_SMOKE_EVIDENCE.json", evidence)
            return
        artifact = (
            root / "validation_predictions/native_action_flow/valid-step-000002.json"
        )
        assert artifact.is_file()
        evidence = dict(
            schema="libero-native-postsmoke-evidence/v1",
            preflight=reference(os.environ["AF_PREFLIGHT_RESULT"]),
            native_metrics_artifact=reference(artifact),
            resolved_config_sha256=sha(root / ".hydra/config.yaml"),
            model_contract_sha256=contract_sha(cfg),
        )
        write(root / "provenance/NATIVE_SMOKE_EVIDENCE.json", evidence)


if __name__ == "__main__":
    main()
