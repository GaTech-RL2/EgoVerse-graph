"""Suite/seed isolation and portable allocated-resource contracts, without models."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from libero_maintained_dispatch_v1 import (  # noqa: E402 - task tool path initialized above
    validate_data_receipt,
)


def test_joint_schedule_reuse_hashes_exact_git_diff_and_rejects_model_changes(tmp_path):
    import hashlib
    import json
    import subprocess

    from scripts.train.libero_native_launch_contract import (
        joint_smoke_schedule_reuse,
        reference,
    )

    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                *args,
            ]
        )

    git("init", "-q")
    source = repo / "egomimic/eval/eval.py"
    source.parent.mkdir(parents=True)
    source.write_text("old schedule\n")
    git("add", ".")
    git("commit", "-qm", "old")
    old = git("rev-parse", "HEAD").decode().strip()
    source.write_text("repaired schedule\n")
    git("commit", "-qam", "repair")
    head = git("rev-parse", "HEAD").decode().strip()
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps({"source_commit": old}))
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "identities": {"source_commit": old},
                "preflight": reference(preflight_path),
            }
        )
    )
    regression = tmp_path / "regression.log"
    regression.write_text("actual CPU evidence fixture\n")
    proof = dict(
        schema="libero-joint-smoke-schedule-reuse/v1",
        scope="explicit_cross_epoch_validation_schedule",
        from_source_commit=old,
        to_source_commit=head,
        target_repo=str(repo),
        smoke=reference(smoke_path),
        cpu_regression_status="PASS",
        changed_paths=["egomimic/eval/eval.py"],
        audited_diff_sha256=hashlib.sha256(
            git("diff", "--no-ext-diff", "--binary", old, head)
        ).hexdigest(),
        cpu_regression=reference(regression),
    )
    proof_path = tmp_path / "proof.json"

    def check():
        proof_path.write_text(json.dumps(proof))
        return joint_smoke_schedule_reuse(
            reference(proof_path),
            smoke_ref=reference(smoke_path),
            target_source=proof["to_source_commit"],
        )

    assert check()[0] == old
    proof["audited_diff_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audited diff differs"):
        check()
    model = repo / "egomimic/model.py"
    model.write_text("changed model\n")
    git("add", ".")
    git("commit", "-qm", "forbidden")
    proof["to_source_commit"] = git("rev-parse", "HEAD").decode().strip()
    proof["changed_paths"] = (
        git("diff", "--name-only", old, "HEAD").decode().splitlines()
    )
    proof["audited_diff_sha256"] = hashlib.sha256(
        git("diff", "--no-ext-diff", "--binary", old, "HEAD")
    ).hexdigest()
    with pytest.raises(ValueError, match="unaudited source paths"):
        check()


from egomimic.benchmarks.libero.native_launch_profiles import (  # noqa: E402 - task tool path initialized above
    PROFILES,
    profile_for_argv,
    profile_for_config,
)


def cfg(suite):
    p = PROFILES[suite]
    return dict(
        name=p.name,
        benchmark=dict(suite=suite),
        data={
            mode + "_datasets": {"libero_panda": {"resolver": {"suite": suite}}}
            for mode in ("train", "valid")
        },
    )


@pytest.mark.parametrize("suite", list(PROFILES))
def test_typed_suite_and_receipt_isolation(suite):
    p = PROFILES[suite]
    c = cfg(suite)
    assert profile_for_config(c) == p
    assert profile_for_argv(["+experiment=" + p.experiment]) == p
    receipt = dict(
        suite=suite,
        replay_path="/replay",
        logical_dataset_sha256="a" * 64,
        episodes=500,
        valid_ratio=0.01,
        split_seed=42,
        train_episodes=495,
        valid_episodes=5,
        action_dim=7,
        task_uids=list(p.task_uids),
    )
    # Use exact maintained receipt schema field names.
    receipt.update(
        dataset_logical_sha256="a" * 64,
        frames=5000,
        action_shape=[5000, 7],
        train_episode_indices=list(range(495)),
        valid_episode_indices=list(range(495, 500)),
    )
    assert (
        validate_data_receipt(
            receipt,
            replay_path="/replay",
            expected_logical_sha="a" * 64,
            profile=p.experiment,
        )["id_overlap"]
        == 0
    )
    for other in PROFILES:
        if other == suite:
            continue
        wrong = copy.deepcopy(c)
        wrong["benchmark"]["suite"] = other
        with pytest.raises(ValueError):
            profile_for_config(wrong)
        with pytest.raises((ValueError, KeyError)):
            validate_data_receipt(
                receipt,
                replay_path="/replay",
                expected_logical_sha="a" * 64,
                profile=PROFILES[other].experiment,
            )


@pytest.mark.parametrize("suite", ["libero_spatial", "libero_goal", "libero_object"])
def test_recipe_only_changes_suite_identity(suite):
    text = (
        ROOT
        / "egomimic/hydra_configs/experiment/libero_historical"
        / (PROFILES[suite].name + ".yaml")
    ).read_text()
    assert "action_flow_libero10_h240_euler50_dithalf_80k_s42" in text
    assert "suite: " + suite in text
    assert PROFILES[suite].replay_environment in text
    assert all(
        key not in text
        for key in ("model:", "trainer:", "normalizer:", "optimizer:", "\nseed:")
    )


def test_catalog_task_uids_match_registry():
    from egomimic.benchmarks.libero.catalog import TASK_IDS, get_tasks

    for suite, p in PROFILES.items():
        assert tuple(TASK_IDS[t][2] for t in get_tasks(suite)) == p.task_uids


def scheduler():
    spec = importlib.util.spec_from_file_location(
        "scheduler_suite", ROOT / "scripts/train/validate_slurm_job_contract.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize(
    "constraint,gpu",
    [
        ("H100", "NVIDIA H100 80GB HBM3"),
        ("H200", "NVIDIA H200"),
        ("H100|H200", "NVIDIA H200"),
    ],
)
@pytest.mark.parametrize(
    "profile",
    [
        PROFILES["libero_spatial"].experiment,
        "libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
    ],
)
def test_ice_native_preserves_real_resource_bindings(constraint, gpu, profile):
    m = scheduler()
    fields = dict(
        JobId="123",
        Account="account",
        Partition="partition",
        QOS="qos",
        NumNodes="1",
        NumTasks="1",
        NumCPUs="8",
        MinCPUsNode="8",
        MinMemoryNode="128G",
        Features=constraint,
        TimeLimit="1-00:00:00",
        ReqTRES="cpu=8,mem=128G,node=1,gres/gpu=1",
    )
    fields["CPUs/Task"] = "8"
    args = dict(
        expected_job_id="123",
        expected_account="account",
        expected_partition="partition",
        expected_qos="qos",
        expected_cpus=8,
        expected_memory="128G",
        expected_time_limit="1-00:00:00",
        expected_constraint=constraint,
        native_profile=profile,
        gpu_probe=dict(
            status="PASSED",
            gpu_name=gpu,
            world_size=1,
            rank=0,
            local_rank=0,
            bf16_supported=True,
            bf16_forward_backward=dict(finite=True),
        ),
    )
    assert m.evaluate_contract(fields, **args)[2] == []
    for field in ("Account", "Partition", "QOS", "Features", "NumCPUs"):
        bad = dict(fields)
        bad[field] = "wrong" if field != "NumCPUs" else "4"
        assert m.evaluate_contract(bad, **args)[2]
    args["gpu_probe"]["bf16_forward_backward"]["finite"] = False
    with pytest.raises(m.ContractError):
        m.evaluate_contract(fields, **args)


@pytest.mark.parametrize("suite", list(PROFILES))
@pytest.mark.parametrize("seed", [42, 43])
def test_metric_training_seed_is_typed_and_not_split_seed(suite, seed):
    from egomimic.benchmarks.libero.action_flow_artifacts import (
        SEED_SHA,
        validate_identity,
    )

    identity = dict(
        suite=suite,
        source="libero_panda",
        action_dim=7,
        action_horizon=16,
        seed=seed,
        sample_count=32,
        energy_seed_bank_sha256=SEED_SHA,
        inference_method="euler",
        inference_steps=50,
        normalization_scope="training_episodes_only",
        effective_batch_size=32,
        homogeneous="not_applicable_single_source",
        source_commit="a" * 40,
        resolved_config_sha256="a" * 64,
        split_sha256="a" * 64,
        normalizer_state_sha256="a" * 64,
        dataset_logical_sha256="a" * 64,
    )
    if seed == 43 and suite not in ("libero_goal", "libero_object"):
        with pytest.raises(ValueError):
            validate_identity(identity)
    else:
        assert validate_identity(identity)


def test_late_preflight_cross_suite_rejected_before_evidence_reads():
    from scripts.train.verify_libero_native_action_flow_smoke import validate_preflight

    pf = dict(
        schema="libero-native-launch-preflight/v1",
        status="PASS",
        profile=PROFILES["libero10"].experiment,
    )
    for suite in ("libero_spatial", "libero_goal", "libero_object"):
        with pytest.raises(ValueError, match="native profile mismatch"):
            validate_preflight(pf, {}, PROFILES[suite].experiment)


@pytest.mark.parametrize(
    "suite", ["libero_spatial", "libero_goal", "libero_object", "libero10"]
)
def test_actual_hydra_recipe_composes_same_science_and_fixed_split(monkeypatch, suite):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    p = PROFILES[suite]
    monkeypatch.setenv(p.replay_environment, "/replay")
    for seed in [42, 43] if suite in ("libero_goal", "libero_object") else [42]:
        with initialize_config_dir(
            version_base="1.3", config_dir=str(ROOT / "egomimic/hydra_configs")
        ):
            c = compose(
                config_name="train_zarr_cartesian",
                overrides=[
                    "hydra/launcher=basic",
                    "+experiment=" + p.experiment,
                    "seed=" + str(seed),
                ],
            )
        assert c.name == p.name and c.benchmark.suite == suite and c.seed == seed
        assert c.data.train_datasets.libero_panda.split_seed == 42
        assert c.data.valid_datasets.libero_panda.split_seed == 42
        assert (
            c.model.hidden_dim == 240
            and c.model.action_horizon == 16
            and c.model.action_dim == 7
        )
        assert c.data.train_dataloader_params.libero_panda.batch_size == 32
        assert c.model.pipeline.stages[6].inference_method == "euler"
        monkeypatch.setenv("LIBERO10_REPLAY_ROOT", "/reference")
        with initialize_config_dir(
            version_base="1.3", config_dir=str(ROOT / "egomimic/hydra_configs")
        ):
            reference = compose(
                config_name="train_zarr_cartesian",
                overrides=[
                    "hydra/launcher=basic",
                    "+experiment=" + PROFILES["libero10"].experiment,
                ],
            )
        for field in ("model", "normalizer", "trainer", "callbacks", "evaluator"):
            assert OmegaConf.to_container(
                c[field], resolve=False
            ) == OmegaConf.to_container(reference[field], resolve=False)


def test_explicit_joint_profile_and_contract_reject_single_source_or_semantic_drift(
    tmp_path, monkeypatch
):
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_profile_for_config,
    )

    spec = importlib.util.spec_from_file_location(
        "joint_contract_test", ROOT / "scripts/train/libero_native_launch_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("LIBERO_ORIGINAL_SPATIAL_ROOT", str(tmp_path / "original"))
    monkeypatch.setenv("LIBERO_SHARPA_SPATIAL_ROOT", str(tmp_path / "sharpa"))
    monkeypatch.setenv("LIBERO_DIAGNOSTIC_SEED_BANK", str(tmp_path / "SEEDS.json"))
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        config = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=[
                "hydra/launcher=basic",
                "+experiment=libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
                "paths.output_dir=" + str(tmp_path),
                "paths.work_dir=" + str(ROOT),
            ],
        )
        config.hydra.runtime.output_dir = str(tmp_path)
        HydraConfig.instance().set_config(config)
        c = OmegaConf.to_container(
            OmegaConf.masked_copy(config, [k for k in config if k != "hydra"]),
            resolve=True,
        )
    assert joint_profile_for_config(c).sources[1] == ("libero_panda_sharpa", 22, 69, 22)
    missing_latent = copy.deepcopy(c)
    missing_latent["evaluator"]["action_flow_diagnostics"]["provenance"].pop(
        "latent_shape"
    )
    with pytest.raises(ValueError, match="diagnostic latent shape"):
        joint_profile_for_config(missing_latent)
    with pytest.raises(ValueError, match="unsupported native"):
        profile_for_config(c)
    contract = module.model_contract(c)
    assert contract["schema"] == "libero-joint-scientific-contract/v1"
    assert contract["train_batch"] == {"libero_panda": 16, "libero_panda_sharpa": 16}
    assert module.contract_sha(c) == module.contract_sha(copy.deepcopy(c))
    relocated = copy.deepcopy(c)
    relocated["callbacks"]["ema"]["final_checkpoint_path"] = (
        "/other/output/checkpoints/last.ckpt"
    )
    relocated["evaluator"]["action_flow_diagnostics"]["artifact_root"] = (
        "/other/output/diagnostics"
    )
    assert module.contract_sha(c) == module.contract_sha(relocated)
    changed_ema = copy.deepcopy(c)
    changed_ema["callbacks"]["ema"]["decay"] = 0.99
    assert module.contract_sha(c) != module.contract_sha(changed_ema)
    changes = [
        (
            "model.pipeline.stages.4.encoders.libero_panda_sharpa.backbone.hidden_dim",
            384,
        ),
        ("model.pipeline.stages.6.field.backbone.depth", 24),
        ("model.pipeline.stages.4.encoders.libero_panda_sharpa.action_dim", 7),
        ("model.pipeline.stages.7.decoders.libero_panda_sharpa.condition_dim", 276),
        ("model.pipeline.stages.6.experts", {}),
        ("model.pipeline.stages.3.num_tokens", 8),
        ("data.train_dataloader_params.libero_panda_sharpa.batch_size", 32),
        ("data.train_datasets.libero_panda_sharpa.valid_ratio", 0.01),
        (
            "data.train_datasets.libero_panda_sharpa.resolver.action_representation",
            "native",
        ),
        ("model.pipeline.stages.6.num_inference_steps", 8),
        ("normalizer._target_", "egomimic.rldb.zarr.libero_dataset.LiberoNormalizer"),
    ]
    for path, replacement in changes:
        wrong = copy.deepcopy(c)
        keys = path.split(".")
        node = wrong
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        node[keys[-1]] = replacement
        with pytest.raises(ValueError, match="joint LIBERO"):
            module.contract_sha(wrong)


@pytest.mark.parametrize("phase", ["normalization", "preflight", "smoke", "full"])
def test_joint_phase_factory_composes_actual_paths_and_private_shared_contract(
    tmp_path, monkeypatch, phase
):
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_phase_argv,
        joint_profile_for_config,
    )

    # These env values must be replaced by the exact typed arguments, including
    # spaces and '=' which must survive the shell/Hydra boundary literally.
    for key in (
        "LIBERO_ORIGINAL_SPATIAL_ROOT",
        "LIBERO_SHARPA_SPATIAL_ROOT",
        "LIBERO_DIAGNOSTIC_SEED_BANK",
    ):
        monkeypatch.setenv(key, "/wrong/unselected")
    original = str(tmp_path / "original data=1")
    sharpa = str(tmp_path / "hand data=2")
    args = dict(
        source_root=str(ROOT),
        original_root=original,
        sharpa_root=sharpa,
        output_root=str(tmp_path / phase),
        context_path=str(tmp_path / "context.json"),
        split_path=str(tmp_path / "split.json"),
        split_sha256="a" * 64,
        normalization_sha256=None if phase == "normalization" else "b" * 64,
        source_commit="c" * 40,
        diagnostic_seed_bank=str(tmp_path / "seeds.json"),
        wandb_entity="test",
        wandb_project="test",
        wandb_run_id="joint-" + phase,
    )
    argv = joint_phase_argv(phase=phase, **args)
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=argv[1:],
        )
        cfg.hydra.runtime.output_dir = str(tmp_path / phase)
        HydraConfig.instance().set_config(cfg)
        resolved = OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True
        )
    joint_profile_for_config(resolved)
    assert resolved["evaluator"]["action_flow_diagnostics"]["provenance"][
        "latent_shape"
    ] == [16, 16]
    assert resolved["trainer"]["log_every_n_steps"] == (1 if phase == "smoke" else 10)
    if phase != "normalization":
        from scripts.train import libero_native_launch_contract as adapter

        def compose_exact(command, *, cwd, text):
            assert command[1:3] == ["-m", "egomimic.trainHydra"]
            assert command[3:-3] == argv
            assert command[-3:] == ["--cfg", "job", "--resolve"]
            assert cwd == ROOT and text is True
            return OmegaConf.to_yaml(OmegaConf.create(resolved))

        monkeypatch.setattr(adapter.subprocess, "check_output", compose_exact)
        destination = tmp_path / (phase + "-composition")
        destination.mkdir()
        composed = adapter.compose_joint_phase(
            args, phase=phase, output=destination, repo=ROOT
        )
        assert composed["launch_ready"] is False
        assert (destination / "exact-phase.argv0").read_bytes() == b"\0".join(
            arg.encode() for arg in argv
        ) + b"\0"
        with pytest.raises(FileExistsError):
            adapter.compose_joint_phase(
                args, phase=phase, output=destination, repo=ROOT
            )
    if phase in {"smoke", "full"}:
        import json

        from scripts.train.libero_native_launch_contract import (
            contract_sha,
            joint_prepare_evidence,
            reference,
        )

        runtime = {
            key: {}
            for key in (
                "python",
                "installed_distributions",
                "core_distribution_versions",
                "dependency_inputs",
                "source",
                "torch_runtime",
            )
        }
        runtime["dependency_inputs"] = []
        runtime_path = tmp_path / "synthetic-runtime.json"
        runtime_path.write_text(json.dumps(runtime))
        identity = {
            key: resolved["run_provenance"][key]
            for key in (
                "source_commit",
                "normalization_sha256",
                "split_manifest_sha256",
            )
        }
        preflight = dict(
            schema="libero-joint-launch-preflight/v1",
            status="PASS",
            profile="libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
            model_contract_sha256=contract_sha(resolved),
            runtime_lock=reference(runtime_path),
            **identity,
        )
        preflight_path = tmp_path / "synthetic-preflight.json"
        preflight_path.write_text(json.dumps(preflight))
        kwargs = dict(
            phase=phase,
            preflight_ref=reference(preflight_path),
            runtime_ref=reference(runtime_path),
        )
        if phase == "full":
            with pytest.raises(ValueError, match="genuine eight-update"):
                joint_prepare_evidence(resolved, **kwargs)
            smoke = dict(
                schema="libero-joint-action-flow-smoke/v1",
                status="PASS",
                profile=preflight["profile"],
                model_contract_sha256=contract_sha(resolved),
                identities=identity,
                preflight=reference(preflight_path),
                strict_online_and_alias_safe_ema_reload=True,
                checkpoint=dict(
                    parameter_count=78182614,
                    global_step=8,
                    joint_ema=dict(alias_safe_strict_reload=True),
                ),
            )
            smoke_path = tmp_path / "synthetic-smoke.json"
            smoke_path.write_text(json.dumps(smoke))
            kwargs["smoke_ref"] = reference(smoke_path)
        assert joint_prepare_evidence(resolved, **kwargs)["launch_ready"] is False
        if phase == "full":
            from scripts.train import libero_native_launch_contract as adapter

            original_smoke = copy.deepcopy(smoke)
            previous = dict(preflight, source_commit="d" * 40)
            previous["model_contract_sha256"] = adapter.joint_contract_at_source(
                resolved, "d" * 40
            )
            previous_runtime = dict(runtime, source={"head": "d" * 40, "dirty": False})
            old_runtime_path = tmp_path / "previous-runtime.json"
            old_runtime_path.write_text(json.dumps(previous_runtime))
            previous["runtime_lock"] = reference(old_runtime_path)
            runtime["source"] = {"head": identity["source_commit"], "dirty": False}
            runtime_path.write_text(json.dumps(runtime))
            preflight["runtime_lock"] = reference(runtime_path)
            preflight_path.write_text(json.dumps(preflight))
            kwargs["runtime_ref"] = reference(runtime_path)
            kwargs["preflight_ref"] = reference(preflight_path)
            previous_path = tmp_path / "previous-preflight.json"
            previous_path.write_text(json.dumps(previous))
            smoke["identities"] = dict(identity, source_commit="d" * 40)
            smoke["model_contract_sha256"] = previous["model_contract_sha256"]
            smoke["preflight"] = reference(previous_path)
            smoke_path.write_text(json.dumps(smoke))
            kwargs["smoke_ref"] = reference(smoke_path)
            with pytest.raises(ValueError, match="smoke proof differs"):
                joint_prepare_evidence(resolved, **kwargs)
            repair_ref = {"path": "separate-repair-proof", "sha256": "e" * 64}
            monkeypatch.setattr(
                adapter,
                "joint_smoke_schedule_reuse",
                lambda *a, **k: ("d" * 40, previous),
            )
            reused = joint_prepare_evidence(
                resolved, **kwargs, smoke_source_repair_ref=repair_ref
            )
            assert reused["smoke_source_repair"] == repair_ref
            assert (
                json.loads(smoke_path.read_text())["identities"]["source_commit"]
                == "d" * 40
            )
            previous["model_contract_sha256"] = "wrong"
            with pytest.raises(ValueError, match="scientific contract differs"):
                joint_prepare_evidence(
                    resolved, **kwargs, smoke_source_repair_ref=repair_ref
                )
            smoke = original_smoke
            smoke["checkpoint"]["global_step"] = 2
            smoke_path.write_text(json.dumps(smoke))
            kwargs["smoke_ref"] = reference(smoke_path)
            with pytest.raises(ValueError, match="smoke proof differs"):
                joint_prepare_evidence(resolved, **kwargs)
    if phase == "preflight":
        from scripts.train.libero_native_launch_contract import contract_sha

        for other_phase in ("smoke", "full"):
            other_args = {
                **args,
                "output_root": str(tmp_path / other_phase),
                "wandb_run_id": "joint-" + other_phase,
            }
            with hydra.initialize_config_dir(
                version_base="1.2", config_dir=str(ROOT / "egomimic/hydra_configs")
            ):
                other = hydra.compose(
                    config_name="train_zarr_cartesian",
                    return_hydra_config=True,
                    overrides=joint_phase_argv(phase=other_phase, **other_args)[1:],
                )
                other.hydra.runtime.output_dir = other_args["output_root"]
                HydraConfig.instance().set_config(other)
                other_resolved = OmegaConf.to_container(
                    OmegaConf.masked_copy(other, [k for k in other if k != "hydra"]),
                    resolve=True,
                )
            assert contract_sha(resolved) == contract_sha(other_resolved)
    if phase == "smoke":
        from scripts.train.libero_native_launch_contract import joint_prepare_evidence

        assert resolved["model"]["gradient_telemetry_cadence"] == 1
        missing_telemetry = copy.deepcopy(resolved)
        missing_telemetry["model"]["gradient_telemetry_cadence"] = 100
        with pytest.raises(ValueError, match="per-step gradient telemetry"):
            joint_prepare_evidence(
                missing_telemetry, phase="smoke", preflight_ref={}, runtime_ref={}
            )

        from scripts.train.verify_libero_native_action_flow_smoke import (
            validate_joint_smoke_config,
        )

        assert (
            validate_joint_smoke_config(resolved, args["output_root"], "c" * 40)[
                "optimizer_updates"
            ]
            == 8
        )
        for key, value in (
            ("trainer.max_steps", 2),
            ("trainer.accumulate_grad_batches", 2),
            ("run_provenance.run_kind", "full"),
            ("logger.wandb.mode", "offline"),
            ("callbacks.model_checkpoint.save_last", True),
            ("run_provenance.source_commit", None),
        ):
            altered = OmegaConf.create(copy.deepcopy(resolved))
            OmegaConf.update(altered, key, value)
            with pytest.raises(
                ValueError, match="joint (smoke config mismatch|LIBERO:)"
            ):
                validate_joint_smoke_config(altered, args["output_root"], "c" * 40)
    for mode in ("train", "valid"):
        for source, root in (
            ("libero_panda", original),
            ("libero_panda_sharpa", sharpa),
        ):
            assert (
                resolved["data"][mode + "_datasets"][source]["resolver"]["folder_path"]
                == root
            )
    assert resolved["callbacks"]["model_checkpoint"]["save_last"] is False
    assert resolved["trainer"]["default_root_dir"] == args["output_root"]
    assert (
        resolved["callbacks"]["model_checkpoint"]["dirpath"]
        == args["output_root"] + "/checkpoints"
    )
    assert (
        resolved["callbacks"]["ema"]["final_checkpoint_path"]
        == args["output_root"] + "/checkpoints/last.ckpt"
    )
    assert resolved["runtime"]["slurm_requeue_owner"] == "runner"
    assert resolved["callbacks"]["early_recovery"]["first_step"] == 1000
    assert resolved["trainer"]["max_steps"] == (8 if phase == "smoke" else 120000)
    if phase == "normalization":
        assert "logger" not in resolved
        # The experiment itself extends the evaluator body; group deletion
        # would leave an invalid target-less partial mapping. Normalization-only
        # trainHydra returns before model/trainer construction or evaluation.
        assert resolved["evaluator"]["_target_"].endswith(".PlanarActionEval")
        assert resolved["norm_stats"]["precomputed_norm_path"] is None
        assert resolved["norm_stats_only"] is True
        assert resolved["norm_stats"]["export_context_path"] == args["context_path"]
        assert resolved["run_provenance"]["normalization_sha256"] is None
    else:
        assert resolved["norm_stats"]["precomputed_norm_path"] == args["context_path"]
        assert resolved["logger"]["wandb"]["id"] == args["wandb_run_id"]
        assert resolved["logger"]["wandb"]["save_dir"] == args["output_root"]
        assert resolved["trainer"]["val_check_interval"] == (
            8 if phase == "smoke" else 15000
        )
        with pytest.raises(ValueError, match="normalization_sha256"):
            joint_phase_argv(phase=phase, **{**args, "normalization_sha256": None})
    for key, bad_value in (
        ("original_root", "relative/root"),
        ("sharpa_root", "/path\nwith-newline"),
        ("source_commit", "unknown"),
        ("wandb_run_id", "unsafe id"),
    ):
        with pytest.raises(ValueError):
            joint_phase_argv(phase=phase, **{**args, key: bad_value})


def test_joint_phase_factory_data_only_snapshot_never_authorizes_training(tmp_path):
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        joint_phase_argv,
        joint_profile_for_config,
    )

    args = dict(
        source_root=str(ROOT),
        original_root=str(tmp_path / "original"),
        sharpa_root=str(tmp_path / "sharpa"),
        output_root=str(tmp_path / "out"),
        context_path=str(tmp_path / "context.json"),
        split_path=str(tmp_path / "split.json"),
        split_sha256="a" * 64,
        normalization_sha256=None,
        source_commit=None,
        diagnostic_seed_bank=str(tmp_path / "bank.json"),
        wandb_entity="test",
        wandb_project="test",
        wandb_run_id="data-only",
        data_preparation_source_sha256="d" * 64,
    )
    argv = joint_phase_argv(phase="normalization", **args)
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=argv[1:],
        )
        cfg.hydra.runtime.output_dir = str(tmp_path / "out")
        HydraConfig.instance().set_config(cfg)
        resolved = OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True
        )
    joint_profile_for_config(resolved)
    assert (
        resolved["norm_stats_only"] is True
        and resolved["run_provenance"]["source_commit"] is None
    )
    assert (
        resolved["run_provenance"]["data_preparation_source_snapshot_sha256"]
        == "d" * 64
    )
    for phase in ("preflight", "smoke", "full"):
        with pytest.raises(ValueError, match="data-only source snapshots"):
            joint_phase_argv(phase=phase, **args)
    with pytest.raises(ValueError, match="exact SHA"):
        joint_phase_argv(
            phase="normalization", **{**args, "data_preparation_source_sha256": None}
        )
