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
        / "egomimic/hydra_configs/experiment/libero"
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
def test_ice_native_preserves_real_resource_bindings(constraint, gpu):
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
        native_profile=PROFILES["libero_spatial"].experiment,
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
