"""Guard owner-selected contracts without launching or reading a training corpus."""

import runpy
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
SUITES = [
    ("libero10", 605121),
    ("libero_spatial", 270054),
    ("libero_object", 325065),
    ("libero_goal", 280056),
]


@pytest.mark.parametrize("suite,steps", SUITES)
def test_libero_current_and_historical_profiles_remain_distinct(suite, steps):
    with initialize_config_dir(
        config_dir=str(ROOT / "egomimic/hydra_configs"), version_base="1.3"
    ):
        af = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=libero/action_flow_{suite}_oat_dp_matched_s42"],
        )
        dp = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=libero/dp_{suite}_oat_batch16_keep_steps_s42"],
        )
        old = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=libero_historical/dp_{suite}_oat_dp_matched_s42"],
        )
    assert (
        af.model.action_horizon,
        af.model.num_latent_tokens,
        af.model.latent_dim,
    ) == (32, 16, 16)
    assert (af.benchmark.n_obs_steps, af.benchmark.n_action_steps) == (2, 16)
    assert af.model.pipeline.stages[0]._target_.endswith("OATObservationStage")
    assert af.model.pipeline.stages[-1].action_velocity_weight == 1.0
    assert af.model.pipeline.stages[5].inference_method == "euler"
    assert (af.model.num_inference_steps, af.model.cfg_scale, af.model.hidden_dim) == (
        50,
        4.0,
        240,
    )
    assert af.data.train_datasets.libero_panda.valid_ratio == 0.1
    assert af.data.valid_datasets.libero_panda.valid_ratio == 0.1
    assert (
        af.benchmark.batch_size,
        af.trainer.accumulate_grad_batches,
        af.trainer.gradient_clip_val,
    ) == (32, 1, 3.0)
    assert af.callbacks.ema._target_.endswith("ActionFlowFixedEMACallback")
    assert af.callbacks.ema.decay == 0.9978 and af.callbacks.ema.use_warmup is False
    assert af.model.optimizer._target_.endswith("ReleasedUniteCompositeOptimizer")
    assert (
        dp.benchmark.batch_size,
        dp.trainer.accumulate_grad_batches,
        dp.trainer.max_steps,
    ) == (16, 1, steps)
    assert dp.benchmark.original_optimizer_step_target == steps
    assert dp.callbacks.batch_budget.global_batch_size == 16
    assert dp.norm_stats.precomputed_norm_path is None
    assert dp.callbacks.model_checkpoint.save_top_k == -1
    assert old.benchmark.batch_size * old.trainer.accumulate_grad_batches == 1024


def test_dp_multiplier_actual_phase_arguments(monkeypatch):
    for k, v in dict(
        DP_COTRAIN_STANDARD="usocket_chain_manual4919_af_obs_multiplier_261m_v1",
        DP_REPO=str(ROOT),
        DP_WANDB_ID="fixture",
        DP_EXPECTED_HEAD="a" * 40,
        DP_NORM_SHA256="b" * 64,
        DP_SPLIT_SHA256="c" * 64,
        DP_CONTENT_MANIFEST="/fixture/content.json",
        DP_CONTENT_SHA256="d" * 64,
        DP_CONTENT_AGGREGATE_SHA256="e" * 64,
    ).items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("DP_SINGLE_SOURCE_STANDARD", raising=False)
    api = runpy.run_path(str(ROOT / "scripts/train/standard_dp_launch.py"))
    for phase in ("normalize", "smoke", "full"):
        argv = api["arguments"](phase, "/fixture/" + phase, "/fixture/norm")
        with initialize_config_dir(
            config_dir=str(ROOT / "egomimic/hydra_configs"), version_base="1.3"
        ):
            cfg = compose(config_name="train_zarr_cartesian", overrides=argv[1:])
        assert cfg.name == "planar_uc_manual4919_dp_261m_af_obs_multiplier"
        assert (
            cfg.planar.batch_size == 32
            and cfg.model.pipeline.stages[5].condition_input_dim == 128
        )
        assert cfg.model.pipeline.stages[2].conditioning_input == "retiming_multiplier"
        assert cfg.model.pipeline.stages[2].get("speed_reference") is None
        assert cfg.callbacks.get("ema") is None
        assert cfg.model.optimizer._target_ == "torch.optim.AdamW"
        assert cfg.model.scheduler.warmup_steps == 500
        assert cfg.norm_stats.sample_frac == 1.0


@pytest.mark.parametrize(
    "cotrain,single",
    [
        ("unknown", ""),
        ("", "unknown"),
        (
            "usocket_chain_manual4919_af_obs_multiplier_261m_v1",
            "chain_manual4919_retimed_v1",
        ),
    ],
)
def test_unrecognized_or_conflicting_profiles_fail_before_launch(
    monkeypatch, cotrain, single
):
    monkeypatch.setenv("DP_COTRAIN_STANDARD", cotrain)
    monkeypatch.setenv("DP_SINGLE_SOURCE_STANDARD", single)
    api = runpy.run_path(str(ROOT / "scripts/train/standard_dp_launch.py"))
    with pytest.raises(ValueError):
        api["recipe"]()
    env = dict(__import__("os").environ)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/train/planar_v2_dp_cotrain_clean_skynet_v1.sbatch"),
        ],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 64


def test_current_pair_helper_composes_all_eight_without_claiming_split_proof(tmp_path):
    api = runpy.run_path(str(ROOT / "tools/libero_oat_pair_config.py"))
    result = api["compose_all"](ROOT, tmp_path)
    assert len(result["rows"]) == 8
    assert result["recipe_contract"] == "current"
    assert (
        result["episode_split_match"]
        == "UNVERIFIED_REQUIRES_MATERIALIZED_EPISODE_LISTS"
    )
    assert result["full_training_ready"] is False
    import json

    for suite, steps in SUITES:
        af = json.loads((tmp_path / f"action_flow-{suite}.json").read_text())
        dp = json.loads((tmp_path / f"dp-{suite}.json").read_text())
        assert af["trainer"]["max_steps"] == 120000
        assert dp["trainer"]["max_steps"] == steps
        assert api["validate"](dp, "dp", suite) == 16
        with pytest.raises(AssertionError):
            api["validate"](dp, "dp", suite, recipe_contract="historical-v7")
        af["trainer"]["max_steps"] = 80000
        with pytest.raises(AssertionError):
            api["validate"](af, "action_flow", suite)
        assert (
            api["validate"](af, "action_flow", suite, recipe_contract="historical-v7")
            == 32
        )


def test_historical_native_names_preserve_aliases_and_reject_current_recipe():
    from egomimic.benchmarks.libero.native_launch_profiles import (
        HISTORICAL_NATIVE_PROFILES,
        PROFILES,
        historical_profile_for_suite,
        profile_for_experiment,
        profile_for_suite,
    )

    assert PROFILES is HISTORICAL_NATIVE_PROFILES
    assert profile_for_suite is historical_profile_for_suite
    for suite, _ in SUITES:
        assert historical_profile_for_suite(suite).experiment.startswith(
            "libero_historical/"
        )
        with pytest.raises(ValueError):
            profile_for_experiment(f"libero/action_flow_{suite}_oat_dp_matched_s42")


@pytest.mark.parametrize("receipt", [None, {"recipe_contract": "historical-v7"}])
def test_current_full_proof_rejects_legacy_gate_before_creating_output(
    monkeypatch, tmp_path, receipt
):
    import json
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    api = runpy.run_path(str(ROOT / "tools/libero_oat_training_proof.py"))
    monkeypatch.setenv("SLURM_STEP_ID", "fixture")
    monkeypatch.setattr(
        api["subprocess"],
        "check_output",
        lambda argv, **kwargs: "a" * 40 if argv[1] == "rev-parse" else "",
    )
    argv = [
        "proof",
        "--root",
        str(tmp_path),
        "--source-commit",
        "a" * 40,
        "--index",
        "0",
        "--phase",
        "full",
        "--attempt",
        "v1",
    ]
    if receipt is not None:
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(receipt))
        argv += ["--readiness-receipt", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises((ValueError, AssertionError)):
        api["main"]()
    assert not (tmp_path / "full-v1").exists()
