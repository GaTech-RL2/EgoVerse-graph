"""Matched launch/config contracts and actual shared train/EMA/inference paths."""

import copy
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.benchmarks.libero.arc_decoders import (
    compare_decoder_runs,
    decoder_training_arguments,
    validate_decoder_config,
)
from egomimic.benchmarks.libero.arc_sweep import profile_settings
from egomimic.benchmarks.libero.cluster import training_arguments
from egomimic.models.arc_diffusion import DECODER_VARIANTS
from scripts.benchmarks.launch_libero_osmo import arc_decoder_workflow

ROOT = Path(__file__).parents[1]


def compose_arguments(arguments):
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[*arguments[3:], "norm_stats.save_cache_dir=out/norm_stats"],
        )


def experiment(variant="shared", profile="stk_2", mode="stk"):
    args = decoder_training_arguments(
        "libero_spatial",
        "data.zarr",
        "out",
        "full",
        5001,
        variant=variant,
        profile=profile,
        arc_mode=mode,
        gpus=8,
    )
    cfg = compose_arguments(args)
    return cfg, {"model": OmegaConf.to_container(cfg.model, resolve=True)}


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
@pytest.mark.parametrize("profile,mode", [("stk_2", "stk"), ("dur_2", "dur")])
def test_config_changes_only_decoder_and_explicit_provenance(variant, profile, mode):
    cfg, tree = experiment(variant, profile, mode)
    validate_decoder_config(
        tree, suite="libero_spatial", variant=variant, profile=profile, arc_mode=mode
    )
    arguments = training_arguments(
        f"arc_{mode}",
        "libero_spatial",
        "data.zarr",
        "out",
        "full",
        5001,
        arc_backbone="oat_dp",
        gpus=8,
    )
    arguments += [
        f"benchmark.{key}={value}"
        for key, value in profile_settings(profile, mode).items()
    ]
    original = compose_arguments(arguments)
    actual_model = copy.deepcopy(tree["model"])
    actual_network = actual_model["pipeline"]["stages"][3]["policy"]["model"]
    actual_network["_target_"] = (
        "egomimic.models.oat.diffusion.GraphDiffusionTransformer"
    )
    actual_network.pop("decoder_variant")
    actual_model["benchmark_protocol"] = OmegaConf.to_container(
        original.model.benchmark_protocol, resolve=True
    )
    assert actual_model == OmegaConf.to_container(original.model, resolve=True)
    for key in ("trainer", "data", "normalizer", "norm_stats", "callbacks"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(
            original[key], resolve=True
        )
    assert cfg.seed == original.seed == 42
    assert cfg.trainer.max_epochs == 5001
    assert (
        cfg.trainer.devices
        * cfg.benchmark.batch_size
        * cfg.trainer.accumulate_grad_batches
        == 1024
    )


@pytest.mark.parametrize("bad", ["variant", "budget", "columns", "codec", "backbone"])
def test_wrong_checkpoint_experiment_is_rejected(bad):
    _, config = experiment()
    protocol = config["model"]["benchmark_protocol"]
    if bad == "variant":
        protocol["arc_decoder_variant"] = "separate"
    elif bad == "budget":
        protocol["arc_decoder_total_layers"] = 8
    elif bad == "columns":
        protocol["arc_timing_columns"] = [2, 9]
    elif bad == "codec":
        config["model"]["pipeline"]["stages"][1]["max_translation"] = 1.6
    else:
        config["model"]["pipeline"]["stages"][3]["policy"]["model"]["n_emb"] = 512
    with pytest.raises(ValueError):
        validate_decoder_config(
            config,
            suite="libero_spatial",
            variant="shared",
            profile="stk_2",
            arc_mode="stk",
        )


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
def test_workflow_keeps_full_training_and_dependent_l40s_evaluation(variant):
    spec = arc_decoder_workflow(
        "a" * 40,
        "decoder-test",
        "libero_spatial",
        variant=variant,
        profile="stk_2",
        arc_mode="stk",
        replay_run="verified-replay",
        gpus=8,
    )["workflow"]
    train, evaluate = spec["tasks"]
    assert (train["name"], evaluate["name"]) == ("train", "evaluate")
    assert evaluate["inputs"] == [{"task": "train"}]
    assert spec["resources"]["default"]["gpu"] == 8
    assert spec["resources"]["evaluation"]["gpu"] == 1
    assert spec["resources"]["evaluation"]["memory"] == "120Gi"
    assert all(
        resource["platform"] == "ovx-l40s" for resource in spec["resources"].values()
    )
    assert train["environment"]["ARC_DECODER_VARIANT"] == variant
    assert train["environment"]["EPOCHS"] == "5001"
    assert train["environment"]["ARC_REPLAY_RUN"] == "verified-replay"
    assert evaluate["environment"]["RUN_KIND"] == "policy_evaluation"
    assert evaluate["environment"]["EVALUATION_WORKERS"] == "5"
    assert "egomimic.benchmarks.libero.arc_decoders" in train["files"][0]["contents"]


@pytest.mark.parametrize("variant", DECODER_VARIANTS)
def test_actual_training_optimizer_ema_resume_and_action_decoding(tmp_path, variant):
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import train
    from tests.test_libero_benchmark import make_replay
    from tests.test_oat_training import config_for

    dataset = tmp_path / "replay.zarr"
    make_replay(dataset)
    cfg = config_for("libero_arc_stream_policy", dataset, tmp_path / "training")
    cfg.benchmark.arc_decoder_variant = variant
    cfg.benchmark.arc_waypoints = 8
    cfg.trainer.precision = "bf16-mixed" if variant == "shape_masked" else "32-true"
    cfg.model.pipeline.stages[3].policy.model.n_emb = 32
    cfg.model.pipeline.stages[3].policy.model.n_layer = 2
    cfg.model.benchmark_protocol.arc_decoder_total_layers = 2
    cfg.model.benchmark_protocol.arc_decoder_width = 32
    _, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    stages = objects["model"].model.pipeline.stages
    rates = {
        id(p): group["lr"]
        for group in objects["trainer"].optimizers[0].param_groups
        for p in group["params"]
    }
    assert all(rates[id(p)] == 5e-5 for p in stages[3].policy.model.parameters())
    assert all(rates[id(p)] == 1e-5 for p in stages[0].encoder.parameters())
    checkpoint = tmp_path / "training/checkpoints/last.ckpt"
    cfg.ckpt_path = str(checkpoint)
    cfg.trainer.max_epochs = 2
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step == 4
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["ema_num_updates"] == 4
    policy, protocol = load_policy(checkpoint, device="cpu")
    assert protocol["arc_decoder_variant"] == variant
    assert json.loads(json.dumps(protocol))["arc_timing_columns"] == [3, 10]
    values = objects["datamodule"].train_datasets["libero_panda"][0]
    values.pop("actions")
    batch = {
        key: value.unsqueeze(0)
        for key, value in values.items()
        if torch.is_tensor(value)
    }
    output = policy.algo.forward_eval({"libero_panda": batch})["libero_panda"][
        "pred_action"
    ]
    assert output.shape == (1, 8, 7) and torch.isfinite(output).all()


@pytest.mark.parametrize("mismatch", [None, "topology", "codec_method"])
def test_final_evaluation_checks_the_saved_decoder_experiment(mismatch):
    from egomimic.benchmarks.libero.evaluate import checkpoint_completion

    _, config = experiment("shape_masked")
    payload = {
        "global_step": 270054,
        "ema_num_updates": 270054,
        "ema_state_dict": {"parameter": torch.ones(1)},
        "normalizer_state": {},
        "benchmark_data_context": {"suite": "libero_spatial"},
        "hyper_parameters": {"config_tree": config},
        "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 5001}}}},
        "training_budget": {
            "epochs": 5001,
            "global_batch_size": 1024,
            "total_optimizer_steps": 270054,
        },
    }
    sha = "a" * 64
    request = {
        "source_run": "decoder-test",
        "source_commit": "b" * 40,
        "method": "arc_stk",
        "suite": "libero_spatial",
        "epochs": 5001,
        "total_optimizer_steps": 270054,
        "arc_decoder_variant": "shape_masked",
        "arc_profile": "stk_2",
        "checkpoint": {
            "sha256": sha,
            "bytes": 1,
            "uri": f"s3://rldb/experiments/arc-oat-20260919/decoder-test/checkpoints/{sha}/last.ckpt",
        },
    }
    if mismatch == "topology":
        request["arc_decoder_variant"] = "shared"
    elif mismatch == "codec_method":
        request["method"] = "arc_dur"
    if mismatch:
        with pytest.raises(ValueError):
            checkpoint_completion(payload, request)
    else:
        assert checkpoint_completion(payload, request)["global_step"] == 270054


@pytest.mark.parametrize(
    "mismatch", [None, "initial_state", "representation", "data_context"]
)
def test_full_result_comparison_requires_identical_inputs(tmp_path, mismatch):
    from dataclasses import asdict

    from egomimic.benchmarks.libero.catalog import LIBERO_COMMIT, OAT_COMMIT
    from egomimic.benchmarks.libero.rollout import rollout_plan

    protocol = {
        "suite": "libero_spatial",
        "method": "arc",
        "max_episode_steps": 550,
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "use_ema": True,
        "oat_commit": OAT_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "data_context": {"id": "fixed"},
        "observations_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "representation": {"mode": "stk", "waypoints": 32},
        "plan": [asdict(spec) for spec in rollout_plan("libero_spatial")],
    }
    rows = [
        dict(
            spec,
            success=False,
            steps=1,
            inference_seconds=[0.1],
            initial_state_sha256="c" * 64,
        )
        for spec in protocol["plan"]
    ]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for directory in (reference, candidate):
        directory.mkdir()
    (reference / "protocol.json").write_text(json.dumps(protocol))
    (reference / "episodes.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    protocol.update(arc_decoder_variant="shape_masked", checkpoint_sha256="d" * 64)
    rows[0]["success"] = True
    if mismatch == "initial_state":
        rows[0]["initial_state_sha256"] = "e" * 64
    elif mismatch == "representation":
        protocol["representation"]["waypoints"] = 36
    elif mismatch == "data_context":
        protocol["data_context"] = {"id": "different"}
    (candidate / "protocol.json").write_text(json.dumps(protocol))
    (candidate / "episodes.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    if mismatch:
        with pytest.raises(ValueError):
            compare_decoder_runs(candidate, reference)
    else:
        result = compare_decoder_runs(candidate, reference)
        assert result["paired_episodes"] == 2500
        assert result["initial_state_mismatches"] == 0
        assert result["success_delta_percentage_points"] == pytest.approx(0.04)
