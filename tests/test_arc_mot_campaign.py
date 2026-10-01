"""MoT graph, matched protocol, launch, and checkpoint coverage."""

import copy
import json

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.benchmarks.libero.arc_decoders import (
    MOT_VARIANTS,
    validate_decoder_config,
)
from scripts.benchmarks.launch_libero_osmo import arc_decoder_workflow
from tests.test_arc_decoder_campaign import experiment


@pytest.mark.parametrize("variant", MOT_VARIANTS)
def test_mot_changes_only_action_model_and_declared_architecture(variant):
    cfg, tree = experiment(variant)
    original, old = experiment("shared")
    validate_decoder_config(
        tree, suite="libero_spatial", variant=variant, profile="stk_2", arc_mode="stk"
    )
    model = copy.deepcopy(tree["model"])
    model["pipeline"]["stages"][3]["policy"]["model"] = old["model"]["pipeline"][
        "stages"
    ][3]["policy"]["model"]
    model["benchmark_protocol"] = old["model"]["benchmark_protocol"]
    assert model == old["model"]
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


@pytest.mark.parametrize("variant", MOT_VARIANTS)
def test_mot_full_workflow_has_gpu_smoke_gate_and_dependent_paired_evaluation(variant):
    spec = arc_decoder_workflow(
        "a" * 40,
        "mot-test",
        "libero_spatial",
        variant=variant,
        profile="stk_2",
        arc_mode="stk",
        replay_run="verified-replay",
        reference_run="arc-dpr-20260925-stk2-spatial",
    )["workflow"]
    train, evaluate = spec["tasks"]
    assert train["environment"]["ARC_DECODER_VARIANT"] == variant
    assert train["environment"]["EPOCHS"] == "5001"
    assert (
        train["environment"]["ARC_DECODER_REFERENCE_RUN"]
        == "arc-dpr-20260925-stk2-spatial"
    )
    assert evaluate["inputs"] == [{"task": "train"}]
    assert evaluate["environment"]["EVALUATION_WORKERS"] == "5"
    assert spec["resources"]["default"]["gpu"] == 8
    assert spec["resources"]["evaluation"]["gpu"] == 1
    assert all(
        resource["platform"] == "ovx-l40s" for resource in spec["resources"].values()
    )


@pytest.mark.parametrize("bad", ["columns", "mask", "width", "feedforward", "variant"])
def test_checkpoint_architecture_cannot_silently_change(bad):
    _, tree = experiment("mot_shape_velocity_masked")
    network = tree["model"]["pipeline"]["stages"][3]["policy"]["model"]
    if bad == "columns":
        network["modality_columns"]["shape"][0] = 3
    elif bad == "mask":
        network["blocked_attention"] = []
    elif bad == "width":
        network["n_emb"] = 256
    elif bad == "feedforward":
        network["dim_feedforward"] = 768
    else:
        tree["model"]["benchmark_protocol"]["arc_decoder_variant"] = (
            "mot_shape_velocity"
        )
    with pytest.raises(ValueError):
        validate_decoder_config(
            tree, suite="libero_spatial", variant="mot_shape_velocity_masked"
        )


@pytest.fixture
def training_tmp_path(tmp_path):
    yield tmp_path
    # Each real optimizer/EMA checkpoint includes the 22M observation encoder.
    # Release only this fixture's generated artifacts between parameterizations.
    for checkpoint in tmp_path.rglob("*.ckpt"):
        checkpoint.unlink()


@pytest.mark.parametrize("variant", MOT_VARIANTS)
def test_shared_training_ema_resume_and_physical_action_decoding(
    training_tmp_path, variant
):
    from egomimic.benchmarks.libero.arc_decoders import mot_architecture
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import train
    from tests.test_libero_benchmark import make_replay
    from tests.test_oat_training import config_for

    tmp_path = training_tmp_path
    dataset = tmp_path / "replay.zarr"
    make_replay(dataset)
    cfg = config_for("libero_arc_mot_policy", dataset, tmp_path / "training")
    cfg.arc_decoder = OmegaConf.create(mot_architecture(variant))
    cfg.arc_decoder.width, cfg.arc_decoder.layers, cfg.arc_decoder.feedforward = (
        32,
        2,
        80,
    )
    cfg.benchmark.arc_waypoints = 8
    cfg.trainer.precision = "bf16-mixed" if variant.endswith("masked") else "32-true"
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
    cfg.ckpt_path, cfg.trainer.max_epochs = str(checkpoint), 2
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step == 4
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["ema_num_updates"] == 4
    policy, protocol = load_policy(checkpoint, device="cpu")
    assert protocol["arc_decoder_variant"] == variant
    assert (
        json.loads(json.dumps(protocol))["arc_mot_modalities"]
        == mot_architecture(variant)["modalities"]
    )
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
