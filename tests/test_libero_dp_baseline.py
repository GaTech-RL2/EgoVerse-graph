"""Raw-action controls exercise shared training, EMA, resume and rollout identity."""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from egomimic.benchmarks.libero.baseline import (
    validate_raw_dp_config,
    verify_checkpoint,
)
from egomimic.benchmarks.libero.cli import policy_method
from scripts.benchmarks.launch_libero_osmo import baseline_workflow
from tests.test_libero_benchmark import make_replay
from tests.test_oat_training import config_for


@pytest.mark.parametrize("backbone,method", [("unet", "dp_unet"), ("oat_dp", "dp_oat")])
@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_raw_dp_train_resume_reload_and_infer(tmp_path, backbone, method, precision):
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import train

    path = tmp_path / "replay.zarr"
    make_replay(path)
    cfg = config_for(f"libero_dp_{backbone}_policy", path, tmp_path / "training")
    cfg.benchmark.suite = "libero_10"
    cfg.benchmark.horizon, cfg.benchmark.n_action_steps = 32, 16
    cfg.trainer.precision = precision
    model = cfg.model.pipeline.stages[3].policy.model
    if backbone == "unet":
        model.down_dims, model.diffusion_step_embed_dim = [16, 32], 32
    else:
        model.n_emb, model.n_layer = 32, 1
    validate_raw_dp_config(OmegaConf.to_container(cfg, resolve=True), method)
    _, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    checkpoint = tmp_path / "training/checkpoints/last.ckpt"
    proof = verify_checkpoint(
        checkpoint, suite="libero_10", method=method, mode="smoke"
    )
    assert proof["ema_num_updates"] == proof["global_step"] == 2
    stages = objects["model"].model.pipeline.stages
    learning_rates = {
        id(p): group["lr"]
        for group in objects["trainer"].optimizers[0].param_groups
        for p in group["params"]
    }
    assert all(
        learning_rates[id(p)] == 5e-5 for p in stages[3].policy.model.parameters()
    )
    assert all(
        learning_rates[id(p)] == 1e-5
        for p in stages[0].encoder.parameters()
        if p.requires_grad
    )
    cfg.ckpt_path, cfg.trainer.max_epochs = str(checkpoint), 2
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step == 4
    policy, protocol = load_policy(checkpoint, device="cpu")
    assert policy_method(policy.algo.pipeline.stages, protocol) == method
    batch = objects["datamodule"].train_datasets["libero_panda"][0]
    batch.pop("actions")
    values = {
        key: value.unsqueeze(0)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
    prediction = policy.algo.forward_eval({"libero_panda": values})["libero_panda"][
        "pred_action"
    ]
    assert prediction.shape == (1, 32, 7) and torch.isfinite(prediction).all()
    assert not any(
        "LiberoArc" in type(stage).__name__ or "Tokenizer" in type(stage).__name__
        for stage in policy.algo.pipeline.stages
    )


@pytest.mark.parametrize("backbone", ["unet", "oat_dp"])
def test_raw_dp_workflow_releases_training_gpus_before_evaluation(backbone):
    spec = baseline_workflow("a" * 40, "plain-dp", "libero_10", backbone=backbone)
    workflow = spec["workflow"]
    train, evaluate = workflow["tasks"]
    assert workflow["resources"]["default"]["gpu"] == 8
    assert workflow["resources"]["evaluation"]["gpu"] == 1
    assert train["environment"]["DP_BACKBONE"] == backbone
    assert train["environment"]["DP_OUTPUT"] == "{{output}}"
    assert evaluate["inputs"] == [{"task": "train"}]
    assert evaluate["environment"]["RUN_ID"] == "plain-dp-eval"
    assert evaluate["environment"]["EVALUATION_WORKERS"] == "5"
    assert "{{input:0}}/evaluation-request.json" in evaluate["files"][0]["contents"]
    assert all(r["platform"] == "ovx-l40s" for r in workflow["resources"].values())
    with pytest.raises(ValueError, match="5001"):
        baseline_workflow(
            "a" * 40, "plain-dp", "libero_10", backbone=backbone, epochs=100
        )


def test_raw_dp_config_rejects_arc_and_control_changes(tmp_path):
    cfg = config_for("libero_dp_unet_policy", "unused.zarr", tmp_path)
    cfg.benchmark.horizon, cfg.benchmark.n_action_steps = 32, 16
    value = OmegaConf.to_container(cfg, resolve=True)
    validate_raw_dp_config(value, "dp_unet")
    for field, wrong in [
        ("dp_backbone", "oat_dp"),
        ("horizon", 24),
        ("action_representation", "arc"),
    ]:
        invalid = copy.deepcopy(value)
        invalid["model"]["benchmark_protocol"][field] = wrong
        with pytest.raises(ValueError, match="raw-action DP"):
            validate_raw_dp_config(invalid, "dp_unet")
    value["model"]["pipeline"]["stages"][1]["_target_"] = (
        "egomimic.pipeline.stages_libero_arc.LiberoArcStage"
    )
    with pytest.raises(ValueError, match="raw-action DP"):
        validate_raw_dp_config(value, "dp_unet")
