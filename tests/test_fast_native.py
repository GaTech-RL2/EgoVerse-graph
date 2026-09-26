"""FAST source parity, variable-length training, and native checkpoint contracts."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.models.oat.policy.fastpolicy import FASTPolicy
from egomimic.models.oat.tokenizer.fast.processing_action_tokenizer import (
    UniversalActionProcessor,
)
from egomimic.models.oat.tokenizer.fast.tokenizer_wrapper import FASTTok
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_fast import FASTPolicyStage
from tests.test_oat_native import TinyObservation
from tests.test_oat_native import reference as reference


@pytest.fixture(scope="module")
def actions():
    rng = np.random.default_rng(42)
    values = rng.uniform(-1, 1, (48, 32, 7)).astype(np.float32)
    values[:4] = 0  # no movement, subquantization movement, and gripper changes
    values[1, :, 0] = np.linspace(0, 0.001, 32)
    values[2, :, 6] = np.where(np.arange(32) < 16, -1, 1)
    values[3] = 1
    return values


@pytest.fixture(scope="module")
def processor(actions):
    return UniversalActionProcessor.fit(list(actions), time_horizon=32, action_dim=7)


def make_small_policy(tokenizer):
    return FASTPolicy(
        {"action": {"shape": [7]}, "obs": {"state": {"shape": [3], "type": "state"}}},
        TinyObservation(),
        tokenizer,
        horizon=32,
        n_action_steps=16,
        n_obs_steps=2,
        embed_dim=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_seq_len=128,
        topk=1,
    )


def test_processor_matches_pinned_huggingface_source(actions, processor):
    path = os.environ.get("FAST_PROCESSOR_REFERENCE")
    if not path:
        pytest.skip(
            "Set FAST_PROCESSOR_REFERENCE to the pinned original processor source"
        )
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "egomimic/models/oat/tokenizer/fast/UPSTREAM.json"
        ).read_text()
    )
    assert (
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        == manifest["files"]["processing_action_tokenizer.py"]
    )
    spec = importlib.util.spec_from_file_location("fast_processor_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.UniversalActionProcessor.fit(
        list(actions), time_horizon=32, action_dim=7
    )
    assert processor(actions) == original(actions)
    np.testing.assert_array_equal(
        processor.decode(processor(actions)), original.decode(original(actions))
    )


def test_wrapper_matches_oat_encoding_and_decoding(
    reference, monkeypatch, processor, actions
):
    module = reference("tokenizer.fast.tokenizer_wrapper")
    monkeypatch.setattr(
        module.AutoProcessor, "from_pretrained", lambda *a, **kw: processor
    )
    upstream = module.FASTTok()
    native = FASTTok.from_processor(processor)
    upstream.set_normalizer(native.normalizer)
    batch = torch.from_numpy(actions)
    assert native.tokenize(batch) == upstream.tokenize(batch)
    torch.testing.assert_close(
        native.detokenize(native.tokenize(batch)),
        upstream.detokenize(upstream.tokenize(batch)),
        rtol=0,
        atol=0,
    )
    # The released decoder returns zero normalized actions for malformed strings.
    bad = [[], [native.vocab_size + 2], native.tokenize(batch)[0][:1]]
    torch.testing.assert_close(
        native.detokenize(bad, 32, 7), upstream.detokenize(bad, 32, 7), rtol=0, atol=0
    )


def test_policy_matches_upstream_loss_gradients_and_generation(
    reference, processor, actions
):
    native = make_small_policy(FASTTok.from_processor(processor))
    original = reference("policy.fastpolicy").FASTPolicy(
        {"action": {"shape": [7]}, "obs": {"state": {"shape": [3], "type": "state"}}},
        TinyObservation(),
        FASTTok.from_processor(processor),
        32,
        16,
        2,
        16,
        1,
        2,
        0.0,
        128,
        1.0,
        1,
    )
    original.load_state_dict(native.state_dict(), strict=True)
    batch = {
        "action": torch.from_numpy(actions[:3]),
        "obs": {"state": torch.randn(3, 2, 3)},
    }
    for forced in (None, [[], [1, 2, 3], [1, 2] * 100]):
        for model in (native, original):
            model.zero_grad()
            if forced is not None:
                model.action_tokenizer.tokenize = lambda _samples: forced
        actual, expected = native(batch), original(batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        expected.backward()
        for p, q in zip(native.parameters(), original.parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    native.eval()
    original.eval()
    with torch.inference_mode():
        torch.manual_seed(44)
        a = native.predict_action(batch["obs"])
        torch.manual_seed(44)
        b = original.predict_action(batch["obs"])
    torch.testing.assert_close(a["action_pred"], b["action_pred"], rtol=0, atol=0)


def test_padding_is_ignored_and_eos_is_supervised(monkeypatch, processor):
    model = make_small_policy(FASTTok.from_processor(processor))
    monkeypatch.setattr(
        model.action_tokenizer, "tokenize", lambda _: [[], [7, 8], [9] * 150]
    )
    logits = torch.randn(3, 128, 1027, requires_grad=True)
    captured = []

    def forward(ids, **kwargs):
        captured.append(ids)
        return logits

    monkeypatch.setattr(model.model, "forward", forward)
    loss = model(
        {"action": torch.zeros(3, 32, 7), "obs": {"state": torch.zeros(3, 2, 3)}}
    )
    loss.backward()
    assert captured[0][0, :3].tolist() == [1024, 1025, 1026]
    assert captured[0][1, :5].tolist() == [1024, 7, 8, 1025, 1026]
    assert logits.grad[0, 0].abs().sum() > 0  # empty action still teaches EOS
    assert logits.grad[0, 1:].abs().sum() == 0
    assert logits.grad[1, 2].abs().sum() > 0
    assert logits.grad[1, 3:].abs().sum() == 0
    assert logits.grad[2, -1].abs().sum() > 0  # overflow ends at EOS


def test_graph_training_and_target_free_inference(processor, actions):
    model = make_small_policy(FASTTok.from_processor(processor))
    graph = PipelineAlgo([FASTPolicyStage(model)], device="cpu")
    batch = {
        "libero_panda": {
            "state": torch.randn(2, 2, 3),
            "actions": torch.from_numpy(actions[:2]),
        }
    }
    output = graph.forward_training(batch)
    graph.compute_losses(output, batch)["loss"].backward()
    assert model.obs_encoder.projection.weight.grad is not None
    assert all(p.grad is None for p in model.action_tokenizer.parameters())
    graph.nets.eval()
    pred = graph.forward_eval(
        {"libero_panda": {"state": batch["libero_panda"]["state"]}}
    )
    assert pred["libero_panda"]["pred_action"].shape == (2, 32, 7)


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_native_fit_train_resume_and_self_contained_reload(tmp_path, precision):
    from egomimic.benchmarks.libero.cli import policy_method
    from egomimic.benchmarks.libero.fast import fit_tokenizer, verify_checkpoint
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import train
    from tests.test_libero_benchmark import make_replay
    from tests.test_oat_training import config_for

    path = tmp_path / "replay.zarr"
    make_replay(path, suite="libero_10")
    report = fit_tokenizer(path, "libero_10", tmp_path / "fit", decoded_cache=False)
    assert (report["train_chunks"], report["validation_chunks"]) == (90, 10)
    assert report["invalid_reconstruction_fraction"] == 0
    cfg = config_for("libero_fastpolicy", path, tmp_path / "train")
    cfg.benchmark.suite = "libero_10"
    cfg.benchmark.horizon, cfg.benchmark.n_action_steps = 32, 16
    artifact = tmp_path / "fit/fast-tokenizer.pt"
    cfg.benchmark.tokenizer_checkpoint = str(artifact)
    cfg.trainer.precision = precision
    _, objects = train(cfg)
    checkpoint = tmp_path / "train/checkpoints/last.ckpt"
    assert (
        verify_checkpoint(checkpoint, suite="libero_10", mode="smoke")["global_step"]
        == 2
    )
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["fast_tokenizer_config"]
    stage = objects["model"].model.pipeline.stages[0]
    rates = {
        id(p): g["lr"]
        for g in objects["trainer"].optimizers[0].param_groups
        for p in g["params"]
    }
    assert all(
        rates[id(p)] == 5e-5 for p in stage.policy.model.parameters() if p.requires_grad
    )
    assert all(
        rates[id(p)] == 1e-5
        for p in stage.policy.obs_encoder.parameters()
        if p.requires_grad
    )
    wrong = copy.deepcopy(payload)
    wrong["fast_tokenizer_config"]["processor_config"]["scale"] = 11
    with pytest.raises(ValueError, match="different fitted BPE"):
        objects["model"].on_load_checkpoint(wrong)
    cfg.ckpt_path, cfg.trainer.max_epochs = str(checkpoint), 2
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step == 4
    artifact.unlink()
    policy, protocol = load_policy(checkpoint, device="cpu")
    assert policy_method(policy.algo.pipeline.stages, protocol) == "fast"
    sample = objects["datamodule"].train_datasets["libero_panda"][0]
    values = {
        k: v.unsqueeze(0)
        for k, v in sample.items()
        if torch.is_tensor(v) and k != "actions"
    }
    pred = policy.algo.forward_eval({"libero_panda": values})["libero_panda"][
        "pred_action"
    ]
    assert pred.shape == (1, 32, 7) and torch.isfinite(pred).all()
    with pytest.raises(ValueError, match="OAT prefix"):
        load_policy(checkpoint, device="cpu", use_k_tokens=4)


def test_fast_launch_preserves_budget_and_separates_evaluation():
    from egomimic.benchmarks.libero.cluster import training_arguments
    from scripts.benchmarks.launch_libero_osmo import fast_workflow

    spec = fast_workflow("a" * 40, "fast-release", "libero_spatial")
    workflow = spec["workflow"]
    train, evaluate = workflow["tasks"]
    assert train["environment"]["RUN_KIND"] == "fast"
    assert train["environment"]["EPOCHS"] == "5001"
    assert evaluate["inputs"] == [{"task": "train"}]
    assert evaluate["environment"]["EVALUATION_WORKERS"] == "5"
    assert workflow["resources"]["default"]["gpu"] == 4
    assert workflow["resources"]["evaluation"]["gpu"] == 1
    assert all(r["platform"] == "ovx-l40s" for r in workflow["resources"].values())
    args = training_arguments(
        "fast", "libero_spatial", "/data", "/evidence", "full", 5001, gpus=4
    )
    assert "benchmark.batch_size=256" in args
    assert "trainer.accumulate_grad_batches=1" in args
    assert "benchmark.tokenizer_checkpoint=/evidence/fast-tokenizer.pt" in args
