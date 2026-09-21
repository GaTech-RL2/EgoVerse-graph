"""Causal replay, spatial windows, graph losses, and native execution contracts."""
import copy
import csv
import numpy as np
import pytest
import torch

from egomimic.models.q_chunking import DecoupledQChunking
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_q_chunking import QChunkingStage
from egomimic.rldb.action_codec import ControlChunkCodec
from egomimic.rldb.goal_replay import GoalReplay


def replay():
    data = {"observations": np.arange(36, dtype=np.float32).reshape(12, 3),
            "actions": np.arange(24, dtype=np.float32).reshape(12, 2) / 25,
            "terminals": np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]),
            "oracle_reps": np.arange(12, dtype=np.float32)[:, None]}
    return GoalReplay(data, backup_horizon=3, discount=.9)


def test_causal_target_goal_discount_and_episode_boundary():
    data = replay()
    batch = data.sample(3, np.random.RandomState(0), indices=[0, 1, 6], goal_indices=[0, 3, 9])
    # OGBench s[t] -> a[t], no robot observation-horizon offset.
    torch.testing.assert_close(batch["high_value_action_chunks"][0, 0], torch.tensor([0., .04]))
    torch.testing.assert_close(batch["high_value_backup_horizon"], torch.tensor([0., 2., 3.]))
    torch.testing.assert_close(batch["high_value_rewards"], torch.tensor([1., .81, 0.]))
    torch.testing.assert_close(batch["high_value_masks"], torch.tensor([0., 0., 1.]))
    with pytest.raises(ValueError, match="boundary"):
        data.sample(1, np.random.RandomState(0), indices=[3], goal_indices=[9])


@pytest.mark.parametrize("kind", ["native", "native_window", "arc"])
def test_constant_actions_and_holds_preserve_native_timing(kind):
    codec = ControlChunkCodec(2, 12, kind=kind, waypoints=4, distance=20)
    actions = torch.tensor([[[.25, -.5]] * 12, [[0., 0.]] * 12])
    tokens, length = codec.encode(actions)
    decoded, decoded_length = codec.decode(tokens)
    torch.testing.assert_close(decoded, actions, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(length, decoded_length)
    assert length.tolist() == [12, 12]


def test_distance_and_rotation_are_physical_not_action_count():
    codec = ControlChunkCodec(3, 12, kind="native_window", distance=.1,
        rotation=.3, translation_indices=[0, 1], rotation_indices=[2], action_scale=[.05, .05, .3])
    actions = torch.zeros(3, 12, 3)
    actions[0, :, 0] = .5  # .025m per native step -> four steps.
    actions[1, :, 2] = .5  # .15rad per native step -> two steps.
    assert codec.window_lengths(actions).tolist() == [4, 2, 12]
    tokens, lengths = codec.encode(actions)
    decoded, decoded_lengths = codec.decode(tokens)
    torch.testing.assert_close(lengths, decoded_lengths)
    for i, n in enumerate(lengths):
        torch.testing.assert_close(decoded[i, :n], actions[i, :n])


def test_arc_preserves_first_future_action_and_variable_duration():
    codec = ControlChunkCodec(2, 12, kind="arc", waypoints=4, distance=.15, action_scale=[.05, .05])
    actions = torch.linspace(0, .8, 24).view(1, 12, 2)
    encoded, length = codec.encode(actions)
    decoded, decoded_length = codec.decode(encoded)
    torch.testing.assert_close(decoded[:, 0], actions[:, 0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(length, decoded_length)
    assert 1 < int(length[0]) < 12


def test_torque_space_has_no_fake_rotation_budget():
    with pytest.raises(ValueError, match="not defined"):
        ControlChunkCodec(21, 5, rotation=.3, path_mode="control")


def test_graph_backward_target_lifecycle_and_inference():
    data = replay().sample(4, np.random.RandomState(42))
    codec = ControlChunkCodec(2, 2)
    stage = QChunkingStage(codec, observation_dim=3, goal_dim=1, action_dim=2,
                          backup_horizon=3, hidden_dims=(16, 16), best_of_n=3, flow_steps=2)
    pipeline = PipelineAlgo([stage], device="cpu")
    output = pipeline.forward_training({"domain": data})
    loss = pipeline.compute_losses(output, {"domain": data})["loss"]
    logs = {k: v for k, v in output["domain"].items() if k.startswith("log/")}
    assert "log/DQC/q_action/mean" in logs and "log/DQC/v/mean" in logs
    assert all(not value.requires_grad for value in logs.values())
    torch.testing.assert_close(loss, sum(v for k, v in output["domain"].items()
                                         if k.startswith("loss/")), rtol=0, atol=0)
    loss.backward()
    assert all(p.grad is None for p in stage.agent.target_action_critic.parameters())
    assert all(p.grad is not None for p in stage.agent.actor_bc.parameters())
    optimizer = torch.optim.Adam(pipeline.nets.parameters(), lr=3e-4)
    stage.agent.update_target_before_optimizer()
    optimizer.step()
    inference = pipeline.forward_eval({"domain": {
        "observations": data["observations"], "goals": data["high_value_goals"]}})["domain"]
    assert inference["pred_action"].shape == (4, 2, 2)
    assert inference["action_lengths"].tolist() == [2] * 4
    assert torch.isfinite(inference["pred_action"]).all()
    assert not any(key.startswith("log/") for key in inference)


@pytest.mark.parametrize("aggregation", ["mean", "min"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_prediction_statistics_and_entropy_include_probability_boundaries(aggregation, batch_size):
    model = DecoupledQChunking(3, 1, 2, 3, 4, hidden_dims=(8,), q_agg=aggregation)
    q = torch.tensor([[0., .2, .8], [.4, .6, 1.]], requires_grad=True)[:, :batch_size]
    v = torch.tensor([0., .5, 1.], requires_grad=True)[:batch_size]
    metrics = model.prediction_metrics(action_q=q, chunk_q=q, target_q=v,
                                       value=v, next_value=v, backup=v)
    expected_q = np.array([.2, .4, .9] if aggregation == "mean" else [0., .2, .8])[:batch_size]
    expected_v = np.array([0., .5, 1.])[:batch_size]
    for name in ["q_action", "q_chunk", "q_target", "v", "v_next", "bellman_target"]:
        values = expected_q if name in {"q_action", "q_chunk"} else expected_v
        for suffix, expected in [("mean", values.mean()), ("std", values.std()),
                                 ("min", values.min()), ("max", values.max())]:
            assert float(metrics[name + "/" + suffix]) == pytest.approx(expected, abs=1e-7)
    entropy = np.log(2) / 3 if batch_size == 3 else 0.
    assert float(metrics["bellman_target/entropy"]) == pytest.approx(entropy)
    for name in ["q_action", "q_chunk"]:
        expected = np.array([.2, .2, .1])[:batch_size].mean()
        assert float(metrics[name + "/ensemble_std"]) == pytest.approx(expected)
    assert all(value.ndim == 0 and torch.isfinite(value) and not value.requires_grad
               for value in metrics.values())


@pytest.mark.parametrize("kind", ["native", "native_window", "arc"])
@pytest.mark.parametrize("use_chunk_critic", [True, False])
def test_prediction_logging_preserves_losses_gradients_updates_and_rng(kind, use_chunk_critic):
    batch = replay().sample(4, np.random.RandomState(13))
    codec = ControlChunkCodec(2, 2, kind=kind, waypoints=2)
    actions, _ = codec.encode(batch["high_value_action_chunks"])
    original = DecoupledQChunking(3, 1, 2, 3, codec.encoded_dim,
                                 hidden_dims=(8, 8), use_chunk_critic=use_chunk_critic)
    instrumented = copy.deepcopy(original)
    before = torch.get_rng_state()

    def update(model, metrics):
        torch.set_rng_state(before)
        calls = []
        hooks = [module.register_forward_hook(lambda *args: calls.append(1))
                 for module in model.modules() if isinstance(module, torch.nn.Linear)]
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        losses = model.losses(batch, actions, metrics=metrics)
        sum(losses.values()).backward()
        model.update_target_before_optimizer()
        optimizer.step()
        for hook in hooks:
            hook.remove()
        return losses, torch.get_rng_state(), len(calls)

    expected, expected_rng, expected_calls = update(original, None)
    metrics = {}
    actual, actual_rng, actual_calls = update(instrumented, metrics)
    assert expected.keys() == actual.keys()
    assert torch.equal(actual_rng, expected_rng)
    assert actual_calls == expected_calls
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    for (name, old), (_, new) in zip(original.named_parameters(), instrumented.named_parameters()):
        torch.testing.assert_close(new, old, rtol=0, atol=0, msg=name)
        if old.grad is None:
            assert new.grad is None
        else:
            torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0, msg=name)
    assert "q_action/mean" in metrics and "v/mean" in metrics
    assert ("q_chunk/mean" in metrics) == use_chunk_critic
    assert all(not value.requires_grad and torch.isfinite(value) for value in metrics.values())


@pytest.fixture
def tiny_rl_config():
    from omegaconf import OmegaConf
    return OmegaConf.create({"model": {"pipeline": {
        "_target_": "egomimic.pipeline.algo.PipelineAlgo", "device": "cpu", "stages": [{
            "_target_": "egomimic.pipeline.stages_q_chunking.QChunkingStage",
            "codec": {"_target_": "egomimic.rldb.action_codec.ControlChunkCodec",
                      "action_dim": 2, "native_horizon": 2},
            "observation_dim": 3, "goal_dim": 1, "action_dim": 2,
            "backup_horizon": 3, "hidden_dims": [16, 16]}]},
        "training_behavior": {"_target_": "egomimic.pl_utils.target_network_behavior.TargetNetworkBehavior",
                              "prediction_log_interval": "${log_interval}"},
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 3e-4}},
        "steps": 4, "log_interval": 2, "checkpoint_interval": 2, "eval_interval": 0})


def test_lightning_checkpoint_resume_preserves_optimizer_and_flow_rng(tmp_path, tiny_rl_config):
    import lightning as L
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.trainRL import RLReceipts
    from egomimic.utils.experiment_artifacts import ArtifactWriter
    cfg = tiny_rl_config
    batch = {"ogbench": replay().sample(4, np.random.RandomState(12))}

    def fit(directory, steps, resume=None, start=0):
        torch.manual_seed(81)
        model = ModelWrapper(config_tree=cfg, enable_grad_norm=False, train_log_on_step=True)
        callback = RLReceipts(cfg, SimpleNamespace(start_step=start, receipts={}), ArtifactWriter(directory))
        trainer = L.Trainer(accelerator="cpu", devices=1, max_steps=steps, max_epochs=-1,
            logger=False, callbacks=[callback], enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, limit_val_batches=0)
        loader = DataLoader([batch] * (steps-start), batch_size=None,
                             generator=torch.Generator().manual_seed(33))
        trainer.fit(model, train_dataloaders=loader, ckpt_path=resume)
        return {key: value.clone() for key, value in model.state_dict().items()}

    expected = fit(tmp_path / "continuous", 4)
    fit(tmp_path / "first", 2)
    checkpoint = tmp_path / "first/checkpoints/step-000000002.ckpt"
    resumed = fit(tmp_path / "resumed", 4, str(checkpoint), 2)
    for key in expected:
        torch.testing.assert_close(expected[key], resumed[key], rtol=0, atol=0)


def test_prediction_metrics_reach_csv_at_interval_after_unaligned_resume(tmp_path, tiny_rl_config):
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.trainRL import RLReceipts
    from egomimic.utils.experiment_artifacts import ArtifactWriter

    cfg = tiny_rl_config
    cfg.checkpoint_interval = 3
    batch = {"ogbench": replay().sample(4, np.random.RandomState(12))}

    def fit(directory, steps, resume=None, start=0):
        cfg.steps = steps
        torch.manual_seed(81)
        model = ModelWrapper(config_tree=cfg, enable_grad_norm=False, train_log_on_step=True)
        callback = RLReceipts(cfg, SimpleNamespace(start_step=start, receipts={}), ArtifactWriter(directory))
        logger = CSVLogger(directory, name="csv")
        trainer = L.Trainer(accelerator="cpu", devices=1, max_steps=steps, max_epochs=-1,
            logger=logger, log_every_n_steps=cfg.log_interval, callbacks=[callback],
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            limit_val_batches=0)
        emissions = []
        stage = model.model.pipeline.stages[0]
        original = stage.agent.prediction_metrics

        def collect(**values):
            emissions.append(trainer.global_step + 1)
            return original(**values)

        stage.agent.prediction_metrics = collect
        trainer.fit(model, train_dataloaders=DataLoader([batch] * (steps-start), batch_size=None),
                    ckpt_path=resume)
        with open(logger.log_dir + "/metrics.csv") as handle:
            rows = list(csv.DictReader(handle))
        return rows, emissions

    first, first_steps = fit(tmp_path / "first", 3)
    checkpoint = tmp_path / "first/checkpoints/step-000000003.ckpt"
    resumed, resumed_steps = fit(tmp_path / "resumed", 5, str(checkpoint), 3)
    assert first_steps == [2] and resumed_steps == [4]
    for rows, expected_step in [(first, 1), (resumed, 3)]:
        prediction_rows = [row for row in rows if row.get("Train/DQC/q_action/mean")]
        assert [int(row["step"]) for row in prediction_rows] == [expected_step]
        row = prediction_rows[0]
        for key in ["Train/DQC/q_chunk/mean", "Train/DQC/q_target/std", "Train/DQC/v/mean",
                    "Train/DQC/v_next/mean", "Train/DQC/bellman_target/entropy"]:
            assert np.isfinite(float(row[key]))
        assert row["Train/DQC/v/mean"] == row["Train/DQC/v/mean/ogbench"]
    assert "log_predictions" not in batch["ogbench"]


@pytest.fixture
def checkpoint_writer(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from egomimic.trainRL import RLReceipts
    from egomimic.utils.experiment_artifacts import ArtifactWriter

    class Remote:
        def __init__(self):
            self.objects = {}

        def upload_file(self, filename, bucket, key, ExtraArgs):
            self.objects[key] = {"body": Path(filename).read_bytes(),
                                 "Metadata": dict(ExtraArgs["Metadata"])}

        def head_object(self, Bucket, Key):
            item = self.objects[Key]
            return {"ContentLength": len(item["body"]), "Metadata": item["Metadata"]}

    writer = ArtifactWriter(tmp_path)
    writer.bucket, writer.prefix, writer.client = "test", "owned-run", Remote()
    cfg = OmegaConf.create({"local_checkpoint_keep": 2})
    callback = RLReceipts(cfg, SimpleNamespace(start_step=0), writer)

    def save(step):
        callback.checkpoint(SimpleNamespace(global_step=step,
            save_checkpoint=lambda path: path.write_bytes(f"checkpoint {step}".encode())))

    return writer, save


@pytest.mark.parametrize("remote", [True, False])
def test_checkpoint_retention_preserves_durable_and_preexisting_files(checkpoint_writer, tmp_path, remote):
    writer, save = checkpoint_writer
    if not remote:
        writer.client = None
    preexisting = tmp_path / "checkpoints/step-000000000.ckpt"
    preexisting.parent.mkdir()
    preexisting.write_bytes(b"pre-existing work")
    for step in range(1, 5):
        save(step)
    assert preexisting.read_bytes() == b"pre-existing work"
    kept = {path.name for path in preexisting.parent.glob("*.ckpt")}
    assert kept == {f"step-{step:09d}.ckpt" for step in ([0, 3, 4] if remote else range(5))}
    if remote:
        assert all(f"owned-run/checkpoints/step-{step:09d}.ckpt" in writer.client.objects
                   for step in range(1, 5))


@pytest.mark.parametrize("damage", ["local", "remote_hash", "remote_size"])
def test_checkpoint_retention_refuses_unverified_copy(checkpoint_writer, tmp_path, damage):
    writer, save = checkpoint_writer
    save(1)
    save(2)
    previous = tmp_path / "checkpoints/step-000000001.ckpt"
    remote = writer.client.objects["owned-run/checkpoints/step-000000001.ckpt"]
    if damage == "local":
        previous.write_bytes(b"modified work")
    elif damage == "remote_hash":
        remote["Metadata"]["sha256"] = "wrong"
    else:
        remote["body"] = b"truncated"
    with pytest.raises(RuntimeError, match="verified durable copy"):
        save(3)
    assert previous.exists()
