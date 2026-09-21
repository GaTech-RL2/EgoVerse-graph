"""Causal replay, spatial windows, graph losses, and native execution contracts."""
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


def test_lightning_checkpoint_resume_preserves_optimizer_and_flow_rng(tmp_path):
    import lightning as L
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.trainRL import RLReceipts
    from egomimic.utils.experiment_artifacts import ArtifactWriter
    cfg = OmegaConf.create({"model": {"pipeline": {
        "_target_": "egomimic.pipeline.algo.PipelineAlgo", "device": "cpu", "stages": [{
            "_target_": "egomimic.pipeline.stages_q_chunking.QChunkingStage",
            "codec": {"_target_": "egomimic.rldb.action_codec.ControlChunkCodec",
                      "action_dim": 2, "native_horizon": 2},
            "observation_dim": 3, "goal_dim": 1, "action_dim": 2,
            "backup_horizon": 3, "hidden_dims": [16, 16]}]},
        "training_behavior": {"_target_": "egomimic.pl_utils.target_network_behavior.TargetNetworkBehavior"},
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 3e-4}},
        "steps": 4, "log_interval": 2, "checkpoint_interval": 2, "eval_interval": 0})
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
