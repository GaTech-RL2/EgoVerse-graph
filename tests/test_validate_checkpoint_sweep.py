from pathlib import Path

import pytest
import torch

from scripts.eval.validate_checkpoint_sweep import (
    Checkpoint,
    discover_checkpoints,
    validation_command,
)


def test_discovers_numbered_checkpoints_in_global_step_order(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "epoch=1-step=20000.ckpt").write_bytes(b"later")
    (checkpoints / "epoch=0-step=10000.ckpt").write_bytes(b"first")
    (checkpoints / "last.ckpt").write_bytes(b"alias")

    found = discover_checkpoints(tmp_path, "**/*.ckpt")

    assert [checkpoint.step for checkpoint in found] == [10_000, 20_000]
    assert [checkpoint.size_bytes for checkpoint in found] == [5, 5]


def test_rejects_ambiguous_duplicate_steps(tmp_path):
    for directory in (tmp_path / "a", tmp_path / "b"):
        directory.mkdir()
        (directory / "step=10000.ckpt").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="multiple checkpoints"):
        discover_checkpoints(tmp_path, "**/*.ckpt")


def test_reads_global_step_from_epoch_named_lightning_checkpoint(tmp_path):
    checkpoint = tmp_path / "epoch_epoch=99.ckpt"
    torch.save({"global_step": 10_000, "state_dict": {}}, checkpoint)

    found = discover_checkpoints(tmp_path, "*.ckpt")

    assert [item.step for item in found] == [10_000]


def test_builds_arc_eval_command_with_checkpoint_step_and_shared_wandb_id(tmp_path):
    checkpoint_path = tmp_path / "epoch=2-step=30000.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = Checkpoint(
        path=str(checkpoint_path.resolve()), step=30_000, size_bytes=10
    )

    command = validation_command(
        python="python",
        experiment="abc_arc/stationery_rl2_hpt_arc_D40_M100_openloop",
        checkpoint=checkpoint,
        output_dir=tmp_path / "results",
        action_mode="arc",
        execute_fraction=0.25,
        limit_val_episodes=4,
        wandb_run_id="arc-offline-val",
        wandb_name="arc offline checkpoint validation",
        wandb_group="stationery",
        extra_overrides=["trainer.precision=32-true"],
    )

    assert command[:3] == [
        "python",
        "egomimic/trainHydra.py",
        "+experiment=abc_arc/stationery_rl2_hpt_arc_D40_M100_openloop",
    ]
    assert "mode=eval" in command
    assert "eval_logger_enabled=true" in command
    assert "trainer.limit_val_batches=1.0" in command
    assert "evaluator.action_mode=arc" in command
    assert "evaluator.execute_fraction=0.25" in command
    assert "evaluator.log_step=30000" in command
    assert "evaluator.limit_val_episodes=4" in command
    assert any(item.startswith('ckpt_path="') for item in command)
    assert 'logger.wandb.id="arc-offline-val"' in command
    assert '+logger.wandb.name="arc offline checkpoint validation"' in command
    assert '+logger.wandb.resume="allow"' in command
    assert command[-1] == "trainer.precision=32-true"


def test_builds_baseline_eval_command_with_explicit_action_mode(tmp_path):
    checkpoint_path = tmp_path / "epoch=2-step=30000.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = Checkpoint(
        path=str(checkpoint_path.resolve()), step=30_000, size_bytes=10
    )

    command = validation_command(
        python="python",
        experiment="abc_arc/stationery_rl2_hpt_baseline_openloop",
        checkpoint=checkpoint,
        output_dir=tmp_path / "results",
        action_mode="baseline",
        execute_fraction=0.30,
        limit_val_episodes=4,
        wandb_run_id="baseline-offline-val",
        wandb_name="baseline-offline-val",
        wandb_group="offline-checkpoint-validation",
        extra_overrides=[],
    )

    assert "evaluator.action_mode=baseline" in command
    assert "evaluator.execute_fraction=0.3" in command
    assert "trainer.limit_val_batches=1.0" in command
    assert "+experiment=abc_arc/stationery_rl2_hpt_baseline_openloop" in command


def test_builds_video_only_command_without_metric_output(tmp_path):
    checkpoint_path = tmp_path / "epoch=2-step=30000.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint = Checkpoint(
        path=str(checkpoint_path.resolve()), step=30_000, size_bytes=10
    )

    command = validation_command(
        python="python",
        experiment="abc_arc/stationery_rl2_hpt_arc_D40_M100_openloop",
        checkpoint=checkpoint,
        output_dir=tmp_path / "output",
        action_mode="arc",
        execute_fraction=0.30,
        limit_val_episodes=4,
        wandb_run_id="arc-video-only",
        wandb_name="arc-video-only",
        wandb_group="offline-checkpoint-video",
        extra_overrides=[],
        video_only=True,
    )

    assert "evaluator.video_only=true" in command
    assert "evaluator.results_path=null" in command
    assert 'logger.wandb.job_type="offline_checkpoint_video"' in command
