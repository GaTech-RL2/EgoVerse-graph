import torch

from egomimic.pipeline.stages_matched_dp import SharedNativeDPStage


def test_shared_native_dp_both_domains_train_infer_and_gradients():
    torch.set_num_threads(1)
    stage = SharedNativeDPStage(
        down_dims=(8, 16, 32),
        step_embed_dim=8,
        n_groups=4,
        inference_steps=2,
        condition_dropout_probability=0,
    )
    robot = stage.branches["yam_bimanual"].stages[1].policy.model
    human = stage.branches["human_bimanual"].stages[1].policy.model
    assert robot.mid_modules is human.mid_modules
    assert robot.down_modules[1] is human.down_modules[1]
    assert robot.down_modules[0][0] is not human.down_modules[0][0]
    assert robot.final_conv[1] is not human.final_conv[1]
    for eid, width in ((7, 14), (3, 138)):
        stage.zero_grad(set_to_none=True)
        condition = torch.randn(1, 256, requires_grad=True)
        out = stage.execute(
            {
                "embodiment": torch.tensor([eid]),
                "condition": condition,
                "target": torch.randn(1, 100, width),
            },
            mode="train",
        )
        loss = out["loss/diffusion_noise"]
        assert torch.isfinite(loss)
        loss.backward()
        assert condition.grad.abs().sum() > 0
        assert (
            sum(
                p.grad.abs().sum()
                for p in robot.mid_modules.parameters()
                if p.grad is not None
            )
            > 0
        )
        with torch.no_grad():
            pred = stage.execute(
                {"embodiment": torch.tensor([eid]), "condition": condition.detach()},
                mode="inference",
            )["pred_action"]
        assert pred.shape == (1, 100, width) and torch.isfinite(pred).all()


def test_strict_dp_checkpoint_preserves_shared_core():
    args = dict(down_dims=(8, 16, 32), step_embed_dim=8, n_groups=4, inference_steps=2)
    old = SharedNativeDPStage(**args)
    new = SharedNativeDPStage(**args)
    new.load_state_dict(old.state_dict(), strict=True)
    a = new.branches["yam_bimanual"].stages[1].policy.model
    b = new.branches["human_bimanual"].stages[1].policy.model
    assert a.mid_modules is b.mid_modules


def test_unaugmented_window_guard_preserves_values_and_complete_anchors():
    import numpy as np

    from egomimic.rldb.zarr.episode_split import complete_window_count
    from egomimic.rldb.zarr.physical_retiming import CompleteNativeWindow

    for horizon in (30, 100):
        guard = CompleteNativeWindow(horizon, ["action"])
        guard.bind_episode({}, {"action": {"horizon": horizon}})
        action = np.arange(horizon)[:, None]
        batch = {"action": action, "_retiming_view": 0}
        result = guard.transform(batch)
        assert result["action"] is action
        assert "retiming_rate" not in result and "requested_speed" not in result
        assert guard.sample_views == 1
        assert complete_window_count(200, guard.required_frames) == 201 - horizon
