"""Released diffusion-source parity and ARC graph training/checkpoint coverage."""

from pathlib import Path

import pytest
import torch
from diffusers import DDIMScheduler

from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.models.oat.diffusion import GraphDiffusionTransformer
from tests.test_oat_native import reference as reference  # noqa: F401


def dimensions(channels=12, horizon=32):
    return dict(
        input_dim=channels,
        output_dim=channels,
        horizon=horizon,
        n_obs_steps=2,
        cond_dim=138,
        n_layer=4,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.1,
        p_drop_attn=0.1,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=0,
    )


def scheduler():
    return DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )


@pytest.mark.parametrize("horizon", [24, 32, 36])
def test_released_backbone_noise_loss_gradients_and_sampling(reference, horizon):
    original_class = reference(
        "model.diffusion.transformer_for_diffusion"
    ).TransformerForDiffusion
    native = GraphDiffusionTransformer(**dimensions(horizon=horizon))
    original = original_class(**dimensions(horizon=horizon))
    original.load_state_dict(native.state_dict(), strict=True)
    condition = torch.randn(2, 2, 138)
    actions, noise = torch.randn(2, horizon, 12), torch.randn(2, horizon, 12)
    timesteps = torch.tensor([13, 87])
    noiser = scheduler()
    noisy = noiser.add_noise(actions, noise, timesteps)
    torch.manual_seed(98)
    predicted = native(noisy, timesteps, condition.flatten(1))
    torch.manual_seed(98)
    expected = original(noisy, timesteps, condition)
    torch.testing.assert_close(predicted, expected, rtol=0, atol=0)
    ((predicted - noise) ** 2).mean().backward()
    ((expected - noise) ** 2).mean().backward()
    for (name, actual), (other_name, wanted) in zip(
        native.named_parameters(), original.named_parameters()
    ):
        assert name == other_name
        if wanted.grad is not None:
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=0, atol=0)
    native.eval()
    original.eval()
    native_policy = DiffusionPolicy(native, scheduler(), horizon, 10)
    upstream_scheduler = scheduler()
    upstream_scheduler.set_timesteps(10)
    with torch.no_grad():
        sampled = native_policy.inference(noise.clone(), condition.flatten(1))
        expected = noise.clone()
        for timestep in upstream_scheduler.timesteps:
            expected = upstream_scheduler.step(
                original(expected, timestep, condition), timestep, expected
            ).prev_sample
    torch.testing.assert_close(sampled, expected, rtol=0, atol=0)


def test_released_config_and_parameter_counts(reference):
    import os

    import yaml
    from hydra import compose, initialize_config_dir

    upstream = yaml.safe_load(
        (
            Path(os.environ["OAT_REFERENCE_ROOT"]) / "oat/config/train_diffpolicy.yaml"
        ).read_text()
    )
    original = reference(
        "model.diffusion.transformer_for_diffusion"
    ).TransformerForDiffusion(**dimensions(channels=7))
    assert sum(p.numel() for p in original.parameters()) == 4_788_231
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "egomimic/hydra_configs"),
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=["+experiment=oat/libero_arc_oat_dp_policy"],
        )
    denoiser = cfg.model.pipeline.stages[3].policy
    model = denoiser.model
    for native, upstream_key in (
        ("n_emb", "embed_dim"),
        ("n_layer", "n_layers"),
        ("n_head", "n_heads"),
        ("p_drop_emb", "dropout"),
        ("p_drop_attn", "dropout"),
    ):
        assert model[native] == upstream["policy"][upstream_key]
    for key, value in upstream["policy"]["noise_scheduler"].items():
        assert denoiser.noise_scheduler[key] == value
    assert denoiser.num_inference_steps == upstream["policy"]["num_inference_steps"]
    assert cfg.trainer.max_epochs == upstream["training"]["num_epochs"]
    for key, upstream_key in (
        ("learning_rate", "policy_lr"),
        ("obs_enc_lr", "obs_enc_lr"),
        ("weight_decay", "weight_decay"),
        ("betas", "betas"),
    ):
        assert cfg.model.training_behavior[key] == upstream["optimizer"][upstream_key]
    for horizon, expected in ((24, 4_788_748), (32, 4_790_796), (36, 4_791_820)):
        adapted = GraphDiffusionTransformer(**dimensions(horizon=horizon))
        assert sum(p.numel() for p in adapted.parameters()) == expected


@pytest.mark.parametrize("mode,precision", [("stk", "32-true"), ("dur", "bf16-mixed")])
def test_arc_dp_train_optimizer_ema_reload_and_resume(tmp_path, mode, precision):
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.trainHydra import train
    from tests.test_libero_benchmark import make_replay
    from tests.test_oat_training import config_for

    path = tmp_path / "replay.zarr"
    make_replay(path)
    cfg = config_for("libero_arc_oat_dp_policy", path, tmp_path / "training")
    cfg.trainer.precision = precision
    cfg.benchmark.arc_mode = mode
    cfg.benchmark.arc_action_dim = 12
    cfg.benchmark.arc_waypoints = 8
    # Real graph/checkpoint path with a smaller trunk; full-size parity is above.
    model = cfg.model.pipeline.stages[3].policy.model
    model.n_emb, model.n_layer = 32, 1
    _, objects = train(cfg)
    trainer = objects["trainer"]
    assert trainer.global_step == 2
    stages = objects["model"].model.pipeline.stages
    optimizer = trainer.optimizers[0]
    learning_rates = {
        id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]
    }
    assert all(
        learning_rates[id(p)] == 5e-5 for p in stages[3].policy.model.parameters()
    )
    assert all(learning_rates[id(p)] == 1e-5 for p in stages[0].encoder.parameters())
    checkpoint = tmp_path / "training/checkpoints/last.ckpt"
    cfg.ckpt_path = str(checkpoint)
    cfg.trainer.max_epochs = 2
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step == 4
    policy, protocol = load_policy(checkpoint, device="cpu")
    assert protocol["arc_backbone"] == "oat_dp"
    values = objects["datamodule"].train_datasets["libero_panda"][0]
    values.pop("actions")
    batch = {
        key: value.unsqueeze(0)
        for key, value in values.items()
        if torch.is_tensor(value)
    }
    output = policy.algo.forward_eval({"libero_panda": batch})
    action = output["libero_panda"]["pred_action"]
    assert action.shape == (1, 8, 7) and torch.isfinite(action).all()


def test_training_launcher_selects_backbone_and_keeps_legacy_default():
    from egomimic.benchmarks.libero.cluster import training_arguments

    args = ("arc_stk", "libero_spatial", "data.zarr", "out", "full", 5001)
    matched = training_arguments(*args, gpus=4, arc_backbone="oat_dp")
    assert "+experiment=oat/libero_arc_oat_dp_policy" in matched
    assert "trainer.devices=4" in matched
    assert "benchmark.batch_size=256" in matched
    assert "trainer.accumulate_grad_batches=1" in matched
    assert "+experiment=oat/libero_arc_stk_policy" in training_arguments(*args)
    with pytest.raises(ValueError, match="Unknown ARC backbone"):
        training_arguments(*args, arc_backbone="invalid")
