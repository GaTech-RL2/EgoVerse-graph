"""Latent epsilon/DDIM contract, sampling identity and coupled gradient regressions."""
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from egomimic.pipeline.stages_action_flow import (
    LatentDiffusionSchedule, LatentDiffusionBridgeStage, ConditionalEpsilonDDIMStage,
)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.calls = 0

    def forward(self, state, time, condition, condition_drop_mask):
        self.calls += 1
        return self.weight * state + condition[:, :1].reshape(-1, 1, 1)


def batch():
    return {
        'action_flow/clean_latent': torch.randn(2, 3, 4, requires_grad=True),
        'sampler/noise': torch.randn(2, 3, 4),
        'condition': torch.randn(2, 5),
    }


def bridge():
    return LatentDiffusionBridgeStage(samples_per_content=2, independent_noise_per_sample=True, condition_dropout_probability=0.1)


def test_corruption_matches_exact_dp_scheduler():
    schedule = LatentDiffusionSchedule()
    dp = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', clip_sample=True, set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    assert torch.equal(schedule.scheduler.alphas_cumprod, dp.alphas_cumprod)
    t = torch.arange(1, 101).float() / 100
    a, s, _, _ = schedule.coefficients(t, 2)
    torch.testing.assert_close(a[:, 0], dp.alphas_cumprod.sqrt(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(s[:, 0], (1 - dp.alphas_cumprod).sqrt(), atol=1e-6, rtol=1e-5)
    assert schedule.scheduler.config.clip_sample is False


def test_noise_is_independent_and_corruption_attached():
    b = batch(); stage = bridge(); stage(b)
    clean = b['action_flow/clean_latent'].index_select(0, b['action_flow/base_index'])
    noise = b['action_flow/diffusion_target_epsilon']
    a, s, da, ds = stage.diffusion.coefficients(b['action_flow/time'], 3)
    torch.testing.assert_close(b['action_flow/state'], a * clean + s * noise)
    torch.testing.assert_close(b['action_flow/target_velocity'], da * clean + ds * noise)
    assert not noise.requires_grad and b['action_flow/state'].requires_grad
    assert not torch.equal(noise[0], noise[1])


def test_epsilon_detaches_only_regression_clean_route():
    torch.manual_seed(7)
    b = batch(); bridge()(b); field = Field()
    stage = ConditionalEpsilonDDIMStage(field=field, flow_clean_gradient_mode='all_stopgrad', cfg_scale=1, num_inference_steps=10)
    stage.execute(b, mode='train')
    loss = b['action_flow/fm_velocity_residual'].square().mean()
    grad = torch.autograd.grad(loss, (field.weight, b['action_flow/clean_latent']), allow_unused=True, retain_graph=True)
    assert grad[0].isfinite() and grad[0].abs() > 0 and grad[1] is None
    # A nonlinear decoder's JVP must train both codec and shared denoiser.
    decoder_scale = torch.tensor(0.7, requires_grad=True)
    state, residual = b['action_flow/state'], b['action_flow/velocity_residual']
    jvp_residual = 2 * decoder_scale * state * residual
    gradients = torch.autograd.grad(jvp_residual.square().mean(), (field.weight, b['action_flow/clean_latent'], decoder_scale))
    assert all(g.isfinite().all() and g.abs().sum() > 0 for g in gradients)


def test_ddim10_cfg1_uses_ten_conditional_calls_and_native_steps():
    field = Field(); stage = ConditionalEpsilonDDIMStage(field=field, cfg_scale=1, num_inference_steps=10)
    b = batch(); stage.execute(b, mode='inference')
    dp = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2', clip_sample=False)
    dp.set_timesteps(10)
    state = b['sampler/noise'].float(); condition = b['condition']
    for i in dp.timesteps:
        epsilon = field.weight * state + condition[:, :1].reshape(-1, 1, 1)
        state = dp.step(epsilon, i, state, eta=0).prev_sample
    assert field.calls == 10
    torch.testing.assert_close(b['action_flow/generated_latent'], state)
    assert torch.equal(stage.diffusion.scheduler.timesteps, dp.timesteps)
    assert b['action_flow/trajectory'].shape[0] == 11


def test_oracle_epsilon_ddim_recovers_clean_latent():
    clean = torch.randn(2, 3, 4)
    schedule = LatentDiffusionSchedule()
    class Oracle(nn.Module):
        def forward(self, state, time, condition, condition_drop_mask):
            a, s, _, _ = schedule.coefficients(time, state.ndim)
            return (state - a * clean) / s
    stage = ConditionalEpsilonDDIMStage(field=Oracle(), cfg_scale=1, num_inference_steps=10)
    b = {'sampler/noise': torch.randn_like(clean), 'condition': torch.zeros(2, 5)}
    stage.execute(b, mode='inference')
    torch.testing.assert_close(b['action_flow/generated_latent'], clean, atol=2e-4, rtol=2e-4)
