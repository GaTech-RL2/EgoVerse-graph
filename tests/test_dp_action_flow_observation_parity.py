"""Reject mismatched proprio, condition widths, masks and multiplier inputs."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision  # Import external registrations before metadata-only counts.
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.rldb.embodiment.pushshapes_retiming import (
    get_standard_dp_retimed_keymap, get_standard_dp_retimed_transform_list,
)
from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.stems.visual_core import VisualCore
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_sampler import FusedObsEncoder, DPStyleObsEncoder, KeyedFeatureProjection
from egomimic.pipeline.stages_speed import SharedSpeedCondition


def config(monkeypatch):
    for name in ('DP_U_DATASET_DIR', 'DP_DATASET_DIR', 'DP_SPLIT_MANIFEST',
                 'DP_SPLIT_SHA256', 'DP_CHAIN_SPLIT_MANIFEST', 'DP_CHAIN_SPLIT_SHA256'):
        monkeypatch.setenv(name, '/unused/' + name)
    root = Path(__file__).parents[1] / 'egomimic/hydra_configs'
    with initialize_config_dir(config_dir=str(root), version_base='1.3'):
        return compose(config_name='train_zarr_cartesian', overrides=[
            'hydra/launcher=basic', '+experiment=pusht/planar_uc_manual4919_dp_261m_af_obs_multiplier'])


def test_resolved_observation_capacity_and_training_contract(monkeypatch):
    c = config(monkeypatch)
    stages = c.model.pipeline.stages
    assert stages[0].input_key == 'state_agent_model'
    assert stages[0].output_dim == 64
    assert all(x.input_dim == 4 for x in stages[0].projections.values())
    assert stages[1].n_obs_steps == 1
    assert stages[1].encoder.obs_specs.proprio_condition.input_dim == 64
    visual = stages[1].encoder.img_encoders.front_img_1
    assert (visual.image_size, visual.crop_height, visual.crop_width) == (96, 84, 84)
    assert visual.pretrained is False and visual.norm_layer == 'group'
    assert visual.pool_type == 'spatial_softmax' and visual.feature_dimension == 64
    assert stages[2].conditioning_input == 'retiming_multiplier'
    assert stages[2].output_key == 'condition'
    denoiser = next(s for s in stages if s['_target_'].endswith('DiffusionDenoiserStage'))
    assert denoiser.condition_input_dim == denoiser.policy.model.cond_dim == 128
    assert denoiser.action_dim == 5 and denoiser.action_horizon == 16
    assert list(denoiser.policy.model.down_dims) == [648, 1296, 2592]
    assert c.planar.active_action_dims.pushshapes_sim_u_socket == 4
    assert c.planar.active_action_dims.pushshapes_sim_chain_gripper == 5
    assert c.norm_stats.norm_mode == "minmax"
    assert c.norm_stats.sample_frac == 1.0
    assert c.norm_stats.precomputed_norm_path is None
    assert c.planar.batch_size == 32
    for group in (c.data.train_datasets, c.data.valid_datasets):
        for item in group.values():
            assert item.resolver.key_map.model_proprio
            assert item.resolver.transform_list.model_proprio


@pytest.mark.parametrize('width', (3, 4))
def test_train_data_preserves_native_metadata_and_common5_targets(width):
    keymap = get_standard_dp_retimed_keymap((1., 2.), model_proprio=True)
    assert keymap['state_agent_obj']['key_type'] == 'metadata_keys'
    assert keymap['state_agent_model']['key_type'] == 'proprio_keys'
    state = np.array([10., 20., np.pi, 30., 40., .7])
    batch = {'state_agent_obj': state.copy(), 'state_agent_model': state.copy(),
             'actions': np.zeros((31, width)), '_retiming_view': 1}
    for t in get_standard_dp_retimed_transform_list((1., 2.), model_proprio=True):
        batch = t.transform(batch)
    np.testing.assert_allclose(batch['state_agent_model'], [10., 20., -1., 0.], atol=1e-6)
    np.testing.assert_array_equal(batch['state_agent_obj'], state)
    assert batch['actions'].shape == (16, 5)
    assert batch['retiming_rate'].item() == 2.


@pytest.mark.parametrize('embodiment', (19, 20))
def test_observation_and_multiplier_gradients_in_both_graph_modes(monkeypatch, embodiment):
    c = config(monkeypatch)
    projection = instantiate(c.model.pipeline.stages[0])
    obs = instantiate(c.model.pipeline.stages[1])
    multiplier = SharedSpeedCondition(condition_dim=128, output_key='condition')
    torch.nn.init.constant_(multiplier.mlp[-1].weight, .1)
    graph = Pipeline([projection, obs, multiplier])
    for mode in ('train', 'inference'):
        state = torch.randn(2, 4, requires_grad=True)
        result = graph.execute({'state_agent_model': state, 'front_img_1': torch.randn(2, 3, 96, 96),
                      'embodiment': torch.tensor([embodiment, embodiment]),
                      'retiming_rate': torch.tensor([[1.], [2.]])}, mode=mode)
        assert result['condition'].shape == (2, 128)
        result['condition'].square().mean().backward()
        assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
        assert multiplier.mlp[0].weight.grad.abs().sum() > 0
        assert next(obs.parameters()).grad.abs().sum() > 0


def test_real_module_parameter_budget_without_weight_allocation(monkeypatch):
    c = config(monkeypatch)
    with torch.device('meta'):
        projection = instantiate(c.model.pipeline.stages[0])
        visual = instantiate(c.model.pipeline.stages[1].encoder.img_encoders.front_img_1)
        multiplier = instantiate(c.model.pipeline.stages[2])
        unet = ConditionalUnet1D(input_dim=5, cond_dim=128, ac_latent_seq=1,
                 diffusion_step_embed_dim=256, down_dims=[648, 1296, 2592], kernel_size=3)
    count = sum(p.numel() for m in (projection, visual, multiplier, unet) for p in m.parameters())
    assert count == 261382885


def test_real_small_dp_uses_128_condition_and_preserves_u_gripper_mask():
    from egomimic.pipeline.stages_diffusion import DiffusionEpsilonLossStage
    from egomimic.models.diffusion_policy import DiffusionPolicy
    from egomimic.models.ddim_scheduler import DDIMScheduler
    multiplier = SharedSpeedCondition(condition_dim=128, output_key='condition')
    condition = multiplier({'condition': torch.randn(2, 128),
                            'retiming_rate': torch.tensor([[1.], [2.]])})['condition']
    unet = ConditionalUnet1D(input_dim=5, cond_dim=128, ac_latent_seq=1,
                     diffusion_step_embed_dim=16, down_dims=[16, 32, 64], kernel_size=3)
    prediction = unet(torch.randn(2, 16, 5), torch.tensor([1, 50]), condition)
    prediction.retain_grad()
    loss_stage = DiffusionEpsilonLossStage(active_action_dims_by_embodiment={
        'pushshapes_sim_u_socket': 4, 'pushshapes_sim_chain_gripper': 5})
    loss = loss_stage({'diffusion/predicted_noise': prediction,
                      'diffusion/noise_target': torch.randn_like(prediction),
                      'target': torch.zeros_like(prediction),
                      'embodiment': torch.tensor([19, 19])})['loss/diffusion_noise']
    loss.backward()
    assert torch.isfinite(loss) and torch.count_nonzero(prediction.grad[..., 4]) == 0
    assert multiplier.mlp[-1].weight.grad.abs().sum() > 0
    policy = DiffusionPolicy(model=unet, action_horizon=16, num_inference_steps=2,
               noise_scheduler=DDIMScheduler(num_train_timesteps=100,
               beta_schedule='squaredcos_cap_v2', prediction_type='epsilon'))
    sample = policy.sample_action(condition.detach(), action_dim=5)
    assert sample.shape == (2, 16, 5) and torch.isfinite(sample).all()
