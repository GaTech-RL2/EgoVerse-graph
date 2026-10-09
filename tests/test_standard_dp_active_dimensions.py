"""U common-five padding must not become a gripper supervision target."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from egomimic.pipeline.stages_diffusion import DiffusionEpsilonLossStage
from egomimic.pipeline.action_dimensions import normalize_active_action_dimensions
from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.eval.energy_score import energy_score
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

U = 'pushshapes_sim_u_socket'
CHAIN = 'pushshapes_sim_chain_gripper'
DIMS = {U: 4, CHAIN: 5}
BLOCKS = {U: [[0, 2], [2, 4]], CHAIN: [[0, 2], [2, 4], [4, 5]]}


def batch_for(name, prediction, noise):
    return {'diffusion/predicted_noise': prediction, 'diffusion/noise_target': noise,
            'target': torch.zeros_like(prediction),
            'embodiment': torch.full((prediction.shape[0],), get_embodiment_id(name))}


def test_u_gripper_loss_and_output_gradient_are_zero():
    pred = torch.ones(2, 16, 5, requires_grad=True)
    noise = torch.zeros_like(pred)
    stage = DiffusionEpsilonLossStage(active_action_dims_by_embodiment=DIMS)
    loss = stage(batch_for(U, pred, noise))['loss/diffusion_noise']
    assert torch.equal(loss, F.mse_loss(pred[..., :4], noise[..., :4]))
    loss.backward()
    assert torch.count_nonzero(pred.grad[..., :4]) == 2 * 16 * 4
    assert torch.count_nonzero(pred.grad[..., 4]) == 0
    changed = pred.detach().clone(); changed[..., 4] = 1e10
    assert torch.equal(stage(batch_for(U, changed, noise))['loss/diffusion_noise'], loss.detach())


def test_chain_retains_exact_full_five_loss_and_gradient():
    pred = torch.ones(2, 16, 5, requires_grad=True)
    noise = torch.zeros_like(pred)
    loss = DiffusionEpsilonLossStage(DIMS)(batch_for(CHAIN, pred, noise))['loss/diffusion_noise']
    assert torch.equal(loss, F.mse_loss(pred, noise))
    loss.backward()
    assert torch.count_nonzero(pred.grad[..., 4]) == 2 * 16


def test_unconfigured_legacy_loss_is_unchanged():
    torch.manual_seed(5)
    pred, noise = torch.randn(2, 16, 5), torch.randn(2, 16, 5)
    assert torch.equal(DiffusionEpsilonLossStage()(batch_for(U, pred, noise))['loss/diffusion_noise'], F.mse_loss(pred, noise))


@pytest.mark.parametrize('bad', [{U: 0}, {U: True}, {U: 4.5}, {}])
def test_invalid_contract_is_rejected(bad):
    with pytest.raises(ValueError): normalize_active_action_dimensions(bad)


def test_mixed_embodiments_and_missing_contract_fail_closed():
    pred, noise = torch.ones(2, 16, 5), torch.zeros(2, 16, 5)
    batch = batch_for(U, pred, noise)
    batch['embodiment'][1] = get_embodiment_id(CHAIN)
    with pytest.raises((ValueError, RuntimeError)):
        DiffusionEpsilonLossStage(DIMS)(batch)
    with pytest.raises(ValueError):
        DiffusionEpsilonLossStage({CHAIN: 5})(batch_for(U, pred, noise))
    with pytest.raises(ValueError):
        DiffusionEpsilonLossStage({U: 6})(batch_for(U, pred, noise))


def evaluator():
    return PlanarActionEval(energy_score_enabled=False,
        active_action_dims_by_embodiment=DIMS, semantic_blocks_by_embodiment=BLOCKS)


def test_energy_score_ignores_u_gripper_but_preserves_chain_contract():
    torch.manual_seed(123)
    samples, target = torch.randn(32, 2, 16, 5), torch.randn(2, 16, 5)
    ev = evaluator()
    u = ev._energy_values(samples, target, get_embodiment_id(U), U)
    changed = samples.clone(); changed[..., 4] += 1e9
    same = ev._energy_values(changed, target, get_embodiment_id(U), U)
    for key in u: assert torch.equal(u[key], same[key])
    chain = ev._energy_values(samples, target, get_embodiment_id(CHAIN), CHAIN)
    native = energy_score(samples, target, BLOCKS[CHAIN])
    for key in chain: assert torch.equal(chain[key], native[key])


def test_validation_normalized_and_native_u_metrics_ignore_padding():
    ev = evaluator()
    collected = {}
    ev.trainer = SimpleNamespace(lightning_module=SimpleNamespace(log_dict=lambda values, **kw: collected.update(values)))
    ev.normalizer = SimpleNamespace(unnormalize=lambda values, embodiment: values)
    class Decoder:
        def decode(self, value):
            return torch.stack((value[..., 0], value[..., 1], torch.atan2(value[..., 3], value[..., 2])), dim=-1)
    ev.native_decoders = {U: Decoder()}
    target = torch.zeros(2, 16, 5); target[..., 2] = 1
    prediction = target.clone(); prediction[..., 0] = 2; prediction[..., 4] = 1e9
    ev.model = SimpleNamespace(forward_eval=lambda batch: {'u': {'pred_action': prediction}})
    batch = {'u': {'actions': target, 'embodiment': torch.full((2,), get_embodiment_id(U))}}
    ev.on_validation_step(batch, 0)
    assert torch.equal(collected['Valid/MSE/'+U], torch.tensor(1.))
    assert torch.allclose(collected['Valid/Native_MSE/'+U], torch.tensor(4./3))
    original = {k:v.clone() for k,v in collected.items()}
    prediction[..., 4] = -1e9
    ev.on_validation_step(batch, 0)
    for key in original: assert torch.equal(original[key], collected[key])
