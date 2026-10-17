"""Tail-error gates, nonoverlapping replay confirmation and Table 7 contracts."""
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from egomimic.eval.control_replay import (
    candidate_rank, median_spatial_budgets, passes_trace_gates, reconstruct,
    reconstruction_statistics, sample_window_indices,
)
from egomimic.rldb.action_codec import ShapeTimeControlChunkCodec

ROOT = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs/benchmark"


def test_tail_gate_rejects_rare_large_error_hidden_by_good_p90():
    original = torch.zeros(100, 3, 1)
    decoded = original.clone()
    decoded[0, 0, 0] = .2
    decoded[:, 1:] = 100  # Unexecuted padding must never affect fidelity.
    stats = reconstruction_statistics(original, decoded, torch.ones(100, dtype=torch.long))
    assert stats["action_rmse_p90"] == 0
    assert abs(stats["action_max_abs"] - .2) < 1e-6
    row = {**stats,"mean_native_steps":1.,"scalar_compression_ratio":.01}
    cfg = OmegaConf.create({"gates":{"action_rmse_p90":.01,"mean_native_steps":1.,"action_max_abs":.1}})
    assert not passes_trace_gates(row,cfg)
    del cfg.gates.action_max_abs
    assert passes_trace_gates(row,cfg)


def test_confirmation_excludes_entire_calibration_windows_and_episode_edges():
    raw = {"terminals":np.array([0]*39+[1]+[0]*39+[1])}
    first = sample_window_indices(raw,5,7,4)
    second = sample_window_indices(raw,5,9,80,first)
    assert (np.abs(second[:,None]-first[None]).min(1) > 5).all()
    assert all((index <= 34) or (40 <= index <= 74) for index in second)
    assert len(second) == len(set(second))


def test_large_m_chunked_replay_keeps_same_controls_and_durations():
    c = ShapeTimeControlChunkCodec(2,25,kind="arc",waypoints=1024,distance=2.)
    actions = torch.randn(7,25,2,generator=torch.Generator().manual_seed(1)).clamp(-1,1)
    expected, lengths = c.encode(actions)
    expected, expected_lengths = c.decode(expected)
    actual, actual_lengths = reconstruct(c,actions,3)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual_lengths,expected_lengths,rtol=0,atol=0)


def test_smallest_faithful_selection_never_uses_scalar_compression():
    cfg = OmegaConf.load(ROOT / "dqc_table7.yaml").calibration
    small = {"codec":{"waypoints":256},"action_rmse_p99":.0009,"action_rmse_mean":.0002,
             "scalar_compression_ratio":.001}
    large = {**small,"codec":{"waypoints":512},"action_rmse_p99":.0004,"scalar_compression_ratio":100.}
    assert candidate_rank(small,cfg) < candidate_rank(large,cfg)


def test_table7_reference_horizons_and_teacher_parameters_are_explicit():
    # External reference: arXiv:2512.10926 Table 7, DQC column.
    expected = {"cube-triple":(5,.93,.8),"cube-quadruple":(5,.93,.8),
        "cube-octuple":(5,.93,.5),"humanoidmaze-giant":(1,.5,.8),
        "puzzle-4x5":(5,.9,.5),"puzzle-4x6":(1,.7,.5)}
    recipe = OmegaConf.load(ROOT / "dqc_table7.yaml")
    suite = OmegaConf.load(ROOT / recipe.suite_recipe)
    for name,(ha,kb,kd) in expected.items():
        protocol = recipe.domain_protocols[name]
        assert (protocol.reference_policy_native_steps,protocol.kappa_b,protocol.kappa_d)==(ha,kb,kd)
        assert protocol.teacher_native_steps == 25
        cfg = OmegaConf.load(ROOT / recipe.base_recipe)
        assert cfg.steps == 1000000 and cfg.batch_size == 4096
        assert suite.domains[name].kappa_b == kb and suite.domains[name].kappa_d == kd
        cfg.codec.action_dim = 5
        cfg.codec.native_horizon = ha
        c = hydra.utils.instantiate(cfg.codec)
        actions = torch.randn(3,25,5).clamp(-1,1)
        z,tau = c.encode(actions)
        decoded,recovered_tau = c.decode(z)
        assert tau.tolist()==[ha]*3 and recovered_tau.tolist()==[ha]*3
        torch.testing.assert_close(decoded,actions[:,:ha],rtol=0,atol=0)
    # Prevent the missing-field failure found by the real-data audit smoke.
    shape = OmegaConf.load(ROOT / "dqc_shape_time.yaml")
    assert shape.protocol.reference_policy_native_steps == 5


def test_median_one_control_window_retains_variable_time():
    speeds = torch.tensor([.05,.1,.2,.3,.5,.7,.9])
    actions = torch.arange(25)[None,:,None]*speeds[:,None,None]
    c = ShapeTimeControlChunkCodec(1,25,kind="arc",path_mode="control",distance=1.,waypoints=16)
    budgets,_ = median_spatial_budgets(c,actions,1,[1.])
    c = ShapeTimeControlChunkCodec(1,25,kind="arc",path_mode="control",waypoints=16,**budgets[0])
    lengths = c.window_lengths(actions)
    assert lengths.float().quantile(.5)==1
    assert lengths.min()==1 and lengths.max()>1
