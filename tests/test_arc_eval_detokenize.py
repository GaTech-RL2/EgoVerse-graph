"""The arc evaluator must detokenize before anything treats rows as poses."""

from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.eval.arc_bimanual_cartesian_eval import ArcBimanualCartesianEval
from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
)

_REPO = Path(__file__).resolve().parents[1]

_D = 0.40
_M = 100
_H = 100


def _evaluator():
    """Bare instance: __init__ needs no trainer, but skip the heavy base setup."""
    ev = ArcBimanualCartesianEval.__new__(ArcBimanualCartesianEval)
    ev.action_key = "actions_cartesian"
    ev.min_distance_unit = _D
    ev.resampled_vector_length = _M
    ev.action_horizon = _H
    ev._tokenizer = TokenizeBimanualArcLengthCartesian(
        action_key="actions_cartesian",
        output_action_key="actions_cartesian",
        min_distance_unit=_D,
        resampled_vector_length=_M,
        preserve_action_key=None,
    )
    return ev


def _raw_chunk(steps: int = 200, seed: int = 0) -> np.ndarray:
    """A (T, 14) bimanual cartesian chunk that actually travels."""
    rng = np.random.default_rng(seed)
    chunk = np.zeros((steps, 14), dtype=np.float64)
    t = np.linspace(0.0, 1.0, steps)
    for base in (0, 7):  # left arm, right arm
        chunk[:, base + 0] = t * 0.6                     # x sweeps 0.6 m
        chunk[:, base + 1] = 0.1 * np.sin(2 * np.pi * t)  # y wiggles
        chunk[:, base + 2] = 0.05 * t
        chunk[:, base + 3] = 0.4 * t                      # yaw
        chunk[:, base + 6] = t                            # gripper opens
    return chunk


def _token(steps: int = 200, seed: int = 0) -> np.ndarray:
    tok = TokenizeBimanualArcLengthCartesian(
        action_key="a", output_action_key="a",
        min_distance_unit=_D, resampled_vector_length=_M,
        preserve_action_key=None,
    )
    return np.asarray(tok.transform({"a": _raw_chunk(steps, seed=seed)})["a"])


# -- the token itself -------------------------------------------------------


def test_tokenizer_emits_m_waypoints_plus_a_velocity_row():
    assert _token().shape == (_M + 1, 14)


def test_velocity_row_is_not_distinguishable_by_magnitude():
    # This is WHY the detokenize-first ordering matters: nothing about the
    # velocity row's scale marks it as a non-pose, so a revert applied to the
    # whole token corrupts it invisibly.
    t = _token()
    # A mid-token waypoint, not row 0: the synthetic chunk starts at the origin
    # so row 0 is all zeros. On real ABC data both scales are ~0.16.
    waypoint_scale = float(np.abs(t[_M // 2]).mean())
    velocity_scale = float(np.abs(t[_M]).mean())
    assert 0.05 < velocity_scale / max(waypoint_scale, 1e-9) < 20.0


# -- the evaluator seam -----------------------------------------------------


def test_base_viz_source_is_identity():
    ev = BimanualCartesianEval.__new__(BimanualCartesianEval)
    x = torch.arange(2 * 3 * 14, dtype=torch.float32).reshape(2, 3, 14)
    assert ev._viz_source(x, 7) is x


def test_arc_viz_source_converts_tokens_to_pose_rows():
    ev = _evaluator()
    batch = torch.from_numpy(np.stack([_token(), _token(seed=1)])).float()
    out = ev._viz_source(batch, 7)
    assert out.shape == (2, _H, 14)
    assert out.dtype == batch.dtype


def test_arc_viz_source_drops_the_velocity_row():
    # The output must be poses only: M+1 rows in, action_horizon rows out, and
    # no row of the result equal to the velocity token.
    ev = _evaluator()
    token = _token()
    out = ev._viz_source(torch.from_numpy(token[None]).float(), 7)[0].numpy()
    velocity = token[_M]
    assert not any(np.allclose(row, velocity, atol=1e-6) for row in out)


def test_arc_viz_source_output_is_monotone_along_the_travelled_axis():
    # x sweeps forward in the source chunk, so the reconstruction should too.
    ev = _evaluator()
    out = ev._viz_source(torch.from_numpy(_token()[None]).float(), 7)[0].numpy()
    dx = np.diff(out[:, 0])
    assert (dx >= -1e-6).all(), "reconstruction moved backwards along x"
    assert out[-1, 0] > out[0, 0]


def test_arc_viz_source_rejects_a_time_indexed_chunk():
    # Guards the misconfiguration of pointing this evaluator at a baseline run.
    ev = _evaluator()
    with pytest.raises(ValueError, match="arc tokens"):
        ev._viz_source(torch.zeros(2, _H, 14), 7)


def test_arc_viz_source_rejects_a_wrong_M():
    ev = _evaluator()
    with pytest.raises(ValueError, match="M=100"):
        ev._viz_source(torch.zeros(2, 26, 14), 7)


def test_arc_round_trip_recovers_the_span_the_token_covers():
    # Detokenize is a reconstruction, not an inverse, but it must land in the
    # right place: the decoded path should cover ~D of travel per arm.
    ev = _evaluator()
    out = ev._viz_source(torch.from_numpy(_token()[None]).double(), 7)[0].numpy()
    travelled = float(np.linalg.norm(np.diff(out[:, :3], axis=0), axis=-1).sum())
    assert 0.5 * _D < travelled < 2.0 * _D, travelled


# -- config consistency -----------------------------------------------------
#
# The detokenizer derives replay duration from D and the velocity token, so if
# the evaluator's D/M drift from the data config's the reconstruction runs at
# the wrong speed while every shape still matches. Nothing at runtime catches
# it, so pin it here.


def _compose(experiment: str):
    from hydra import compose, initialize_config_dir

    root = str(_REPO / "egomimic/hydra_configs")
    with initialize_config_dir(version_base=None, config_dir=root):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}", "++paths.root_dir=."],
        )


def test_arc_experiment_evaluator_matches_its_data_tokenizer():
    cfg = _compose("abc_arc/abc_fstshirt_arc_bc")
    tok = cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    assert cfg.evaluator.min_distance_unit == tok.min_distance_unit
    assert cfg.evaluator.resampled_vector_length == tok.resampled_vector_length


def test_arc_experiment_model_horizon_matches_the_token_row_count():
    cfg = _compose("abc_arc/abc_fstshirt_arc_bc")
    tok = cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    expected_rows = int(tok.resampled_vector_length) + 1
    assert cfg.abc.arc_token_rows == expected_rows
    # All three diffusion stages must agree, or training aborts on batch one.
    for stage in cfg.model.pipeline.stages:
        if "action_horizon" in stage:
            assert int(stage.action_horizon) == expected_rows


def test_arc_experiment_uses_the_arc_evaluator_not_the_baseline_one():
    cfg = _compose("abc_arc/abc_fstshirt_arc_bc")
    assert cfg.evaluator._target_.endswith("ArcBimanualCartesianEval")


def test_baseline_experiment_still_uses_the_time_indexed_evaluator():
    # The baseline must NOT pick up the arc evaluator: its chunks are poses.
    # `override /evaluator: null` drops the key entirely, so select, don't index.
    from omegaconf import OmegaConf

    cfg = _compose("abc_arc/abc_fstshirt_bc")
    target = OmegaConf.select(cfg, "evaluator._target_")
    assert target is None or not str(target).endswith("ArcBimanualCartesianEval")
