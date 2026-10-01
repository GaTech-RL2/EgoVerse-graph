"""Independent geometry and integration regressions for the three-stack merge."""

import numpy as np
import pytest
import torch

from egomimic.eval.open_loop_sim import OpenLoopSimEval
from egomimic.rldb.embodiment.bimanual_arc import get_keymap
from egomimic.rldb.zarr.arc_length_tokenizer import TokenizeBimanualArcLengthCartesian
from egomimic.rldb.zarr.group_balance_sampler import build_group_balance_sampler
from egomimic.robot.arc_decoder import BimanualIntervalArcDecoder


def test_same_shape_hybrid_mode_change_invalidates_saved_contract():
    from copy import deepcopy

    from omegaconf import OmegaConf

    from egomimic.pipeline.checkpoint_binding import (
        checkpoint_binding,
        validate_checkpoint_binding,
    )
    from egomimic.pipeline.inference_config import (
        build_inference_config,
    )
    from egomimic.pl_utils.data_context import DataContext
    from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

    recipe = (
        CONFIGS
        / "experiment/abc_arc/robot_bc/abc_multitask4_hpt300_hybrid_visual_openloop.yaml"
    )
    with compose_for_audit(recipe) as cfg:
        original = build_inference_config(cfg)
        changed = deepcopy(cfg)
        changed.abc.arc_chunking_mode = "race"
        assert changed.hpt.action_horizon == cfg.hpt.action_horizon
        assert changed.hpt.token_dim == cfg.hpt.token_dim
        assert (
            original["inference_contract_sha256"]
            != build_inference_config(changed)["inference_contract_sha256"]
        )
        context = DataContext(
            None,
            {},
            (),
            OmegaConf.to_container(cfg.model.data_requirements, resolve=True),
        )
        checkpoint = {
            "inference_binding": checkpoint_binding(cfg, context),
            "data_context": context.snapshot(),
        }
        with pytest.raises(ValueError, match="binding mismatch"):
            validate_checkpoint_binding(checkpoint, changed, context)


@pytest.mark.parametrize("layout", ["stacked", "wide"])
def test_training_evaluation_and_robot_decoder_preserve_independent_clocks(layout):
    # Hand-authored motion, independent of the encoder: left translates at 1 m/s
    # and turns at 2 rad/s, right translates at .25 m/s and turns at .5 rad/s.
    # Each reaches its own endpoint and holds; rotation is intrinsic ZYX yaw.
    poses = np.zeros((3, 14))
    poses[:, 0] = [0, 0.1, 0.2]
    poses[:, 7] = [0, 0.05, 0.1]
    poses[:, 3] = [0, 0.2, 0.4]
    poses[:, 10] = [0, 0.1, 0.2]
    poses[:, (6, 13)] = 0.7
    rates = np.zeros_like(poses)
    rates[:, (0, 7, 3, 10)] = [1.0, 0.25, 2.0, 0.5]
    token = np.concatenate((poses, rates), axis=0 if layout == "stacked" else 1)
    settings = dict(
        min_distance_unit=0.4,
        resampled_vector_length=3,
        dt=0.1,
        velocity_mode="per_waypoint",
        velocity_layout=layout,
        rotation_distance_unit=0.8,
        arc_chunking_mode="multistream",
    )
    training = TokenizeBimanualArcLengthCartesian(**settings).detokenize(
        token, action_horizon=4
    )
    deployed = BimanualIntervalArcDecoder(**settings, action_horizon=4)(token[None])[0]
    evaluator = OpenLoopSimEval(
        action_mode="arc",
        control_dt=0.1,
        control_horizon=4,
        execute_fraction=1.0,
        min_distance_unit=0.4,
        resampled_vector_length=3,
        rotation_distance_unit=0.8,
        arc_chunking_mode="multistream",
    )
    evaluated, steps = evaluator._decode_prediction_with_steps(token, max_steps=4)
    time = np.arange(4) * 0.1
    expected = np.zeros((4, 14))
    expected[:, 0] = np.minimum(time, 0.2)
    expected[:, 7] = np.minimum(time * 0.25, 0.1)
    expected[:, 3] = np.minimum(time * 2.0, 0.4)
    expected[:, 10] = np.minimum(time * 0.5, 0.2)
    expected[:, (6, 13)] = 0.7
    assert steps == 4
    for decoded in (training, deployed, evaluated):
        np.testing.assert_allclose(decoded, expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("value", [0, -1, True, 2.5, float("nan")])
def test_source_window_override_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        get_keymap(horizon=200, embodiment="yam", yam_source_frames=value)


@pytest.mark.parametrize(
    "embodiment,keymap_mode",
    [("human", "cartesian"), ("yam", "arc_tokenizer_cartesian")],
)
def test_source_window_override_cannot_be_silently_ignored(embodiment, keymap_mode):
    with pytest.raises(ValueError, match="only to YAM cartesian"):
        get_keymap(
            horizon=200,
            embodiment=embodiment,
            keymap_mode=keymap_mode,
            yam_source_frames=400,
        )


class GroupData:
    index_map = [("large", index) for index in range(900)] + [
        ("small", index) for index in range(100)
    ]


def test_group_sampler_preserves_shares_and_exact_distributed_draw(tmp_path):
    small = tmp_path / "small.txt"
    small.write_text("small\n")
    options = dict(
        group_episode_files={"small": str(small)},
        group_fractions={"small": 0.7, "large": 0.3},
        default_group="large",
        num_samples=10000,
        seed=123,
    )
    whole = build_group_balance_sampler(GroupData(), **options)
    ranks = [
        build_group_balance_sampler(GroupData(), **options, num_replicas=2, rank=rank)
        for rank in (0, 1)
    ]
    draws = list(whole)
    assert list(ranks[0]) == draws[::2]
    assert list(ranks[1]) == draws[1::2]
    assert abs(np.mean(np.array(draws) >= 900) - 0.7) < 0.02
    whole.set_epoch(1)
    assert list(whole) != draws
    whole.set_epoch(0)
    assert list(whole) == draws
    assert all(
        isinstance(sampler, torch.utils.data.DistributedSampler) for sampler in ranks
    )


@pytest.mark.parametrize("fraction", [float("nan"), float("inf"), -0.1, 0.0])
def test_group_sampler_rejects_nonfinite_or_nonpositive_fractions(tmp_path, fraction):
    with pytest.raises(ValueError, match="positive and sum to 1"):
        build_group_balance_sampler(GroupData(), {}, {"large": fraction}, "large")
