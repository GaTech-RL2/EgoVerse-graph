"""Stationery source parity (the six M 200 / D 0.33 recipes) and integration with main's graph/data lifecycle."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.open_loop_sim import OpenLoopSimEval
from egomimic.pipeline.inference_config import build_inference_config
from egomimic.rldb.embodiment.bimanual_arc import get_keymap, get_transform_list
from egomimic.rldb.zarr.arc_projection_tokenizer import TokenizeBimanualArcProjection
from egomimic.rldb.zarr.data_module import _check_fixed_ranges
from egomimic.rldb.zarr.e1_arc_tokenizer import TokenizeBimanualArcLengthE1
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.robot.arc_decoder import BimanualArcDecoder
from scripts.e1.validate_stattempo_configs import compose_recipe, fingerprint

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/stattempo_contracts.json").read_text()
)
# Codec modes the runtime keeps (their recipes were removed from this branch);
# the native-preprocessing parity test still covers every one of them.
VARIANTS = {
    "arcdurhyb": "durhyb",
    "arcvelhyb": "profhyb",
    "arcdurtri": "durtri",
    "arcveltri": "proftri",
    "arcdurevt": "durevt",
    "arcvelevt": "stkevt",
    "arcdurref": "durref",
    "arcvelref": "stkref",
}
# The kept recipes and the decoder layout each declares.
RECIPES = {
    "bc_rl2_stattempo_slowpace366_arcdur_m200_d033": ("dur", "e1_dur"),
    "bc_rl2_stattempo_slowpace366_arcvel_m200_d033": ("profile", "e1_profile"),
    "bc_rl2_stattempo_slowpace366_arcdur_abstime_m200_d033": ("dur", "e1_dur"),
    "bc_rl2_stattempo_slowpace366_arcvel_nosmooth_m200_d033": ("profile", "e1_profile"),
    "bc_rl2_stattempo_slowpace366_arc_projection_dur_m200_d033": (
        "dur",
        "e1_projection_dur",
    ),
    "bc_rl2_stattempo_slowpace366_arc_projection_vel_m200_d033": (
        "profile",
        "e1_projection_vel",
    ),
}


def chunk(kind):
    t = np.arange(100) / 30
    actions = np.zeros((100, 14))
    if kind != "hold":
        u = np.maximum(t - 0.6, 0) if kind == "delayed" else t
        for off in (0, 7):
            actions[:, off] = 0.12 * u
            actions[:, off + 1] = 0.025 * np.sin(u * 3)
            actions[:, off + 3] = 0.8 * u
            actions[:, off + 6] = np.clip((u - 1) * 1.3, 0, 1)
    return actions


def codec(mode, **kwargs):
    return TokenizeBimanualArcLengthE1(
        min_distance_unit=0.4,
        resampled_vector_length=100,
        dt=1 / 30,
        velocity_norm="path",
        velocity_mode=mode,
        rotation_distance_unit=2 * np.pi,
        **kwargs,
    )


@pytest.mark.parametrize("name", FIXTURE["configs"])
def test_resolved_recipes_preserve_original_scientific_contract(name):
    cfg = compose_recipe(name)
    assert (
        fingerprint(OmegaConf.to_container(cfg, resolve=True))
        == FIXTURE["configs"][name]
    )
    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        for leaf in split.values():
            instantiate(leaf.resolver.key_map)
            instantiate(leaf.resolver.transform_list)
    evaluator = instantiate(cfg.evaluator)
    assert evaluator is not None


@pytest.mark.parametrize("case", FIXTURE["codecs"])
def test_tokens_and_decoded_clocks_match_original_source(case):
    kind, mode = case.split("/")
    # The original source smoothed speeds over 7 frames; recipes now default to 1.
    c = codec(mode, speed_smooth_frames=7)
    tokens = c.transform({"actions_cartesian": chunk(kind)})["actions_cartesian"]
    decoded = c.detokenize(tokens, action_horizon=100)
    payload = json.dumps(
        [np.round(tokens, 10).tolist(), np.round(decoded, 10).tolist()],
        separators=(",", ":"),
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == FIXTURE["codecs"][case]


@pytest.mark.parametrize("name,expected", RECIPES.items())
def test_declared_decoder_and_tick_prefix_use_the_training_codec(name, expected):
    mode, layout = expected
    cfg = compose_recipe(name)
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "ready", artifact.get("reason")
    decoder = instantiate(cfg.model.inference.profiles.default.adapter.decoder)
    assert isinstance(decoder, BimanualArcDecoder)
    # The contract declares fastest-stream termination on; with it off the decode
    # is the training codec's whole-token decode.
    assert decoder.token_layout == layout and decoder.first_stream == 1
    decoder.first_stream = 0
    training = instantiate(
        cfg.data.train_datasets.yam_bimanual.resolver.transform_list
    )[-1]
    assert type(decoder.codec) is type(training)
    assert training.velocity_mode == mode and training.M == 200
    assert training.tokenizer.config.min_distance_unit == 0.33
    if isinstance(training, TokenizeBimanualArcProjection):
        np.testing.assert_array_equal(
            decoder.codec.projection_offset_m, training.projection_offset_m
        )
    tokens = training.transform({"actions_cartesian": chunk("delayed")})[
        "actions_cartesian"
    ]
    np.testing.assert_allclose(
        decoder(tokens)[0], training.detokenize(tokens, action_horizon=100)
    )
    evaluator = instantiate(cfg.evaluator)
    assert isinstance(evaluator, OpenLoopSimEval) and evaluator.tick_prefix
    decoded, steps = evaluator._decode_prediction_with_steps(tokens)
    assert decoded.shape == (steps, 14) and np.isfinite(decoded).all()


@pytest.mark.parametrize(
    "name",
    [
        "bc_rl2_stattempo_slowpace366_arcdur_m200_d033",
        "bc_rl2_stattempo_slowpace366_arc_projection_vel_m200_d033",
    ],
)
def test_small_graph_backward_and_sampling(name):
    torch.set_num_threads(1)
    cfg = compose_recipe(name)
    for key, val in {
        "embed_dim": 32,
        "attention_heads": 4,
        "attention_head_dim": 8,
        "trunk_blocks": 1,
        "flow_blocks": 1,
        "flow_hidden_dim": 32,
    }.items():
        cfg.model[key] = val
    cfg.model.graph.sampler.num_inference_steps = 2
    for stems in [
        cfg.model.graph.stem.stems,
        cfg.model.graph.stem.domain_stems.yam_bimanual,
    ]:
        for stem in stems.values():
            if "weights" in stem:
                stem.weights = None
    model = instantiate(cfg.model.pipeline, device="cpu")
    row = {
        "embodiment": torch.tensor([7, 7]),
        "observations.state.ee_pose": torch.randn(2, 1, 14),
        "actions_cartesian": torch.randn(2, 100, cfg.model.action_token_dim),
    }
    for camera in ("front_img_1", "left_wrist_img", "right_wrist_img"):
        row["observations.images." + camera] = torch.rand(2, 1, 3, 32, 32)
    row["actions_cartesian"] = torch.randn(
        2, int(cfg.model.action_horizon), cfg.model.action_token_dim
    )
    batch = {"opaque": row}
    result = model.forward_training(batch)
    loss = model.compute_losses(result, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    model.nets.eval()
    with torch.no_grad():
        prediction = model.forward_eval(batch)["opaque"]["pred_action"]
    assert prediction.shape == row["actions_cartesian"].shape == (2, 200, 16)
    assert torch.isfinite(prediction).all()


def test_explicit_source_window_does_not_change_main_defaults():
    def horizons(value):
        return {v["horizon"] for v in value.values() if "horizon" in v}

    assert horizons(get_keymap(45, embodiment="yam")) == {45}
    assert horizons(get_keymap(45, embodiment="yam", yam_source_frames=100)) == {100}
    transforms = get_transform_list(
        "arcdurref", chunk_length=45, embodiment="yam", yam_source_frames=100
    )
    assert transforms[-1].token_dim == 24
    assert transforms[-2].n_rows == 100


def test_fixed_ranges_roundtrip_and_restored_scale_guard():
    stats = {k: np.zeros((2, 3)) for k in ("quantile_1", "quantile_99", "min", "max")}
    fixed = {"action": {"low": [-0.75, -1, 0], "high": [0.75, 1, 1]}}
    scaled, _ = MultiDataset._fixed_range_quantiles(stats, **fixed["action"])
    owner = SimpleNamespace(norm_stats={7: {"action": scaled}})
    _check_fixed_ranges(owner, fixed)
    owner.norm_stats[7]["action"]["quantile_99"][0, 0] += 0.1
    with pytest.raises(ValueError, match="fixed_ranges"):
        _check_fixed_ranges(owner, fixed)
    for low, high in [
        ([0] * 2, [1] * 2),
        ([0] * 3, [0] * 3),
        ([0] * 3, [1, 1, float("inf")]),
    ]:
        with pytest.raises(ValueError, match="fixed ranges"):
            MultiDataset._fixed_range_quantiles(stats, low, high)


def test_quantile_widening_only_changes_degenerate_columns():
    stats = {
        "quantile_1": np.array([0.0, 0.0]),
        "quantile_99": np.array([0.0, 1.0]),
        "min": np.array([0.0, -1.0]),
        "max": np.array([4.0, 2.0]),
    }
    wide, n = MultiDataset._widen_degenerate_quantiles(stats)
    assert n == 1
    np.testing.assert_array_equal(wide["quantile_99"], [4.0, 1.0])
    np.testing.assert_array_equal(stats["quantile_99"], [0.0, 1.0])


@pytest.mark.parametrize("name", FIXTURE["models"])
def test_full_model_checkpoint_names_shapes_and_parameter_counts(name):
    pipeline = OmegaConf.to_container(compose_recipe(name).model.pipeline, resolve=True)

    def offline(value):
        if isinstance(value, dict):
            if value.get("_target_", "").endswith("E1ImageStem"):
                value["weights"] = None
            if "pretrained" in value:
                value["pretrained"] = False
            for child in value.values():
                offline(child)
        elif isinstance(value, list):
            for child in value:
                offline(child)

    offline(pipeline)
    pipeline["device"] = "meta"
    with torch.device("meta"):
        model = instantiate(pipeline)
    schema = {
        k: [list(v.shape), str(v.dtype)] for k, v in model.nets.state_dict().items()
    }
    expected = FIXTURE["models"][name]
    assert sum(p.numel() for p in model.nets.parameters()) == expected["parameters"]
    assert (
        hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        == expected["state_schema_sha256"]
    )


@pytest.mark.parametrize("variant", ["time", "arcdur", "arcvel", *VARIANTS])
def test_native_yam_full_preprocessing_preserves_source_rows(variant):
    from scipy.spatial.transform import Rotation

    from egomimic.rldb.embodiment.bimanual_arc import VELOCITY_MODES

    actions = chunk("motion")
    raw = {}
    for arm, off in (("left", 0), ("right", 7)):
        quat = Rotation.from_euler("ZYX", actions[:, off + 3 : off + 6]).as_quat()
        poses = np.concatenate(
            [actions[:, off : off + 3], quat[:, [3, 0, 1, 2]]], axis=1
        )
        raw[f"{arm}.cmd_ee_pose"] = poses
        raw[f"{arm}.obs_ee_pose"] = poses[0].copy()
        raw[f"{arm}.cmd_gripper"] = actions[:, off + 6 : off + 7].copy()
        raw[f"{arm}.obs_gripper"] = actions[0, off + 6 : off + 7].copy()
    transforms = get_transform_list(
        variant, chunk_length=200, embodiment="yam", yam_source_frames=100
    )
    for transform in transforms:
        raw = transform.transform(raw)
    np.testing.assert_allclose(raw["actions_time"], actions, atol=1e-6)
    expected = (
        actions
        if variant == "time"
        else codec(VELOCITY_MODES[variant]).transform(
            {"actions_cartesian": actions.copy()}
        )["actions_cartesian"]
    )
    np.testing.assert_allclose(raw["actions_cartesian"], expected, atol=1e-6)


@pytest.mark.parametrize("kind", ["pose", "linear"])
def test_native_interpolation_copies_rows_without_legacy_stride(kind):
    from egomimic.rldb.zarr.action_chunk_transforms import (
        InterpolateLinear,
        InterpolatePose,
    )

    cls = InterpolatePose if kind == "pose" else InterpolateLinear
    source = np.arange(35, dtype=np.float64).reshape(5, 7)
    transform = cls(None, "source", "output", stride=3)
    result = transform.transform({"source": source})["output"]
    np.testing.assert_array_equal(result, source)
    assert not np.shares_memory(result, source)
