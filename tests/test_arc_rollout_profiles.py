"""Model-owned rollout contracts for donor ARC layouts; no checkpoint or hardware loads."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.inference_config import build_inference_config
from egomimic.pipeline.inference_controls import (
    bind_policy_controls,
    configure_profile_controls,
)
from egomimic.robot.graph_policy import GraphRobotPolicy

REPO = Path(os.environ.get("EGO_REPO", Path(__file__).resolve().parents[1]))


def config(recipe, *overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "egomimic/hydra_configs")
    ):
        return compose(
            "train_zarr_cartesian", overrides=["+experiment=" + recipe, *overrides]
        )


def profile(cfg):
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "ready", artifact.get("reason")
    graph = artifact["inference_graph"]
    p = graph["profiles"]["default"]
    return graph, p


def instantiate_decoder(graph, p):
    decoder = instantiate(OmegaConf.create(p["adapter"]["decoder"]))
    decoder.validate_inference_contract(
        graph["native_output"]["shape"], graph["output"]["shape"]
    )
    return decoder


def bind_declared_controls(cfg, graph, p, decoder):
    # A tiny settings owner exercises the real generic binder without creating
    # the model's large neural network or loading any checkpoint.
    sampler = SimpleNamespace(
        num_inference_steps=50, policy=SimpleNamespace(num_inference_steps=100)
    )
    backend = SimpleNamespace(pipeline=SimpleNamespace(stage_by_id=lambda _: sampler))
    bindings = configure_profile_controls(backend, cfg, {"default": p})
    policy = SimpleNamespace(
        adapter=SimpleNamespace(decoder=decoder), replan_every=30, max_valid_samples=1
    )
    return bind_policy_controls(
        policy, bindings, allowed_paths=GraphRobotPolicy.control_paths
    )


@pytest.mark.parametrize(
    "variant,cols",
    [
        ("arcvel", 16),
        ("arcdur", 16),
        ("arclogdur", 16),
        ("arcdurhyb", 18),
        ("arcvelhyb", 18),
        ("arcdurtri", 20),
        ("arcveltri", 20),
    ],
)
@pytest.mark.parametrize("family", ["wrists", "front"])
def test_e1_declared_arc_profile_native_shapes_and_defaults(variant, cols, family):
    if family == "wrists":
        cfg = config(
            "yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdur",
            "e1.variant=" + variant,
            "model.action_token_dim=" + str(cols),
            "++model.rotation_distance_unit=6.283185307179586",
        )
    else:
        cfg = config(
            "yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdur",
            "model=e1/hpt_flow",
            "e1.variant=" + variant,
            "hpt.action_dim=" + str(cols),
        )
    graph, p = profile(cfg)
    assert graph["native_output"]["shape"] == [100, cols]
    assert graph["output"]["shape"] == [100, 14]
    assert p["overrides"]["inference_steps"]["default"] == 20
    assert p["overrides"]["execute_waypoint_percent"]["default"] == 50
    assert p["overrides"]["execute_waypoint_percent"]["min"] == 2
    assert p["overrides"]["execute_waypoint_percent"]["step"] == 1
    assert p["overrides"]["multistream_fastest_stream"]["default"] == 1
    decoder = instantiate_decoder(graph, p)
    bind_declared_controls(cfg, graph, p, decoder)
    for spec in p["overrides"].values():
        if spec["target"]["kind"] == "policy_attribute" and spec["target"][
            "attribute_path"
        ].startswith("adapter.decoder."):
            attribute = spec["target"]["attribute_path"].rsplit(".", 1)[1]
            decoder.validate_inference_control(attribute, spec["default"])


@pytest.mark.parametrize("family,steps", [("wrists", 50), ("front", 10), ("dp", 100)])
def test_time_profile_has_tempo_controls_and_original_sampler_default(family, steps):
    overrides = []
    recipe = "yam_arc_grid/scratch_rl2_stattempo_elmoaidan_time"
    if family == "front":
        overrides = ["model=e1/hpt_flow"]
    if family == "dp":
        recipe = "yam_arc_grid/scratch_rl2_stattempo_slowpace_time_dp180pt"
    cfg = config(recipe, *overrides)
    graph, p = profile(cfg)
    assert p["overrides"]["inference_steps"]["default"] == steps
    assert "multistream_fastest_stream" not in p["overrides"]
    assert "execute_waypoint_percent" not in p["overrides"]
    assert p["adapter"]["decoder"]["_target_"].endswith("TimeChunkRetimer")
    bind_declared_controls(cfg, graph, p, instantiate_decoder(graph, p))


def test_dp_arc_preserves_diffusion_step_default():
    graph, p = profile(config("yam_arc_grid/scratch_rl2_towels394_arcdur_dp300_lambda"))
    assert p["overrides"]["inference_steps"]["default"] == 100
    assert (
        p["overrides"]["inference_steps"]["target"]["attribute_path"]
        == "policy.num_inference_steps"
    )
    instantiate_decoder(graph, p)


@pytest.mark.parametrize(
    "layout,rows,cols,mode",
    [
        ("wide", 100, 28, "per_waypoint"),
        ("stacked", 200, 14, "per_waypoint"),
        ("clock", 100, 18, "duration"),
    ],
)
def test_m28_codec_family_uses_declared_waypoints_and_native_layout(
    layout, rows, cols, mode
):
    cfg = config(
        "abc_arc/robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop",
        "abc.arc_chunking_mode=multistream",
        "abc.arc_token_rows=" + str(rows),
        "abc.arc_token_dim=" + str(cols),
        "abc.arc_velocity_mode=" + mode,
        "run_provenance.action_contract.velocity_layout=" + layout,
    )
    graph, p = profile(cfg)
    assert graph["native_output"]["shape"] == [rows, cols]
    assert graph["output"]["shape"] == [100, 14]
    decoder = instantiate_decoder(graph, p)
    bind_declared_controls(cfg, graph, p, decoder)
    assert decoder.M == 100  # M28 is a codec name, not M=28.
    assert p["overrides"]["execute_waypoint_percent"]["min"] == 2
    assert p["overrides"]["execute_waypoint_percent"]["step"] == 1
    assert "arc_speed_percent" not in p["overrides"]
    assert "arc_hold_speed_percent" not in p["overrides"]


@pytest.mark.parametrize("layout,rows,cols", [("wide", 100, 28), ("stacked", 200, 14)])
def test_explicit_pr193_lab_template_keeps_whole_native_token(layout, rows, cols):
    cfg = config(
        "yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdur",
        "+model_contract=arc_lab_pr193",
        "model.lab_arc.velocity_layout=" + layout,
        "model.lab_arc.token_rows=" + str(rows),
        "model.action_horizon=" + str(rows),
        "model.action_token_dim=" + str(cols),
    )
    graph, p = profile(cfg)
    assert graph["native_output"]["shape"] == [rows, cols]
    assert graph["output"]["shape"] == [100, 14]
    assert p["adapter"]["decoder"]["token_layout"] == "lab_pw_" + layout
    assert "arc_hold_speed_percent" not in p["overrides"]
    bind_declared_controls(cfg, graph, p, instantiate_decoder(graph, p))


def test_visual_time_preserves_grid_conversion_before_retiming():
    cfg = config("abc_arc/robot_bc/stationery_rl2_hpt300_visual_baseline_openloop")
    graph, p = profile(cfg)
    assert p["overrides"]["inference_steps"]["default"] == 50
    decoder = instantiate_decoder(graph, p)
    bind_declared_controls(cfg, graph, p, decoder)
    assert len(decoder.decoders) == 2
    assert (
        p["overrides"]["arc_speed_percent"]["target"]["attribute_path"]
        == "adapter.decoder.decoders.1.speed_percent"
    )
    assert "multistream_fastest_stream" not in p["overrides"]


def test_non_100_waypoints_require_explicit_whole_prefix_bounds():
    cfg = config("yam_arc_grid/scratch_rl2_stattempo_elmoaidan_arcdur", "e1.M=3")
    graph, p = profile(cfg)
    with pytest.raises(Exception, match="whole|waypoint|percent"):
        instantiate(OmegaConf.create(p["adapter"]["decoder"]))
    # Actual M28, distinct from the M28 codec family: only 25%-spaced prefixes.
    cfg.e1.M = 28
    cfg.model.action_horizon = 28
    cfg.model.arc_rollout.execute_percent = {
        "min": 25,
        "max": 100,
        "step": 25,
        "default": 50,
    }
    graph, p = profile(cfg)
    assert p["overrides"]["execute_waypoint_percent"]["min"] == 25
    assert p["overrides"]["execute_waypoint_percent"]["step"] == 25
    instantiate_decoder(graph, p)


@pytest.mark.parametrize(
    "action,chunk,velocity,layout,rows,cols",
    [
        (
            "arc_tokenizer_cartesian",
            "joint_distance",
            "per_waypoint",
            "stacked",
            200,
            14,
        ),
        ("arc_tokenizer_cartesian", "joint_distance", "duration", "stacked", 200, 14),
        ("arc_tokenizer_cartesian", "race", "per_waypoint", "stacked", 200, 14),
        ("arc_tokenizer_cartesian", "race", "duration", "stacked", 200, 14),
        (
            "hybrid_arc_tokenizer_cartesian",
            "joint_distance",
            "per_waypoint",
            "stacked",
            200,
            14,
        ),
        ("hybrid_arc_tokenizer_cartesian", "race", "per_waypoint", "stacked", 200, 14),
        (
            "hybrid_arc_tokenizer_cartesian",
            "multistream",
            "per_waypoint",
            "wide",
            100,
            28,
        ),
        (
            "hybrid_arc_tokenizer_cartesian",
            "multistream",
            "per_waypoint",
            "stacked",
            200,
            14,
        ),
        ("hybrid_arc_tokenizer_cartesian", "multistream", "duration", "wide", 100, 28),
        (
            "hybrid_arc_tokenizer_cartesian",
            "multistream",
            "duration",
            "stacked",
            200,
            14,
        ),
        ("hybrid_arc_tokenizer_cartesian", "multistream", "duration", "clock", 100, 18),
    ],
)
def test_visual_declared_mode_matrix_binds_exact_canonical_codec(
    action, chunk, velocity, layout, rows, cols
):
    cfg = config(
        "abc_arc/robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop",
        "abc.action_mode=" + action,
        "abc.arc_chunking_mode=" + chunk,
        "abc.arc_velocity_mode=" + velocity,
        "abc.arc_token_rows=" + str(rows),
        "abc.arc_token_dim=" + str(cols),
        "run_provenance.action_contract.velocity_layout=" + layout,
    )
    graph, p = profile(cfg)
    decoder = instantiate_decoder(graph, p)
    bind_declared_controls(cfg, graph, p, decoder)
    assert decoder.codec_version == "canonical200"
    assert graph["compatibility"]["tokenizer"]["codec_version"] == "canonical200"
    assert graph["native_output"]["shape"] == [rows, cols]
    assert graph["output"]["shape"] == [100, 14]


@pytest.mark.parametrize(
    "action,chunk,velocity,layout,reason",
    [
        (
            "hybrid_arc_tokenizer_cartesian",
            "multistream",
            "mean",
            "stacked",
            "Chunk-mean",
        ),
        (
            "hybrid_arc_tokenizer_cartesian",
            "multistream",
            "per_waypoint",
            "clock",
            "duration",
        ),
        (
            "arc_tokenizer_cartesian",
            "multistream",
            "per_waypoint",
            "wide",
            "hybrid rotation",
        ),
        (
            "hybrid_arc_tokenizer_cartesian",
            "joint_distance",
            "duration",
            "stacked",
            "per_waypoint",
        ),
        ("hybrid_arc_tokenizer_cartesian", "race", "per_waypoint", "wide", "stacked"),
    ],
)
def test_unsupported_visual_clock_combinations_fail_in_declaration(
    action, chunk, velocity, layout, reason
):
    cfg = config(
        "abc_arc/robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop",
        "abc.action_mode=" + action,
        "abc.arc_chunking_mode=" + chunk,
        "abc.arc_velocity_mode=" + velocity,
        "run_provenance.action_contract.velocity_layout=" + layout,
    )
    artifact = build_inference_config(cfg)
    assert artifact["status"] == "unsupported"
    assert reason in artifact["reason"]


def test_historical_codec_requires_explicit_verified_binding_metadata():
    overrides = [
        "+model_contract=arc_m28_historical",
        "abc.arc_chunking_mode=multistream",
    ]
    cfg = config(
        "abc_arc/robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop", *overrides
    )
    with pytest.raises(Exception, match="historical_arc_verified_data_identity"):
        build_inference_config(cfg)
    # Offline synthetic metadata proves explicit selection, not checkpoint migration.
    cfg = config(
        "abc_arc/robot_bc/stationery_rl2_hpt300_visual_hybrid_openloop",
        *overrides,
        "model.historical_arc_verified_data_identity=offline-unbound-test",
        "model.historical_arc_data_requirements={preprocessing:{7:{transforms:{codec_version:m28_99be4af0}}}}",
    )
    graph, p = profile(cfg)
    decoder = instantiate_decoder(graph, p)
    bind_declared_controls(cfg, graph, p, decoder)
    assert decoder.codec_version == "m28_99be4af0"
    assert graph["compatibility"]["tokenizer"]["codec_version"] == "m28_99be4af0"
