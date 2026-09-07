"""The configured HPT graph: composition, contracts and a real forward pass."""

import importlib.util
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = _REPO / "egomimic/hydra_configs"
_EXPERIMENT = "abc/yam_fstshirt_hpt"

_SPEC = importlib.util.spec_from_file_location(
    "config_graph_hpt", _REPO / "tools/config_graph.py"
)
config_graph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(config_graph)

_EXPERIMENT_YAML = _CONFIGS / "experiment" / "abc" / "yam_fstshirt_hpt.yaml"


def _cfg():
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={_EXPERIMENT}", "++paths.root_dir=."],
        )


@pytest.fixture(scope="module")
def cfg():
    return _cfg()


@pytest.fixture(scope="module")
def algo(cfg):
    return hydra.utils.instantiate(cfg.model.pipeline)


def test_the_stage_list_is_the_decomposed_hpt_dataflow(algo):
    assert [type(s).__name__ for s in algo.pipeline.stages] == [
        "HPTStemStage",
        "HPTTrunkStage",
        "ActionTargetBuilder",
        "FlowNoisingStage",
        "FlowDenoiserStage",
        "FlowVelocityLossStage",
    ]


def test_embed_dim_is_one_shared_bus_width(cfg):
    """Every stem, the trunk and the head must read the same width.

    A mismatch is the failure this config is most likely to grow, and the
    stem stage only catches it at the first batch.
    """
    width = cfg.hpt.embed_dim
    stems = cfg.model.pipeline.stages[0].stems
    for name, stem in stems.items():
        assert stem.output_dim == width, name
        assert stem.specs.cross_attn.modality_embed_dim == width, name
    trunk_stage = cfg.model.pipeline.stages[1]
    assert trunk_stage.embed_dim == width
    assert trunk_stage.trunk.embed_dim == width
    assert trunk_stage.trunk.attn_target.embed_dim == width
    denoiser = cfg.model.pipeline.stages[4]
    assert denoiser.condition_input_dim == width
    assert denoiser.model.cond_dim == width


def test_action_shape_agrees_across_the_head(cfg):
    horizon, dim = cfg.hpt.action_horizon, cfg.hpt.action_dim
    noising, denoiser = cfg.model.pipeline.stages[3], cfg.model.pipeline.stages[4]
    assert (noising.action_horizon, noising.action_dim) == (horizon, dim)
    assert (denoiser.action_horizon, denoiser.action_dim) == (horizon, dim)
    assert denoiser.model.act_seq == horizon
    assert denoiser.model.act_dim == dim


def test_the_head_is_told_the_condition_needs_a_token_axis(cfg):
    # CrossTransformer cross-attends over the condition; the trunk emits a
    # pooled vector, so this flag is what reconciles them.
    assert cfg.model.pipeline.stages[4].condition_as_tokens is True


def test_action_key_matches_what_the_yam_transform_list_emits(cfg):
    assert cfg.model.pipeline.stages[2].action_key == "actions_cartesian"


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_the_configured_graph_lints_clean(mode):
    assert config_graph.build_graph(_EXPERIMENT_YAML, mode=mode)["lint"] == []


def test_inference_drops_the_training_only_stages():
    graph = config_graph.build_graph(_EXPERIMENT_YAML, mode="inference")
    assert {e["t"] for e in graph["skipped_stages"]} == {
        "ActionTargetBuilder",
        "FlowNoisingStage",
        "FlowVelocityLossStage",
    }
    assert "pred_action" in {k for n in graph["nodes"] for k in n["out"]}


def test_the_encoder_survives_into_inference():
    """Stems and trunk must NOT be mode-restricted: sampling needs them."""
    graph = config_graph.build_graph(_EXPERIMENT_YAML, mode="inference")
    names = [n["t"] for n in graph["nodes"]]
    assert names[:2] == ["HPTStemStage", "HPTTrunkStage"]


def _batch(cfg, size: int = 2) -> dict:
    return {
        "observations.images.front_img_1": torch.rand(size, 3, 480, 640),
        "observations.images.left_wrist_img": torch.rand(size, 3, 480, 640),
        "observations.images.right_wrist_img": torch.rand(size, 3, 480, 640),
        "observations.state.ee_pose": torch.rand(size, cfg.hpt.action_dim),
        "actions_cartesian": torch.rand(
            size, cfg.hpt.action_horizon, cfg.hpt.action_dim
        ),
        "embodiment": 7,
    }


def test_a_real_batch_trains_and_reaches_every_parameter(cfg, algo):
    algo.device = torch.device("cpu")
    algo.nets.to("cpu")
    batch = {"yam_bimanual": _batch(cfg)}
    out = algo.forward_training(batch)
    loss = algo.compute_losses(out, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    trainable = [p for p in algo.nets.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None for p in trainable), (
        "some HPT parameters got no gradient, so a stage is detached"
    )


def test_the_token_count_is_modalities_times_latents_plus_the_action_token(cfg, algo):
    algo.device = torch.device("cpu")
    algo.nets.to("cpu")
    out = algo.forward_training({"yam_bimanual": _batch(cfg)})["yam_bimanual"]
    latents = cfg.hpt.stem_specs.cross_attn.crossattn_latent
    modalities = len(cfg.model.pipeline.stages[0].stems)
    assert out["hpt/tokens"].shape == (2, modalities * latents, cfg.hpt.embed_dim)
    assert out["condition"].shape == (2, cfg.hpt.embed_dim)


def test_inference_samples_an_action_chunk(cfg, algo):
    algo.device = torch.device("cpu")
    algo.nets.to("cpu")
    observations = {k: v for k, v in _batch(cfg).items() if k != "actions_cartesian"}
    out = algo.forward_eval({"yam_bimanual": observations})["yam_bimanual"]
    assert out["pred_action"].shape == (2, cfg.hpt.action_horizon, cfg.hpt.action_dim)
    assert torch.isfinite(out["pred_action"]).all()


def test_the_dp_counterpart_still_uses_diffusion(cfg):
    """The two abc experiments must stay distinct architectures."""
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        dp = compose(
            config_name="train_zarr_cartesian",
            overrides=["+experiment=abc/yam_fstshirt_dp", "++paths.root_dir=."],
        )
    dp_stages = [s._target_.rsplit(".", 1)[-1] for s in dp.model.pipeline.stages]
    hpt_stages = [s._target_.rsplit(".", 1)[-1] for s in cfg.model.pipeline.stages]
    assert "DiffusionDenoiserStage" in dp_stages
    assert "FlowDenoiserStage" in hpt_stages
    assert "HPTStemStage" not in dp_stages
    # Same data, so a comparison between them isolates the architecture.
    assert dp.data._target_ == cfg.data._target_
