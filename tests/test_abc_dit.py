"""ABC-DiT graph: both multitask recipes compose, and a tiny model trains/samples."""

from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir

from egomimic.pipeline.stages_flow import FlowDenoiserStage

_CONFIGS = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
_BASELINE = "abc_arc/robot_bc/abc_multitask4_abcdit_baseline_visual_openloop"
_HYBRID = "abc_arc/robot_bc/abc_multitask4_abcdit_hybrid_visual_openloop"
# Keeps the DiT tiny; the ViT-B/16 backbone shape is fixed by the checkpoint.
_TINY = [
    "abc_dit.hidden_size=64",
    "abc_dit.depth=2",
    "abc_dit.num_heads=4",
    "abc_dit.backbone_checkpoint=null",
    "abc_dit.text_checkpoint=null",
]


def _cfg(experiment, extra=()):
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}", "++paths.root_dir=.", *extra],
        )


@pytest.mark.parametrize(
    "experiment, horizon, representation",
    [(_BASELINE, 100, "cartesian"), (_HYBRID, 200, "hybrid_arc_tokenizer_cartesian")],
)
def test_recipes_compose_with_the_right_action_contract(
    experiment, horizon, representation
):
    cfg = _cfg(experiment)
    stages = cfg.model.pipeline.stages
    assert [s._target_.rsplit(".", 1)[1] for s in stages] == [
        "ABCObservationStage",
        "ActionTargetBuilder",
        "FlowNoisingStage",
        "FlowDenoiserStage",
        "FlowVelocityLossStage",
    ]
    assert stages[2].action_horizon == stages[3].action_horizon == horizon
    assert stages[3].model.action_horizon == horizon
    assert stages[3].condition_input_dim == cfg.abc_dit.hidden_size
    assert cfg.run_provenance.action_contract.representation == representation
    assert cfg.abc_dit.backbone_checkpoint.endswith(
        "dinov3_vitb16_pretrain_lvd1689m.pth"
    )
    assert cfg.abc_dit.text_checkpoint.endswith("clip-vit-base-patch32")
    assert stages[0].task_key == "task"
    assert cfg.data.train_datasets.yam_bimanual.resolver.key_map.task_key == "task"


def test_a_tiny_hybrid_graph_trains_and_samples():
    cfg = _cfg(_HYBRID, _TINY)
    algo = hydra.utils.instantiate(cfg.model.pipeline)
    algo.device = torch.device("cpu")
    algo.nets.to("cpu")
    # Stand-in for frozen CLIP text: one fixed 512-d vector per prompt.
    prompts = {}
    algo.pipeline.stages[0].encoder.text_encoder = lambda texts: torch.stack(
        [prompts.setdefault(t, torch.randn(512)) for t in texts]
    )
    horizon, dim = cfg.abc_dit.action_horizon, cfg.abc_dit.action_dim
    obs = {
        "observations.images.front_img_1": torch.rand(2, 3, 480, 640),
        "observations.images.left_wrist_img": torch.rand(2, 3, 480, 640),
        "observations.images.right_wrist_img": torch.rand(2, 3, 480, 640),
        "observations.state.ee_pose": torch.rand(2, dim),
        "task": ["fold and stack the towels", "sort stationery into containers"],
        "embodiment": 7,
    }
    batch = {"yam_bimanual": {**obs, "actions_cartesian": torch.rand(2, horizon, dim)}}
    out = algo.forward_training(batch)
    loss = algo.compute_losses(out, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    dit = next(s for s in algo.pipeline.stages if isinstance(s, FlowDenoiserStage))
    assert all(p.grad is not None for p in dit.model.parameters())

    sampled = algo.pipeline.execute(dict(obs), mode="inference")
    assert sampled["pred_action"].shape == (2, horizon, dim)


def test_the_task_prompt_is_read_from_episode_metadata_but_not_normalized():
    from egomimic.rldb.embodiment.yam import Yam

    key_map = Yam.get_keymap("cartesian", task_key="task")
    assert key_map["task"] == {
        "key_type": "episode_metadata",
        "zarr_key": "task_description",
    }
    assert "task" not in Yam.get_keymap("cartesian", norm_mode=True, task_key="task")
