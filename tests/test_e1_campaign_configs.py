from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


CONFIGS = Path(__file__).parents[1] / "egomimic/hydra_configs"
EXPERIMENTS = sorted(
    path.stem for path in (CONFIGS / "experiment/e1").glob("*.yaml")
    if not path.stem.endswith("_base")
)


def config(experiment):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name="train_zarr_cartesian", overrides=[
            f"+experiment=e1/{experiment}", "e1.spread=smoke",
            "e1.train_root=/tmp/e1/train", "e1.valid_root=/tmp/e1/valid",
            "evaluator.results_path=/tmp/e1/results.json", "e1.image_weights=null",
        ])


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_campaign_model_matches_codec(experiment):
    cfg = config(experiment)
    model = OmegaConf.to_container(cfg.model, resolve=True)
    evaluator = instantiate(cfg.evaluator)
    assert evaluator.variant == cfg.e1.variant
    expected = (101, 14) if cfg.e1.variant == "arcmean" else (
        100, 14 if cfg.e1.variant == "time" else 16
    )
    noising = model["pipeline"]["stages"][3]
    head = model["pipeline"]["stages"][4]["model"]
    assert (noising["action_horizon"], noising["action_dim"]) == expected
    assert (head["act_seq"], head["act_dim"]) == expected
    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        for dataset in split.values():
            instantiate(dataset.resolver.key_map)
            instantiate(dataset.resolver.transform_list)


@pytest.mark.parametrize("experiment,embodiment", [("fold_time", 3), ("abc_arcmean", 7), ("abcs_arclogdur", 7)])
def test_graph_training_and_inference(experiment, embodiment):
    torch.set_num_threads(1)
    cfg = config(experiment)
    cfg.hpt.num_blocks = 1
    cfg.hpt.embed_dim = 32
    cfg.hpt.stem_specs.cross_attn.crossattn_latent = 2
    cfg.hpt.stem_specs.cross_attn.crossattn_dim_head = 8
    cfg.model.pipeline.stages[4].num_inference_steps = 2
    cfg.model.pipeline.stages[4].model.nblocks = 1
    policy = instantiate(cfg.model.pipeline)
    row = {
        "embodiment": torch.tensor([embodiment, embodiment]),
        "observations.state.ee_pose": torch.randn(2, 1, 14),
        "observations.images.front_img_1": torch.rand(2, 1, 3, 32, 32),
        "actions_cartesian": torch.randn(2, cfg.e1.action_horizon, cfg.e1.action_dim),
    }
    if experiment.startswith("abcs_"):
        for camera in ["left_wrist_img", "right_wrist_img"]:
            row["observations.images." + camera] = torch.rand(2, 1, 3, 32, 32)
    batch = {"opaque-source": row}
    result = policy.forward_training(batch)
    loss = policy.compute_losses(result, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    policy.nets.eval()
    prediction = policy.forward_eval(batch)["opaque-source"]["pred_action"]
    assert prediction.shape == row["actions_cartesian"].shape
    assert torch.isfinite(prediction).all()


def test_full_validation_fraction_is_preserved():
    cfg = config("abc_time")
    cfg.evaluator.limit_val_batches = 1.0
    assert type(instantiate(cfg.evaluator).override_dict["limit_val_batches"]) is float
