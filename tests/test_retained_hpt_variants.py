"""Retained multi-head, keypoint and interpolated-target behavior gates."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from egomimic.pipeline.inference_config import (
    build_inference_config,
    validate_model_data_context,
)
from egomimic.pipeline.sequence_decoder import UniformTimeGridDecoder
from tests.fixtures.synthetic_episodes import write_episode
from tests.test_retained_recipe_steps import CONFIGS, small_cpu_model


@pytest.mark.parametrize(
    "name",
    [
        "hpt_cotrain_flow_shared_head",
        "hpt_cotrain_flow_seperate_head",
        "hpt_cotrain_enc_dec_base",
    ],
)
def test_both_domains_train_and_decode(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        config = compose(
            config_name="train_zarr_cartesian", overrides=[f"model={name}"]
        )
    cpu = small_cpu_model(config, "hpt")

    def reduce_blocks(node):
        if OmegaConf.is_dict(node):
            for key in ("nblocks", "num_layers"):
                if key in node:
                    node[key] = 1
            if "num_inference_steps" in node:
                node.num_inference_steps = 2
            for value in node.values():
                reduce_blocks(value)
        elif OmegaConf.is_list(node):
            for value in node:
                reduce_blocks(value)

    reduce_blocks(cpu.model.pipeline)
    graph = instantiate(cpu.model.pipeline, device="cpu")
    optimizer = torch.optim.AdamW(graph.nets.parameters(), lr=1e-4)
    for embodiment, width, domain in [
        (6, 14, "eva_bimanual"),
        (3, 12, "human_bimanual"),
    ]:
        batch = {
            "actions_cartesian": torch.randn(2, 100, width),
            "observations.state.ee_pose": torch.randn(2, width),
            "embodiment": torch.tensor([embodiment, embodiment]),
        }
        for key in ("front_img_1", "left_wrist_img", "right_wrist_img"):
            batch[f"observations.images.{key}"] = torch.rand(2, 3, 32, 32)
        for _ in range(2):
            optimizer.zero_grad()
            predictions = graph.forward_training({domain: batch})
            loss = graph.compute_losses(predictions, {domain: batch})["loss"]
            assert torch.isfinite(loss)
            loss.backward()
            assert graph.pipeline.stage_by_id("trunk").action_token.grad is not None
            optimizer.step()
        graph.nets.eval()
        result = graph.forward_eval({domain: batch})[domain]["pred_action"]
        assert result.shape == (2, 100, width)
        assert torch.isfinite(result).all()
        config.model.deployment_domain = domain
        artifact = build_inference_config(config)
        assert artifact["status"] == "ready"


@pytest.mark.parametrize("wrist", [False, True])
def test_keypoints_real_transforms_and_overlay(wrist, tmp_path):
    name = "train_zarr_keypoint_wrist" if wrist else "train_zarr_keypoints"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name=name)
    for i in range(3):
        write_episode(tmp_path, "aria", T=8, H=32, W=32, seed=i)
    ds = cfg.data.train_datasets.human_bimanual
    with open_dict(ds):
        ds.resolver._target_ = (
            "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
        )
        ds.resolver.folder_path = str(tmp_path)
        ds.filters = None
        ds.bounds_check = False
    with open_dict(cfg.data):
        cfg.data.valid_datasets = {"human_bimanual": deepcopy(ds)}
        cfg.data.train_dataloader_params = {
            "human_bimanual": {"batch_size": 2, "num_workers": 0}
        }
        cfg.data.valid_dataloader_params = {
            "human_bimanual": {"batch_size": 2, "num_workers": 0}
        }
    dm = instantiate(cfg.data, _recursive_=False)
    context = dm.prepare_context(mode="train", normalization={"num_workers": 0})
    validate_model_data_context(cfg, context)
    mismatched = deepcopy(cfg)
    mismatched.model.data_requirements.preprocessing["3"].transforms.coord_frame = (
        "camframe" if wrist else "eef_frame"
    )
    with pytest.raises(ValueError, match="coord_frame"):
        validate_model_data_context(mismatched, context)
    evaluator = instantiate(cfg.evaluator)
    dm.configure_evaluation(evaluator.data_requirements())
    graph = instantiate(small_cpu_model(cfg, "hpt").model.pipeline, device="cpu")
    context.bind(graph, evaluator)
    evaluator.model = graph
    evaluator.trainer = SimpleNamespace(
        current_epoch=0,
        default_root_dir=str(tmp_path),
        is_global_zero=True,
        lightning_module=SimpleNamespace(log_dict=lambda *a, **k: None),
    )
    evaluator.on_validation_start()
    batch = graph.process_batch_for_training(next(iter(dm.val_dataloader()))[0])
    source = next(iter(batch.values()))
    assert source["actions_keypoints"].shape == (2, 100, 138)
    loss = graph.compute_losses(graph.forward_training(batch), batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    graph.nets.eval()
    metrics = evaluator.on_validation_step(batch, 0)
    assert "Valid/human_bimanual_actions_keypoints_paired_mse_avg" in metrics
    assert (
        evaluator._frame_records
    ), "The retained MANO overlay must actually produce frames"


def test_time_grid_preserves_duration_and_angle_wrap():
    decoder = UniformTimeGridDecoder([100, 2], 44 / (99 * 30), 45, 1 / 30, [1])
    native = np.stack([np.linspace(0, 44, 100), np.linspace(3.0, 3.4, 100)], axis=-1)[
        None
    ]
    native[..., 1] = (native[..., 1] + np.pi) % (2 * np.pi) - np.pi
    output = decoder(native)
    np.testing.assert_allclose(output[0, :, 0], np.arange(45), atol=1e-10)
    np.testing.assert_allclose(
        np.unwrap(output[0, :, 1]), np.linspace(3.0, 3.4, 45), atol=1e-10
    )
    with pytest.raises(ValueError, match="exceeds"):
        UniformTimeGridDecoder([100, 2], 44 / (99 * 30), 100, 1 / 30)
