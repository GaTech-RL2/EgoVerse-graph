"""Retained PI schedules, mixed-source normalization and painted predictions."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from egomimic.pipeline.inference_config import build_inference_config
from tests.fixtures.synthetic_episodes import write_episode
from tests.test_pi05_graph import tiny_openpi as _tiny_openpi
from tests.test_retained_recipe_steps import CONFIGS, small_cpu_model

tiny_openpi = _tiny_openpi


def test_retained_pi_schedules_match_pinned_source_and_base_fails_closed():
    source = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/integration/evidence/legacy-model-recipes.json"
        ).read_text()
    )["models"]
    for name, original in source.items():
        if not name.startswith("pi0.5"):
            continue
        with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
            cfg = compose(
                config_name="train_zarr_cartesian", overrides=[f"model={name}"]
            )
        for field in ("optimizer", "scheduler", "scheduler_interval"):
            value = cfg.model[field]
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            assert value == original[field]
        artifact = build_inference_config(cfg)
        if name == "pi0.5_base":
            assert artifact["status"] == "unsupported"
            with pytest.raises(Exception, match="incomplete base fragment"):
                instantiate(cfg.model.pipeline)
        else:
            assert artifact["status"] == "ready"


def test_cotrain_keeps_normalization_isolated_and_renders_prompt(
    tiny_openpi, monkeypatch, tmp_path
):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "model=pi0.5_cotrain_eva_aria",
                "data=cotrain_pi_base",
                "evaluator=eval_pi",
            ],
        )
    datasets = {}
    for vendor, domain in (("eva", "eva_bimanual"), ("aria", "human_bimanual")):
        root = tmp_path / vendor
        for index in range(3):
            write_episode(root, vendor, T=8, H=64, W=64, seed=index)
        ds = cfg.data.train_datasets[domain]
        with open_dict(ds):
            ds.resolver._target_ = (
                "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
            )
            ds.resolver.folder_path = str(root)
            ds.filters = None
            ds.mode = "total"
            ds.bounds_check = False
        datasets[domain] = deepcopy(ds)
    with open_dict(cfg.data):
        cfg.data.valid_datasets = datasets
        for field in ("train_dataloader_params", "valid_dataloader_params"):
            cfg.data[field] = {
                name: {"batch_size": 2, "num_workers": 0} for name in datasets
            }
    cfg.evaluator.rkl_samples = 1
    dm = instantiate(cfg.data, _recursive_=False)
    context = dm.prepare_context(mode="train", normalization={"num_workers": 0})
    assert context.normalizer.key_shape("actions_cartesian", 6) == (100, 14)
    assert context.normalizer.key_shape("actions_cartesian", 3) == (100, 12)
    evaluator = instantiate(cfg.evaluator)
    dm.configure_evaluation(evaluator.data_requirements())
    graph = instantiate(small_cpu_model(cfg, "pi").model.pipeline, device="cpu")
    context.bind(graph, evaluator)
    batch = graph.process_batch_for_training(
        {
            source: next(iter(loader))
            for source, loader in dm.train_dataloader().iterables.items()
        }
    )
    optimizer = instantiate(cfg.model.optimizer)(params=graph.nets.parameters())
    for _ in range(2):
        optimizer.zero_grad()
        predictions = graph.forward_training(batch)
        losses = graph.compute_losses(predictions, batch)
        torch.testing.assert_close(
            losses["loss"],
            torch.stack([value["loss/pi05"] for value in predictions.values()]).mean(),
        )
        losses["loss"].backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in graph.nets.parameters()
            if p.requires_grad
        )
        optimizer.step()
    graph.nets.eval()
    import egomimic.rldb.embodiment.embodiment as renderer

    painted = []
    original = renderer._viz_annotations

    def paint(*args, **kwargs):
        painted.append(kwargs.get("annotations"))
        return original(*args, **kwargs)

    monkeypatch.setattr(renderer, "_viz_annotations", paint)
    evaluator.model = graph
    evaluator.trainer = SimpleNamespace(
        current_epoch=0,
        default_root_dir=str(tmp_path),
        is_global_zero=True,
        lightning_module=SimpleNamespace(log_dict=lambda *a, **kw: None),
    )
    evaluator.on_validation_start()
    metrics = evaluator.on_validation_step(batch, 0)
    assert metrics and evaluator._frame_records
    assert any(
        "pick up the red cube" in str(text) or "place it in the bin" in str(text)
        for text in painted
    )
    evaluator.on_validation_end()
    assert len(list(tmp_path.rglob("*.mp4"))) >= 2
