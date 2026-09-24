"""Real loader, normalizer, graph and optimizer steps across retained sources."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.pipeline.inference_config import (
    build_inference_config,
    export_configured_inference_artifact,
)
from egomimic.pipeline.inference_session import InferenceSession
from egomimic.pl_utils.pl_model import ModelWrapper
from tests.fixtures.synthetic_episodes import write_episode
from tests.test_pi05_graph import tiny_openpi as _tiny_openpi

tiny_openpi = _tiny_openpi

CONFIGS = Path(__file__).parents[1] / "egomimic/hydra_configs"


def recipe(vendor, family):
    model = f"hpt_bc_flow_{vendor}" if family == "hpt" else f"pi0.5_bc_{vendor}"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"model={model}", f"data={vendor}"],
        )


def small_cpu_model(config, family):
    config = deepcopy(config)
    with open_dict(config.model):
        for stage in config.model.pipeline.stages:
            if family == "pi":
                stage.policy.config.pytorch_weight_path = None
                stage.policy.config.pytorch_training_precision = "float32"
            if "trunk" in stage:
                stage.trunk.num_blocks = 1
            if "model" in stage:
                stage.model.nblocks = 1
                stage.num_inference_steps = 2
            for key in ("stems", "domain_stems"):

                def disable_download(node):
                    if not OmegaConf.is_dict(node):
                        return
                    if "encoder" in node:
                        with open_dict(node.encoder):
                            node.encoder.weights = None
                    for value in node.values():
                        disable_download(value)

                if key in stage:
                    disable_download(stage[key])
    return config


@pytest.mark.parametrize("vendor", ["eva", "aria", "mecka", "scale"])
@pytest.mark.parametrize("family", ["hpt", "pi"])
def test_two_optimizer_steps_and_strict_roundtrip(
    vendor, family, tiny_openpi, tmp_path
):
    config = recipe(vendor, family)
    domain = "eva_bimanual" if vendor == "eva" else "human_bimanual"
    for index in range(4):
        write_episode(tmp_path, vendor, T=8, H=32, W=32, seed=index)
    dataset = config.data.train_datasets[domain]
    with open_dict(dataset):
        dataset.resolver._target_ = (
            "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
        )
        dataset.resolver.folder_path = str(tmp_path)
        dataset.resolver.key_map.annotation_key = "annotations"
        dataset.filters = {
            "_target_": "egomimic.rldb.filters.DatasetFilter",
            "episode_hashes": [f"{vendor}_{i:02}" for i in range(3)],
        }
        dataset.bounds_check = False
    valid = deepcopy(dataset)
    valid.filters.episode_hashes = [f"{vendor}_03"]
    # Materialize the old train/valid alias before separating the held-out set.
    with open_dict(config.data):
        config.data.valid_datasets = {domain: valid}
        config.data.train_dataloader_params = {
            domain: {"batch_size": 2, "num_workers": 0}
        }
        config.data.valid_dataloader_params = {
            domain: {"batch_size": 2, "num_workers": 0}
        }
    dm = instantiate(config.data, _recursive_=False)
    context = dm.prepare_context(
        mode="train",
        normalization={"norm_mode": "quantile", "num_workers": 0},
        normalizer=None,
    )
    assert not (
        set(dm.train_datasets[domain].datasets)
        & set(dm.valid_datasets[domain].datasets)
    )
    assert dm.valid_datasets[domain].norm_stats is context.normalizer.norm_stats
    cpu = small_cpu_model(config, family)
    wrapper = ModelWrapper(config_tree=cpu)
    graph = wrapper.model
    context.bind(graph)
    wrapper.data_context = context
    optimizer = instantiate(config.model.optimizer)(params=wrapper.parameters())
    before = [
        parameter.detach().clone()
        for parameter in wrapper.parameters()
        if parameter.requires_grad
    ]
    loader = iter(dm.train_dataloader().iterables[domain])
    for _ in range(2):
        batch = graph.process_batch_for_training({domain: next(loader)})
        optimizer.zero_grad()
        predictions = graph.forward_training(batch)
        loss = graph.compute_losses(predictions, batch)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        parameters = [p for p in wrapper.parameters() if p.requires_grad]
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in parameters
        )
        optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, parameters, strict=True))
    graph.nets.eval()
    observed = graph.process_batch_for_training({domain: next(loader)})
    observed[domain].pop("actions_cartesian")
    output = graph.forward_eval(observed)[domain]["pred_action"]
    assert output.shape == (2, 100, 14 if vendor == "eva" else 12)
    assert torch.isfinite(output).all()
    saved = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(saved)
    restored = instantiate(cpu.model.pipeline, device="cpu")
    context.bind(restored)
    strict_load_pipeline_checkpoint(restored, saved)
    for name, tensor in graph.nets.state_dict().items():
        torch.testing.assert_close(restored.nets.state_dict()[name], tensor)
    assert build_inference_config(config)["status"] == "ready"
    with open_dict(cpu):
        cpu.inference_config.output_path = str(tmp_path / "inference-config.yaml")
    export_configured_inference_artifact(cpu, data_context=context)
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(saved, checkpoint_path)
    identity = int(observed[domain]["embodiment"][0])
    session = InferenceSession.load(
        cpu,
        checkpoint_path=checkpoint_path,
        context_path=tmp_path / "data-context.json",
        identity=identity,
        normalize_inputs=False,
    )
    controls = session.inference_controls()
    session.apply_inference_overrides({"inference_steps": 2})
    actions = session.predict(observed[domain])
    assert tuple(actions.shape[1:]) == tuple(cpu.model.inference.output.shape)
    assert torch.isfinite(actions).all()
    assert session.execution_plan(actions).shape[1] == controls["replan_every"]["value"]
