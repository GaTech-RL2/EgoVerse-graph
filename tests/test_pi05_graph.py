"""Graph integration with the actual campaign adapter and a tiny OpenPI backend.

Weights, tokenizer downloads and robot/data access are deliberately unnecessary.
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.campaigns.pi05.data import PI05Dataset
from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.pl_utils.pl_model import ModelWrapper

CONFIGS = Path(__file__).parents[1] / "egomimic/hydra_configs"


def config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(
            config_name="train_zarr_cartesian_pi",
            overrides=["pi05.pretrained_weights=null", *overrides],
        )


@pytest.mark.parametrize(
    "group,name",
    [
        (group, path.stem)
        for group in ["model", "data", "evaluator"]
        for path in sorted((CONFIGS / group / "pi05").glob("*.yaml"))
    ],
)
def test_pi_configs_resolve_without_openpi(group, name):
    cfg = config([f"{group}=pi05/{name}"])
    instantiate(cfg.model.pipeline)  # backend remains deferred
    instantiate(cfg.evaluator)
    for ds in cfg.data.train_datasets.values():
        if ds is None:
            continue
        instantiate(ds.resolver.key_map)
        instantiate(ds.resolver.transform_list)


@pytest.fixture
def tiny_openpi(monkeypatch):
    modules = {}
    for name in [
        "openpi",
        "openpi.models",
        "openpi.models.pi0_config",
        "openpi.models_pytorch",
        "openpi.models_pytorch.pi0_pytorch",
        "openpi.shared",
        "openpi.shared.image_tools",
    ]:
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    modules["openpi.models.pi0_config"].Pi0Config = lambda **kwargs: SimpleNamespace(
        **kwargs
    )

    class Network(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.config = config

        def forward(self, observation, action):
            assert not torch.is_inference_mode_enabled()
            assert action.shape[-1] == 32
            assert observation.tokenized_prompt.shape[0] == len(action)
            return (action - self.weight).square().mean()

        def sample_actions(self, device, observation, noise, num_steps):
            assert not torch.is_inference_mode_enabled()
            return torch.zeros(
                len(observation.state), self.config.action_horizon, 32, device=device
            )

    modules["openpi.models_pytorch.pi0_pytorch"].PI0Pytorch = Network
    modules["openpi.shared.image_tools"].resize_with_pad_torch = lambda image, h, w: (
        torch.nn.functional.interpolate(image, (h, w))
    )

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            return {
                "input_ids": torch.ones(len(prompts), 8, dtype=torch.long),
                "attention_mask": torch.ones(len(prompts), 8, dtype=torch.long),
            }

    # Reload so this fixture cannot leave a previous fixture's backend attached.
    sys.modules.pop("egomimic.campaigns.pi05.policy", None)
    module = importlib.import_module("egomimic.campaigns.pi05.policy")
    monkeypatch.setattr(
        module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer()
    )
    yield
    sys.modules.pop("egomimic.campaigns.pi05.policy", None)


def normalizer():
    norm = PI05Dataset(state={}, norm_mode="quantile")
    kinds = {
        "actions_cartesian": "action_keys",
        "observations.state.ee_pose": "proprio_keys",
        "base_0_rgb": "camera_keys",
        "annotations": "annotation_keys",
    }
    norm.key_types = {6: kinds}
    norm.zarr_keys = {6: {k: k for k in kinds}}
    norm.shapes = {
        6: {"actions_cartesian": (100, 20), "observations.state.ee_pose": (20,)}
    }
    norm.norm_stats = {
        6: {
            key: {"quantile_1": np.zeros(shape), "quantile_99": np.full(shape, 10.0)}
            for key, shape in norm.shapes[6].items()
        }
    }
    norm.embodiments = {6}
    return norm


def batch():
    return {
        "opaque-input": {
            "embodiment": torch.tensor([6, 6]),
            "base_0_rgb": torch.rand(2, 3, 32, 32),
            "observations.state.ee_pose": torch.zeros(2, 20),
            "actions_cartesian": torch.zeros(2, 100, 20),
            "annotations": [["Sort pens"], ["Sort rulers"]],
        }
    }


def test_pi_training_inference_and_strict_checkpoint(tiny_openpi, tmp_path):
    cfg = config()
    graph = instantiate(cfg.model.pipeline)
    with pytest.raises(RuntimeError, match="Bind PI05Stage"):
        graph.forward_training(batch())
    norm = normalizer()
    graph.bind_data_context(normalizer=norm)
    stage = graph.pipeline.stages[0]
    params = list(graph.nets.parameters())
    assert params and params[0] is stage.backend.model.weight
    wrapper = ModelWrapper(pipeline=graph)
    train = graph.forward_training(batch())
    loss = graph.compute_losses(train, batch())["loss"]
    loss.backward()
    assert params[0].grad is not None and torch.isfinite(params[0].grad)
    inference = batch()
    inference["opaque-input"].pop("actions_cartesian")
    result = graph.forward_eval(inference)["opaque-input"]["pred_action"]
    assert result.shape == (2, 100, 20)
    torch.testing.assert_close(result, torch.zeros_like(result))
    saved = {"state_dict": wrapper.state_dict()}
    restored = instantiate(cfg.model.pipeline)
    restored.bind_data_context(normalizer=norm)
    strict_load_pipeline_checkpoint(restored, saved)
    torch.testing.assert_close(next(restored.nets.parameters()), params[0])
    path = tmp_path / "source.ckpt"
    torch.save(
        {
            "state_dict": {
                "nets." + k: v for k, v in stage.policy_nets.state_dict().items()
            }
        },
        path,
    )
    stage.load_initial_weights(path)
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="complete matching"):
        stage.load_initial_weights(path)
    with pytest.raises(RuntimeError, match="different normalizer"):
        graph.bind_data_context(normalizer=normalizer())


def test_pi_evaluator_works_through_lightning_wrapper(tiny_openpi, tmp_path):
    cfg = config()
    norm = normalizer()
    graph = instantiate(cfg.model.pipeline)
    graph.bind_data_context(normalizer=norm)
    graph.pipeline.stages[0].backend.rkl_samples = 1
    wrapper = ModelWrapper(pipeline=graph)
    logged = []
    logger = SimpleNamespace(
        device=torch.device("cpu"),
        log_dict=lambda metrics, **kwargs: logged.append(metrics),
    )
    evaluator = instantiate(cfg.evaluator)
    evaluator.trainer = SimpleNamespace(
        current_epoch=0,
        default_root_dir=str(tmp_path),
        is_global_zero=True,
        world_size=1,
        lightning_module=logger,
    )
    evaluator.model = wrapper
    evaluator.bind_data_context(normalizer=norm)
    evaluator.on_validation_start()
    # Zero normalized output and targets must give zero native MSE. Revert lists
    # are disabled here; their numeric round-trip is covered by the source tests.
    evaluator.options["transform_lists"] = {}
    values = evaluator.on_validation_step(batch(), 0)
    assert values["Valid/eva_bimanual_actions_cartesian_paired_mse_avg"] == 0
    assert len(logged) == 1
    evaluator.set_validation_group("train_viz")
    train_values = evaluator.on_validation_step(batch(), 0)
    assert all(key.startswith("train_viz/") for key in train_values)
    evaluator.on_validation_end()
