"""Exercise real language stems/data/overlays with only the HF backend replaced."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from tests.fixtures.synthetic_episodes import write_episode
from tests.test_retained_recipe_steps import CONFIGS, small_cpu_model


class TextBackend(torch.nn.Module):
    config = SimpleNamespace(hidden_size=32)

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 32)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.mark.parametrize("variant", ["pooled", "pertoken"])
def test_language_optimizer_and_actual_annotation_overlay(
    variant, monkeypatch, tmp_path
):
    prompts_seen = []

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            prompts_seen.extend(prompts)
            ids = torch.tensor(
                [[ord(c) % 64 for c in (text + "     ")[:5]] for text in prompts]
            )
            return BatchEncoding(
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            )

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **kw: TextBackend())
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        config = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"model=hpt_bc_pickplace_qwen_{variant}",
                "data=bc_pickplace_eva_qwen",
                "evaluator=eval_hpt",
                "evaluator/viz@evaluator.viz_func=cotrain_lang",
            ],
        )
    for index in range(3):
        write_episode(tmp_path, "eva", T=8, H=64, W=64, seed=index)
    dataset = config.data.train_datasets.eva_bimanual
    with open_dict(dataset):
        dataset.resolver._target_ = (
            "egomimic.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver"
        )
        dataset.resolver.folder_path = str(tmp_path)
        dataset.filters = None
        dataset.bounds_check = False
    with open_dict(config.data):
        config.data.valid_datasets = {"eva_bimanual": deepcopy(dataset)}
        config.data.train_dataloader_params = {
            "eva_bimanual": {"batch_size": 2, "num_workers": 0}
        }
        config.data.valid_dataloader_params = {
            "eva_bimanual": {"batch_size": 2, "num_workers": 0}
        }
    config.evaluator.rkl_samples = 1
    dm = instantiate(config.data, _recursive_=False)
    context = dm.prepare_context(mode="train", normalization={"num_workers": 0})
    evaluator = instantiate(config.evaluator)
    dm.configure_evaluation(evaluator.data_requirements())
    graph = instantiate(small_cpu_model(config, "hpt").model.pipeline, device="cpu")
    context.bind(graph, evaluator)
    source = next(iter(dm.train_dataloader().iterables["eva_bimanual"]))
    batch = graph.process_batch_for_training({"eva_bimanual": source})
    optimizer = torch.optim.Adam(graph.nets.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad()
        loss = graph.compute_losses(graph.forward_training(batch), batch)["loss"]
        loss.backward()
        stem = graph.pipeline.stage_by_id("stems").stems["observations__annotation"]
        assert stem.proj.weight.grad.abs().sum() > 0
        assert stem.encoder.embedding.weight.grad is None
        optimizer.step()
    assert prompts_seen and all(
        text in ("pick up the red cube", "place it in the bin") for text in prompts_seen
    )
    graph.nets.eval()
    # Empty annotations must retain one prompt per sample, including at inference.
    empty = deepcopy(batch)
    empty["eva_bimanual"]["annotations"] = [[], []]
    assert graph.forward_eval(empty)["eva_bimanual"]["observations.annotation"] == [
        "",
        "",
    ]
    painted = []
    import egomimic.rldb.embodiment.embodiment as renderer

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
    evaluator.on_validation_step(batch, 0)
    assert evaluator._frame_records, "Language validation must actually render frames"
    assert painted and any(
        "pick up the red cube" in str(text) or "place it in the bin" in str(text)
        for text in painted
    )
    evaluator.on_validation_end()
    assert list(
        tmp_path.rglob("*.mp4")
    ), "The language overlay must survive video encoding"
