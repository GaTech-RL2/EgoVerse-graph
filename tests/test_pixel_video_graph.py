"""Real diffusion graph contracts, shared weights, and legacy numerical parity."""

import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.pixel_diffusion import (
    LinearDDIMImageSampler,
    LinearGaussianImageNoising,
    PlainPixelMSE,
)
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_pixel_video import (
    PixelVideoDenoiserStage,
    PixelVideoLossStage,
    PixelVideoNoisingStage,
    PixelVideoSamplerStage,
)
from tools.config_graph import build_graph

IDS = {
    "pixel_noising": 0,
    "pixel_denoiser": 1,
    "pixel_loss": 2,
    "pixel_sampler": 3,
    "image_proprio": 1,
}


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_batch():
    image = {
        "value": torch.linspace(-1, 1, 2 * 4 * 12).reshape(2, 4, 12),
        "time": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]]),
        "index": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]]),
        "valid": torch.tensor([[True, True, True, True], [True, True, True, False]]),
        "observed": torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        ),
        "xy": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]).repeat(
            2, 1, 1
        ),
        "camera": torch.tensor([[0, 1, 0, 1], [0, 1, 0, 0]]),
    }
    image["value"][1, -1] = float("nan")
    return {"world": {"image": image}, "embodiment": ["robot", "robot"]}


def small_model(mode="continuous", reduction="mean_per_example"):
    return ImageProprioModel(
        objective="visual_only",
        phase_a_condition_on_proprio=False,
        image_dim=12,
        width=16,
        depth=1,
        heads=2,
        mlp_ratio=2,
        max_cameras=2,
        max_tokens=32,
        time_mode=mode,
        loss_reduction=reduction,
    )


def pipeline(model, *, loss=None, steps=3, prediction_key="image_proprio_prediction"):
    return Pipeline(
        [
            PixelVideoNoisingStage(LinearGaussianImageNoising()),
            PixelVideoDenoiserStage(model),
            PixelVideoLossStage(loss or PlainPixelMSE(reduction=model.loss_reduction)),
            PixelVideoSamplerStage(LinearDDIMImageSampler(steps), prediction_key),
        ],
        stage_ids=IDS,
    )


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
@pytest.mark.parametrize("reduction", ["token_mean", "mean_per_example"])
def test_split_training_matches_legacy_loss_gradients_and_rng(mode, reduction):
    model = small_model(mode, reduction)
    legacy = copy.deepcopy(model)
    graph = pipeline(model)
    batch = fixture_batch()
    torch.manual_seed(19)
    expected = legacy.training_losses(batch["world"], "robot")["total"]
    expected_rng = torch.get_rng_state().clone()
    expected.backward()
    torch.manual_seed(19)
    output = graph.execute(batch, mode="train")
    actual = sum_losses(output)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    actual.backward()
    for (name, actual_parameter), (expected_name, expected_parameter) in zip(
        model.named_parameters(), legacy.named_parameters()
    ):
        assert name == expected_name
        if expected_parameter.grad is None:
            assert actual_parameter.grad is None
        else:
            torch.testing.assert_close(
                actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0
            )
    assert "optimizer_step" not in batch
    assert "image_proprio_prediction" not in output
    assert output["pixel_video/denoiser"] is model
    assert set(batch) == {"world", "embodiment"}
    assert not any(key.startswith("pixel_video/") for key in batch)


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_sampler_matches_legacy_and_uses_one_registered_model(mode):
    model = small_model(mode)
    graph = pipeline(model)
    batch = fixture_batch()
    keys_before = tuple(graph.state_dict())
    assert keys_before and all(key.startswith("stages.1.model.") for key in keys_before)
    assert sum(p.numel() for p in graph.parameters()) == sum(
        p.numel() for p in model.parameters()
    )
    assert graph.stage_by_id("image_proprio") is graph.stage_by_id("pixel_denoiser")
    torch.manual_seed(23)
    expected = model.sample(batch["world"], "robot", steps=3)["image"]
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(23)
    output = graph.execute(batch, mode="inference")
    actual = output["image_proprio_prediction"]["image"]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    image = batch["world"]["image"]
    assert torch.equal(actual[image["observed"]], image["value"][image["observed"]])
    assert not actual[~image["valid"]].any()
    assert tuple(graph.state_dict()) == keys_before
    assert not any(key.startswith("loss/") for key in output)
    restored = pipeline(small_model(mode))
    restored.load_state_dict(graph.state_dict(), strict=True)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, graph.state_dict()[key])


def test_loss_is_replaceable_and_receives_noise_time_and_masks():
    class RecordingObjective:
        def __call__(self, prediction, target, mask, **kwargs):
            self.received = prediction, target, mask, kwargs
            return PlainPixelMSE("mean_per_example")(prediction, target, mask, **kwargs)

    loss = RecordingObjective()
    graph = pipeline(small_model(), loss=loss)
    output = graph.execute(fixture_batch(), mode="train")
    prediction, target, mask, kwargs = loss.received
    assert prediction is output["pixel_video/prediction"]
    assert target is output["pixel_video/target"]
    assert mask is output["pixel_video/loss_mask"]
    assert kwargs["noise_levels"] is output["pixel_video/noise_levels"]
    metadata = kwargs["metadata"]
    assert metadata is output["pixel_video/noise_metadata"]
    assert set(metadata) == {"timestamps", "valid", "observed"}
    assert torch.equal(mask, metadata["valid"] & ~metadata["observed"])
    assert not kwargs["noise_levels"][metadata["observed"]].any()
    assert kwargs["check_target"] is False
    sum_losses(output).backward()


def test_sampler_can_publish_intermediate_for_separate_readout():
    graph = pipeline(small_model(), prediction_key="pixel_video/sampled_prediction")
    output = graph.execute(fixture_batch(), mode="inference")
    assert "image_proprio_prediction" not in output
    assert set(output["pixel_video/sampled_prediction"]) == {"image"}
    assert graph.stage_by_id("pixel_sampler").contract("inference")[1] == (
        "pixel_video/sampled_prediction",
    )


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_actual_recipe_fresh_canonical_graph_has_real_diffusion_edges(tmp_path, mode):
    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment=pixel_proprio/video_{mode}",
                "hydra/launcher=basic",
                "data.manifest_path=/fixture/manifest.json",
                "paths.output_dir=/fixture/output",
                "paths.work_dir=/fixture/source",
                "+pixel_execution=ddp4_prefetch",
                "+world_model_execution=regional_compile",
            ],
        )
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert plain["model"]["pipeline"]["stage_ids"] == IDS
    stages = plain["model"]["pipeline"]["stages"]
    assert all(
        "warmup_steps" not in stage and "pose_ramp_steps" not in stage
        for stage in stages
    )
    assert stages[2]["loss"]["reduction"] == "mean_per_example"
    assert stages[3]["sampler"]["steps"] == 20
    assert stages[1]["model"]["max_tokens"] == (3200 if mode == "discrete" else 25600)
    # The canonical builder's generic seed inventory needs only actual dataset
    # outputs. Stage-one no longer reads injected optimizer-step metadata.
    plain["data"]["train_datasets"] = {
        "libero_gripper": {"batch_keys": ["world", "embodiment"]}
    }
    config_path = tmp_path / f"{mode}.yaml"
    OmegaConf.save(OmegaConf.create(plain), config_path)
    with torch.device("meta"):
        training = build_graph(config_path, mode="train")
        inference = build_graph(config_path, mode="inference")
    assert training["lint"] == inference["lint"] == []
    assert [node["t"] for node in training["nodes"]] == [
        "PixelVideoNoisingStage",
        "PixelVideoDenoiserStage",
        "PixelVideoLossStage",
    ]
    assert [node["t"] for node in inference["nodes"]] == [
        "PixelVideoDenoiserStage",
        "PixelVideoSamplerStage",
    ]
    assert {(edge["a"], edge["b"]) for edge in training["edges"]} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert inference["edges"] == [
        {"a": 0, "b": 1, "k": "pixel_video/denoiser", "s": "shared"}
    ]
    assert (
        set(training["seed_keys"])
        == set(inference["seed_keys"])
        == {"world", "embodiment"}
    )
