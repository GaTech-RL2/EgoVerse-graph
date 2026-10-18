"""Explicit readout graph, strict initialization and frozen-image AdamW behavior."""

import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError

from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.pixel_diffusion import (
    LinearDDIMImageSampler,
    LinearGaussianImageNoising,
    PlainPixelMSE,
)
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_pixel_readout import (
    PixelReadoutDenoiserStage,
    PixelReadoutImageDenoiserStage,
    PixelReadoutImageLossStage,
    PixelReadoutMSEStage,
    PixelReadoutNoisingStage,
    PixelStateActionReadoutStage,
)
from egomimic.pipeline.stages_pixel_video import (
    PixelVideoNoisingStage,
    PixelVideoSamplerStage,
)
from scripts.world_model.initialize_pixel_readout import (
    initialize_readout,
    sha256_file,
    transfer_shared_weights,
    validate_transition,
)
from tools.config_graph import build_graph

IDS = {
    "pixel_noising": 0,
    "pixel_denoiser": 1,
    "pixel_loss": 2,
    "readout_noising": 3,
    "readout_denoiser": 4,
    "pixel_sampler": 5,
    "state_action_readout": 6,
    "measured_state_loss": 7,
    "control_command_loss": 8,
    "image_proprio": 1,
}
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"


@pytest.fixture(autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def arguments(mode="continuous"):
    return dict(
        objective="visual_state_action",
        phase_a_condition_on_proprio=False,
        pose_dims={"robot": 3},
        action_dims={"robot": 2},
        image_dim=12,
        width=16,
        depth=2,
        heads=2,
        mlp_ratio=2,
        max_cameras=2,
        max_tokens=16,
        time_mode=mode,
        loss_reduction="mean_per_example",
    )


def fixture_batch():
    image = {
        "value": torch.linspace(-1, 1, 2 * 4 * 12).reshape(2, 4, 12),
        "time": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
        "index": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        "valid": torch.tensor([[True, True, True, True], [True, True, True, False]]),
        "observed": torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        ),
        "xy": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]).repeat(
            2, 1, 1
        ),
        "camera": torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
    }
    image["value"][1, -1] = float("nan")
    pose = {
        "value": torch.tensor(
            [[[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], [[-0.1, 0.2, 0.3], [0.3, -0.5, 0.6]]]
        ),
        "time": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        "index": torch.tensor([[0, 1], [0, 1]]),
        "valid": torch.ones(2, 2, dtype=torch.bool),
        "observed": torch.tensor([[True, False], [True, False]]),
    }
    action = {
        **{k: v.clone() for k, v in pose.items() if k != "value"},
        "value": torch.tensor([[[0.2, -0.3], [-0.4, 0.5]], [[-0.2, 0.1], [0.3, -0.4]]]),
        "observed": torch.zeros(2, 2, dtype=torch.bool),
    }
    return {
        "world": {"image": image, "proprio": pose, "action": action},
        "embodiment": ["robot", "robot"],
    }


def stages(model, training_mode="frozen_image"):
    return [
        PixelVideoNoisingStage(LinearGaussianImageNoising()),
        PixelReadoutDenoiserStage(model, training_mode),
        PixelReadoutImageLossStage(PlainPixelMSE(model.loss_reduction)),
        PixelReadoutNoisingStage(LinearGaussianImageNoising(0.2)),
        PixelReadoutImageDenoiserStage(),
        PixelVideoSamplerStage(
            LinearDDIMImageSampler(3), "pixel_video/sampled_prediction"
        ),
        PixelStateActionReadoutStage(),
        PixelReadoutMSEStage(PlainPixelMSE(model.loss_reduction), "proprio"),
        PixelReadoutMSEStage(PlainPixelMSE(model.loss_reduction), "action"),
    ]


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_joint_graph_preserves_legacy_three_losses_gradients_and_rng(mode):
    model = ImageProprioModel(**arguments(mode))
    legacy = copy.deepcopy(model)
    graph = Pipeline(stages(model, "joint"), stage_ids=IDS)
    batch = fixture_batch()
    torch.manual_seed(61)
    expected = legacy.training_losses(batch["world"], "robot")
    expected_rng = torch.get_rng_state().clone()
    expected["total"].backward()
    torch.manual_seed(61)
    result = graph.execute(batch, mode="train")
    assert set(k for k in result if k.startswith("loss/")) == {
        "loss/pixel_video",
        "loss/measured_state",
        "loss/control_command",
    }
    assert "log/image_proprio_total" not in result
    for key, legacy_key in (
        ("pixel_video", "image"),
        ("measured_state", "pose"),
        ("control_command", "action"),
    ):
        torch.testing.assert_close(
            result[f"loss/{key}"], expected[legacy_key], rtol=0, atol=0
        )
    torch.testing.assert_close(sum_losses(result), expected["total"], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert result["pixel_readout/noise_levels"].max() <= 0.2
    sum_losses(result).backward()
    for name, parameter in model.named_parameters():
        reference = dict(legacy.named_parameters())[name]
        assert parameter.grad is not None and reference.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_frozen_image_adamw_updates_only_new_branch_and_preserves_samples(mode):
    target_args = arguments(mode)
    source_args = {
        **target_args,
        "objective": "visual_only",
        "pose_dims": None,
        "action_dims": None,
    }
    source = ImageProprioModel(**source_args)
    model = ImageProprioModel(**target_args)
    transfer_shared_weights(source, model)
    graph = Pipeline(stages(model), stage_ids=IDS)
    assert all(key.startswith("stages.1.model.") for key in graph.state_dict())
    assert sum(p.numel() for p in graph.parameters()) == sum(
        p.numel() for p in model.parameters()
    )
    assert graph.stage_by_id("image_proprio") is graph.stage_by_id("pixel_denoiser")
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    batch = fixture_batch()
    sampler = LinearDDIMImageSampler(3)
    torch.manual_seed(78)
    expected_images = sampler(model, batch["world"], "robot")
    torch.manual_seed(78)
    source_images = sampler(source, {"image": batch["world"]["image"]}, "robot")
    torch.testing.assert_close(expected_images, source_images, rtol=0, atol=0)
    # Include frozen tensors in AdamW too: no gradient means no decay/update.
    optimizer = torch.optim.AdamW(graph.parameters(), lr=5e-5, weight_decay=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = graph.execute(batch, mode="train")
        assert not output["loss/pixel_video"].requires_grad
        loss = sum_losses(output)
        assert loss.requires_grad and loss.dtype == torch.float32
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert (
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                ), name
            else:
                assert parameter.grad is None, name
        optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, before[name]), name
        else:
            assert not torch.equal(parameter, before[name]), name
    assert not model.phase.weight.requires_grad
    assert torch.equal(model.phase.weight, before["phase.weight"])
    for name, tensor in source.state_dict().items():
        actual = model.state_dict()[name]
        if name == "phase.weight":
            actual = actual[:1]
        assert torch.equal(tensor, actual), name
    torch.manual_seed(78)
    actual_images = sampler(model, batch["world"], "robot")
    torch.testing.assert_close(actual_images, expected_images, rtol=0, atol=0)
    # Restoring ordinary tensors into the same configured freeze policy is strict.
    restored = Pipeline(stages(ImageProprioModel(**target_args)), stage_ids=IDS)
    restored.load_state_dict(graph.state_dict(), strict=True)
    assert [p.requires_grad for p in restored.parameters()] == [
        p.requires_grad for p in graph.parameters()
    ]


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_inference_matches_legacy_sampling_and_never_consumes_query_labels(mode):
    model = ImageProprioModel(**arguments(mode))
    graph = Pipeline(stages(model), stage_ids=IDS)
    batch = fixture_batch()
    torch.manual_seed(91)
    expected = model.sample(batch["world"], "robot", steps=3)
    changed = copy.deepcopy(batch)
    changed["world"]["proprio"]["value"][~changed["world"]["proprio"]["observed"]] = (
        float("nan")
    )
    changed["world"]["action"]["value"].fill_(float("nan"))
    torch.manual_seed(91)
    result = graph.execute(changed, mode="inference")
    assert not any(key.startswith("loss/") for key in result)
    assert result["pixel_video/denoiser"] is model
    for key in ("image", "proprio", "action"):
        torch.testing.assert_close(
            result["image_proprio_prediction"][key], expected[key], rtol=0, atol=0
        )


def test_measured_and_command_targets_are_separate_and_missing_labels_masked():
    graph = Pipeline(stages(ImageProprioModel(**arguments())), stage_ids=IDS)
    batch = fixture_batch()
    batch["world"]["proprio"]["target_valid"] = batch["world"]["proprio"][
        "observed"
    ].clone()
    batch["world"]["proprio"]["value"][:, 1] = float("nan")
    result = graph.execute(batch, mode="train")
    assert result["loss/measured_state"] == 0
    assert result["loss/control_command"] > 0
    assert torch.isfinite(sum_losses(result))
    # This graph has two experts, with separate state and command output heads.
    model = graph.stage_by_id("pixel_denoiser").model
    assert all(len(block.qkv) == 2 for block in model.blocks)
    assert model.pose_out["robot"] is not model.action_out["robot"]


def recipe(mode, *, require_initialization=True):
    overrides = [
        f"+experiment=pixel_proprio/readout_{mode}",
        "hydra/launcher=basic",
        "world_model_attention=native",
        "data.manifest_path=/fixture/actions.json",
        "paths.output_dir=/fixture/output",
        "paths.work_dir=/fixture/source",
    ]
    if require_initialization:
        overrides.extend(
            [
                "pixel_readout.initial_weights=/fixture/readout.pt",
                "pixel_readout.initial_weights_sha256=" + "a" * 64,
            ]
        )
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name="train_zarr_cartesian", overrides=overrides)


@pytest.mark.parametrize(
    "mode,pose_dim,action_dim,capacity",
    [("continuous", 9, 7, 25856), ("discrete", 36, 28, 3232)],
)
def test_actual_stage2_recipes_graph_and_counts(
    tmp_path, mode, pose_dim, action_dim, capacity
):
    cfg = recipe(mode)
    plain = OmegaConf.to_container(cfg, resolve=True)
    pipeline = plain["model"]["pipeline"]
    expected_ids = {
        "pixel_noising": 0,
        "pixel_denoiser": 1,
        "pixel_prediction": 2,
        "pixel_loss": 3,
        "readout_noising": 4,
        "readout_denoiser": 5,
        "pixel_sampler": 6,
        "state_action_readout": 7,
        "measured_state_loss": 8,
        "control_command_loss": 9,
        "image_proprio": 1,
    }
    assert pipeline["stage_ids"] == expected_ids
    assert len(pipeline["stages"]) == 10
    model_cfg = pipeline["stages"][1]["model"]
    assert model_cfg["pose_dims"] == {"libero_gripper": pose_dim}
    assert model_cfg["action_dims"] == {"libero_gripper": action_dim}
    assert model_cfg["max_tokens"] == capacity
    assert not cfg.data.image_only and cfg.data.task == "visual_state_action"
    assert cfg.evaluator.objective == "visual_state_action"
    assert cfg.model.optimizer.lr == 5e-5 and cfg.model.optimizer.weight_decay == 0.01
    assert pipeline["stages"][1]["training_mode"] == "frozen_image"
    assert pipeline["stages"][4]["noising"]["maximum"] == 0.2
    assert [pipeline["stages"][i]["weight"] for i in (3, 8, 9)] == [1.0, 1.0, 1.0]
    assert pipeline["initialization"] == [
        {
            "stage_id": "image_proprio",
            "module_path": "model",
            "source": "/fixture/readout.pt",
            "sha256": "a" * 64,
        }
    ]
    with torch.device("meta"):
        stage = instantiate(pipeline["stages"][1])
    frozen = sum(p.numel() for p in stage.parameters() if not p.requires_grad)
    assert frozen == 88_178_688  # Exactly the copied AdaLN/3-axis-RoPE image weights.
    assert stage.model.noise_conditioning == "adaln_zero"
    assert not hasattr(stage.model, "phase")
    assert sum(p.numel() for p in stage.parameters()) > frozen
    # Exact stage-one model declaration is compatible with the strict migrator.
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        source_cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment=pixel_proprio/video_{mode}",
                "hydra/launcher=basic",
                "world_model_attention=native",
            ],
        )
    validate_transition(
        OmegaConf.to_container(source_cfg.model.pipeline.stages[1].model, resolve=True),
        model_cfg,
    )
    plain["data"]["train_datasets"] = {
        "libero_gripper": {"batch_keys": ["world", "embodiment"]}
    }
    path = tmp_path / "graph.yaml"
    OmegaConf.save(OmegaConf.create(plain), path)
    with torch.device("meta"):
        training = build_graph(path, mode="train")
        inference = build_graph(path, mode="inference")
    assert training["lint"] == inference["lint"] == []
    assert len(training["nodes"]) == 9 and len(inference["nodes"]) == 3
    assert [node["t"] for node in inference["nodes"]] == [
        "PixelReadoutModelStage",
        "PixelReadoutSampledImageStage",
        "PixelStateActionReadoutStage",
    ]


def test_recipe_cannot_resolve_without_exact_initialization_artifact():
    cfg = recipe("continuous", require_initialization=False)
    with pytest.raises(InterpolationToMissingValueError):
        OmegaConf.to_container(cfg.model, resolve=True)


def test_strict_index_one_checkpoint_expansion_loads_into_frozen_graph(tmp_path):
    target_args = arguments()
    source_args = {
        **target_args,
        "objective": "visual_only",
        "pose_dims": None,
        "action_dims": None,
    }
    source = ImageProprioModel(**source_args)
    checkpoint = tmp_path / "stage1.ckpt"
    torch.save(
        {
            "state_dict": {
                f"nets.pipeline.stages.1.model.{name}": value
                for name, value in source.state_dict().items()
            },
            "hyper_parameters": {
                "config": {
                    "model": {"pipeline": {"stages": [{}, {"model": source_args}]}}
                }
            },
        },
        checkpoint,
    )
    receipt = initialize_readout(
        source_checkpoint=checkpoint,
        source_sha256=sha256_file(checkpoint),
        source_config=source_args,
        target_config=target_args,
        output_path=tmp_path / "expanded.pt",
        receipt_path=tmp_path / "expanded.json",
        seed=42,
        state_dict_path=("state_dict",),
        source_prefix="nets.pipeline.stages.1.model.",
        source_config_path=(
            "hyper_parameters",
            "config",
            "model",
            "pipeline",
            "stages",
            1,
            "model",
        ),
    )
    model = ImageProprioModel(**target_args)
    algo = PipelineAlgo(
        stages(model),
        device="cpu",
        stage_ids=IDS,
        initialization=receipt["initialization"],
    )
    assert receipt["phase_a_preservation"]["bit_exact"]
    assert algo.initialization_receipts[0]["sha256"] == receipt["output"]["sha256"]
    state = torch.load(receipt["output"]["path"], weights_only=True)
    assert all(
        torch.equal(value, state[name]) for name, value in model.state_dict().items()
    )
    assert (
        not model.image_in.weight.requires_grad
        and model.pose_in["robot"].weight.requires_grad
    )
    bad = copy.deepcopy(receipt["initialization"])
    bad[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        PipelineAlgo(
            stages(ImageProprioModel(**target_args)),
            device="cpu",
            stage_ids=IDS,
            initialization=bad,
        )


def test_freeze_rejects_unknown_parameters_or_policy():
    model = ImageProprioModel(**arguments())
    model.unclassified = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="unknown or shared"):
        PixelReadoutDenoiserStage(model)
    with pytest.raises(ValueError, match="frozen_image or joint"):
        PixelReadoutDenoiserStage(ImageProprioModel(**arguments()), "partial")
