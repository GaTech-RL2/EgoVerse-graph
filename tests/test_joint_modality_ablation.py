"""Counterfactual stream removal must preserve paired RGB experimental controls."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.image_proprio_eval import ImageProprioEval
from egomimic.models.joint_state_action import (
    CoupledStateActionNoising,
    JointStateActionSampler,
)
from egomimic.models.pixel_diffusion import PlainPixelMSE
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_joint_state_action import (
    JointDenoiserStage,
    JointLossStage,
    JointNoisingStage,
)
from tests.test_joint_state_action import make_nonzero, model, world

CELLS = {
    "image_only": ("image",),
    "image_proprio": ("image", "proprio"),
    "image_action": ("image", "action"),
    "image_proprio_action": ("image", "proprio", "action"),
}


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def image_parameters(network):
    result = {}
    for key, value in network.state_dict().items():
        if key.startswith(("inputs.image.", "outputs.image.", "camera.")):
            result[key] = value
        elif key.startswith(("noise_embeddings.0.", "norms.0.", "final_adaln.0.")):
            result[key] = value
        elif key.startswith("blocks.") and any(
            "." + bank + ".0." in key
            for bank in ["norm1", "qkv", "out", "norm2", "ffn", "adaln"]
        ):
            result[key] = value
    return result


def test_common_image_initialization_and_constructor_rng_are_bitwise_paired():
    reference = None
    rng = None
    for streams in CELLS.values():
        torch.manual_seed(42)
        network = model(active_streams=streams)
        selected = image_parameters(network)
        assert selected
        if reference is None:
            reference = {k: v.clone() for k, v in selected.items()}
            rng = torch.get_rng_state()
        else:
            assert selected.keys() == reference.keys()
            for k, v in selected.items():
                torch.testing.assert_close(v, reference[k], atol=0, rtol=0)
            assert torch.equal(rng, torch.get_rng_state())
        assert set(network.inputs) == set(network.outputs) == set(streams)
        assert len(network.blocks[0].qkv) == len(streams)
        assert sum(network.expert_parameter_counts().values()) == sum(
            p.numel() for p in network.parameters()
        )
        assert all(p.requires_grad for p in network.parameters())


@pytest.mark.parametrize("streams", list(CELLS.values()))
def test_removed_streams_cannot_influence_images_and_every_remaining_parameter_optimizes(
    streams,
):
    batch = world()
    network = model(active_streams=streams)
    make_nonzero(network)
    noised = CoupledStateActionNoising()(batch)
    baseline = network(batch, "robot", noised["values"], noised["levels"])
    assert tuple(baseline) == streams
    changed = copy.deepcopy(noised["values"])
    changed_world = copy.deepcopy(batch)
    for name in set(batch) - set(streams):
        changed[name].fill_(1000)
        changed_world[name]["value"].fill_(-1000)
    actual = network(changed_world, "robot", changed, noised["levels"])
    torch.testing.assert_close(actual["image"], baseline["image"], atol=0, rtol=0)
    stages = [
        JointNoisingStage(CoupledStateActionNoising()),
        JointDenoiserStage(network),
    ]
    stages += [JointLossStage(name, PlainPixelMSE()) for name in streams]
    graph = Pipeline(stages)
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-6)
    result = graph(dict(world=batch, embodiment="robot"), mode="train")
    loss = sum_losses(result)
    assert set(k for k in result if k.startswith("loss/")) == {
        "loss/" + k for k in streams
    }
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in network.parameters()
    )
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in network.parameters())


def test_sampling_pairs_initial_image_noise_rng_and_observed_clamping():
    batch = world()
    for stream in batch.values():
        stream["value"][~stream["observed"]] = float("nan")
    first_reference = None
    after_reference = None
    for streams in CELLS.values():
        network = model(active_streams=streams)
        make_nonzero(network)
        initial = []

        def hook(module, args):
            if not initial:
                initial.append(args[2]["image"].clone())

        handle = network.register_forward_pre_hook(hook)
        torch.manual_seed(42)
        prediction = JointStateActionSampler(state_steps=3, action_steps=3)(
            network, batch, "robot"
        )
        handle.remove()
        assert tuple(prediction) == streams
        for name, value in prediction.items():
            assert torch.isfinite(value).all()
            stream = batch[name]
            assert torch.equal(
                value[stream["observed"]], stream["value"][stream["observed"]]
            )
        if first_reference is None:
            first_reference = initial[0]
            after_reference = torch.get_rng_state()
        else:
            torch.testing.assert_close(initial[0], first_reference, atol=0, rtol=0)
            assert torch.equal(after_reference, torch.get_rng_state())


def test_full_default_state_dict_and_behavior_remain_unchanged():
    torch.manual_seed(42)
    implicit = model()
    torch.manual_seed(42)
    explicit = model(active_streams=("image", "proprio", "action"))
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
    batch = world()
    noised = CoupledStateActionNoising()(batch)
    for name, value in implicit(
        batch, "robot", noised["values"], noised["levels"]
    ).items():
        torch.testing.assert_close(
            value,
            explicit(batch, "robot", noised["values"], noised["levels"])[name],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "bad",
    [
        (),
        ("proprio",),
        ("image", "image"),
        ("image", "action", "proprio"),
        ("image", "bad"),
    ],
)
def test_invalid_stream_selection_fails_closed(bad):
    with pytest.raises(ValueError, match="image-first subset"):
        model(active_streams=bad)


@pytest.mark.parametrize("streams", list(CELLS.values()))
def test_evaluator_consumes_exact_active_outputs_and_records_only_their_metrics(
    streams,
):
    data = world()
    kinds = ["forecast", "completion"]
    item = dict(world=data, task_kind=kinds, embodiment="robot")
    predictions = {
        k: torch.nan_to_num(v["value"]) + 0.1 for k, v in data.items() if k in streams
    }
    logged = {}
    model_stub = SimpleNamespace(
        device=torch.device("cpu"),
        global_step=3,
        trainer=SimpleNamespace(sanity_checking=False),
        forward_eval=lambda batch: {"robot": {"image_proprio_prediction": predictions}},
        log=lambda k, v, **kw: logged.update({k: v}),
    )
    evaluator = ImageProprioEval(
        objective="visual_state_action", prediction_streams=streams
    )
    evaluator.model = model_stub
    evaluator.on_validation_start()
    evaluator.on_validation_step({"robot": item}, 0)
    evaluator.log_epoch_metrics()
    evaluator.on_validation_end()
    assert evaluator.has_completed_validation(3)
    assert all(any("/" + k + "_" in metric for k in streams) for metric in logged)
    for absent in set(data) - set(streams):
        assert not any("/" + absent + "_" in metric for metric in logged)
    predictions["unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError, match="differ from model output"):
        evaluator.on_validation_step({"robot": item}, 0)


@pytest.mark.parametrize(
    "temporal,world_size,batch_size,pose,action",
    [("discrete", 2, 16, 36, 28), ("continuous", 4, 8, 9, 7)],
)
def test_factorial_configs_match_controls_and_construct_expected_parameters(
    temporal, world_size, batch_size, pose, action
):
    root = Path(__file__).parents[1] / "egomimic/hydra_configs"
    reference = None
    ids = []
    for cell, streams in CELLS.items():
        with initialize_config_dir(config_dir=str(root), version_base=None):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[
                    f"+experiment=pixel_proprio/ablation/{temporal}_{cell}",
                    "hydra/launcher=basic",
                    "data.manifest_path=/fixture/manifest.json",
                    "paths.output_dir=/fixture/output",
                ],
            )
        assert cfg.seed == 42 and cfg.trainer.max_steps == 100000
        assert (
            cfg.trainer.devices == cfg.data.distributed_world_size == world_size
            and cfg.data.batch_size == batch_size
        )
        assert (
            world_size * batch_size == 32
            and cfg.model.optimizer.lr == 5e-6
            and cfg.model.scheduler.warmup_steps == 3000
        )
        assert (
            cfg.callbacks.model_checkpoint.save_top_k == -1
            and cfg.callbacks.model_checkpoint.save_last == "link"
        )
        assert (
            cfg.callbacks.model_checkpoint.every_n_train_steps
            == cfg.trainer.val_check_interval
            == 20000
        )
        assert (
            not cfg.callbacks.model_checkpoint.every_n_epochs
            and not cfg.callbacks.model_checkpoint.save_on_train_epoch_end
        )
        assert cfg.evaluator.prediction_streams == list(streams)
        resolved = OmegaConf.to_container(cfg.model.pipeline, resolve=True)
        args = resolved["stages"][1]["model"]
        assert args["active_streams"] == list(streams)
        assert [
            s["stream"]
            for s in resolved["stages"]
            if s["_target_"].endswith("JointLossStage")
        ] == list(streams)
        with torch.device("meta"):
            network = instantiate(args)
        expected = dict(
            image=88178688,
            proprio=13956388 if temporal == "discrete" else 13942537,
            action=13952284 if temporal == "discrete" else 13941511,
        )
        assert network.expert_parameter_counts() == {k: expected[k] for k in streams}
        assert sum(p.numel() for p in network.parameters()) == sum(
            expected[k] for k in streams
        )
        controls = {
            name: OmegaConf.to_container(node, resolve=True)
            for name, node in dict(
                data=cfg.data,
                optimizer=cfg.model.optimizer,
                scheduler=cfg.model.scheduler,
                trainer=cfg.trainer,
                callbacks=cfg.callbacks,
            ).items()
        }
        if reference is None:
            reference = controls
        else:
            assert reference == controls
        ids.append(cfg.logger.wandb.id)
    assert len(set(ids)) == 4


@pytest.mark.parametrize("streams", list(CELLS.values()))
def test_active_banks_compile_and_keep_fp32_mixed_precision_contract(streams):
    batch = world()
    noised = CoupledStateActionNoising()(batch)
    network = model(
        active_streams=streams,
        token_execution="packed_valid",
        compile_mode="default",
        compile_backend="eager",
        compile_dynamic=True,
    )
    make_nonzero(network)
    # Local Torch2.7 lacks the required compiled-autocast backward policy.
    # Exercise compiled FP32 and eager BF16 here; the exact production runtime
    # must separately pass combined FA4/compiled/BF16 on allocated GPUs.
    compiled = network(batch, "robot", noised["values"], noised["levels"])
    sum(value.square().sum() for value in compiled.values()).backward()
    eager = model(active_streams=streams, token_execution="packed_valid")
    eager.load_state_dict(network.state_dict(), strict=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        predictions = eager(batch, "robot", noised["values"], noised["levels"])
        loss = sum(
            PlainPixelMSE()(predictions[k], noised["targets"][k], noised["masks"][k])
            for k in streams
        )
    assert tuple(predictions) == streams
    assert all(value.dtype == torch.bfloat16 for value in predictions.values())
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(
        p.dtype == p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
        for p in list(network.parameters()) + list(eager.parameters())
    )


def test_paired_control_digest_covers_geometry_masks_and_actual_image_noise():
    from egomimic.pl_utils.joint_smoke import (
        image_parameter_state,
        paired_image_digest,
        weights_digest,
    )

    batch = world()
    noised = CoupledStateActionNoising()(batch)
    before = torch.get_rng_state()
    reference = paired_image_digest(batch, noised["values"], noised["levels"])
    assert torch.equal(before, torch.get_rng_state())
    for key in (
        "value",
        "index",
        "time",
        "xy",
        "camera",
        "observed",
        "valid",
        "target_valid",
    ):
        changed = copy.deepcopy(batch)
        item = changed["image"][key]
        if item.dtype == torch.bool:
            item.flatten()[0] = ~item.flatten()[0]
        else:
            item.flatten()[0] += 1
        assert (
            paired_image_digest(changed, noised["values"], noised["levels"])
            != reference
        )
    digests = []
    for streams in CELLS.values():
        torch.manual_seed(42)
        digests.append(
            weights_digest(image_parameter_state(model(active_streams=streams)))
        )
    assert len(set(digests)) == 1
