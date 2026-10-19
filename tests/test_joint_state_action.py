"""Scientific contracts for the fresh three-expert joint diffusion setup."""

import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.models.joint_state_action import (
    CoupledStateActionNoising,
    JointStateActionModel,
    JointStateActionSampler,
)
from egomimic.models.pixel_diffusion import PlainPixelMSE
from egomimic.models.world_model import MoTBlock
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_joint_state_action import (
    JointDenoiserStage,
    JointLossStage,
    JointNoisingStage,
    JointSamplerStage,
)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def world():
    result = {}
    for name, repeat, dim in (("image", 4, 12), ("proprio", 1, 3), ("action", 1, 2)):
        time = torch.tensor([[-0.1, 0.2, 1.7], [-0.4, 0.5, 0.0]]).repeat_interleave(
            repeat, 1
        )
        valid = torch.tensor(
            [[1, 1, 1], [1, 1, 0]], dtype=torch.bool
        ).repeat_interleave(repeat, 1)
        observed = torch.zeros_like(valid)
        if name != "action":
            observed[:, :repeat] = True
        result[name] = dict(
            value=torch.randn(2, 3 * repeat, dim),
            time=time,
            index=torch.arange(3)
            .float()[None]
            .expand(2, -1)
            .repeat_interleave(repeat, 1),
            valid=valid,
            observed=observed,
            target_valid=valid.clone(),
        )
        result[name]["value"][~valid] = float("nan")
    result["image"].update(
        xy=torch.zeros(2, 12, 2), camera=torch.zeros(2, 12, dtype=torch.long)
    )
    return result


def model(**kwargs):
    return JointStateActionModel(
        pose_dims={"robot": 3},
        action_dims={"robot": 2},
        image_dim=12,
        width=16,
        proprio_width=5,
        action_width=7,
        heads=2,
        depth=2,
        mlp_ratio=2,
        max_tokens=18,
        rope_axis_dims=(4, 2, 2),
        spatial_grid_size=(2, 2),
        **kwargs,
    )


def test_private_widths_keep_one_common_attention_space():
    network = model()
    block = network.blocks[0]
    assert block.expert_widths == (16, 5, 7)
    assert [layer.in_features for layer in block.qkv] == [16, 5, 7]
    assert [layer.out_features for layer in block.qkv] == [48, 48, 48]
    assert [layer.out_features for layer in block.out] == [16, 5, 7]
    assert [layers[0].out_features for layers in block.ffn] == [32, 10, 14]
    # Vector residual widths need not be divisible by the attention head count.
    assert block.heads == 2 and block.head_dim == 8


def test_shared_mot_default_is_unchanged_and_supports_private_widths():
    torch.manual_seed(42)
    implicit = MoTBlock(16, 2)
    torch.manual_seed(42)
    explicit = MoTBlock(16, 2, expert_widths=(16, 16, 16))
    for key, value in implicit.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
    small = MoTBlock(16, 2, expert_widths=(16, 5, 7))
    streams = [torch.randn(2, 3, d) for d in (16, 5, 7)]
    times = [torch.arange(3).float().expand(2, -1) for _ in streams]
    valid = [torch.ones(2, 3, dtype=torch.bool) for _ in streams]
    outputs = small(streams, times, valid)
    assert [out.shape for out in outputs] == [x.shape for x in streams]
    sum(out.square().mean() for out in outputs).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters()
    )
    with pytest.raises(ValueError, match="one positive residual width"):
        MoTBlock(16, 2, expert_widths=(16, 5))


def make_nonzero(model):
    # Nonzero gates/heads exercise cross-modal attention rather than passing
    # only because AdaLN-Zero initially masks its residuals.
    with torch.no_grad():
        for block in model.blocks:
            for mod in block.adaln:
                mod[-1].bias.fill_(0.1)
        for head in model.outputs.modules():
            if isinstance(head, torch.nn.Linear):
                head.weight.normal_(std=0.02)


def test_coupled_strength_independent_draws_and_padding():
    batch = world()
    torch.manual_seed(42)
    noised = CoupledStateActionNoising()(batch)
    for b, count in ((0, 3), (1, 2)):
        for t in range(count):
            level = noised["levels"]["proprio"][b, t]
            assert (noised["levels"]["image"][b, t * 4 : (t + 1) * 4] == level).all()
        assert not torch.equal(
            noised["levels"]["proprio"][b, 1:count],
            noised["levels"]["action"][b, 1:count],
        )
    for k, s in batch.items():
        assert torch.equal(
            noised["values"][k][s["observed"]], s["value"][s["observed"]]
        )
        assert torch.isfinite(noised["values"][k]).all()
        assert (noised["values"][k][~s["valid"]] == 0).all()
    bad = copy.deepcopy(batch)
    bad["proprio"]["observed"][0, 1] = True
    with pytest.raises(ValueError, match="masks must be tied"):
        CoupledStateActionNoising()(bad)


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_packed_matches_padded_outputs_gradients_and_no_cross_example_attention(mode):
    batch = world()
    noised = CoupledStateActionNoising()(batch)
    packed, padded = (
        model(time_mode=mode),
        model(time_mode=mode, token_execution="padded"),
    )
    make_nonzero(packed)
    padded.load_state_dict(packed.state_dict(), strict=True)
    a = packed(batch, "robot", noised["values"], noised["levels"])
    b = padded(batch, "robot", noised["values"], noised["levels"])
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=2e-7, rtol=2e-5)
    sum(x.square().sum() for x in a.values()).backward()
    sum(x.square().sum() for x in b.values()).backward()
    for (_, x), (_, y) in zip(packed.named_parameters(), padded.named_parameters()):
        assert x.grad is not None and y.grad is not None
        torch.testing.assert_close(x.grad, y.grad, atol=3e-6, rtol=3e-4)
    changed = copy.deepcopy(noised["values"])
    changed["action"][1] += 10
    c = packed(batch, "robot", changed, noised["levels"])
    for k in a:
        torch.testing.assert_close(a[k][0], c[k][0], atol=0, rtol=0)
    bad = copy.deepcopy(noised["levels"])
    bad["proprio"][0, 1] += 0.01
    with pytest.raises(ValueError, match="noise strengths must be tied"):
        packed(batch, "robot", noised["values"], bad)


def test_real_optimization_all_experts_and_sampler_clamping_without_gt_queries():
    batch = world()
    network = model()
    stages = [
        JointNoisingStage(CoupledStateActionNoising()),
        JointDenoiserStage(network),
    ]
    stages += [
        JointLossStage(k, PlainPixelMSE()) for k in ("image", "proprio", "action")
    ]
    stages += [
        JointSamplerStage(JointStateActionSampler(state_steps=3, action_steps=2))
    ]
    graph = Pipeline(stages)
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-6)
    for _ in range(3):
        result = graph(dict(world=batch, embodiment="robot"), mode="train")
        loss = sum_losses(result)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in network.parameters()
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    inference = copy.deepcopy(batch)
    for s in inference.values():
        s["value"][~s["observed"]] = float("nan")
    out = graph(dict(world=inference, embodiment="robot"), mode="inference")[
        "image_proprio_prediction"
    ]
    for k, s in inference.items():
        assert torch.isfinite(out[k]).all()
        assert torch.equal(out[k][s["observed"]], s["value"][s["observed"]])
    assert not any(not p.requires_grad for p in network.parameters())


@pytest.mark.parametrize("execution", ["padded", "packed_valid"])
def test_compiled_regions_and_bf16_mixed_keep_fp32_weights_and_loss(execution):
    batch = world()
    noised = CoupledStateActionNoising()(batch)
    eager = model(token_execution=execution)
    make_nonzero(eager)
    compiled = model(
        token_execution=execution,
        compile_mode="default",
        compile_backend="eager",
        compile_dynamic=True,
    )
    compiled.load_state_dict(eager.state_dict(), strict=True)
    a = eager(batch, "robot", noised["values"], noised["levels"])
    b = compiled(batch, "robot", noised["values"], noised["levels"])
    for k in a:
        torch.testing.assert_close(a[k], b[k])
    sum(value.square().sum() for value in b.values()).backward()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = eager(batch, "robot", noised["values"], noised["levels"])
        loss = sum(
            PlainPixelMSE()(mixed[k], noised["targets"][k], noised["masks"][k])
            for k in mixed
        )
    assert loss.dtype == torch.float32
    loss.backward()
    assert all(
        p.dtype == torch.float32 and p.grad.dtype == torch.float32
        for p in compiled.parameters()
    )


@pytest.mark.parametrize(
    "arm,gpus,per_rank,pose,action",
    [("continuous", 4, 8, 9, 7), ("discrete", 2, 16, 36, 28)],
)
def test_exact_pair_config(arm, gpus, per_rank, pose, action):
    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"+experiment=pixel_proprio/joint_{arm}",
                "hydra/launcher=basic",
                "data.manifest_path=/fixture/manifest.json",
                "paths.output_dir=/fixture/output",
            ],
        )
    assert cfg.trainer.devices == cfg.data.distributed_world_size == gpus
    assert cfg.data.batch_size == per_rank and gpus * per_rank == 32
    assert cfg.trainer.max_steps == 300000 and cfg.model.optimizer.lr == 5e-6
    assert (
        cfg.model.scheduler._target_
        == "egomimic.utils.schedulers.linear_warmup_constant_scheduler"
    )
    assert cfg.model.scheduler.warmup_steps == 3000
    assert cfg.model.scheduler_interval == "step" and cfg.model.scheduler_frequency == 1
    assert cfg.data.couple_state_observations
    assert (
        cfg.joint.pose_dims.libero_gripper == pose
        and cfg.joint.action_dims.libero_gripper == action
    )
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert (
        cfg.callbacks.model_checkpoint.every_n_train_steps
        == cfg.trainer.val_check_interval
        == 20000
    )
    assert not cfg.model.pipeline.get("initialization")
    resolved = OmegaConf.to_container(cfg.model.pipeline, resolve=True)
    arguments = resolved["stages"][1]["model"]
    with torch.device("meta"):
        exact = instantiate(arguments)
    counts = exact.expert_parameter_counts()
    assert counts["image"] == 88178688
    assert counts["proprio"] <= 20000000 and counts["action"] <= 20000000
    assert exact.expert_widths == (768, 256, 256)
    assert sum(counts.values()) == sum(p.numel() for p in exact.parameters())
    arguments.update(
        image_dim=12,
        width=16,
        proprio_width=64,
        action_width=64,
        heads=2,
        depth=2,
        mlp_ratio=2,
        rope_axis_dims=[4, 2, 2],
        spatial_grid_size=[2, 2],
        compile_mode="none",
        attention_backend="sdpa",
        attention_dtype="native",
    )
    graph = instantiate(resolved)
    assert len(graph.pipeline.stages[1].model.blocks[0].qkv) == 3
