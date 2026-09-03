import importlib.util
import sys
import types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from egomimic.eval.checkpoint_loading import (
    extract_pipeline_nets_state,
    strict_load_pipeline_checkpoint,
)
from egomimic.eval.energy_score import energy_score
from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.models.denoising_nets import CrossTransformer, PaperConditionalUnet1D
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.utils.ema_callback import EMACallback


def _load_planar_stages_without_shared_pipeline():
    """The isolated slice intentionally receives Stage from the preceding PR."""
    module_name = "egomimic.pipeline.core"
    previous = sys.modules.get(module_name)
    stub = types.ModuleType(module_name)
    stub.Stage = nn.Module
    stub.resolve_homogeneous_scalar = resolve_homogeneous_scalar
    sys.modules[module_name] = stub
    try:
        path = Path(__file__).parents[1] / "egomimic/pipeline/stages_planar.py"
        spec = importlib.util.spec_from_file_location("_planar_stages_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


class TinyDenoiser(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.proj_u = nn.Linear(latent_dim, hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent, time, condition):
        del time
        return self.proj_d(torch.tanh(self.proj_u(latent) + condition[..., :4]))


def test_planar_flow_sampler_train_and_eval_contract():
    module = _load_planar_stages_without_shared_pipeline()
    sampler = module.PlanarFlowSampler(
        TinyDenoiser(3, 4),
        condition_input_dim=2,
        action_horizon=5,
        action_dims={"pushshapes_sim_u_socket": 5},
        latent_dim=3,
        condition_dim=4,
        decoder_hidden_dim=8,
        denoiser_hidden_dim=4,
        gradient_checkpointing=False,
        num_inference_steps=4,
    )
    batch = {
        "embodiment": torch.tensor([19, 19]),
        "condition": torch.randn(2, 2),
        "sampler/noise": torch.randn(2, 5, 3),
    }
    result = sampler(batch.copy())
    assert result["pred_action"].shape == (2, 5, 5)
    torch.testing.assert_close(
        sampler.last_integration_step_sizes.sum(dim=-1), torch.ones(2)
    )
    sampler.train()
    batches_seen = int(sampler.training_batches_seen)
    result = sampler.execute(batch.copy(), mode="inference")
    assert result["log/sampler_unroll_steps"] == 4
    assert int(sampler.training_batches_seen) == batches_seen
    with pytest.raises(ValueError, match="homogeneous"):
        sampler({**batch, "embodiment": torch.tensor([19, 20])})


def test_planar_decoder_preserves_legacy_sequential_state_keys():
    module = _load_planar_stages_without_shared_pipeline()
    decoder = module.TokenwiseActionDecoder(
        latent_dim=3,
        hidden_dim=8,
        action_dim=5,
        num_layers=3,
    )
    assert set(decoder.state_dict()) == {
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    }


def test_paper_unet_preserves_action_shape():
    model = PaperConditionalUnet1D(
        input_dim=5,
        global_cond_dim=12,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        n_groups=4,
    )
    output = model(torch.randn(2, 16, 5), torch.tensor([1, 2]), torch.randn(2, 12))
    assert output.shape == (2, 16, 5)


@pytest.mark.parametrize(
    ("time_conditioning", "embedding_width"),
    (("concat", 4), ("additive", 8)),
)
def test_cross_transformer_time_conditioning_preserves_expected_state_shapes(
    time_conditioning, embedding_width
):
    model = CrossTransformer(
        nblocks=1,
        cond_dim=6,
        hidden_dim=8,
        act_dim=4,
        act_seq=3,
        n_heads=2,
        dropout=0.0,
        mlp_layers=1,
        mlp_ratio=2,
        time_conditioning=time_conditioning,
    )
    assert model.proj_u.weight.shape == (embedding_width, 4)
    assert model.state_dict()["pos_emb"].shape == (1, 3, embedding_width)
    output = model(
        torch.randn(2, 3, 4),
        torch.rand(2),
        torch.randn(2, 2, 6),
    )
    assert output.shape == (2, 3, 4)


def test_cross_transformer_rejects_invalid_or_rank_deficient_conditioning():
    kwargs = {
        "nblocks": 1,
        "cond_dim": 6,
        "act_seq": 3,
        "n_heads": 1,
        "dropout": 0.0,
        "mlp_layers": 1,
        "mlp_ratio": 2,
    }
    with pytest.raises(ValueError, match="either 'concat' or 'additive'"):
        CrossTransformer(
            hidden_dim=8,
            act_dim=4,
            time_conditioning="unknown",
            **kwargs,
        )
    with pytest.raises(ValueError, match="even hidden_dim"):
        CrossTransformer(
            hidden_dim=7,
            act_dim=3,
            time_conditioning="concat",
            **kwargs,
        )
    with pytest.raises(ValueError, match="Rank-deficient denoiser input"):
        CrossTransformer(
            hidden_dim=8,
            act_dim=5,
            time_conditioning="concat",
            **kwargs,
        )
    additive = CrossTransformer(
        hidden_dim=8,
        act_dim=5,
        time_conditioning="additive",
        **kwargs,
    )
    assert additive.proj_u.weight.shape == (8, 5)


def test_energy_score_separates_accuracy_and_diversity():
    target = torch.zeros(2, 3, 5)
    samples = torch.stack((target, target + 1, target - 1))
    values = energy_score(samples, target)
    assert values["accuracy"] > 0
    assert values["diversity"] > values["accuracy"]
    torch.testing.assert_close(
        values["score"], values["accuracy"] - 0.5 * values["diversity"]
    )


def test_planar_evaluator_uses_natural_batch_metadata_and_nested_results():
    target = torch.zeros(2, 16, 5)
    batch = {
        "opaque-stream": {
            "actions": target,
            "embodiment": torch.tensor([19, 19]),
        }
    }

    class IdentityNormalizer:
        @staticmethod
        def unnormalize(values, embodiment_id):
            assert embodiment_id == 19
            return values

    evaluator = PlanarActionEval(energy_score_enabled=False)
    evaluator.bind_data_context(normalizer=IdentityNormalizer())
    evaluator.model = SimpleNamespace(
        forward_eval=lambda grouped: {
            source: {"pred_action": source_batch["actions"] + 1}
            for source, source_batch in grouped.items()
        }
    )
    logged = {}
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        )
    )

    evaluator.on_validation_step(batch, batch_idx=0)

    assert logged["Valid/MSE"].item() == pytest.approx(1.0)
    assert logged["Valid/MSE/pushshapes_sim_u_socket"].item() == pytest.approx(1.0)


def test_strict_checkpoint_loader_overlays_ema_and_retains_online_buffers():
    class BufferedPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 1)
            self.register_buffer("running_scale", torch.tensor(3.0))

    class Algo:
        nets = nn.ModuleDict({"pipeline": BufferedPolicy()})

    algo = Algo()
    expected = OrderedDict(
        (key, value.clone()) for key, value in algo.nets.state_dict().items()
    )
    online = OrderedDict(
        (f"model.nets.{key}", value + 1) for key, value in expected.items()
    )
    parameter_keys = set(dict(algo.nets.named_parameters()))
    ema = OrderedDict(
        (f"nets.{key}", expected[key] + 2) for key in sorted(parameter_keys)
    )
    checkpoint = {"state_dict": online, "ema_state_dict": ema}
    assert set(extract_pipeline_nets_state(checkpoint)) == set(expected)
    strict_load_pipeline_checkpoint(algo, checkpoint, use_ema=True)
    for key, value in algo.nets.state_dict().items():
        offset = 2 if key in parameter_keys else 1
        torch.testing.assert_close(value, expected[key] + offset)
    with pytest.raises(ValueError):
        strict_load_pipeline_checkpoint(algo, {"state_dict": {}}, use_ema=False)


def test_strict_checkpoint_loader_rejects_inexact_or_conflicting_ema_keys():
    class Algo:
        nets = nn.ModuleDict({"pipeline": nn.Linear(2, 1)})

    algo = Algo()
    expected = algo.nets.state_dict()
    online = OrderedDict(
        (f"model.nets.{key}", value.clone()) for key, value in expected.items()
    )
    ema = OrderedDict(
        (f"model.nets.{key}", value.clone())
        for key, value in algo.nets.named_parameters()
    )

    missing = OrderedDict(ema)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="EMA parameter key mismatch"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": missing},
            use_ema=True,
        )

    extra = OrderedDict(ema)
    extra["model.nets.pipeline.unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError, match="EMA parameter key mismatch"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": extra},
            use_ema=True,
        )

    conflicting = OrderedDict(ema)
    key = next(iter(dict(algo.nets.named_parameters())))
    conflicting[f"nets.{key}"] = expected[key] + 1
    with pytest.raises(ValueError, match="conflicting checkpoint aliases"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": conflicting},
            use_ema=True,
        )


def _factorized_checkpoint_fixture(domain="domain"):
    class BufferedPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 1)
            self.register_buffer("running_scale", torch.tensor(3.0))

    class DenoiserStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = BufferedPolicy()

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList(
                [nn.Identity(), nn.Identity(), nn.Identity(), DenoiserStage()]
            )

    from egomimic.pipeline.algo import PipelineAlgo

    class Algo(PipelineAlgo):
        def __init__(self):
            self.nets = nn.ModuleDict({"pipeline": Policy()})

    algo = Algo()
    expected = OrderedDict(
        (key, value.clone()) for key, value in algo.nets.state_dict().items()
    )
    old_prefix = f"policy.stages.1.policies.{domain}."
    new_prefix = "pipeline.stages.3.policy."
    online = OrderedDict(
        (
            "model.nets." + key.replace(new_prefix, old_prefix, 1),
            value + 1,
        )
        for key, value in expected.items()
    )
    parameter_keys = set(dict(algo.nets.named_parameters()))
    ema = OrderedDict(
        (
            "nets." + key.replace(new_prefix, old_prefix, 1),
            expected[key] + 2,
        )
        for key in sorted(parameter_keys)
    )
    return algo, expected, online, ema, old_prefix, new_prefix


def _registered_ema_checkpoint(
    online,
    expected,
    old_prefix,
    new_prefix,
    *,
    registered_prefix="ema_nets.",
):
    state_dict = OrderedDict(online)
    for key, value in expected.items():
        legacy_key = key.replace(new_prefix, old_prefix, 1)
        state_dict[registered_prefix + legacy_key] = value + 2
    # These belong to the historical Lightning wrapper, not algo.nets.
    state_dict["ema_optimization_step"] = torch.tensor(17)
    state_dict["ema_decay"] = torch.tensor(0.999)
    return {"state_dict": state_dict}


@pytest.mark.parametrize(
    "domain",
    ["pushshapes_sim_chain_gripper", "pushshapes_sim_u_socket"],
)
def test_strict_checkpoint_loader_remaps_online_and_ema_exactly(domain):
    algo, expected, online, ema, old_prefix, new_prefix = (
        _factorized_checkpoint_fixture(domain)
    )
    parameter_keys = set(dict(algo.nets.named_parameters()))

    strict_load_pipeline_checkpoint(
        algo,
        {"state_dict": online, "ema_state_dict": ema},
        use_ema=True,
        prefix_rewrites=OrderedDict([(old_prefix, new_prefix)]),
    )

    for key, value in algo.nets.state_dict().items():
        offset = 2 if key in parameter_keys else 1
        torch.testing.assert_close(value, expected[key] + offset)


@pytest.mark.parametrize(
    "domain",
    ["pushshapes_sim_chain_gripper", "pushshapes_sim_u_socket"],
)
@pytest.mark.parametrize("registered_prefix", ["ema_nets.", "model.ema_nets."])
def test_strict_checkpoint_loader_remaps_registered_ema_exactly(
    domain, registered_prefix
):
    algo, expected, online, _, old_prefix, new_prefix = _factorized_checkpoint_fixture(
        domain
    )
    checkpoint = _registered_ema_checkpoint(
        online,
        expected,
        old_prefix,
        new_prefix,
        registered_prefix=registered_prefix,
    )
    strict_load_pipeline_checkpoint(
        algo,
        checkpoint,
        use_ema=True,
        prefix_rewrites=OrderedDict([(old_prefix, new_prefix)]),
    )

    for key, value in algo.nets.state_dict().items():
        # The legacy wrapper evaluated its complete registered EMA tree,
        # including its copied buffers.
        torch.testing.assert_close(value, expected[key] + 2)


def test_strict_checkpoint_loader_accepts_equal_registered_ema_aliases():
    algo, expected, online, _, old_prefix, new_prefix = _factorized_checkpoint_fixture(
        "pushshapes_sim_chain_gripper"
    )
    checkpoint = _registered_ema_checkpoint(online, expected, old_prefix, new_prefix)
    for key, value in list(checkpoint["state_dict"].items()):
        if key.startswith("ema_nets."):
            checkpoint["state_dict"]["model." + key] = value.clone()

    strict_load_pipeline_checkpoint(
        algo,
        checkpoint,
        use_ema=True,
        prefix_rewrites={old_prefix: new_prefix},
    )


def test_strict_checkpoint_loader_rejects_dual_ema_selection():
    algo, expected, online, ema, old_prefix, new_prefix = (
        _factorized_checkpoint_fixture("pushshapes_sim_chain_gripper")
    )
    checkpoint = _registered_ema_checkpoint(online, expected, old_prefix, new_prefix)
    checkpoint["ema_state_dict"] = ema
    with pytest.raises(ValueError, match="both registered EMA state"):
        strict_load_pipeline_checkpoint(
            algo,
            checkpoint,
            use_ema=True,
            prefix_rewrites={old_prefix: new_prefix},
        )

    # Dual EMA metadata is irrelevant when the caller explicitly selects the
    # exact online tree, so online-only loading remains backward compatible.
    strict_load_pipeline_checkpoint(
        algo,
        checkpoint,
        use_ema=False,
        prefix_rewrites={old_prefix: new_prefix},
    )
    for key, value in algo.nets.state_dict().items():
        torch.testing.assert_close(value, expected[key] + 1)


def test_strict_checkpoint_loader_rejects_conflicting_registered_ema_aliases():
    algo, expected, online, _, old_prefix, new_prefix = _factorized_checkpoint_fixture(
        "pushshapes_sim_chain_gripper"
    )

    checkpoint = _registered_ema_checkpoint(online, expected, old_prefix, new_prefix)
    registered_key = next(
        key for key in checkpoint["state_dict"] if key.startswith("ema_nets.")
    )
    checkpoint["state_dict"]["model." + registered_key] = (
        checkpoint["state_dict"][registered_key] + 1
    )
    with pytest.raises(
        ValueError, match="conflicting checkpoint aliases in registered EMA state_dict"
    ):
        strict_load_pipeline_checkpoint(
            algo,
            checkpoint,
            use_ema=True,
            prefix_rewrites={old_prefix: new_prefix},
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_strict_checkpoint_loader_rejects_inexact_registered_ema(mutation):
    algo, expected, online, _, old_prefix, new_prefix = _factorized_checkpoint_fixture(
        "pushshapes_sim_u_socket"
    )
    checkpoint = _registered_ema_checkpoint(online, expected, old_prefix, new_prefix)
    registered_keys = [
        key for key in checkpoint["state_dict"] if key.startswith("ema_nets.")
    ]
    if mutation == "missing":
        checkpoint["state_dict"].pop(registered_keys[0])
    else:
        checkpoint["state_dict"]["ema_nets." + old_prefix + "unexpected"] = torch.zeros(
            1
        )

    with pytest.raises(ValueError, match="registered EMA state_dict key mismatch"):
        strict_load_pipeline_checkpoint(
            algo,
            checkpoint,
            use_ema=True,
            prefix_rewrites={old_prefix: new_prefix},
        )


def test_strict_checkpoint_loader_rejects_incomplete_or_ambiguous_rewrites():
    algo, _, online, ema, old_prefix, new_prefix = _factorized_checkpoint_fixture()
    with pytest.raises(ValueError, match="Pipeline checkpoint key mismatch"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": ema},
            use_ema=True,
            prefix_rewrites={
                old_prefix + "projection.weight": new_prefix + "projection.weight"
            },
        )

    with pytest.raises(ValueError, match="ambiguous state_dict prefix rewrite"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": ema},
            use_ema=True,
            prefix_rewrites=OrderedDict(
                [
                    (old_prefix, new_prefix),
                    (old_prefix + "projection.", new_prefix + "projection."),
                ]
            ),
        )


def test_strict_checkpoint_loader_rejects_rewrite_collisions():
    algo, _, online, ema, old_prefix, new_prefix = _factorized_checkpoint_fixture()
    legacy_key = next(iter(online))
    current_key = legacy_key.replace(old_prefix, new_prefix, 1)
    online[current_key] = online[legacy_key].clone()

    with pytest.raises(ValueError, match="state_dict prefix rewrite collision"):
        strict_load_pipeline_checkpoint(
            algo,
            {"state_dict": online, "ema_state_dict": ema},
            use_ema=True,
            prefix_rewrites={old_prefix: new_prefix},
        )


def test_train_eval_pipeline_load_is_strict():
    from egomimic.trainHydra import _load_eval_checkpoint

    algo, _, online, ema, old_prefix, new_prefix = _factorized_checkpoint_fixture()

    class PipelineWrapper:
        model = algo

        def load_state_dict(self, *args, **kwargs):
            raise AssertionError(
                "Pipeline checkpoints must not use wrapper strict=False"
            )

    incomplete = OrderedDict(online)
    incomplete.pop(next(iter(incomplete)))
    cfg = OmegaConf.create(
        {
            "eval_checkpoint": {
                "use_ema": True,
                "prefix_rewrites": {old_prefix: new_prefix},
            }
        }
    )
    with pytest.raises(ValueError, match="Pipeline checkpoint key mismatch"):
        _load_eval_checkpoint(
            PipelineWrapper(),
            {"state_dict": incomplete, "ema_state_dict": ema},
            cfg,
        )

def test_eval_mode_removes_training_time_ema_callback():
    from egomimic.trainHydra import _callbacks_for_mode

    ema_callback = EMACallback()
    retained_callback = object()
    assert _callbacks_for_mode([ema_callback, retained_callback], "eval") == [
        retained_callback
    ]
    assert _callbacks_for_mode([ema_callback, retained_callback], "train") == [
        ema_callback,
        retained_callback,
    ]


def test_paper_ema_warmup_and_fail_closed_resume():
    callback = EMACallback(decay=0.9999, use_warmup=True, power=0.75)
    assert callback._schedule(1) == 0
    assert 0 < callback._schedule(2) < callback._schedule(100)
    with pytest.raises(RuntimeError):
        callback.on_load_checkpoint(None, None, {"global_step": 3})
