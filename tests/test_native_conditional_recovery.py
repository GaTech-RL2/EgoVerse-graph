import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.pl_utils.joint_smoke import ConditionalOptimizerUsage
from egomimic.pl_utils import world_model_behavior as wb
from scripts.train.joint.native_recovery import (
    conditional_record, restored_conditional_counts, install_rng_observer,
    state_digest, preserved_rng, expected_metrics, validate_hidden_sample,
    summarize_native_phases, checkpoint_science,
)
from scripts.train.joint.recovery_check import ResumeOptimizerProof, equal_state, phase_config


class TinyConditional(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.ones(2))
        self.proprio_banks = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2)
                                               for name in ("eva_bimanual", "human_bimanual")})
        self.action = torch.nn.Linear(2, 2)
        self.camera = torch.nn.Linear(2, 2)
        self.never = torch.nn.Parameter(torch.ones(2))
        self.pose_dims = {"eva_bimanual": 2, "human_bimanual": 2}
        self.action_dims = {"eva_bimanual": 2}

    def active_streams(self, emb):
        return ("image", "proprio", "action", "camera") if emb == "eva_bimanual" else ("image", "proprio", "camera")

    def loss(self, emb):
        value = self.shared.sum() + self.proprio_banks[emb](self.shared).sum() + self.camera(self.shared).sum()
        if emb == "eva_bimanual":
            value = value + self.action(self.shared).sum()
        return value.square()


def trained():
    torch.manual_seed(7)
    net = TinyConditional()
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-6)
    counts = {name: 0 for name, _ in net.named_parameters()}
    for emb in ("eva_bimanual", "human_bimanual") * 2:
        optimizer.zero_grad(set_to_none=True)
        net.loss(emb).backward()
        for name, p in net.named_parameters():
            counts[name] += int(p.grad is not None)
        optimizer.step()
    return net, optimizer, counts


def test_actual_conditional_adam_accepts_shared_private_and_never_used():
    net, optimizer, counts = trained()
    record = conditional_record(optimizer, net.named_parameters(), counts, 4)
    assert counts["shared"] == 4 and counts["never"] == 0
    assert counts["action.weight"] == 2
    assert counts["proprio_banks.human_bimanual.weight"] == 2
    assert optimizer.state.get(net.never) is None
    clone = TinyConditional()
    clone.load_state_dict(net.state_dict(), strict=True)
    restored = torch.optim.AdamW(clone.parameters(), lr=5e-6)
    restored.load_state_dict(optimizer.state_dict())
    cb = ConditionalOptimizerUsage()
    trainer = SimpleNamespace(global_step=4, optimizers=[restored], callbacks=[cb])
    checkpoint = dict(global_step=4, optimizer_states=[optimizer.state_dict()],
                      native_conditional_optimizer_usage=record)
    cb.on_load_checkpoint(trainer, clone, checkpoint)
    cb.on_fit_start(trainer, clone)
    actual, found = restored_conditional_counts(trainer, clone, checkpoint)
    assert actual == counts and found is cb
    restored.zero_grad(set_to_none=True)
    clone.loss("human_bimanual").backward()
    cb.on_before_optimizer_step(trainer, clone, restored)
    restored.step()
    trainer.global_step = 5
    after = conditional_record(restored, clone.named_parameters(), cb.counts, 5)
    assert after["parameter_update_counts"]["action.weight"] == 2
    assert after["parameter_update_counts"]["shared"] == 5


@pytest.mark.parametrize("damage", ["step", "missing_count", "bool_count", "moment_nan", "moment_dtype", "moment_shape", "unused_state", "order", "extra_state"])
def test_conditional_state_rejects_corruption(damage):
    net, optimizer, counts = trained()
    if damage == "step": optimizer.state[net.shared]["step"] += 1
    elif damage == "missing_count": counts.pop("shared")
    elif damage == "bool_count": counts["shared"] = True
    elif damage == "moment_nan": optimizer.state[net.shared]["exp_avg"][0] = float("nan")
    elif damage == "moment_dtype": optimizer.state[net.shared]["exp_avg"] = optimizer.state[net.shared]["exp_avg"].half()
    elif damage == "moment_shape": optimizer.state[net.shared]["exp_avg"] = torch.zeros(3)
    elif damage == "unused_state": optimizer.state[net.never] = copy.deepcopy(optimizer.state[net.shared])
    elif damage == "order": optimizer.param_groups[0]["params"].reverse()
    elif damage == "extra_state": optimizer.state[987654] = copy.deepcopy(optimizer.state[net.shared])
    with pytest.raises((ValueError, KeyError)):
        conditional_record(optimizer, net.named_parameters(), counts, 4)


@pytest.mark.parametrize("emb", ["eva_bimanual", "human_bimanual"])
def test_resume_uses_multimodal_gradient_checks_and_counts(tmp_path, emb):
    net = TinyConditional()
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-6)
    callback = ResumeOptimizerProof(output=tmp_path, world_size=4, start_step=50,
                                    stop_step=52, checkpoint="unbound", multimodal=True)
    callback.network = net
    callback.lr_policy = dict(peak_lr=5e-6, warmup_optimizer_steps=0, schedule="constant")
    callback.optimizer_lrs = []
    callback.parameter_usage = {n: 0 for n, _ in net.named_parameters()}
    callback.embodiment_batches = {e: 0 for e in net.pose_dims}
    callback.current_embodiment = emb
    net.training_behavior = SimpleNamespace(_pending_rng=None)
    net.loss(emb).backward()
    callback.on_before_optimizer_step(SimpleNamespace(global_step=50), net, optimizer)
    assert callback.gradients_checked and callback.parameter_usage["never"] == 0
    assert callback.parameter_usage["action.weight"] == int(emb == "eva_bimanual")
    assert callback.embodiment_batches[emb] == 1
    assert callback.multimodal


def test_legacy_constructor_remains_opt_out(tmp_path):
    assert not ResumeOptimizerProof(output=tmp_path, world_size=4, start_step=50,
                                    stop_step=52, checkpoint="unbound").multimodal


@pytest.mark.parametrize("rank", range(4))
def test_actual_scientific_cpu_rng_restore_observed_before_draw(monkeypatch, rank):
    original = wb._capture_rng_state(current_cuda_only=True)
    cuda = [torch.arange(8, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda[0].clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: cuda.__setitem__(0, value.clone()))
    monkeypatch.setattr(wb, "_distributed_world", lambda: (4, rank))
    try:
        states = []
        for seed in range(4):
            torch.manual_seed(130 + seed); np.random.seed(130 + seed); random.seed(130 + seed)
            states.append(wb._capture_rng_state(current_cuda_only=True))
        saved = dict(schema="mot_rng_state.distributed.v1", world_size=4, ranks=states)
        wb._restore_rng_state(states[rank])
        expected_draw = (torch.rand(2), np.random.rand(2), random.random())
        torch.manual_seed(999); np.random.seed(999); random.seed(999)

        class Context:
            global_step = 50
            trainer = SimpleNamespace(datamodule=SimpleNamespace(checkpointable_sampling=False))
            def _default_training_step(self, batch, index):
                return torch.rand(2), np.random.rand(2), random.random()

        context = Context()
        context.training_behavior = wb.WorldModelTrainingBehavior(42)
        context.training_behavior.bind(context)
        context.training_behavior.on_load_checkpoint({"mot_rng_state": saved})
        record, cleanup = install_rng_observer(context, saved, equal_state)
        actual = context.training_behavior.training_step({"human_bimanual": {}}, 0)
        assert equal_state(actual, expected_draw)
        assert record["observed"] and record["actual_sha256"] == record["expected_sha256"]
        cleanup()
    finally:
        wb._restore_rng_state({**original, "torch_cuda": []})


def test_rng_observer_fails_if_restore_was_not_executed(monkeypatch):
    saved = wb._capture_rng_state(current_cuda_only=True)
    saved["torch_cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    monkeypatch.setattr(wb, "_rank_rng_state", lambda value: saved)
    module = SimpleNamespace(training_behavior=SimpleNamespace(_pending_rng={}),
                             _default_training_step=lambda: pytest.fail("must not run"))
    record, cleanup = install_rng_observer(module, {}, equal_state)
    with pytest.raises(ValueError, match="did not execute"):
        module._default_training_step()
    assert not record["observed"]
    cleanup()


def test_rng_preserved_even_on_sampler_error():
    before = wb._capture_rng_state(current_cuda_only=True)
    with pytest.raises(RuntimeError):
        with preserved_rng():
            torch.rand(3); np.random.rand(3); random.random()
            raise RuntimeError("sample failure")
    assert equal_state(wb._capture_rng_state(current_cuda_only=True), before)
    assert state_digest(before) == state_digest(copy.deepcopy(before))


def original_config(mode):
    return OmegaConf.create(dict(
        model={"optimizer": {"lr": 5e-6}, "loss": {"mode": mode, "scale": 1.7}},
        data={"sampling_strategy": "full_episode_uniform", "boundary_policy": None if mode != "boundary" else {"jitter": 4}},
        ckpt_path=None, paths={"output_dir": "/original"}, val_at_start=False, val_at_end=False,
        trainer={"max_steps": 50, "val_check_interval": 50},
        callbacks={"joint_smoke": {"multimodal": True}, "model_checkpoint": {},
                   "usage": {"_target_": "egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage"}},
        logger={"csv": {"save_dir": "/original"}}, evaluator={"pixel_diagnostics": None}))


@pytest.mark.parametrize("mode", ["plain", "fused", "boundary"])
@pytest.mark.parametrize("steps", [(50, 52), (52, 56)])
def test_phase_changes_only_diagnostic_lifetime_preserving_science(mode, steps):
    original = original_config(mode)
    before = OmegaConf.to_container(original, resolve=True)
    cfg = phase_config(original, output="/diag", checkpoint="/ckpt", start_step=steps[0],
                       stop_step=steps[1], service_output="/proof", world=4)
    assert OmegaConf.to_container(original, resolve=True) == before
    assert cfg.model == original.model and cfg.data == original.data
    assert cfg.callbacks.resume_proof.multimodal
    assert cfg.trainer.max_steps == 56 and cfg.trainer.val_check_interval == 2
    assert ("allocation_boundary" in cfg.callbacks) == (steps[1] == 52)
    assert cfg.callbacks.model_checkpoint.save_top_k == -1


def test_phase_rejects_attempt_to_use_smoke_path_for_production_crash():
    with pytest.raises(ValueError, match="not production crash"):
        phase_config(original_config("plain"), output="/diag", checkpoint="/ckpt", start_step=20000,
                     stop_step=20002, service_output="/proof", world=4)


@pytest.mark.parametrize("mode", ["plain", "fused", "boundary"])
def test_checkpoint_science_pins_objective_and_sampling_even_if_weights_fit(mode):
    saved = original_config(mode)
    resumed = copy.deepcopy(saved)
    resumed.trainer.max_steps = 56
    checkpoint_science(saved, resumed)
    resumed.model.loss.scale = 2.3
    with pytest.raises(ValueError, match="model/loss/data/sampling"):
        checkpoint_science(saved, resumed)
    resumed = copy.deepcopy(saved)
    resumed.data.sampling_strategy = "different_policy"
    with pytest.raises(ValueError, match="model/loss/data/sampling"):
        checkpoint_science(saved, resumed)


def hidden_fixture():
    stream = {"value": torch.tensor([[[3.], [float("nan")], [float("nan")]]]),
              "observed": torch.tensor([[True, False, False]]),
              "valid": torch.tensor([[True, True, False]])}
    return {"world": {"image": stream}, "language_features": torch.ones(1, 1024), "pixel_metadata": [{}]}, {"image": torch.tensor([[[3.], [2.], [0.]]])}


@pytest.mark.parametrize("damage", [None, "leak", "nonfinite", "observed", "padding", "extra"])
def test_hidden_sample_requires_finite_outputs_clamped_context_and_no_labels(damage):
    sample, output = hidden_fixture()
    if damage == "leak": sample["world"]["image"]["value"][0, 1] = 8
    elif damage == "nonfinite": output["image"][0, 1] = float("nan")
    elif damage == "observed": output["image"][0, 0] = 4
    elif damage == "padding": output["image"][0, 2] = 7
    elif damage == "extra": output["action"] = output["image"]
    if damage is None: validate_hidden_sample(sample, output, ["image"])
    else:
        with pytest.raises(ValueError): validate_hidden_sample(sample, output, ["image"])


def phase_rows():
    net = TinyConditional()
    streams = {e: list(net.active_streams(e)) for e in net.pose_dims}
    result = []
    for start, stop in ((50, 52), (52, 56)):
        row = dict(multimodal=True, private_bank_optimizer_steps_verified=True, human_action_absent=True,
                   active_streams_by_embodiment=streams,
                   initial_parameter_usage={"shared": start, "private": start // 2, "unused": 0},
                   conditional_optimizer_record=dict(schema="native-conditional-optimizer-usage-v1", global_step=stop,
                     parameter_names=["shared", "private", "unused"],
                     parameter_update_counts={"shared": stop, "private": stop // 2, "unused": 0}),
                   embodiment_optimizer_batches={e: (stop-start)//2 for e in streams},
                   actual_rank_rng_restore=dict(observed=True, actual_sha256="a"*64, expected_sha256="a"*64),
                   validation_metrics={k: 0.01 for k in expected_metrics(net)},
                   hidden_label_samples={e: dict(sample_input_sha256="b"*64, state_steps=20,
                       action_steps=20 if e in net.action_dims else 0, streams=v,
                       finite=True, hidden_targets_nan=True) for e, v in streams.items()})
        result.append([copy.deepcopy(row) for _ in range(4)])
    return result


def test_summary_accepts_complete_synthetic_fixture_only():
    summary = summarize_native_phases(phase_rows())
    assert summary["diagnostic_scope"] == "scheduled-smoke-checkpoint-only"
    assert summary["private_bank_optimizer_steps_verified"]


@pytest.mark.parametrize("damage", ["legacy", "human_action", "rng", "nan_metric", "missing_metric", "missing_rank", "counter", "sample", "second_restore", "absent_human"])
def test_summary_rejects_incomplete_or_corrupt_all_rank_evidence(damage):
    phases = phase_rows(); row = phases[1][0]
    if damage == "legacy": row.pop("multimodal")
    elif damage == "human_action": row["active_streams_by_embodiment"]["human_bimanual"].append("action")
    elif damage == "rng": row["actual_rank_rng_restore"]["observed"] = False
    elif damage == "nan_metric": row["validation_metrics"][next(iter(row["validation_metrics"]))] = float("nan")
    elif damage == "missing_metric": row["validation_metrics"].popitem()
    elif damage == "missing_rank": phases[1].pop()
    elif damage == "counter": row["conditional_optimizer_record"]["parameter_update_counts"]["private"] = 56
    elif damage == "sample": row["hidden_label_samples"]["human_bimanual"]["hidden_targets_nan"] = False
    elif damage == "second_restore":
        for r in phases[1]: r["initial_parameter_usage"]["private"] = 25
    elif damage == "absent_human":
        for phase, count in zip(phases, (2, 4)):
            for r in phase: r["embodiment_optimizer_batches"] = {"eva_bimanual": count, "human_bimanual": 0}
    with pytest.raises(ValueError): summarize_native_phases(phases)
