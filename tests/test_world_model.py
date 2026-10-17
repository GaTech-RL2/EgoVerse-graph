import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from egomimic.models.world_model import MoTWorldModel
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_world_model import WorldModelStage
from egomimic.rldb.world_model import (
    HomogeneousWorldBatchSampler,
    PreparedWorldDataset,
    collate_world,
    points_to_reference,
    sample_times,
)


def make_batch(pose_dim=7, batch_size=2):
    result = {}
    for name, dim, n in (("image", 8, 6), ("track", 3, 4), ("proprio", pose_dim, 2)):
        time = (
            torch.arange(2)
            .repeat_interleave(n // 2)
            .float()
            .expand(batch_size, -1)
            .clone()
        )
        result[name] = dict(
            value=torch.randn(batch_size, n, dim),
            time=time,
            index=time.clone(),
            valid=torch.ones(batch_size, n, dtype=torch.bool),
            observed=time == 0,
            target_valid=torch.ones(batch_size, n, dtype=torch.bool),
        )
        if name != "proprio":
            result[name].update(
                xy=torch.rand(batch_size, n, 2),
                camera=torch.zeros(batch_size, n, dtype=torch.long),
            )
        if name == "track":
            result[name].update(
                point_id=torch.tensor([0, 1, 0, 1]).expand(batch_size, -1),
                confidence=torch.ones(batch_size, n),
            )
    return result


def make_model(**kwargs):
    return MoTWorldModel(
        {"robot": 7, "human": 11},
        image_dim=8,
        width=24,
        heads=3,
        depth=2,
        max_tracks=8,
        max_tokens=100,
        **kwargs,
    )


@pytest.mark.parametrize("chunk,rematerialize", [(3, False), (0, True), (3, True)])
def test_memory_controls_preserve_outputs_losses_gradients_and_update(chunk, rematerialize):
    torch.manual_seed(48)
    baseline = make_model().float()
    optimized = make_model(
        attention_query_chunk_size=chunk, activation_checkpointing=rematerialize
    ).float()
    optimized.load_state_dict(baseline.state_dict(), strict=True)
    batch = make_batch()
    # Include padding in one example and arbitrary observed/target ordering.
    for name in ("image", "track"):
        batch[name]["valid"][0, -1] = False
        batch[name]["target_valid"][0, -1] = False
        batch[name]["observed"][0, -1] = False
        batch[name]["observed"][1] = batch[name]["observed"][1].flip(0)
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in ("image", "track")}
    values = {n: batch[n]["value"] for n in ("image", "track")}
    with torch.no_grad():
        expected = baseline(batch, "robot", values, levels)
        actual = optimized(batch, "robot", values, levels)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=2e-5)
    losses = []
    for model in (baseline, optimized):
        torch.manual_seed(99)
        loss = model.training_losses(batch, "robot")
        loss["total"].backward()
        losses.append(loss)
    for name in losses[0]:
        torch.testing.assert_close(losses[0][name], losses[1][name], atol=2e-6, rtol=2e-5)
    for (name, a), (other, b) in zip(baseline.named_parameters(), optimized.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=1e-4, msg=name)
    for model in (baseline, optimized):
        torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01).step()
    for name, a in baseline.state_dict().items():
        torch.testing.assert_close(a, optimized.state_dict()[name], atol=2e-6, rtol=2e-5, msg=name)


def test_chunked_attention_keeps_all_keys(monkeypatch):
    from egomimic.models.world_model import MoTBlock
    calls = []
    original = MoTBlock._attend
    def observe(q, k, v, mask):
        calls.append((q.shape[2], k.shape[2], v.shape[2]))
        return original(q, k, v, mask)
    monkeypatch.setattr(MoTBlock, "_attend", staticmethod(observe))
    model, batch = make_model(attention_query_chunk_size=3), make_batch()
    with torch.no_grad():
        model(batch, "robot", {n: batch[n]["value"] for n in ("image", "track")},
              {n: torch.zeros_like(batch[n]["time"]) for n in ("image", "track")})
    assert calls and all(q <= 3 and k == v == 12 for q, k, v in calls)


def test_pose_gradient_reaches_image_and_track_generators():
    torch.manual_seed(7)
    model = make_model()
    losses = model.training_losses(make_batch(), "robot")
    losses["pose"].backward()
    for module in (model.image_out, model.track_out, model.image_in, model.track_in):
        assert module.weight.grad is not None
        assert module.weight.grad.norm() > 0
    for block in model.blocks:
        assert all(m.weight.grad.norm() > 0 for m in block.qkv)
    assert model.pose_out["human"].weight.grad is None


@pytest.mark.parametrize("active", ["image", "track", "proprio"])
def test_each_modality_alone_excludes_other_streams_and_their_gradients(active):
    """Isolated expert execution is a code contract, not learned real-data quality."""
    torch.manual_seed(42)
    model, batch = make_model().float(), make_batch()
    names = ("image", "track", "proprio")
    for name in names:
        if name != active:
            for key in ("valid", "observed", "target_valid"):
                batch[name][key].zero_()
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in names[:2]}
    values = {n: batch[n]["value"] for n in names[:2]}
    original = model(batch, "robot", values, levels)[active]
    changed = copy.deepcopy(batch)
    for name in names:
        if name != active:
            changed[name]["value"].fill_(float("nan"))
    mutated = model(
        changed, "robot", {n: changed[n]["value"] for n in names[:2]}, levels
    )[active]
    torch.testing.assert_close(original, mutated, rtol=0, atol=0)
    original.square().mean().backward()
    for block in model.blocks:
        for i, name in enumerate(names):
            grad = block.qkv[i].weight.grad
            if name == active:
                assert (
                    grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
                )
            else:
                assert grad is None or torch.equal(grad, torch.zeros_like(grad))


def test_oracle_visual_pose_readout_without_generator_uses_both_modalities():
    """Clean visual inputs isolate the readout; hidden pose targets cannot leak."""
    torch.manual_seed(42)
    model, batch = make_model().float(), make_batch()
    visual = {n: batch[n]["value"].clone().requires_grad_() for n in ("image", "track")}
    batch["proprio"]["value"].requires_grad_()
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in visual}
    prediction = model(batch, "robot", visual, levels)["proprio"]
    query = ~batch["proprio"]["observed"]
    prediction[query].square().mean().backward()
    for value in visual.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.norm() > 0
    pose_grad = batch["proprio"]["value"].grad
    assert pose_grad[batch["proprio"]["observed"]].norm() > 0
    assert torch.equal(pose_grad[query], torch.zeros_like(pose_grad[query]))
    assert model.image_out.weight.grad is None
    assert model.track_out.weight.grad is None


def test_visual_velocity_target_inverts_to_clean_image_and_track_independently():
    from egomimic.models.world_model import noise_coefficients

    torch.manual_seed(42)
    for width in (768, 3):
        clean, noise = torch.randn(2, 4, width), torch.randn(2, 4, width)
        level = torch.tensor([[0.0, 0.1, 0.9, 1.0]]).expand(2, -1)
        alpha, sigma = noise_coefficients(level)
        noisy = alpha * clean + sigma * noise
        velocity = alpha * noise - sigma * clean
        restored = alpha * noisy - sigma * velocity
        torch.testing.assert_close(restored, clean, rtol=1e-5, atol=1e-6)


def test_zero_initialized_adapters_preserve_adapter_free_predictions():
    model = make_model(image_adapter_dim=4).eval()
    baseline = make_model(image_adapter_dim=0).eval()
    baseline.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith("image_adapters.")
        },
        strict=True,
    )
    batch = make_batch()
    a = model.sample(batch, "robot", 2, torch.Generator().manual_seed(5))
    b = baseline.sample(batch, "robot", 2, torch.Generator().manual_seed(5))
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    # Old checkpoints must not silently skip the new trainable parameters.
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(baseline.state_dict(), strict=True)


@pytest.mark.parametrize(
    "selected", ["yam", "human", "libero_sharpa", "libero_gripper"]
)
def test_image_adapters_route_and_learn_for_all_four_embodiments(selected):
    torch.manual_seed(14)
    dims = dict(yam=7, human=11, libero_sharpa=9, libero_gripper=5)
    model = MoTWorldModel(
        dims,
        image_dim=8,
        width=24,
        heads=3,
        depth=2,
        max_tracks=8,
        max_tokens=100,
        image_adapter_dim=4,
    )
    untouched = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith("image_adapters.") and f".{selected}." not in name
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Pose loss alone must reach adapters through generated visual features.
        model.training_losses(make_batch(dims[selected]), selected)["pose"].backward()
        for block in model.image_adapters:
            adapter = block[selected]
            assert adapter.up.weight.grad.norm() > 0
            if step:
                assert adapter.down.weight.grad.norm() > 0
                assert adapter.norm.weight.grad.norm() > 0
            for other in dims.keys() - {selected}:
                assert all(p.grad is None for p in block[other].parameters())
        assert model.blocks[0].qkv[0].weight.grad.norm() > 0
        optimizer.step()
    for name, p in model.named_parameters():
        if name in untouched:
            torch.testing.assert_close(p, untouched[name], rtol=0, atol=0)


def test_adapter_effect_is_embodiment_specific_and_checkpointed():
    model = make_model(image_adapter_dim=4).eval()
    batches = {"robot": make_batch(7), "human": make_batch(11)}

    def predict(embodiment):
        return model.sample(
            batches[embodiment], embodiment, 2, torch.Generator().manual_seed(9)
        )["image"]

    before = {e: predict(e) for e in batches}
    with torch.no_grad():
        for block in model.image_adapters:
            block["robot"].up.weight.normal_(std=0.1)
    assert not torch.equal(before["robot"], predict("robot"))
    torch.testing.assert_close(before["human"], predict("human"), rtol=0, atol=0)
    restored = make_model(image_adapter_dim=4).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        predict("robot"),
        restored.sample(batches["robot"], "robot", 2, torch.Generator().manual_seed(9))[
            "image"
        ],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dimension", [-1, 1.5, True])
def test_invalid_image_adapter_width_is_rejected(dimension):
    with pytest.raises(ValueError, match="image_adapter_dim"):
        make_model(image_adapter_dim=dimension)


@pytest.mark.parametrize("embodiment,dimension", [("robot", 7), ("human", 11)])
def test_actual_optimizer_and_strict_reload(embodiment, dimension):
    torch.manual_seed(10)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.track_out.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        loss = model.training_losses(make_batch(dimension), embodiment)["total"]
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, model.track_out.weight)
    replacement = make_model()
    replacement.load_state_dict(model.state_dict(), strict=True)
    new_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3)
    new_optimizer.load_state_dict(optimizer.state_dict())
    assert len(new_optimizer.state) == len(optimizer.state)
    batch = make_batch(dimension)
    a = model.sample(batch, embodiment, 3, torch.Generator().manual_seed(8))
    b = replacement.sample(batch, embodiment, 3, torch.Generator().manual_seed(8))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_sampler_does_not_read_withheld_ground_truth():
    model = make_model().eval()
    batch = make_batch()
    changed = copy.deepcopy(batch)
    for stream in changed.values():
        stream["value"][~stream["observed"]] = float("nan")
        stream["target_valid"][:] = False
        if "confidence" in stream:
            stream["confidence"][:] = 0
    a = model.sample(batch, "robot", 2, torch.Generator().manual_seed(4))
    b = model.sample(changed, "robot", 2, torch.Generator().manual_seed(4))
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        torch.testing.assert_close(
            a[name][batch[name]["observed"]],
            batch[name]["value"][batch[name]["observed"]],
        )


def test_hidden_pose_does_not_condition_first_pass():
    model = make_model()
    batch = make_batch()
    other = copy.deepcopy(batch)
    other["proprio"]["value"][~other["proprio"]["observed"]] += 900
    values = {n: batch[n]["value"] for n in ("image", "track")}
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in values}
    a, b = model(batch, "robot", values, levels), model(other, "robot", values, levels)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


def refinement_batch():
    batch = make_batch(batch_size=1)
    for stream in batch.values():
        for key, value in list(stream.items()):
            stream[key] = torch.cat((value, value.clone()), dim=1)
        time = (
            torch.arange(4)
            .repeat_interleave(stream["value"].shape[1] // 4)[None]
            .float()
        )
        stream["time"], stream["index"] = time, time.clone()
        stream["observed"] = time == 0
    batch["refinement"] = {}
    for name in ("image", "track"):
        s = batch[name]
        strength = torch.tensor([0.0, 0.0, 0.4, 1.0])[s["time"].long()]
        previous = torch.randn_like(s["value"])
        previous[strength == 1] = float("nan")  # Fresh queries need no previous value.
        batch["refinement"][name] = dict(strength=strength, prediction=previous)
        s["value"][~s["observed"]] = float("nan")  # Never use held-out ground truth.
    return batch


@pytest.mark.parametrize("mode,mixed", [("continuous", False), ("discrete", True)])
def test_selective_refinement_schedule_and_clean_anchor_clamping(mode, mixed):
    from egomimic.models.world_model import noise_coefficients

    model = make_model(time_mode=mode).eval()
    batch = refinement_batch()
    original = copy.deepcopy(batch)
    calls = []
    handle = model.register_forward_pre_hook(
        lambda module, args: calls.append(
            (
                {k: v.clone() for k, v in args[2].items()},
                {k: v.clone() for k, v in args[3].items()},
            )
        )
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
        output = model.sample(batch, "robot", 4, torch.Generator().manual_seed(29))
    handle.remove()
    assert len(calls) == 5  # Four joint denoising passes, then pose readout.
    rng = torch.Generator().manual_seed(29)
    for name in ("image", "track"):
        s, control = batch[name], batch["refinement"][name]
        strength, previous = control["strength"], control["prediction"]
        noise = torch.randn(s["value"].shape, generator=rng)
        partial = strength == 0.4
        alpha, sigma = noise_coefficients(strength)
        expected = alpha * previous + sigma * noise
        torch.testing.assert_close(calls[0][0][name][partial], expected[partial])
        fresh = strength == 1
        torch.testing.assert_close(
            calls[0][0][name][fresh], noise[fresh], rtol=0, atol=0
        )
        retained = (strength == 0) & ~s["observed"]
        for values, levels in calls:
            torch.testing.assert_close(
                values[name][retained], previous[retained], rtol=0, atol=0
            )
            torch.testing.assert_close(
                values[name][s["observed"]], s["value"][s["observed"]], rtol=0, atol=0
            )
        for j, (_, levels) in enumerate(calls[:4]):
            torch.testing.assert_close(
                levels[name], strength * ((4 - j) / 4), rtol=0, atol=0
            )
        torch.testing.assert_close(
            output[name][retained], previous[retained], rtol=0, atol=0
        )
        assert torch.isfinite(output[name]).all()
        assert not torch.equal(output[name][partial], previous[partial])
        for key in s:
            torch.testing.assert_close(
                s[key], original[name][key], rtol=0, atol=0, equal_nan=True
            )
    assert torch.isfinite(output["proprio"]).all()


def test_refinement_fresh_controls_equal_default_and_ignore_seed_values():
    model = make_model().eval()
    batch = make_batch()
    baseline = model.sample(batch, "robot", 3, torch.Generator().manual_seed(31))
    for name in ("image", "track"):
        s = batch[name]
        batch.setdefault("refinement", {})[name] = dict(
            strength=(~s["observed"]).float(),
            prediction=torch.full_like(s["value"], float("nan")),
        )
    controlled = model.sample(batch, "robot", 3, torch.Generator().manual_seed(31))
    for name in baseline:
        torch.testing.assert_close(baseline[name], controlled[name], rtol=0, atol=0)


def test_repeat_refinement_and_new_timestamps_through_pipeline():
    model = make_model().eval()
    batch = refinement_batch()
    first = model.sample(batch, "robot", 2, torch.Generator().manual_seed(3))
    # Preserve the generated frame at t=2, revise t=1, generate t=3 afresh.
    for name in ("image", "track"):
        batch["refinement"][name] = dict(
            strength=torch.tensor([0.0, 0.6, 0.0, 1.0])[batch[name]["time"].long()],
            prediction=first[name],
        )
        # Change the fresh query's physical time, preserving other token identities.
        batch[name]["time"][batch[name]["time"] == 3] = 4.5
    algo = PipelineAlgo([WorldModelStage(model, sampler_steps=2)], device="cpu")
    raw = {"robot": {"world": batch, "embodiment": "robot"}}
    second = algo.forward_eval(raw)["robot"]["world_prediction"]
    for name in ("image", "track"):
        retained = batch[name]["time"] == 2
        torch.testing.assert_close(
            second[name][retained], first[name][retained], rtol=0, atol=0
        )
        assert torch.isfinite(second[name]).all()


@pytest.mark.parametrize(
    "error",
    [
        "range",
        "observed",
        "missing",
        "nan",
        "frame",
        "shape",
        "proprio",
    ],
)
def test_refinement_invalid_requests_fail_before_rng_use(error):
    model, batch = make_model(), refinement_batch()
    c = batch["refinement"]["image"]
    if error == "range":
        c["strength"][0, -1] = 1.1
    elif error == "observed":
        c["strength"][0, 0] = 0.1
    elif error == "missing":
        del c["prediction"]
    elif error == "nan":
        c["prediction"][0, 3] = float("nan")
    elif error == "frame":
        c["strength"][0, 3] = 0.2
    elif error == "shape":
        c["strength"] = c["strength"][:, :-1]
    else:
        batch["refinement"]["proprio"] = {}
    generator = torch.Generator().manual_seed(19)
    state = generator.get_state().clone()
    with pytest.raises(ValueError):
        model.sample(batch, "robot", 2, generator)
    assert torch.equal(state, generator.get_state())


def test_padding_invariance():
    model = make_model().eval()
    batch = make_batch()
    padded = copy.deepcopy(batch)
    for stream in padded.values():
        for key, x in list(stream.items()):
            extra = torch.zeros_like(x[:, :1])
            if key == "value":
                extra[:] = float("nan")
            stream[key] = torch.cat((x, extra), 1)
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in ("image", "track")}
    values = {n: batch[n]["value"] for n in levels}
    a = model(batch, "robot", values, levels)
    b = model(
        padded,
        "robot",
        {n: padded[n]["value"] for n in levels},
        {n: torch.zeros_like(padded[n]["time"]) for n in levels},
    )
    for name in a:
        torch.testing.assert_close(a[name], b[name][:, :-1], atol=2e-6, rtol=2e-5)


def test_common_time_shift_invariance_and_discrete_mode():
    model = make_model().eval()
    batch, shifted = make_batch(), None
    shifted = copy.deepcopy(batch)
    for s in shifted.values():
        s["time"] += 3.5
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in ("image", "track")}
    values = {n: batch[n]["value"] for n in levels}
    a, b = (
        model(batch, "robot", values, levels),
        model(shifted, "robot", values, levels),
    )
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=2e-6, rtol=2e-5)
    discrete = make_model(time_mode="discrete")
    discrete.load_state_dict(model.state_dict())
    for name in a:
        torch.testing.assert_close(
            a[name], discrete(batch, "robot", values, levels)[name]
        )
    del shifted["image"]["index"]
    with pytest.raises(ValueError, match="indices"):
        discrete(shifted, "robot", values, levels)


def test_budget_guard_and_warmup_in_pipeline():
    model = make_model()
    stage = WorldModelStage(model, warmup_steps=3, pose_ramp_steps=2)
    assert [stage.pose_scale(i) for i in range(6)] == [0, 0, 0, 0, 0.5, 1]
    algo = PipelineAlgo([stage], device="cpu")
    raw = {"robot": {"world": make_batch(), "embodiment": "robot", "optimizer_step": 0}}
    predictions = algo.forward_training(raw)
    loss = algo.compute_losses(predictions, raw)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert predictions["robot"]["log/world_pose"] == 0
    assert "optimizer_step" not in raw["robot"]["world"]
    model.max_tokens = 2
    with pytest.raises(ValueError, match="budget"):
        model.sample(make_batch(), "robot")


def test_temporal_sampling_and_rigid_frame():
    counts = set()
    rng = np.random.default_rng(2)
    for _ in range(50):
        selected = sample_times(100, rng)
        counts.add(len(selected))
        assert (np.diff(selected) > 0).all()
    assert counts == {2, 3, 4}
    selected = sample_times(100, rng, mode="discrete", stride=3)
    assert (np.diff(selected) == 3).all()
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(
        points_to_reference(np.zeros((2, 3)), transform), [[1, 2, 3]] * 2
    )
    transform[0, 0] = -1
    with pytest.raises(ValueError, match="handedness"):
        points_to_reference(np.zeros((2, 3)), transform)


def test_sixteen_timestamps_preserve_physical_window():
    times = np.linspace(0, 12, 361, dtype=np.float64)
    counts = set()
    for seed in range(200):
        selected = sample_times(
            len(times), np.random.default_rng(seed), anchor=180, max_frames=16,
            timestamps=times, max_past_seconds=3, max_future_seconds=3,
        )
        counts.add(len(selected))
        assert 180 in selected
        assert (np.diff(selected) > 0).all()
        assert (np.abs(times[selected] - times[180]) <= 3).all()
    assert counts == set(range(2, 17))


def test_physical_time_bounds_use_seconds_and_cover_both_sides():
    times = np.array([0, 2, 5.9, 6, 7, 9, 10, 12, 12.1, 40], dtype=np.float64)
    counts, seen = set(), set()
    for seed in range(100):
        selected = sample_times(
            len(times), np.random.default_rng(seed), anchor=5, timestamps=times,
            max_past_seconds=3, max_future_seconds=3,
        )
        assert 5 in selected
        assert (np.diff(selected) > 0).all()
        assert ((times[selected] - times[5] >= -3) & (times[selected] - times[5] <= 3)).all()
        counts.add(len(selected))
        seen.update(selected)
    assert counts == {2, 3, 4}
    assert seen == {3, 4, 5, 6, 7}
    # The exact recorded endpoints are included; no nearest frame may cross them.
    selected = sample_times(
        5, np.random.default_rng(1), anchor=2, min_frames=3, max_frames=3,
        timestamps=np.array([6.9, 7, 10, 13, 13.1]),
        max_past_seconds=3, max_future_seconds=3,
    )
    np.testing.assert_array_equal(selected, [1, 2, 3])


@pytest.mark.parametrize("anchor,expected", [(0, [0, 1, 2, 3]), (7, [4, 5, 6, 7])])
def test_physical_sampling_clips_episode_edges_without_gap_escape(anchor, expected):
    times = np.array([0, 1, 2, 3, 100, 101, 102, 103], dtype=np.float64)
    selected = sample_times(
        len(times), np.random.default_rng(9), anchor=anchor,
        min_frames=4, max_frames=4, timestamps=times,
        max_past_seconds=3, max_future_seconds=3,
    )
    np.testing.assert_array_equal(selected, expected)
    with pytest.raises(ValueError, match="Too few recorded"):
        sample_times(
            3, np.random.default_rng(1), anchor=1, timestamps=[0, 10, 20],
            max_past_seconds=3, max_future_seconds=3,
        )


@pytest.mark.parametrize("past,future", [(None, 3), (3, None), (-1, 3), (3, float("nan")),
                                         (True, 3), (0, 0), (3, float("inf"))])
def test_physical_sampling_rejects_invalid_bounds(past, future):
    with pytest.raises(ValueError, match="Time bounds"):
        sample_times(4, np.random.default_rng(1), timestamps=[0, 1, 2, 3],
                     max_past_seconds=past, max_future_seconds=future)


def test_physical_sampling_requires_valid_times_and_rejects_discrete_bounds():
    for times in (None, [0, 1], [0, 1, 1, 3], [0, 1, np.nan, 3]):
        with pytest.raises(ValueError, match="Physical timestamps"):
            sample_times(4, np.random.default_rng(1), timestamps=times,
                         max_past_seconds=3, max_future_seconds=3)
    with pytest.raises(ValueError, match="only to continuous"):
        sample_times(4, np.random.default_rng(1), mode="discrete",
                     max_past_seconds=3, max_future_seconds=3)


def cache_manifest(tmp_path):
    entries = []
    for embodiment, d in (("robot", 7), ("human", 11)):
        for split in ("train", "val"):
            path = tmp_path / f"{embodiment}-{split}.npz"
            rng = np.random.default_rng(1)
            np.savez(
                path,
                time=np.arange(8, dtype=np.float64) * 0.1 + 1e9,
                image=rng.normal(size=(8, 2, 3, 8)),
                image_xy=rng.random((3, 2)),
                track=rng.normal(size=(8, 2, 3)),
                track_xy=rng.random((2, 2)),
                track_camera=np.zeros(2, dtype=np.int64),
                track_confidence=np.ones((8, 2)),
                pose=rng.normal(size=(8, d)),
                pose_valid=np.ones(8, dtype=bool),
            )
            entries.append(
                dict(
                    embodiment=embodiment,
                    episode_id=path.stem,
                    path=path.name,
                    frames=8,
                    split=split,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    frame="fixed_reference",
                    pose_kind="measured_pose",
                )
            )
    manifest = dict(
        schema="mot-world-prepared-v1",
        dino_identity={
            "model": "test-fixture",
            "revision": "fixture",
            "image_size": [2, 6],
            "patch_grid": [1, 3],
            "feature_layer": "last",
        },
        normalization={
            "scope": "train_only",
            "sha256": "0" * 64,
            "arrays_normalized": True,
        },
        pose_schemas={
            name: {"dimension": dim, "fields": [f"coordinate_{i}" for i in range(dim)]}
            for name, dim in (("robot", 7), ("human", 11))
        },
        track_provenance={
            "interpolated_endpoints": False,
            "method": "synthetic_fixture",
            "reference_frame": "world",
            "calibration_sha256": "0" * 64,
        },
        episodes=entries,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_prepared_cache_pipeline_and_reject_overlap(tmp_path):
    path = cache_manifest(tmp_path)
    dataset = PreparedWorldDataset(path)
    sampler = HomogeneousWorldBatchSampler(dataset, batch_size=2, shuffle=False)
    for indices in sampler:
        batch = collate_world([dataset[i] for i in indices])
        embodiment, values = next(iter(batch.items()))
        result = make_model().training_losses(values["world"], embodiment)
        assert torch.isfinite(result["total"])
    manifest = json.loads(path.read_text())
    manifest["episodes"].append(manifest["episodes"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        PreparedWorldDataset(path)


def test_bounded_dataset_filters_isolated_anchors_and_restores_context(tmp_path):
    from egomimic.rldb.world_model_datamodule import WorldModelDataModule
    path = cache_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    for episode in manifest["episodes"]:
        file = path.parent / episode["path"]
        with np.load(file, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        arrays["time"] = np.array([0, 10, 20, 30, 100, 100.1, 100.2, 100.3])
        np.savez(file, **arrays)
        episode["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    dataset = PreparedWorldDataset(path, split="val", max_past_seconds=3, max_future_seconds=3)
    assert dataset.counts == [4, 4]
    for anchors in dataset.anchor_indices:
        np.testing.assert_array_equal(anchors, [4, 5, 6, 7])
    for index in range(len(dataset)):
        sample = dataset[index]
        episode = dataset.episodes[index // 4]
        with np.load(path.parent / episode["path"], allow_pickle=False) as saved:
            expected_anchor_pose = saved["pose"][4 + index % 4].astype(np.float32)
        poses = sample["world"]["proprio"]["value"].numpy()
        assert any(np.array_equal(p, expected_anchor_pose) for p in poses)
        assert sample["world"]["proprio"]["time"].abs().max() < .31
    data = WorldModelDataModule(str(path), max_past_seconds=3, max_future_seconds=3)
    context = data.prepare_context(mode="train", normalization=None, normalizer=None)
    assert context.state["sampling"]["max_past_seconds"] == 3
    assert context.state["sampling"]["max_future_seconds"] == 3
    changed = WorldModelDataModule(str(path), max_past_seconds=2, max_future_seconds=3)
    with pytest.raises(ValueError, match="checkpoint/data context mismatch"):
        changed.prepare_context(mode="train", normalization=None, normalizer=None,
                                restored_state=context.state)


def test_reject_interpolated_track_targets(tmp_path):
    path = cache_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["track_provenance"]["interpolated_endpoints"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="interpolation"):
        PreparedWorldDataset(path)


def test_bf16_mixed_finite_gradients():
    model = make_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.training_losses(make_batch(), "robot")["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_hydra_both_recipes_and_required_schema():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    for mode in ("continuous", "discrete"):
        with initialize_config_dir(config_dir=str(root), version_base=None):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[f"+experiment=world_model/{mode}", "hydra/launcher=basic"],
            )
        assert cfg.data.mode == mode
        assert cfg.data.min_frames == 2
        assert cfg.data.max_frames == 16
        assert cfg.data.discrete_frames == 4
        assert cfg.model.pipeline.stages[0].model.max_tokens == 8192
        assert cfg.model.pipeline.stages[0].model.time_mode == mode
        assert cfg.trainer.check_val_every_n_epoch is None
        assert cfg.callbacks.model_checkpoint.every_n_train_steps == 15000
        expected_bound = 3.0 if mode == "continuous" else None
        assert cfg.data.max_past_seconds == expected_bound
        assert cfg.data.max_future_seconds == expected_bound
        assert OmegaConf.is_missing(cfg.model.pipeline.stages[0].model, "pose_dims")
        assert cfg.evaluator._target_.endswith("WorldModelEval")


def test_shared_lightning_fit_validate_and_resume(tmp_path):
    from lightning import Trainer
    from omegaconf import OmegaConf

    from egomimic.eval.world_model_eval import WorldModelEval
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.world_model_datamodule import WorldModelDataModule

    path = cache_manifest(tmp_path)
    data = WorldModelDataModule(str(path), batch_size=2, max_frames=3)
    context = data.prepare_context(mode="train", normalization=None, normalizer=None)
    with pytest.raises(ValueError, match="mismatch"):
        data.prepare_context(
            mode="train", normalization=None, normalizer=None, restored_state={}
        )
    cfg = OmegaConf.create(
        {
            "model": {
                "pipeline": {
                    "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                    "device": "cpu",
                    "stages": [
                        {
                            "_target_": "egomimic.pipeline.stages_world_model.WorldModelStage",
                            "warmup_steps": 0,
                            "pose_ramp_steps": 0,
                            "sampler_steps": 2,
                            "model": {
                                "_target_": "egomimic.models.world_model.MoTWorldModel",
                                "pose_dims": {"robot": 7, "human": 11},
                                "image_dim": 8,
                                "width": 24,
                                "heads": 3,
                                "depth": 1,
                                "max_tracks": 8,
                            },
                        }
                    ],
                },
                "training_behavior": {
                    "_target_": "egomimic.pl_utils.world_model_behavior.WorldModelTrainingBehavior"
                },
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "_partial_": True,
                    "lr": 0.001,
                },
            }
        }
    )
    wrapper = ModelWrapper(config_tree=cfg, evaluator=WorldModelEval())
    wrapper.data_context = context
    context.bind(wrapper.model, wrapper.evaluator)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        limit_val_batches=1,
        val_check_interval=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(wrapper, datamodule=data)
    assert trainer.global_step == 2
    assert any(k.startswith("val/") for k in trainer.callback_metrics)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = ModelWrapper(config_tree=cfg, evaluator=WorldModelEval())
    restored.data_context = context
    trainer2 = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        max_epochs=2,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer2.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert trainer2.global_step == 3


def test_invalid_unobserved_track_targets_are_masked():
    model = make_model()
    batch = make_batch()
    batch["track"]["value"][:, -1] = float("nan")
    batch["track"]["target_valid"][:, -1] = False
    batch["track"]["confidence"][:, -1] = 0
    with pytest.raises(ValueError, match="zero-fill is forbidden"):
        model.training_losses(batch, "robot")
    loss = model.training_losses(
        batch, "robot", missing_track_policy="conditional_sample", imputation_steps=2
    )["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_noise_levels_are_per_frame_and_observed_are_clean():
    stream = make_batch()["image"]
    levels = MoTWorldModel.frame_noise_levels(stream)
    assert torch.equal(levels[:, :3], torch.zeros_like(levels[:, :3]))
    assert torch.equal(levels[:, 3], levels[:, 4])
    assert torch.equal(levels[:, 4], levels[:, 5])
    assert ((levels[:, 3:] > 0) & (levels[:, 3:] < 1)).all()


def test_arbitrary_conditioning_includes_backcasting_and_endpoint_interpolation(
    tmp_path,
):
    torch.manual_seed(42)
    dataset = PreparedWorldDataset(
        cache_manifest(tmp_path),
        min_frames=4,
        max_frames=4,
        forecasting_probability=0,
    )
    backcast = interpolation = False
    for i in range(100):
        stream = dataset[i % len(dataset)]["world"]["image"]
        times, mask = stream["time"], stream["observed"]
        unique = torch.unique(times, sorted=True)
        observed = torch.stack([mask[times == t].all() for t in unique])
        assert observed.any() and (~observed).any()
        backcast |= bool(~observed[0] & observed[-1])
        interpolation |= bool(observed[0] & observed[-1] & (~observed[1:-1]).any())
    assert backcast, "No image query before the earliest image observation"
    assert interpolation, "No withheld image between observed endpoint images"


def test_fractional_confidence_mse_is_a_weighted_mean():
    from egomimic.models.world_model import masked_mse

    pred = torch.tensor([[[2.0], [4.0]]], requires_grad=True)
    target = torch.zeros_like(pred)
    weight = torch.tensor([[0.1, 0.2]])
    loss = masked_mse(pred, target, weight)
    torch.testing.assert_close(loss, torch.tensor(12.0))
    torch.testing.assert_close(loss, masked_mse(pred, target, weight * 0.1))
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    pred.grad = None
    empty = masked_mse(pred, torch.full_like(target, float("nan")), weight * 0)
    assert empty.item() == 0
    empty.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_declared_dino_patch_grid_must_match_cache(tmp_path):
    path = cache_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["dino_identity"]["patch_grid"] = [2, 3]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="patch grid"):
        PreparedWorldDataset(path)[0]


@pytest.mark.parametrize("shape", [[0, 3], [True, 3], [3], "1x3", [1.0, 3]])
def test_declared_dino_grid_requires_positive_integer_shape(tmp_path, shape):
    path = cache_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["dino_identity"]["patch_grid"] = shape
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="two positive integers"):
        PreparedWorldDataset(path)


def test_complete_tracks_bypass_imputation_and_preserve_rng(monkeypatch):
    model, batch = make_model(), make_batch()

    def forbidden(*args, **kwargs):
        raise AssertionError("Complete tracks must not invoke a teacher")

    monkeypatch.setattr(model, "sample", forbidden)
    torch.manual_seed(19)
    a = model.training_losses(batch, "robot")
    rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    b = model.training_losses(batch, "robot", missing_track_policy="conditional_sample")
    assert torch.equal(rng, torch.get_rng_state())
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert a["imputed_track_tokens"] == a["imputation_forwards"] == 0


def test_missing_tracks_use_detached_completion_not_zero_geometry(monkeypatch):
    from egomimic.models.world_model import noise_coefficients

    model, batch = make_model(), make_batch()
    batch["track"]["value"][:, -1] = float("nan")
    batch["track"]["target_valid"][:, -1] = False
    batch["track"]["confidence"][:, -1] = 0
    before = copy.deepcopy(batch)
    completed = torch.full_like(batch["track"]["value"], 7.0, requires_grad=True)

    def teacher(teacher_batch, embodiment, steps):
        assert not torch.is_grad_enabled()
        assert steps == 2
        for name in ("image", "track"):
            assert torch.isfinite(teacher_batch[name]["value"]).all()
        pose = teacher_batch["proprio"]
        assert torch.equal(pose["observed"], batch["proprio"]["observed"])
        assert (pose["value"][~pose["observed"]] == 0).all()
        assert not teacher_batch["track"]["observed"][:, -1].any()
        return {"track": completed}

    monkeypatch.setattr(model, "sample", teacher)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    calls = []
    original = model.forward

    def capture(b, e, values, levels):
        calls.append((values, levels))
        return original(b, e, values, levels)

    monkeypatch.setattr(model, "forward", capture)
    loss = model.training_losses(
        batch, "robot", missing_track_policy="conditional_sample", imputation_steps=2
    )
    values, levels = calls[0]
    alpha, _ = noise_coefficients(levels["track"])
    torch.testing.assert_close(values["track"][:, -1], 7 * alpha[:, -1].expand(-1, 3))
    assert torch.equal(levels["track"][:, -1], levels["track"][:, -2])
    assert loss["imputed_track_tokens"] == 2
    assert loss["imputation_forwards"] == 3
    loss["total"].backward()
    assert completed.grad is None
    assert model.track_out.weight.grad.norm() > 0
    for name in batch:
        for key in batch[name]:
            torch.testing.assert_close(
                batch[name][key], before[name][key], equal_nan=True
            )


@pytest.mark.parametrize("bf16", [False, True])
def test_conditional_completion_ignores_placeholders_and_withheld_poses(bf16):
    model, batch = make_model(), make_batch()
    # Even a finite coordinate with zero confidence must not become a target.
    batch["track"]["confidence"][:, -1] = 0
    batch["track"]["value"][:, -1] = float("nan")
    other = copy.deepcopy(batch)
    other["track"]["value"][:, -1] = 1e9
    other["proprio"]["value"][~other["proprio"]["observed"]] = float("nan")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        torch.manual_seed(77)
        a = model.training_losses(
            batch,
            "robot",
            pose_weight=0,
            missing_track_policy="conditional_sample",
            imputation_steps=2,
        )
        torch.manual_seed(77)
        b = model.training_losses(
            other,
            "robot",
            pose_weight=0,
            missing_track_policy="conditional_sample",
            imputation_steps=2,
        )
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    a["total"].backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_training_rejects_unavailable_observations_and_image_targets():
    model, batch = make_model(), make_batch()
    batch["track"]["confidence"][:, 0] = 0
    with pytest.raises(ValueError, match="Observed visual"):
        model.training_losses(batch, "robot", missing_track_policy="conditional_sample")
    batch = make_batch()
    batch["image"]["target_valid"][:, -1] = False
    with pytest.raises(ValueError, match="Missing image targets"):
        model.training_losses(batch, "robot", missing_track_policy="conditional_sample")


def test_pipeline_exposes_missing_track_policy_and_cost():
    model, world = make_model(), make_batch()
    world["track"]["target_valid"][:, -1] = False
    world["track"]["value"][:, -1] = float("nan")
    stage = WorldModelStage(
        model, missing_track_policy="conditional_sample", imputation_steps=2
    )
    result = stage.execute(
        dict(world=world, embodiment="robot", optimizer_step=0), mode="train"
    )
    assert torch.isfinite(result["loss/world"])
    assert result["log/world_imputed_track_tokens"] == 2
    assert result["log/world_imputation_forwards"] == 3
