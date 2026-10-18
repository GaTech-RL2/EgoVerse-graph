"""Benchmark protocol checks; these CPU tests make no H100 speed claim."""

import json
import sys
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from egomimic.models.image_proprio import ImageProprioModel
from scripts.world_model import benchmark_pixel_execution as benchmark


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_mixed_schedule_matches_frame_histogram_across_batch_sizes(batch_size):
    schedule = benchmark.frame_schedule("mixed", batch_size)
    histogram = Counter(frame for batch in schedule for frame in batch)
    assert set(histogram) == {2, 16, 64, 128}
    assert len(set(histogram.values())) == 1
    assert all(len(batch) == batch_size for batch in schedule)
    assert benchmark.complete_cycles(10, schedule) % len(schedule) == 0
    assert benchmark.complete_cycles(10, schedule) >= 10


@pytest.mark.parametrize("scenario,capacity", [("fixed16", 3200), ("mixed", 25600), ("worst128", 25600)])
def test_benchmark_binds_exact_architecture_and_explicit_loss(scenario, capacity):
    arguments = benchmark.model_arguments("flash4", scenario)
    with torch.device("meta"):
        model = ImageProprioModel(**arguments)
    assert sum(p.numel() for p in model.parameters()) == 88_178_688
    assert not model.use_phase_embedding
    assert model.noise_conditioning == "adaln_zero"
    assert model.position_encoding == "spatiotemporal_rope"
    assert not hasattr(model, "xy")
    assert not any(name.startswith("phase.") for name in model.state_dict())
    assert model.max_tokens == capacity
    assert model.loss_reduction == "mean_per_example"
    assert model.pose_dims == model.action_dims == {}
    assert model.modalities == ("image",)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def small_batch():
    return benchmark.make_batch((2, 3), seed=17, device=torch.device("cpu"), image_size=4, patch_size=2)


def small_model():
    return ImageProprioModel(
        image_dim=12, width=16, depth=1, heads=2, max_cameras=2, max_tokens=32,
        objective="visual_only", phase_a_condition_on_proprio=False,
        attention_backend="sdpa", attention_dtype="bfloat16",
        loss_reduction="mean_per_example",
    )


def test_geometry_padding_and_unpadded_independent_examples():
    batch = small_batch()
    image = batch["image"]
    assert image["value"].shape == (2, 24, 12)
    assert image["valid"].sum(1).tolist() == [16, 24]
    assert image["observed"].sum(1).tolist() == [8, 8]
    assert image["time"].dtype == torch.float32
    assert image["value"].dtype == torch.float32
    assert image["camera"].unique().tolist() == [0, 1]
    single = benchmark.single_example(batch, 0)["image"]
    assert single["value"].shape == (1, 16, 12)
    assert single["valid"].all()
    torch.testing.assert_close(single["value"], image["value"][0:1, :16])


def test_mean_per_example_is_not_padding_or_token_weighted():
    batch = small_batch()
    batch["image"]["value"].zero_()
    batch["image"]["value"][1] = 1
    loss = benchmark.example_mean_loss(torch.zeros_like(batch["image"]["value"]), batch)
    assert loss == 0.5  # Unequal query-token counts must not weight example 1 twice.


def test_batched_probe_checks_outputs_gradients_and_cross_example_isolation():
    # Flash4's context means standard BF16 autocast. The tiny CPU model uses
    # native SDPA explicitly; hardware/kernel certification happens on CUDA.
    result = benchmark.check_outputs(small_model(), small_batch(), backend="flash4",
                                     seed=42, tolerance=0.03, gradients=True)
    assert result["output_relative_l2"] <= 0.03
    assert result["gradient_relative_l2"] <= 0.03
    assert result["cross_example_relative_l2"] <= 1e-6


def test_measured_training_loss_matches_explicit_mean_independent_example_contract(monkeypatch):
    model, batch = small_model(), small_batch()
    noisy, levels = benchmark.fixed_corruption(batch, 42)
    monkeypatch.setattr(model, "_noisy_images", lambda *args, **kwargs: (noisy, levels))
    with benchmark.execution_context("flash4", torch.device("cpu")):
        prediction = model(batch, "libero_gripper", noisy, levels, phase="A")["image"]
        explicit = benchmark.example_mean_loss(prediction, batch)
        losses = model.training_losses(batch, "libero_gripper", pose_weight=0.0)
    assert losses["total"].dtype == torch.float32
    torch.testing.assert_close(losses["total"], explicit, rtol=0, atol=0)


def test_eager_compiled_gate_accepts_equal_execution_and_rejects_bad_backward():
    # Exercise the comparison gate without claiming CPU execution certifies
    # CUDA Inductor. The scheduled harness supplies an actual compiled model.
    eager, candidate = small_model(), small_model()
    candidate.load_state_dict(eager.state_dict(), strict=True)
    result = benchmark.check_eager_compiled(eager, candidate, small_batch(),
                                           backend="flash4", seed=42, tolerance=0.03)
    assert result["same_state_verified"]
    assert result["output_relative_l2"] == result["gradient_relative_l2"] == 0
    assert result["loss_relative_error"] == 0
    hooks = [p.register_hook(lambda gradient: gradient * 2) for p in candidate.parameters()]
    with pytest.raises(benchmark.CorrectnessError, match="Compiled gradient differs from eager"):
        benchmark.check_eager_compiled(eager, candidate, small_batch(),
                                       backend="flash4", seed=42, tolerance=0.03)
    assert all(p.grad is None for model in (eager, candidate) for p in model.parameters())
    for hook in hooks:
        hook.remove()


def test_eager_compiled_gate_rejects_unmatched_weights():
    with pytest.raises(benchmark.CorrectnessError, match="initial state differs"):
        benchmark.check_eager_compiled(small_model(), small_model(), small_batch(),
                                       backend="flash4", seed=42, tolerance=0.03)


def test_probe_rejects_cross_example_interference():
    model = small_model()
    forward = model.forward

    def leaking_forward(batch, *args, **kwargs):
        result = forward(batch, *args, **kwargs)
        result["image"] = result["image"] + batch["image"]["value"].mean() * 10
        return result

    model.forward = leaking_forward
    with pytest.raises(benchmark.CorrectnessError, match="independent B1|leakage"):
        benchmark.check_outputs(model, small_batch(), backend="flash4", seed=42,
                                tolerance=0.03, gradients=True)


def test_leakage_gate_does_not_confuse_grad_mode_rounding_with_interference():
    model = small_model()
    forward = model.forward

    def mode_dependent_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        if not torch.is_grad_enabled():
            result["image"] = result["image"] + 0.001
        return result

    model.forward = mode_dependent_forward
    result = benchmark.check_outputs(model, small_batch(), backend="flash4", seed=42,
                                     tolerance=0.03, gradients=True)
    assert 1e-6 < result["grad_vs_no_grad_relative_l2"] < 0.03
    assert result["cross_example_relative_l2"] == 0


def test_cpu_worker_is_explicitly_unsupported_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    request = tmp_path / "request.json"
    benchmark.write_once(request, {"backend": "flash4"})
    benchmark.worker(request)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "UNSUPPORTED"
    assert "no CPU benchmark fallback" in result["error"]
    with pytest.raises(FileExistsError):
        benchmark.worker(request)


@pytest.mark.parametrize("legacy", [False, True])
def test_fa3_inventory_matches_canonical_or_absent_package_alias(tmp_path, monkeypatch, legacy):
    module_file = tmp_path / "interface.py"
    module_file.write_text("# exact fake backend source\n")
    calls = []

    def import_module(name):
        calls.append(name)
        if legacy and name == "flash_attn_3.flash_attn_interface":
            raise ModuleNotFoundError("No canonical package", name="flash_attn_3")
        return SimpleNamespace(__file__=str(module_file))

    monkeypatch.setattr(benchmark.importlib, "import_module", import_module)
    inventory = benchmark.backend_inventory("flash3")
    expected = ["flash_attn_3.flash_attn_interface"] + (["flash_attn_interface"] if legacy else [])
    assert calls == expected
    assert inventory["module"] == expected[-1]
    assert inventory["sha256"] == benchmark.digest(module_file)


def test_fa3_installed_but_broken_dependency_never_uses_alias(monkeypatch):
    calls = []

    def import_module(name):
        calls.append(name)
        raise ModuleNotFoundError("Missing installed backend dependency", name="flash_attn_3_cuda")

    monkeypatch.setattr(benchmark.importlib, "import_module", import_module)
    with pytest.raises(benchmark.UnsupportedBackend, match="Required backend import failed"):
        benchmark.backend_inventory("flash3")
    assert calls == ["flash_attn_3.flash_attn_interface"]


def test_worker_subprocess_and_timeout_are_bounded(tmp_path):
    successful = tmp_path / "success"
    successful.mkdir()
    code, timed_out = benchmark.run_worker_process([sys.executable, "-c", "print('done')"], successful, 10)
    assert code == 0 and not timed_out
    assert (successful / "stdout.log").read_text().strip() == "done"
    timeout = tmp_path / "timeout"
    timeout.mkdir()
    code, timed_out = benchmark.run_worker_process([sys.executable, "-c", "import time; time.sleep(30)"], timeout, 0.1)
    assert timed_out and code != 0


@pytest.mark.parametrize("token_execution,suffix", [("padded", ""), ("packed_valid", "-packed_valid")])
def test_token_execution_cli_is_explicit_and_keeps_legacy_padded_ids(monkeypatch, capsys, token_execution, suffix):
    monkeypatch.setattr(sys, "argv", [
        "benchmark_pixel_execution.py", "--dry-run", "--backends", "flash3",
        "--scenarios", "mixed", "--batch-sizes", "2", "--compile-mode", "default",
        "--compile-dynamic", "--token-execution", token_execution,
    ])
    benchmark.main()
    plan = json.loads(capsys.readouterr().out)
    candidate, = plan["candidates"]
    assert candidate["id"] == "flash3-mixed-b2-default" + suffix
    assert candidate["token_execution"] == token_execution
    assert candidate["compile_mode"] == "default" and candidate["compile_dynamic"]
    arguments = benchmark.model_arguments(
        candidate["backend"], candidate["scenario"], compile_mode=candidate["compile_mode"],
        compile_dynamic=candidate["compile_dynamic"], token_execution=candidate["token_execution"],
    )
    assert arguments["token_execution"] == token_execution
    # The compiled/eager gate changes compilation only, not token layout.
    assert {**arguments, "compile_mode": "none"}["token_execution"] == token_execution


def test_token_execution_cli_defaults_to_archived_padded_id(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "benchmark_pixel_execution.py", "--dry-run", "--backends", "flash4",
        "--scenarios", "fixed16", "--batch-sizes", "1",
    ])
    benchmark.main()
    candidate, = json.loads(capsys.readouterr().out)["candidates"]
    assert candidate["id"] == "flash4-fixed16-b1-none"
    assert candidate["token_execution"] == "padded"
    assert benchmark.model_arguments("flash4", "fixed16")["token_execution"] == "padded"


def test_token_execution_invalid_choice_fails_closed():
    with pytest.raises(ValueError, match="Unknown execution candidate"):
        benchmark.model_arguments("flash4", "mixed", token_execution="automatic")


def test_compiled_reference_cannot_change_token_execution():
    eager, compiled = small_model(), small_model()
    compiled.load_state_dict(eager.state_dict(), strict=True)
    compiled.token_execution = "packed_valid"
    with pytest.raises(benchmark.CorrectnessError, match="token execution differs"):
        benchmark.check_eager_compiled(eager, compiled, small_batch(), backend="flash4", seed=42, tolerance=.03)


def test_worker_reports_unsupported_ragged_backend_without_fallback(tmp_path, monkeypatch):
    def unsupported(*args):
        raise ValueError("Ragged packed_valid execution requires flash3 or flash4")
    monkeypatch.setattr(benchmark, "run_candidate", unsupported)
    request = tmp_path / "request.json"
    benchmark.write_once(request, {"backend": "sdpa", "token_execution": "packed_valid"})
    benchmark.worker(request)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "UNSUPPORTED"
    assert result["candidate"]["token_execution"] == "packed_valid"


def test_adaln_correctness_probe_is_nontrivial_and_restores_state_and_rng():
    model = ImageProprioModel(
        image_dim=12, width=16, depth=1, heads=2, max_cameras=2, max_tokens=32,
        objective="visual_only", phase_a_condition_on_proprio=False,
        use_phase_embedding=False, noise_conditioning="adaln_zero",
        attention_backend="sdpa", attention_dtype="bfloat16",
        loss_reduction="mean_per_example",
    )
    batch = small_batch()
    noisy, levels = benchmark.fixed_corruption(batch, 42)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    assert not model(batch, "libero_gripper", noisy, levels)["image"].any()
    with benchmark.nontrivial_conditioning_probe(model, 19) as receipt:
        assert receipt["nonzero_adaln_probe"]
        first = model(batch, "libero_gripper", noisy, levels)["image"]
        changed_levels = (1 - levels).masked_fill(batch["image"]["observed"], 0)
        second = model(batch, "libero_gripper", noisy, changed_levels)["image"]
        assert first.abs().sum() > 0
        assert not torch.allclose(first, second)
        result = benchmark.check_outputs(model, batch, backend="flash4", seed=42,
                                         tolerance=0.03, gradients=True)
        assert result["gradient_relative_l2"] < 0.03
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(RuntimeError, match="intentional probe failure"):
        with benchmark.nontrivial_conditioning_probe(model, 20):
            raise RuntimeError("intentional probe failure")
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
