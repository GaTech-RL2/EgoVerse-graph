"""Historical behavior contracts, not a claim of full training equivalence."""

import ast
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper
from egomimic.rldb.zarr import zarr_dataset_multi as dm
from tests.test_bounds_check_gate import _dataset


def module(mode):
    return MultiDataModuleWrapper({}, {}, {}, {}, compatibility_mode=mode)


def test_invalid_mode_fails_closed():
    for make in (
        module,
        lambda mode: PipelineAlgo([], device="cpu", compatibility_mode=mode),
        lambda mode: dm.MultiDataset(state={}, compatibility_mode=mode),
    ):
        with pytest.raises(ValueError, match="compatibility_mode"):
            make("typo")


def test_bounds_false_is_explicitly_overridden_only_in_legacy_mode():
    ds = dm.MultiDataset(state={}, bounds_check=False, compatibility_mode="legacy_c12")
    assert ds.bounds_check is True
    assert ds.bounds_semantics == "legacy_full_vector"
    assert ds.fallback_policy == "legacy_random"
    assert dm.MultiDataset(state={}, bounds_check=False).bounds_check is False


@pytest.mark.parametrize("bad_value", [1e6, float("nan"), float("inf"), -1e-7])
def test_legacy_filter_rejects_nonfinite_and_strict_outliers(bad_value):
    ds, leaf = _dataset(bounds_check=False, bad_value=bad_value)
    # Constructor-derived settings, transplanted onto a tiny normalized fixture.
    settings = dm.MultiDataset(
        state={}, bounds_check=False, compatibility_mode="legacy_c12"
    )
    for key in (
        "bounds_check",
        "bounds_semantics",
        "fallback_policy",
        "compatibility_mode",
    ):
        setattr(ds, key, getattr(settings, key))
    ds[1]
    assert leaf.served == [1, 0]


def test_loader_matches_default_shuffle_worker_seed_and_global_rng():
    legacy = module("legacy_c12")
    for workers in (0, 2):
        results = []
        for generator in (None, legacy._loader_generator("train", "source")):
            torch.manual_seed(42)
            loader = torch.utils.data.DataLoader(
                torch.arange(64),
                batch_size=8,
                shuffle=True,
                num_workers=workers,
                generator=generator,
            )
            iterator = iter(loader)
            worker_seed = iterator._base_seed
            batches = torch.cat(list(iterator))
            results.append((worker_seed, batches, torch.get_rng_state().clone()))
        assert results[0][0] == results[1][0]
        assert torch.equal(results[0][1], results[1][1])
        assert torch.equal(results[0][2], results[1][2])
    assert not legacy._loader_generators


def test_checkpoint_cannot_silently_change_loader_semantics():
    for left, right in (("current", "legacy_c12"), ("legacy_c12", "current")):
        with pytest.raises(ValueError, match="compatibility mode"):
            module(right).load_state_dict(module(left).state_dict())
    legacy = module("legacy_c12")
    legacy.load_state_dict(legacy.state_dict())


def test_input_dtype_preserved_recursively_only_in_legacy_mode():
    value = torch.tensor([1.0000000001], dtype=torch.float64)
    for mode, expected in (("current", torch.float32), ("legacy_c12", torch.float64)):
        algo = PipelineAlgo([], device="cpu", compatibility_mode=mode)
        batch = algo._move_value({"a": [value, (value,)], "i": torch.tensor([1])})
        assert batch["a"][0].dtype == expected
        assert batch["a"][1][0].dtype == expected
        assert batch["i"].dtype == torch.int64
        if mode == "legacy_c12":
            assert torch.equal(batch["a"][0], value)


def test_retry_against_pinned_historical_source():
    path = os.environ.get("C12_DATASET_SOURCE")
    if not path:
        pytest.skip("set C12_DATASET_SOURCE to the pinned historical dataset module")
    tree = ast.parse(Path(path).read_text())
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_fallback_idx"
    )
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiDataset"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_next_after_failure"
    )
    ns = dict(vars(dm))
    exec(
        compile(ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"),
        ns,
    )
    ds = dm.MultiDataset(state={}, compatibility_mode="legacy_c12")
    ds.index_map = [("a", i) for i in range(64)] + [("b", i) for i in range(4)]
    ds._global_indices_by_dataset = {"a": list(range(64)), "b": list(range(64, 68))}
    original_rng = random.getstate()
    try:
        for seed in (0, 42, 999):
            for attempt in (None, 0, 24, 25, 26, 62, 63, 999):
                results = []
                for method in (
                    ns["_next_after_failure"],
                    dm.MultiDataset._next_after_failure,
                ):
                    random.seed(seed)
                    try:
                        result = method(ds, 0, "a", attempt, reason="fixture")
                    except RuntimeError as exc:
                        result = (type(exc).__name__, str(exc))
                    results.append((result, random.getstate()))
                assert results[0] == results[1], (seed, attempt)
    finally:
        random.setstate(original_rng)




@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_metric_dtype_matches_historical_reducer_without_changing_gradient(dtype):
    from collections import OrderedDict

    from egomimic.pl_utils.training_behavior_action_flow import (
        ActionFlowTrainingBehavior,
    )
    from egomimic.pl_utils.training_metrics import reduce_component_means

    path = os.environ.get("C12_DATASET_SOURCE")
    if not path:
        pytest.skip("set C12_DATASET_SOURCE to the pinned historical dataset module")
    source = Path(path).parents[2] / "pl_utils/pl_model_action_flow.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ActionFlowModelWrapper"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_reduce_component_means"
    )
    method.decorator_list = []
    ns = {"torch": torch, "OrderedDict": OrderedDict, "Mapping": dict}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    value = torch.tensor(0.123456789, dtype=dtype, requires_grad=True)
    state = torch.get_rng_state().clone()
    expected, count = ns["_reduce_component_means"]({"loss": value}, 17)
    actual, actual_count = reduce_component_means(
        {"loss": value}, 17, label="fixture", preserve_input_dtype=True
    )
    assert count == actual_count
    assert actual["loss"].dtype == expected["loss"].dtype
    assert torch.equal(actual["loss"], expected["loss"])
    assert value.grad is None and not actual["loss"].requires_grad
    assert torch.equal(state, torch.get_rng_state())
    assert (
        ActionFlowTrainingBehavior(compatibility_mode="legacy_c12").compatibility_mode
        == "legacy_c12"
    )
    with pytest.raises(ValueError, match="compatibility_mode"):
        ActionFlowTrainingBehavior(compatibility_mode="invalid")


def test_normalization_collection_matches_historical_precision():
    path = os.environ.get("C12_DATASET_SOURCE")
    if not path:
        pytest.skip("set C12_DATASET_SOURCE to the pinned historical dataset module")
    tree = ast.parse(Path(path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiDataset"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_collect_norm_samples"
    )
    ns = dict(vars(dm))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    for dtype in (torch.float32, torch.float64):
        samples = torch.tensor(
            [[1.0000000001], [1.0000000003], [1.0000000009]], dtype=dtype
        )
        ds = dm.MultiDataset(state={}, compatibility_mode="legacy_c12")
        ds.keyname_to_zarr_key = lambda key, embodiment: key
        args = ([{"action": samples}], ["action"], "fixture", 2, 3, 0)
        expected = ns["_collect_norm_samples"](ds, *args)["action"][0]
        actual = ds._collect_norm_samples(*args)["action"][0]
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        for key, value in ds._compute_stats_for_array(expected).items():
            np.testing.assert_array_equal(
                ds._compute_stats_for_array(actual)[key], value
            )
        ds.compatibility_mode = "current"
        assert ds._collect_norm_samples(*args)["action"][0].dtype == np.float32
