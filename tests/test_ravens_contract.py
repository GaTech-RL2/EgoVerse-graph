"""Guard action/observation alignment and reuse of the selected LIBERO models."""
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import pytest

from egomimic.benchmarks.ravens.continuous import validate_suction_transitions
from egomimic.benchmarks.ravens.continuous import defer_dependency


def test_optional_dependency_loads_actual_module_only_when_used(tmp_path, monkeypatch):
    import importlib
    import sys
    name = "ravens_lazy_dependency_fixture"
    (tmp_path / f"{name}.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    proxy = defer_dependency(name)
    try:
        assert "VALUE" not in vars(proxy)
        assert importlib.import_module(name) is proxy
        assert proxy.VALUE == 42
        loaded = sys.modules[name]
        assert loaded is not proxy
        assert proxy.VALUE == loaded.VALUE
        assert sys.modules[name] is loaded
    finally:
        sys.modules.pop(name, None)


def test_suction_recording_requires_decisions_before_state_changes():
    states = np.array([[0., 0.], [1., 0.]], np.float32)
    actions = np.zeros((2, 7), np.float32)
    actions[:, 6] = [1., -1.]
    validate_suction_transitions(states, actions)
    leaked_states = np.array([[1., 0.], [0., 0.]], np.float32)
    with pytest.raises(ValueError, match="pre-action suction"):
        validate_suction_transitions(leaked_states, actions)


@pytest.mark.parametrize("family", ["action_flow", "dp"])
def test_ravens_reuses_libero_model_and_changes_benchmark_contract(family):
    source = Path(__file__).resolve().parents[1]
    with hydra.initialize_config_dir(version_base="1.2", config_dir=str(source / "egomimic/hydra_configs")):
        common = ["hydra/launcher=basic", "benchmark.dataset=/UNBOUND", "paths.output_dir=/UNBOUND",
                  f"paths.work_dir={source}"]
        original = hydra.compose(config_name="train_zarr_cartesian", overrides=[
            *common, f"+experiment=libero/{family}_libero_spatial_oat_dp_matched_s42"])
        ravens = hydra.compose(config_name="train_zarr_cartesian", overrides=[
            *common, f"+experiment=ravens/{family}", "evaluator.source_commit=UNBOUND",
            "evaluator.config_sha256=UNBOUND"])
        model = OmegaConf.to_container(ravens.model, resolve=True)
        reference = OmegaConf.to_container(original.model, resolve=True)
        # Protocol suite is a benchmark identity, not a model architecture change.
        for value in (model, reference):
            value.pop("benchmark_protocol", None)
        assert model == reference
        assert list(ravens.data.train_datasets) == ["ravens_ur5"]
        assert list(ravens.data.valid_datasets) == ["ravens_ur5"]
        assert ravens.data.train_datasets.ravens_ur5.split_seed == 42
        assert ravens.data.train_datasets.ravens_ur5.valid_ratio == .1
        assert ravens.benchmark.horizon == 32
        assert ravens.benchmark.n_obs_steps == 2
        assert ravens.benchmark.n_action_steps == 16
        assert ravens.callbacks.model_checkpoint.save_top_k == -1
        assert ravens.callbacks.model_checkpoint.every_n_train_steps == 15000
        assert ravens.trainer.val_check_interval == 15000
        assert ravens.trainer.accumulate_grad_batches == 1
        assert ravens.callbacks.batch_budget.global_batch_size == 32
