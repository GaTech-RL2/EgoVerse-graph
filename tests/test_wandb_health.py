import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "scripts/train/check_wandb_health.py"
SPEC = importlib.util.spec_from_file_location("check_wandb_health", PATH)
health = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(health)


class FakeRun:
    path = ["team", "project", "run"]
    url = "https://wandb.ai/team/project/runs/run"
    state = "running"
    lastHistoryStep = 20545
    config = {"run_provenance": {"source_commit": "exact-source"}}

    def __init__(self, rows, summary=None):
        self.rows = rows
        self.summary = summary or {}
        self.scan_arguments = None

    def scan_history(self, **kwargs):
        self.scan_arguments = kwargs
        assert kwargs["keys"] is None  # Sparse keys must never be intersected.
        return iter(self.rows)


class FakeApi:
    def __init__(self, run):
        self.run_object = run

    def run(self, path):
        assert path == "team/project/run"
        return self.run_object


def inspect(run, **kwargs):
    return health.collect_health(FakeApi(run), "team/project/run", **kwargs)


def test_sparse_latest_metrics_and_optimizer_axis_use_bounded_history():
    run = FakeRun([
        {"_step": 20544, "trainer/global_step": 200000, "Train/loss": 0.2},
        {"_step": 20545, "trainer/global_step": 200010, "Schedule/flow": 0.01},
    ])
    result = inspect(run, metrics=["Train/loss", "Schedule/flow"], window=20,
                     expected_config={"run_provenance.source_commit": "exact-source"})
    assert result["status"] == "PASS"
    assert result["optimizer_step"] == 200010
    assert result["wandb_history_step"] == 20545
    assert result["metrics"]["Train/loss"]["optimizer_step"] == 200000
    assert run.scan_arguments == {"keys": None, "min_step": 20526,
                                  "max_step": 20546, "page_size": 20}


def test_latest_nan_is_not_hidden_by_an_earlier_finite_value():
    run = FakeRun([
        {"_step": 20544, "trainer/global_step": 20, "loss": 0.1},
        {"_step": 20545, "trainer/global_step": 30, "loss": float("nan")},
    ])
    result = inspect(run, metrics=["loss"])
    assert result["status"] == "ERROR"
    assert result["metrics"]["loss"]["finite"] is False


def test_summary_only_metric_is_reported_without_inventing_a_step():
    run = FakeRun([{"_step": 20545, "trainer/global_step": 20}], {"Valid/loss": 0.3})
    result = inspect(run, metrics=["Valid/loss"])
    assert result["status"] == "WARN"
    assert result["metrics"]["Valid/loss"]["source"] == "summary_only"
    assert result["metrics"]["Valid/loss"]["optimizer_step"] is None


def test_missing_and_wrong_config_identity_fail_closed():
    run = FakeRun([{"trainer/global_step": 20}])
    result = inspect(run, expected_config={"run_provenance.source_commit": "wrong", "missing": 1})
    assert result["status"] == "ERROR"
    assert len(result["errors"]) == 2


def test_history_counter_is_never_substituted_for_missing_optimizer_step():
    result = inspect(FakeRun([{"_step": 20545, "loss": 0.1}]), metrics=["loss"])
    assert result["status"] == "INCOMPLETE"
    assert result["optimizer_step"] is None


def test_recent_resume_regression_and_stale_metrics_are_visible():
    run = FakeRun([
        {"_step": 20544, "trainer/global_step": 200000, "_timestamp": 10},
        {"_step": 20545, "trainer/global_step": 160000, "_timestamp": 20},
    ])
    result = inspect(run, max_age_seconds=30, now=100)
    assert result["status"] == "WARN"
    assert result["optimizer_step_regressions"] == [{"from": 200000, "to": 160000, "wandb_step": 20545}]
    assert result["latest_history_age_seconds"] == 80


def test_no_unbounded_history_request():
    with pytest.raises(ValueError, match="window"):
        inspect(FakeRun([]), window=1000000)
