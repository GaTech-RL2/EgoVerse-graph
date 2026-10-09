"""Every new YAML is automatically included in the offline composition gate."""

from hydra.core.hydra_config import HydraConfig

from scripts.audit_components import audit_components
from scripts.audit_hydra_configs import CONFIGS, audit


def test_every_shipped_yaml_composes_and_resolves():
    previous = HydraConfig.instance().cfg
    records = audit()
    assert HydraConfig.instance().cfg is previous
    paths = {str(path.relative_to(CONFIGS)) for path in CONFIGS.rglob("*.yaml")}
    assert {record["path"] for record in records} == paths
    failures = [record for record in records if record["status"] != "passed"]
    assert not failures, failures


def test_every_shipped_component_constructs_without_network_or_weights():
    previous = HydraConfig.instance().cfg
    records = audit_components()
    assert HydraConfig.instance().cfg is previous
    assert {record["path"] for record in records} == {
        str(path.relative_to(CONFIGS)) for path in CONFIGS.rglob("*.yaml")
    }
    failures = [record for record in records if record["status"] != "passed"]
    assert not failures, failures


def test_scale_filter_defers_cloud_access_and_preserves_resolution_memo(monkeypatch):
    import pandas as pd

    from egomimic.rldb.filters import ScaleAnnotationDatasetFilter
    from egomimic.rldb.resolve_memo import resolve_once
    from egomimic.utils import scale_utils

    monkeypatch.delenv("SCALE_API_KEY", raising=False)
    first = ScaleAnnotationDatasetFilter("fixture")
    assert first._tasks is None
    calls = []
    monkeypatch.setenv("SCALE_API_KEY", "fixture-only")
    monkeypatch.setattr(
        scale_utils,
        "get_completed_tasks",
        lambda name, key: calls.append((name, key)) or ["keep"],
    )
    monkeypatch.setattr(
        scale_utils,
        "build_df_from_tasks",
        lambda tasks: pd.DataFrame({"SEQUENCE_ID": tasks}),
    )
    with resolve_once():
        second = ScaleAnnotationDatasetFilter("fixture")
        assert first.matches({"episode_hash": "keep"})
        assert second.cache_key() == first.cache_key()
        assert not second.matches({"episode_hash": "missing"})
        assert not second.matches({"episode_hash": "keep", "is_deleted": True})
        assert calls == [("fixture", "fixture-only")]
    with resolve_once():
        assert ScaleAnnotationDatasetFilter("fixture").matches({"episode_hash": "keep"})
    assert len(calls) == 2
