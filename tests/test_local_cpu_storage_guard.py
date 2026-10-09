from types import SimpleNamespace

import pytest

from scripts.local_cpu_storage_guard import pytest_runtest_setup


@pytest.mark.parametrize("free_gib", [3, 4, 8])
def test_per_test_headroom_gate(monkeypatch, free_gib):
    monkeypatch.delenv("LOCAL_CPU_TEST_MIN_FREE_KIB", raising=False)
    monkeypatch.setattr(
        "scripts.local_cpu_storage_guard.shutil.disk_usage",
        lambda path: SimpleNamespace(free=free_gib * 1024**3),
    )
    if free_gib < 4:
        with pytest.raises(pytest.UsageError, match="blocked before test"):
            pytest_runtest_setup(None)
    else:
        pytest_runtest_setup(None)


def test_per_test_floor_cannot_be_lowered(monkeypatch):
    monkeypatch.setenv("LOCAL_CPU_TEST_MIN_FREE_KIB", "1")
    with pytest.raises(pytest.UsageError, match="cannot be below"):
        pytest_runtest_setup(None)
