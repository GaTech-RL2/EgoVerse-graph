"""Fail before each local test when retained synthetic checkpoints fill disk."""

import os
import shutil
import tempfile

import pytest


def pytest_runtest_setup(item):
    del item
    floor_kib = int(os.environ.get("LOCAL_CPU_TEST_MIN_FREE_KIB", "4194304"))
    if floor_kib < 4194304:
        raise pytest.UsageError("Local CPU storage floor cannot be below 4 GiB")
    if shutil.disk_usage(tempfile.gettempdir()).free < floor_kib * 1024:
        raise pytest.UsageError(
            "Local CPU validation blocked before test: insufficient temporary "
            "storage; preserve results, collect completed synthetic fixtures, "
            "then resume unfinished tests only"
        )
