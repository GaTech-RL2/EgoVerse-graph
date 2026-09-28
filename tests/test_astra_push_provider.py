"""Archive malformed replies and bound retries without real provider requests."""

import io
import json
import urllib.error
from pathlib import Path

import pytest

from egomimic.experiments.astra_push.provider import MODEL, AstraProvider


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Opener:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def open(self, *args, **kwargs):
        self.calls += 1
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return Reply(reply)


def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "fixture-key-not-a-credential")
    monkeypatch.setattr(
        "egomimic.experiments.astra_push.provider.time.sleep", lambda _: None
    )
    return AstraProvider(tmp_path)


def test_raw_bad_json_is_retained_and_request_is_not_reissued(tmp_path, monkeypatch):
    provider = client(tmp_path, monkeypatch)
    provider.opener = Opener([b'{"broken":'])
    with pytest.raises(ValueError, match="raw response archived"):
        provider.request("logical-1", system="fixture", user={})
    raw = json.loads((tmp_path / "logical-1/attempt-0.json").read_text())
    assert raw["raw_response"] == '{"broken":'
    assert (tmp_path / "logical-1/validation-failure.json").exists()
    with pytest.raises(FileExistsError):
        provider.request("logical-1", system="fixture", user={})
    assert provider.opener.calls == 1


def test_catalog_retries_rate_limit_twice_without_generation(tmp_path, monkeypatch):
    provider = client(tmp_path, monkeypatch)
    failures = [
        urllib.error.HTTPError("fixture", 429, "Too many", {}, io.BytesIO(b"capacity"))
        for _ in range(3)
    ]
    provider.opener = Opener(failures)
    with pytest.raises(RuntimeError, match="no generation call"):
        provider.catalog()
    assert provider.opener.calls == 3
    assert len(list(Path(tmp_path).glob("catalog-attempt-*.json"))) == 3
    assert not list(Path(tmp_path).rglob("request.json"))


def test_catalog_requires_the_requested_model(tmp_path, monkeypatch):
    provider = client(tmp_path, monkeypatch)
    provider.opener = Opener([json.dumps({"data": [{"id": MODEL}]}).encode()])
    provider.catalog()
    assert json.loads((tmp_path / "catalog.json").read_text())["listed"]
