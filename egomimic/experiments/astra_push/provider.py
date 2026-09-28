"""Astra gateway transport with archived physical attempts and no model fallback."""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from egomimic.experiments.astra_push.artifacts import publish_json
from egomimic.experiments.astra_push.schemas import SceneSpec

ENDPOINT = "https://inference-api.nvidia.com/v1"
MODEL = "azure/openai/gpt-6-astra"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Refusing to redirect a credentialed provider request")


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


class AstraProvider:
    def __init__(self, archive, *, model=MODEL, timeout=170):
        if model != MODEL:
            raise ValueError("This manifest permits only the specified Astra model")
        self.archive = Path(archive)
        self.model, self.timeout = model, timeout
        self.key = os.environ.get("NVIDIA_INFERENCE_API_KEY")
        if not self.key:
            raise RuntimeError(
                "NVIDIA_INFERENCE_API_KEY is not injected; no provider call made"
            )
        self.opener = urllib.request.build_opener(NoRedirect())

    def catalog(self):
        req = urllib.request.Request(
            ENDPOINT + "/models", headers={"Authorization": f"Bearer {self.key}"}
        )
        for attempt in range(3):
            try:
                with self.opener.open(req, timeout=self.timeout) as response:
                    raw = (
                        response.read(8 * 1024 * 1024)
                        .decode()
                        .replace(self.key, "[REDACTED]")
                    )
                publish_json(
                    self.archive / f"catalog-attempt-{attempt}.json",
                    {"status": "response", "raw_response": raw},
                )
                catalog = strict_json(raw)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                code = getattr(exc, "code", 0)
                record = {
                    "status": "transport_failure",
                    "http_status": code,
                    "exception": type(exc).__name__,
                }
                if isinstance(exc, urllib.error.HTTPError):
                    record["provider_error"] = (
                        exc.read(1024 * 1024)
                        .decode(errors="replace")
                        .replace(self.key, "[REDACTED]")
                    )
                    record["retry_after"] = exc.headers.get("Retry-After")
                publish_json(self.archive / f"catalog-attempt-{attempt}.json", record)
                if attempt == 2 or (
                    code and code not in {408, 429, 500, 502, 503, 504}
                ):
                    raise RuntimeError(
                        f"Astra catalog unavailable (HTTP {code}); no generation call made"
                    ) from None
                delay = record.get("retry_after", "")
                time.sleep(
                    min(30, max(2**attempt, int(delay)))
                    if str(delay).isdigit()
                    else 2**attempt
                )
        ids = [entry["id"] for entry in catalog["data"]]
        publish_json(
            self.archive / "catalog.json",
            {"model": self.model, "listed": self.model in ids},
        )
        if self.model not in ids:
            raise RuntimeError(
                "Requested Astra model is not in the authenticated catalog"
            )

    def request(self, request_id, *, system, user, max_tokens=8192):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
            raise ValueError("Provider output token cap must be in [1,8192]")
        if not isinstance(request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", request_id
        ):
            raise ValueError("Request ID must be a path component")
        path = self.archive / request_id
        # Existence blocks duplicate billed requests after an interrupted call.
        path.mkdir(parents=True, exist_ok=False)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, allow_nan=False)},
            ],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        }
        publish_json(
            path / "request.json",
            {"request_id": request_id, "body": body, "logical_calls": 1},
        )
        for attempt in range(3):
            started = time.monotonic()
            req = urllib.request.Request(
                ENDPOINT + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with self.opener.open(req, timeout=self.timeout) as response:
                    raw = (
                        response.read(8 * 1024 * 1024)
                        .decode()
                        .replace(self.key, "[REDACTED]")
                    )
                publish_json(
                    path / f"attempt-{attempt}.json",
                    {
                        "status": "response",
                        "seconds": time.monotonic() - started,
                        "raw_response": raw,
                    },
                )
                try:
                    result = strict_json(raw)
                    choice = result["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Provider response did not finish normally")
                    answer = strict_json(choice["message"]["content"])
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    publish_json(
                        path / "validation-failure.json",
                        {"exception": type(exc).__name__, "message": str(exc)},
                    )
                    raise ValueError(
                        "Invalid provider response; raw response archived"
                    ) from None
                publish_json(path / "answer.json", answer)
                return answer, {
                    "model": result.get("model"),
                    "requested_model": self.model,
                    "response_id": result.get("id"),
                    "usage": result.get("usage"),
                    "transport_attempts": attempt + 1,
                }
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                record = {
                    "status": "transport_failure",
                    "exception": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                    "seconds": time.monotonic() - started,
                }
                if isinstance(exc, urllib.error.HTTPError):
                    raw = (
                        exc.read(1024 * 1024)
                        .decode(errors="replace")
                        .replace(self.key, "[REDACTED]")
                    )
                    record["provider_error"] = raw
                publish_json(path / f"attempt-{attempt}.json", record)
                code = getattr(exc, "code", 0)
                if attempt == 2 or (
                    code and code not in {408, 429, 500, 502, 503, 504}
                ):
                    raise RuntimeError(
                        f"Astra transport failed ({type(exc).__name__}, HTTP {code}); request archived"
                    ) from None
                time.sleep(2**attempt)
        raise AssertionError("Unreachable")


def probe(output):
    client = AstraProvider(output)
    client.catalog()
    answer, metadata = client.request(
        "commissioning-w01-0001",
        system="You are Astra designing bounded LIBERO pushing scenes. Return only the requested SceneSpec JSON. No Python or rewards. This is a commissioning proposal, not a measured rollout result.",
        user={
            "schema": SceneSpec.model_json_schema(),
            "stage": "S1",
            "constraints": "One red or blue 40 mm cube, one target named target. Placement half-width 0.03 to 0.05 m; target half-width 0.04 to 0.06 m. Optional one reference_fixture at least 5 cm clear of the direct push corridor. Geometry within |x|+half<=0.28, |y|+half<=0.32; all rectangular regions disjoint. Nominal cube-center to target-center distance 0.05 to 0.15 m. Fixed Panda, table, cameras and physics. Propose a new bounded layout; do not claim novelty or success has been verified.",
            "required_schema_version": "astrapush-1",
        },
    )
    scene = SceneSpec.model_validate(answer)
    publish_json(Path(output) / "scene.json", scene.model_dump(mode="json"))
    publish_json(
        Path(output) / "receipt.json",
        {
            **metadata,
            "phase": "commissioning",
            "logical_calls_used": 1,
            "remaining_phase_call_cap": 7,
            "structured_scene_valid": True,
            "physical_witness": False,
            "structural_signature": scene.structural_signature(),
        },
    )
    print(
        json.dumps(
            {
                "provider_model": metadata["model"],
                "structured_scene_valid": True,
                "logical_calls": 1,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    probe(args.output)
