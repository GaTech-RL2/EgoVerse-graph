"""Render the source-pinned one-GPU AstraPush engineering preflight."""

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(
    commit, name, *, provider_probe=True, generation_archive=None, full_experiment=False
):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full immutable source commit is required")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,60}", name):
        raise ValueError("Invalid workflow name")
    if full_experiment and (provider_probe or generation_archive):
        raise ValueError(
            "Full experiment uses the frozen commissioning evidence and Codex transport"
        )
    if generation_archive is not None:
        if (
            provider_probe
            or not re.fullmatch(
                r"docs/experiments/astra-hpt-libero/evidence/[a-zA-Z0-9_/-]+",
                generation_archive,
            )
            or ".." in generation_archive.split("/")
        ):
            raise ValueError(
                "Subagent input must be a pinned evidence directory; disable gateway probe"
            )
    result = {
        "workflow": {
            "name": name,
            "tasks": [
                {
                    "name": "foundation-preflight",
                    "image": "nvcr.io/nvidia/pytorch:25.06-py3",
                    "resource": "single",
                    "credentials": {
                        "egoverse-github": {"GITHUB_TOKEN": "github_token"},
                        "astra-reversal-inference-20260924": {
                            "NVIDIA_INFERENCE_API_KEY": "api_key"
                        },
                        "grabber-arc-r2-20260916": {
                            "R2_ACCESS_KEY_ID": "r2_access_key_id",
                            "R2_SECRET_ACCESS_KEY": "r2_secret_access_key",
                            "R2_ENDPOINT_URL": "r2_endpoint_url",
                        },
                    },
                    "environment": {
                        "SOURCE_COMMIT": commit,
                        "RUN_PROVIDER_PROBE": "1" if provider_probe else "0",
                        "GATE_OUTPUT": "{{output}}",
                        "ARTIFACT_PREFIX": f"experiments/astra-hpt-libero-20260928/{name}/{commit}/preflight",
                    },
                    "command": ["bash"],
                    "args": ["/tmp/entry.sh"],
                    "files": [
                        {
                            "path": "/tmp/entry.sh",
                            "contents": (
                                ROOT / "scripts/astra_push/osmo_entry.sh"
                            ).read_text(),
                        }
                    ],
                }
            ],
            "resources": {
                "single": {
                    "gpu": 1,
                    "cpu": 8,
                    "memory": "96Gi",
                    "storage": "200Gi",
                    "platform": "ovx-l40",
                }
            },
            "timeout": {"queue_timeout": "1h", "exec_timeout": "3h"},
        }
    }
    if not provider_probe:
        del result["workflow"]["tasks"][0]["credentials"][
            "astra-reversal-inference-20260924"
        ]
    if generation_archive:
        result["workflow"]["tasks"][0]["environment"]["ASTRA_GENERATION_ARCHIVE"] = (
            generation_archive
        )
    if full_experiment:
        task = result["workflow"]["tasks"][0]
        task["name"] = "learner-curriculum"
        task["environment"]["ASTRA_FULL_RUN"] = "1"
        task["environment"]["ARTIFACT_PREFIX"] = (
            f"experiments/astra-hpt-libero-20260928/{name}/{commit}/full"
        )
        result["workflow"]["timeout"]["exec_timeout"] = "24h"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-experiment", action="store_true")
    parser.add_argument(
        "--generation-archive", help="Source-pinned Codex Astra commissioning exchange"
    )
    parser.add_argument(
        "--skip-provider",
        action="store_true",
        help="Run engineering checks while the confirmed provider budget blocker is unresolved",
    )
    args = parser.parse_args()
    with args.output.open("x") as stream:
        yaml.safe_dump(
            workflow(
                args.source_commit,
                args.name,
                provider_probe=not args.skip_provider,
                generation_archive=args.generation_archive,
                full_experiment=args.full_experiment,
            ),
            stream,
            sort_keys=False,
        )
