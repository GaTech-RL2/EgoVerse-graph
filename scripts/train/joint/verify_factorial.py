"""Prove common image initialization and RGB draws from all four actual GPU smokes."""

import argparse
import copy
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.train.joint.contract import bound, need, sha, verify_evidence, write_once

CELLS = ("image_only", "image_proprio", "image_action", "image_proprio_action")


def verify(requests):
    need(
        len(requests) == 4 and {r["experiment"] for r in requests} == set(CELLS),
        "Need exactly four factorial arms",
    )
    reference = None
    rows = []
    for request in requests:
        completion, _ = verify_evidence(request)
        cfg = OmegaConf.to_container(
            OmegaConf.create(bound(request["smoke"]["resolved"], False)), resolve=True
        )
        controls = {
            key: copy.deepcopy(cfg[key])
            for key in (
                "data",
                "seed",
                "runtime",
                "world_model_attention",
                "joint",
                "trainer",
            )
        }
        smoke = bound(request["smoke"]["request"])
        controls = json.loads(
            json.dumps(controls)
            .replace(smoke["output"], "<output>")
            .replace(smoke["local_logs"], "<logs>")
        )
        paired = dict(
            controls=controls,
            source_revision=request["source_revision"],
            data_context=completion["ranks"][0]["data_context_sha256"],
            initial_image=[
                r["initial_image_weights_sha256"] for r in completion["ranks"]
            ],
            rgb_batches=[r["paired_rgb_batch_sha256"] for r in completion["ranks"]],
        )
        if reference is None:
            reference = paired
        need(
            paired == reference,
            "Unpaired data, masks, actual noise, initial RGB weights or runtime",
        )
        rows.append(
            dict(
                experiment=request["experiment"],
                smoke_completion_sha256=request["smoke"]["completion"]["sha256"],
                smoke_request_sha256=request["smoke"]["request"]["sha256"],
                smoke_job=request["smoke"]["job_id"],
            )
        )
    return dict(
        schema="joint-factorial-gpu-controls-v1",
        status="PASS",
        source_revision=requests[0]["source_revision"],
        experiments=sorted(CELLS),
        paired=reference,
        evidence=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs=4, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify([json.loads(path.read_text()) for path in args.requests])
    result["request_files"] = [
        {"path": str(p), "sha256": sha(p)} for p in args.requests
    ]
    write_once(args.output, result)
    print(json.dumps(dict(status="PASS", output=str(args.output))))


if __name__ == "__main__":
    main()
