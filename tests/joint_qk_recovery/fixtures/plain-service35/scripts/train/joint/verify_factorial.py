"""Prove common image initialization and RGB draws from all four actual GPU smokes."""

import argparse
import copy
import json
from pathlib import Path

from omegaconf import OmegaConf

from scripts.train.joint.contract import bound, need, sha, verify_evidence, verify_model_execution, write_once

CELLS = ("image_only", "image_proprio", "image_action", "image_proprio_action")


def image_controls(request, cfg, controls):
    """Project only the declared native proprio consequences out of paired controls.

    The full context remains independently bound by verify_evidence for each run.
    Manifest content, every remaining data setting and actual RGB/mask/noise
    digests must still match. Legacy joint requests retain exact context equality.
    """
    if request.get("model_execution") is None:
        return False
    verify_model_execution(request, cfg)
    proprio = request["active_streams"] == ["image", "proprio"]
    need(request["experiment"] == ("image_proprio" if proprio else "image_only"),
         "Native experiment/stream mismatch")
    data = controls["data"]
    need(data["image_only"] is (not proprio)
         and data["couple_state_observations"] is proprio,
         "Unexpected native modality data flags")
    data.pop("image_only")
    data.pop("couple_state_observations")
    frames = data["discrete_frames"] if data["mode"] == "discrete" else data["max_frames"]
    need(isinstance(frames, int) and frames > 0, "Invalid native token budget")
    controls["joint"]["max_tokens"] -= frames * int(proprio)
    need(controls["joint"]["max_tokens"] > 0, "Missing image token budget")
    return True


def verify(requests):
    cells = set(CELLS) if len(requests) == 4 else {"image_only", "image_proprio"}
    need(len(requests) in (2, 4) and {r["experiment"] for r in requests} == cells,
         "Need all four factorial arms or the exact image/proprio pair")
    need(len({r.get("arm", "discrete") for r in requests}) == 1, "Paired controls require one time mode")
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
        native = image_controls(request, cfg, controls)
        paired = dict(
            controls=controls,
            source_revision=request["source_revision"],
            data_context=(dict(scope="native-modality-specific-contexts",
                               data_manifest_sha256=request["data_manifest_sha256"],
                               model_execution=request["model_execution"])
                          if native else completion["ranks"][0]["data_context_sha256"]),
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
                data_context_sha256=completion["ranks"][0]["data_context_sha256"],
            )
        )
    return dict(
        schema="joint-factorial-gpu-controls-v1",
        status="PASS",
        source_revision=requests[0]["source_revision"],
        experiments=sorted(cells),
        paired=reference,
        evidence=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", nargs="+", type=Path)
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
