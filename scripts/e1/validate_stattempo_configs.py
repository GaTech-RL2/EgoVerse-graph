"""CPU-only contract audit of the consolidated stationery config family.

The fixture hashes were exported from stattempo-streams-ref at 51cd2bf8.
Only the documented structural migrations are removed before comparison.
No episode resolution, model weights, SQL or network access is required.
"""

import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "egomimic/hydra_configs"


def canonical_contract(value, path=()):
    if isinstance(value, list):
        return [canonical_contract(v, path) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if path == ("model",) and key == "graph":
            continue  # Named components are already resolved into pipeline.stages.
        if key == "clip_sample_range" and child is None:
            continue  # Explicit opt-out is the sampler's original default.
        if key == "yam_source_frames" and path[-1:] in (
            ("transforms",),
            ("transform_list",),
        ):
            assert child == 100
            continue  # Previously hard-coded in the stationery adapter.
        if path == ("e1",) and key in {
            "reference_translation_range",
            "reference_rotation_range",
        }:
            continue  # Already resolved into norm_stats.fixed_ranges.
        if not path and key == "eval_logger_enabled" and child is False:
            continue  # Retired by main; not used by training.
        if path == ("evaluator",) and (
            (
                key
                in {
                    "dtw_max_segment_steps",
                    "gripper_distance_unit",
                    "records_dump_path",
                    "velocity_layout",
                }
                and child is None
            )
            or (key == "video_overlay_mode" and child == "executed_chunk")
        ):
            # Defaults added by the shared open-loop evaluator after these runs
            # were recorded: unset diagnostics and the overlay style of videos
            # these recipes disable.
            continue
        result[key] = canonical_contract(child, (*path, key))
    return result


def fingerprint(value):
    payload = json.dumps(
        canonical_contract(value), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def compose_recipe(name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(
            "train_zarr_cartesian",
            overrides=[
                f"+experiment=yam_arc_grid/{name}",
                "paths.output_dir=/tmp/stattempo-parity",
                "paths.work_dir=/tmp/stattempo-parity",
            ],
        )


def main():
    fixture = json.loads((REPO / "tests/fixtures/stattempo_contracts.json").read_text())
    for name, expected in fixture["configs"].items():
        value = OmegaConf.to_container(compose_recipe(name), resolve=True)
        actual = fingerprint(value)
        if actual != expected:
            raise ValueError(
                f"Resolved recipe differs from {fixture['source']}: {name}"
            )
    print(
        f"PASS: {len(fixture['configs'])} resolved stationery recipes match {fixture['source']}"
    )


if __name__ == "__main__":
    main()
