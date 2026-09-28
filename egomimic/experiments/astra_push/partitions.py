"""Prospectively frozen trusted evaluation grammar and training split firewall."""

import json
from pathlib import Path

import numpy as np
import yaml

from egomimic.experiments.astra_push.artifacts import named_seed, publish_json
from egomimic.experiments.astra_push.schemas import (
    SceneSpec,
    TaskSpec,
    canonical_hash,
    instruction_for,
    instruction_realizations,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "hydra_configs/astra_push/full_experiment.yaml"
)


def config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def family_features(scene):
    """Ignore IDs/colors and global translations; retain relative metric geometry."""
    scene = SceneSpec.model_validate(scene)
    cubes = sorted(
        scene.cubes, key=lambda c: (c.placement.center_xy[1], c.placement.center_xy[0])
    )
    origin = np.asarray(cubes[0].placement.center_xy)
    rows, composition = [], []
    for cube in cubes:
        rows.extend(
            [
                *(np.asarray(cube.placement.center_xy) - origin),
                cube.placement.half_width,
            ]
        )
        composition.append("cube")
    for target in sorted(scene.targets, key=lambda t: t.name):
        rows.extend([*(np.asarray(target.center_xy) - origin), target.half_width])
        composition.append("target")
    for fixture in sorted(scene.fixtures, key=lambda f: f.name):
        rows.extend([*(np.asarray(fixture.center_xy) - origin), 0.02])
        composition.append(fixture.name)
    return composition, np.asarray(rows, dtype=np.float64)


def family_hash(scene):
    composition, values = family_features(scene)
    precision = config()["evaluation"]["family_precision_m"]
    return canonical_hash(
        {
            "version": 1,
            "composition": composition,
            "geometry": np.rint(values / precision).astype(int).tolist(),
        }
    )


def same_family(a, b):
    ac, av = family_features(a)
    bc, bv = family_features(b)
    return ac == bc and bool(
        np.max(np.abs(av - bv))
        <= config()["evaluation"]["family_exclusion_tolerance_m"]
    )


def check_training_scene(scene, bank):
    """Return no held-out content to the generator when rejecting a proposal."""
    if any(
        same_family(scene, row["scene"]) or family_hash(scene) == row["family_hash"]
        for row in bank["templates"]
    ):
        raise ValueError("split_firewall: family_conflict")


def definitions(output, *, forbidden_scenes=()):
    output = Path(output)
    if output.exists():
        return json.loads(output.read_text())
    cfg = config()
    ecfg, geom = cfg["evaluation"], cfg["evaluation"]["geometry"]
    rng = np.random.default_rng(named_seed("trusted_frozen_partitions", cfg["seed"]))
    rows = []
    for partition in ("training-control", "development", "sealed"):
        counts = {
            "training-control": (
                ecfg["control_templates_per_stage"],
                ecfg["control_states_per_template"],
            ),
            "development": (
                ecfg["development_pair_templates_per_stage"],
                ecfg["development_states_per_template"],
            ),
            "sealed": (
                ecfg["sealed_pair_templates_per_stage"],
                ecfg["sealed_states_per_template"],
            ),
        }
        count, states = counts[partition]
        for stage in (
            ("S1", "S2", "S3") if partition == "training-control" else ("S2", "S3")
        ):
            for i in range(count):
                key = f"{partition.replace('-', '_')}_{stage.lower()}_{i:02d}"
                for proposal_index in range(10000):
                    colors = (
                        ["red", "blue"]
                        if stage == "S2"
                        else [str(rng.choice(["red", "blue"]))] * 2
                    )
                    if stage == "S2" and i % 2:
                        colors.reverse()
                    cubes, targets = [], []
                    for side in range(1 if stage == "S1" else 2):
                        x = float(rng.uniform(*geom["initial_x_range"]))
                        y = (
                            0.0
                            if stage == "S1"
                            else (-1 if side == 0 else 1)
                            * float(rng.uniform(*geom["lane_y_range"]))
                        )
                        cubes.append(
                            {
                                "name": f"cube_{side}",
                                "color": colors[side],
                                "placement": {
                                    "center_xy": [x, y],
                                    "half_width": geom["placement_half_width"],
                                },
                            }
                        )
                        targets.append(
                            {
                                "name": "target"
                                if stage == "S1"
                                else ("left" if side == 0 else "right"),
                                "center_xy": [
                                    x + float(rng.uniform(*geom["push_dx_range"])),
                                    y
                                    + float(rng.uniform(*geom["target_y_delta_range"])),
                                ],
                                "half_width": float(
                                    rng.uniform(*geom["target_half_width_range"])
                                ),
                            }
                        )
                    fixtures = (
                        [
                            {
                                "name": "marker",
                                "center_xy": [
                                    float(rng.uniform(*geom["marker_x_range"])),
                                    0.0,
                                ],
                            }
                        ]
                        if stage == "S3"
                        else []
                    )
                    if i % 2:
                        fixtures.append(
                            {"name": "reference_fixture", "center_xy": [0.20, 0.24]}
                        )
                    try:
                        scene = SceneSpec(
                            schema_version="astrapush-1",
                            scene_id=key,
                            stage=stage,
                            cubes=cubes,
                            targets=targets,
                            fixtures=fixtures,
                        )
                        tasks = []
                        for side in range(len(cubes)):
                            referent = (
                                "block"
                                if stage == "S1"
                                else colors[side]
                                if stage == "S2"
                                else ("left" if side == 0 else "right")
                            )
                            destination = (
                                "target"
                                if stage == "S1"
                                else ("left" if side == 0 else "right")
                            )
                            text = instruction_for(stage, referent, destination)
                            language_index = {
                                "training-control": 0,
                                "development": 1,
                                "sealed": 2,
                            }[partition]
                            task = TaskSpec(
                                schema_version="astrapush-1",
                                task_id=f"{key}_{side}",
                                scene_hash=canonical_hash(scene),
                                referent=referent,
                                destination=destination,
                                instruction=instruction_realizations(text)[
                                    language_index
                                ],
                            )
                            tasks.append(
                                task.validate_scene(scene).model_dump(mode="json")
                            )
                        others = [r["scene"] for r in rows] + list(forbidden_scenes)
                        if any(
                            same_family(scene, other)
                            or family_hash(scene) == family_hash(other)
                            for other in others
                        ):
                            continue
                    except ValueError:
                        continue
                    rows.append(
                        {
                            "id": key,
                            "partition": partition,
                            "stage": stage,
                            "scene": scene.model_dump(mode="json"),
                            "tasks": tasks,
                            "family_hash": family_hash(scene),
                            "state_seeds": [
                                named_seed(f"frozen:{key}:{j}") for j in range(states)
                            ],
                            "control_instruction_indices": [
                                (i + j) % len(tasks) for j in range(states)
                            ],
                            "proposal_index": proposal_index,
                        }
                    )
                    break
                else:
                    raise RuntimeError(
                        "Cannot construct the prospectively fixed partition bank"
                    )
    bank = {
        "version": cfg["version"],
        "templates": rows,
        "authorship": "trusted_deterministic_grammar_no_generator",
        "language_holdout": "development reverses clause order; sealed changes sentence separator/case; bounded vocabulary, not open-domain language",
        "family_definition": "translation/label/color invariant relative geometry; centimetre bins plus 7.5mm exclusion tolerance",
        "counts_per_checkpoint": {
            "training-control": 36,
            "development": 32,
            "sealed": 96,
        },
    }
    publish_json(output, bank)
    return bank
