from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from egomimic.experiments.astra_push.render_probe import starter
from egomimic.experiments.astra_push.schemas import SceneSpec
from egomimic.experiments.astra_push.semantics import SuccessEvaluator


def evaluator():
    scene, task = starter("S2")
    initial = {c.name: np.array([*c.placement.center_xy, 0.92]) for c in scene.cubes}
    return SuccessEvaluator(scene, task, initial), initial


def step(checker, positions):
    return checker.update(positions, {name: np.eye(3) for name in positions})


def test_correct_witness_needs_ten_consecutive_steps():
    checker, positions = evaluator()
    positions["cube_a"] = np.array([*checker.target.center_xy, 0.92])
    for _ in range(9):
        assert not step(checker, positions)["success"]
    assert step(checker, positions)["success"]


def test_transient_goal_does_not_accumulate_disjoint_holds():
    checker, positions = evaluator()
    initial = positions["cube_a"].copy()
    for _ in range(4):
        positions["cube_a"] = np.array([*checker.target.center_xy, 0.92])
        for _ in range(9):
            assert not step(checker, positions)["success"]
        positions["cube_a"] = initial
        step(checker, positions)
    assert checker.max_hold == 9


@pytest.mark.parametrize(
    "failure", ["wrong_object", "wrong_target", "lift", "move_then_return"]
)
def test_adversarial_trajectories_fail(failure):
    checker, positions = evaluator()
    target = np.array([*checker.target.center_xy, 0.92])
    if failure == "wrong_object":
        positions["cube_b"] = target
    elif failure == "wrong_target":
        other = next(t for t in checker.scene.targets if t.name != checker.destination)
        positions["cube_a"] = np.array([*other.center_xy, 0.92])
    else:
        positions["cube_a"] = target
        if failure == "lift":
            positions["cube_a"][2] += 0.011
            step(checker, positions)
            positions["cube_a"][2] -= 0.011
        else:
            positions["cube_b"][0] += 0.016
            step(checker, positions)
            positions["cube_b"][0] -= 0.016
    for _ in range(12):
        assert not step(checker, positions)["success"]


def test_fully_inside_checks_rotated_corners():
    checker, positions = evaluator()
    positions["cube_a"] = np.array(
        [checker.target.center_xy[0] + 0.018, checker.target.center_xy[1], 0.92]
    )
    angle = np.pi / 4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    for _ in range(12):
        result = checker.update(positions, {"cube_a": rotation, "cube_b": np.eye(3)})
    assert not result["success"]


def test_scene_novelty_excludes_renaming_and_color_only_changes():
    scene, _ = starter("S2")
    value = scene.model_dump()
    value["scene_id"] = "renamed"
    for i, cube in enumerate(value["cubes"]):
        cube["name"] = f"renamed_{i}"
        cube["color"] = "red" if cube["color"] == "blue" else "blue"
    renamed = SceneSpec.model_validate(value)
    assert renamed.structural_signature() == scene.structural_signature()
    value["targets"][0]["center_xy"][0] += 0.01
    assert (
        SceneSpec.model_validate(value).structural_signature()
        != scene.structural_signature()
    )


def test_scene_schema_rejects_generated_code_unknown_assets_and_ambiguous_relations():
    scene, _ = starter("S3")
    value = scene.model_dump()
    invalid = deepcopy(value)
    invalid["reward_code"] = "return True"
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(invalid)
    invalid = deepcopy(value)
    invalid["cubes"][0]["color"] = "mesh_from_url"
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(invalid)
    invalid = deepcopy(value)
    invalid["fixtures"][0]["center_xy"][1] = 0.1
    with pytest.raises(ValidationError, match="unique sides"):
        SceneSpec.model_validate(invalid)
