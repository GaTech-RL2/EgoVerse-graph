"""Typed data grammar. Generated Python, reward code, and arbitrary assets are forbidden."""

import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "astrapush-1"
Stage = Literal["S1", "S2", "S3"]
Side = Literal["left", "right"]
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")]
XY = Annotated[list[float], Field(min_length=2, max_length=2)]


def canonical_hash(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, strict=True
    )


class Placement(Contract):
    center_xy: XY
    half_width: Annotated[float, Field(ge=0.03, le=0.05)]


class Cube(Contract):
    name: Name
    color: Literal["red", "blue"]
    placement: Placement


class Target(Contract):
    name: Literal["target", "left", "right"]
    center_xy: XY
    half_width: Annotated[float, Field(ge=0.04, le=0.06)]


class Fixture(Contract):
    name: Literal["marker", "reference_fixture"]
    center_xy: XY


class SceneSpec(Contract):
    schema_version: Literal["astrapush-1"]
    scene_id: Name
    stage: Stage
    cubes: Annotated[list[Cube], Field(min_length=1, max_length=2)]
    targets: Annotated[list[Target], Field(min_length=1, max_length=2)]
    fixtures: Annotated[list[Fixture], Field(max_length=2)]

    @model_validator(mode="after")
    def geometry(self):
        if len(self.cubes) != (1 if self.stage == "S1" else 2):
            raise ValueError("Stage requires one cube for S1 and two for S2/S3")
        expected = {"target"} if self.stage == "S1" else {"left", "right"}
        if (
            len(self.targets) != len(expected)
            or {t.name for t in self.targets} != expected
        ):
            raise ValueError("Stage target names/count do not match grammar")
        names = [c.name for c in self.cubes] + [f.name for f in self.fixtures]
        if len(set(names)) != len(names):
            raise ValueError("Scene names must be unique")
        if len({f.name for f in self.fixtures}) != len(self.fixtures):
            raise ValueError("Only one marker and one reference fixture are permitted")
        if self.stage == "S2" and {c.color for c in self.cubes} != {"red", "blue"}:
            raise ValueError("S2 needs one red and one blue cube")
        if self.stage == "S3" and len({c.color for c in self.cubes}) != 1:
            raise ValueError("S3 cubes must have the same color")
        rectangles = [
            (c.placement.center_xy, c.placement.half_width) for c in self.cubes
        ]
        rectangles += [(t.center_xy, t.half_width) for t in self.targets]
        rectangles += [(f.center_xy, 0.02) for f in self.fixtures]
        for xy, half in rectangles:
            if abs(xy[0]) + half > 0.28 or abs(xy[1]) + half > 0.32:
                raise ValueError("Geometry is outside the commissioned tabletop bounds")
        # Placement regions and fixtures must not overlap. Goal markings may
        # overlap a push corridor but cannot cover the initial cube regions.
        for i, (a, ah) in enumerate(rectangles):
            for b, bh in rectangles[i + 1 :]:
                if all(abs(a[k] - b[k]) < ah + bh for k in (0, 1)):
                    raise ValueError("Initial regions, targets or fixtures overlap")
        if self.stage != "S1":
            targets = {t.name: t for t in self.targets}
            if targets["left"].center_xy[1] >= targets["right"].center_xy[1]:
                raise ValueError("Left is negative table Y relative to right")
        if self.stage == "S3":
            markers = [f for f in self.fixtures if f.name == "marker"]
            if len(markers) != 1:
                raise ValueError("S3 requires exactly one marker")
            y = markers[0].center_xy[1]
            left = [
                c
                for c in self.cubes
                if c.placement.center_xy[1] + c.placement.half_width <= y - 0.05
            ]
            right = [
                c
                for c in self.cubes
                if c.placement.center_xy[1] - c.placement.half_width >= y + 0.05
            ]
            if len(left) != 1 or len(right) != 1:
                raise ValueError("S3 needs unique sides with a 5 cm region margin")
        return self

    def structural_signature(self):
        """Version 1, precision 0.1 mm; labels/colors/reset coordinates excluded."""

        def xy(point):
            return [round(float(x), 4) for x in point]

        cubes = sorted(
            self.cubes, key=lambda c: (*c.placement.center_xy, c.placement.half_width)
        )
        fixtures = sorted(self.fixtures, key=lambda f: (f.name, *f.center_xy))
        regions = [
            {
                "asset": "cube_40mm",
                "xy": xy(c.placement.center_xy),
                "half": round(c.placement.half_width, 4),
            }
            for c in cubes
        ]
        relations = []
        for i, c in enumerate(cubes):
            for j, f in enumerate(fixtures):
                if f.name == "marker":
                    side = (
                        "left" if c.placement.center_xy[1] < f.center_xy[1] else "right"
                    )
                    relations.append([i, side, j])
        value = {
            "version": 1,
            "regions": regions,
            "fixtures": [{"asset": f.name, "xy": xy(f.center_xy)} for f in fixtures],
            "targets": sorted(
                (xy(t.center_xy) + [round(t.half_width, 4)] for t in self.targets)
            ),
            "relations": relations,
        }
        return canonical_hash(value)


class TaskSpec(Contract):
    schema_version: Literal["astrapush-1"]
    task_id: Name
    scene_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    referent: Literal["block", "red", "blue", "left", "right"]
    destination: Literal["target", "left", "right"]
    instruction: Annotated[str, Field(min_length=1, max_length=254)]

    def validate_scene(self, scene):
        if self.scene_hash != canonical_hash(scene):
            raise ValueError("Task refers to a different immutable scene")
        expected = instruction_for(scene.stage, self.referent, self.destination)
        if self.instruction != expected:
            raise ValueError("Instruction must match the trusted semantic grammar")
        if len(self.instruction.encode("utf-8")) > 254:
            raise ValueError("Instruction exceeds byte-language capacity")
        if scene.stage == "S1":
            cube = scene.cubes[0]
        elif scene.stage == "S2":
            cube = next(c for c in scene.cubes if c.color == self.referent)
        else:
            marker = next(f for f in scene.fixtures if f.name == "marker")
            cube = next(
                c
                for c in scene.cubes
                if (c.placement.center_xy[1] < marker.center_xy[1])
                == (self.referent == "left")
            )
        target = next(t for t in scene.targets if t.name == self.destination)
        distance = math.dist(cube.placement.center_xy, target.center_xy)
        if not 0.05 <= distance <= 0.15:
            raise ValueError("Nominal direct push must be 5–15 cm")
        return self


def instruction_for(stage, referent, destination):
    if stage == "S1" and referent == "block" and destination == "target":
        return "Push the block into the target."
    if (
        stage == "S2"
        and referent in {"red", "blue"}
        and destination in {"left", "right"}
    ):
        other = "blue" if referent == "red" else "red"
        return f"Push the {referent} block to the {destination} target. Leave the {other} block in place."
    if (
        stage == "S3"
        and referent in {"left", "right"}
        and destination in {"left", "right"}
    ):
        return f"Push the block {referent} of the marker to the {destination} target. Leave the other block in place."
    raise ValueError("Referent/destination are outside the stage grammar")


class Skill(Contract):
    skill: Literal["approach", "align_behind", "contact", "push_toward", "settle"]
    object: Name
    target: Literal["target", "left", "right"]
    speed_m_s: Annotated[float, Field(gt=0, le=0.1)]
    timeout_steps: Annotated[int, Field(ge=1, le=150)]


class TeacherProgram(Contract):
    schema_version: Literal["astrapush-1"]
    skills: Annotated[list[Skill], Field(min_length=5, max_length=5)]

    def validate_scene(self, scene, task):
        if [s.skill for s in self.skills] != [
            "approach",
            "align_behind",
            "contact",
            "push_toward",
            "settle",
        ]:
            raise ValueError("Teacher must use the five ordered pushing skills")
        if len({s.object for s in self.skills}) != 1 or self.skills[0].object not in {
            c.name for c in scene.cubes
        }:
            raise ValueError("Teacher must consistently name one scene cube")
        if (
            any(s.target != task.destination for s in self.skills)
            or sum(s.timeout_steps for s in self.skills) > 150
        ):
            raise ValueError("Teacher destination or time budget is invalid")
        # Object correctness is independently measured by the evaluator, so
        # a syntactically valid but wrong teacher cannot redefine success.
        return self
