"""Validated generated-scene commissioning, separate from imitation training."""

from collections import Counter
from typing import Annotated, Literal

from pydantic import Field, model_validator

from egomimic.experiments.astra_push.decisions import TaskProposal
from egomimic.experiments.astra_push.render_probe import starter
from egomimic.experiments.astra_push.schemas import (
    Contract,
    Name,
    SceneSpec,
    TeacherProgram,
    canonical_hash,
)


def composition_relations(scene):
    """Compare authored composition/relations without names, colors or positions."""
    marker = next((f for f in scene.fixtures if f.name == "marker"), None)
    return canonical_hash(
        {
            "cubes": len(scene.cubes),
            "targets": len(scene.targets),
            "fixtures": sorted(f.name for f in scene.fixtures),
            "marker_sides": sorted(
                "left" if c.placement.center_xy[1] < marker.center_xy[1] else "right"
                for c in scene.cubes
            )
            if marker
            else [],
        }
    )


class GateTemplate(Contract):
    template_id: Name
    scene: SceneSpec
    task: TaskProposal
    teacher: TeacherProgram
    novelty_rationale: Annotated[str, Field(min_length=1, max_length=1500)]

    @model_validator(mode="after")
    def valid_program(self):
        task = self.task.bind(self.scene)
        self.teacher.validate_scene(self.scene, task)
        return self


class CommissioningProposal(Contract):
    schema_version: Literal["astrapush-commissioning-1"]
    templates: Annotated[list[GateTemplate], Field(min_length=6, max_length=6)]

    @model_validator(mode="after")
    def six_novel_scenes(self):
        if Counter(t.scene.stage for t in self.templates) != {
            "S1": 2,
            "S2": 2,
            "S3": 2,
        }:
            raise ValueError("Commissioning needs exactly two scenes per stage")
        if len({t.template_id for t in self.templates}) != 6:
            raise ValueError("Commissioning template IDs must be unique")
        signatures = {t.scene.structural_signature() for t in self.templates}
        if len(signatures) != 6:
            raise ValueError("Commissioning requires six structural signatures")
        starters = {s: starter(s)[0] for s in ("S1", "S2", "S3")}
        if signatures & {s.structural_signature() for s in starters.values()}:
            raise ValueError("Generated scene duplicates an engineering starter")
        changed = sum(
            composition_relations(t.scene)
            != composition_relations(starters[t.scene.stage])
            for t in self.templates
        )
        if changed < 3:
            raise ValueError("At least three scenes must change composition/relations")
        return self

    def validate_revision(self, previous, accepted_templates):
        """A repair cannot replace a successful commissioning witness family."""
        old = {t.template_id: t for t in previous.templates}
        new = {t.template_id: t for t in self.templates}
        if set(old) != set(new) or any(
            old[k].scene.stage != new[k].scene.stage for k in old
        ):
            raise ValueError("Commissioning repairs must retain template IDs/stages")
        if any(new[k] != old[k] for k in accepted_templates):
            raise ValueError("A commissioned template cannot be rewritten")
        return self
