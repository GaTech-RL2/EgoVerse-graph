"""Typed curriculum choices and a generation-only view for the uniform arm."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from egomimic.experiments.astra_push.schemas import (
    Contract,
    Name,
    SceneSpec,
    TaskSpec,
    TeacherProgram,
    canonical_hash,
    instruction_for,
)

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonnegative = Annotated[int, Field(ge=0)]


class TaskProposal(Contract):
    task_id: Name
    referent: Literal["block", "red", "blue", "left", "right"]
    destination: Literal["target", "left", "right"]
    instruction: str

    def bind(self, scene):
        if self.instruction != instruction_for(
            scene.stage, self.referent, self.destination
        ):
            raise ValueError(
                "Generated training tasks must use the canonical language realization"
            )
        task = TaskSpec(
            schema_version="astrapush-1",
            scene_hash=canonical_hash(scene),
            **self.model_dump(),
        )
        return task.validate_scene(scene)


class TemplateProposal(Contract):
    template_id: Name
    accepted_episodes: Annotated[int, Field(gt=0, le=75)]
    scene: SceneSpec
    task: TaskProposal
    teacher: TeacherProgram

    @model_validator(mode="after")
    def semantic_contract(self):
        task = self.task.bind(self.scene)
        self.teacher.validate_scene(self.scene, task)
        return self


class Allocations(Contract):
    S1: Nonnegative
    S2: Nonnegative
    S3: Nonnegative


class CurriculumDecision(Contract):
    schema_version: Literal["astrapush-decision-1"]
    arm: Literal["A", "U"]
    round: Annotated[int, Field(ge=1, le=4)]
    report_hash: SHA256 | None = None
    allocations: Allocations
    templates: Annotated[list[TemplateProposal], Field(min_length=5, max_length=5)]
    rationale: Annotated[str, Field(min_length=1, max_length=6000)]
    predicted_change: Annotated[str, Field(min_length=1, max_length=3000)]
    stage_intent: dict[
        Literal["S1", "S2", "S3"], Literal["advance", "consolidate", "revisit"]
    ]

    @model_validator(mode="after")
    def matched_budgets(self):
        allocations = self.allocations.model_dump()
        if sum(allocations.values()) != 75:
            raise ValueError(
                "Each round must request exactly seventy-five accepted episodes"
            )
        if len({t.template_id for t in self.templates}) != 5:
            raise ValueError("Five distinct positive-allocation templates are required")
        by_stage = {
            k: sum(t.accepted_episodes for t in self.templates if t.scene.stage == k)
            for k in allocations
        }
        if by_stage != allocations:
            raise ValueError("Template quotas do not sum to the stage allocations")
        if set(self.stage_intent) != {
            k for k, value in allocations.items() if value > 0
        }:
            raise ValueError("Intent must be recorded exactly for active stages")
        if self.arm == "U":
            if (
                allocations != {"S1": 25, "S2": 25, "S3": 25}
                or self.report_hash is not None
            ):
                raise ValueError(
                    "Uniform allocation is fixed and cannot reference learner feedback"
                )
        elif self.report_hash is None:
            raise ValueError(
                "Adaptive rationale must reference its frozen learner report"
            )
        return self

    def validate_context(self, *, arm, round_index, report_hash=None):
        if (self.arm, self.round, self.report_hash) != (arm, round_index, report_hash):
            raise ValueError(
                "Decision belongs to a different arm, round or learner report"
            )
        return self

    def validate_revision(self, previous):
        if (
            self.arm != previous.arm
            or self.round != previous.round
            or self.allocations != previous.allocations
        ):
            raise ValueError(
                "Repairs cannot change a partly acquired round's allocations"
            )
        before = {
            t.template_id: (t.scene.stage, t.accepted_episodes)
            for t in previous.templates
        }
        after = {
            t.template_id: (t.scene.stage, t.accepted_episodes) for t in self.templates
        }
        if before != after:
            raise ValueError(
                "Repairs cannot donate or replace committed template quotas"
            )
        return self


class ProbeCounts(Contract):
    trials: Literal[12]
    successes: Annotated[int, Field(ge=0, le=12)]
    wrong_object: Annotated[int, Field(ge=0, le=12)]
    wrong_target: Annotated[int, Field(ge=0, le=12)]
    lift_violation: Annotated[int, Field(ge=0, le=12)]
    preservation_violation: Annotated[int, Field(ge=0, le=12)]
    displacement_failure: Annotated[int, Field(ge=0, le=12)]
    overshoot_failure: Annotated[int, Field(ge=0, le=12)]


class LearnerReport(Contract):
    schema_version: Literal["astrapush-control-report-1"]
    partition: Literal["training-control"]
    arm: Literal["common", "A", "U"]
    checkpoint_hash: SHA256
    probe_manifest_hash: SHA256
    completed_round: Annotated[int, Field(ge=0, le=4)]
    stages: dict[Literal["S1", "S2", "S3"], ProbeCounts]
    previous_report_hash: SHA256 | None
    previous_allocation: Allocations | None
    frame_sequence_hashes: Annotated[list[SHA256], Field(max_length=6)]
    inferred_error_categories: Annotated[list[str], Field(max_length=8)]
    stage_success_deltas: dict[Literal["S1", "S2", "S3"], int] = Field(
        default_factory=dict
    )
    teacher_yield: dict[str, dict[str, int]] = Field(default_factory=dict)
    remaining_rounds: Annotated[int, Field(ge=0, le=4)] = 4
    case_successes: dict[str, bool] = Field(default_factory=dict)
    stage_forgetting_counts: dict[Literal["S1", "S2", "S3"], Nonnegative] = Field(
        default_factory=dict
    )
    remaining_accepted_episodes: Nonnegative = 300
    remaining_optimizer_updates: Nonnegative = 2000

    @model_validator(mode="after")
    def all_stages(self):
        if set(self.stages) != {"S1", "S2", "S3"}:
            raise ValueError(
                "Report must distinguish zero successes from missing stage probes"
            )
        return self


class GenerationArchiveEntry(Contract):
    source_arm: Literal["common", "A", "U"]
    scene_hash: SHA256
    structural_signature: SHA256
    stage: Literal["S1", "S2", "S3"]


class ValidityFailure(Contract):
    template_id: Name
    kind: Literal[
        "schema", "geometry", "duplicate", "reset", "teacher", "split_firewall"
    ]
    detail_code: Literal[
        "invalid_contract",
        "out_of_bounds",
        "blocked_corridor",
        "signature_seen",
        "reset_failed",
        "teacher_failed",
        "family_conflict",
    ]


def generation_context(
    *, arm, round_index, generation_archive, validity_failures, report=None
):
    if arm not in {"A", "U"} or round_index not in (1, 2, 3, 4):
        raise ValueError("Unknown arm/round")
    archive = [GenerationArchiveEntry.model_validate(r) for r in generation_archive]
    failures = [ValidityFailure.model_validate(r) for r in validity_failures]
    if any(r.source_arm not in {arm, "common"} for r in archive):
        raise ValueError("An arm cannot see the other arm's generation archive")
    context = {
        "arm": arm,
        "round": round_index,
        "decision_schema": CurriculumDecision.model_json_schema(),
        "generation_archive": [r.model_dump(mode="json") for r in archive],
        "validity_failures": [r.model_dump(mode="json") for r in failures],
    }
    if arm == "U":
        if report is not None:
            raise ValueError(
                "The uniform generator must never receive learner performance"
            )
        context["fixed_allocations"] = {"S1": 25, "S2": 25, "S3": 25}
    else:
        report = LearnerReport.model_validate(report)
        if report.completed_round != round_index - 1 or report.arm != (
            "common" if round_index == 1 else "A"
        ):
            raise ValueError(
                "Adaptive feedback must be from the immediately preceding checkpoint"
            )
        context["learner_report"] = report.model_dump(mode="json")
        context["report_hash"] = canonical_hash(report)
    return context
