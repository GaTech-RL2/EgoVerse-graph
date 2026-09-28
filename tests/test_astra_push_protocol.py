"""No budget refunds after interruption, no quota theft, and no U feedback leak."""

import copy

import pytest
from pydantic import ValidationError

from egomimic.experiments.astra_push.controller import engineering_program
from egomimic.experiments.astra_push.decisions import (
    CurriculumDecision,
    generation_context,
)
from egomimic.experiments.astra_push.ledger import (
    BudgetExhausted,
    Ledger,
    PendingOperation,
)
from egomimic.experiments.astra_push.render_probe import starter


def test_interrupted_operation_consumes_budget_and_resume_does_not_duplicate(tmp_path):
    path, manifest = tmp_path / "ledger.sqlite", {"fixture": 1}
    ledger = Ledger(path, manifest)
    ledger.add_phase("A1", {"attempt": 2, "generation": 1}, quotas={"one": 1})
    ledger.reserve("call1", "A1", "generation", {"prompt": "fixed"})
    ledger.close()
    ledger = Ledger(path, manifest)
    with pytest.raises(PendingOperation):
        ledger.reserve("call1", "A1", "generation", {"prompt": "fixed"})
    with pytest.raises(BudgetExhausted):
        ledger.reserve("call2", "A1", "generation", {"prompt": "fixed"})
    ledger.complete("call1", status="succeeded", result={"response_hash": "a" * 64})
    assert (
        ledger.reserve("call1", "A1", "generation", {"prompt": "fixed"})["execute"]
        is False
    )
    ledger.reserve("attempt1", "A1", "attempt", {}, template="one")
    ledger.complete(
        "attempt1",
        status="failed",
        result={"accepted": False, "reason": "reset_failed"},
    )
    ledger.reserve("attempt2", "A1", "attempt", {}, template="one")
    evidence = {
        "accepted": True,
        "episode_hash": "1" * 64,
        "semantic_receipt_hash": "2" * 64,
    }
    ledger.complete("attempt2", status="succeeded", result=evidence)
    ledger.complete("attempt2", status="succeeded", result=evidence)
    assert ledger.phase_complete("A1")
    with pytest.raises(ValueError, match="immutable"):
        ledger.complete("attempt1", status="succeeded", result=evidence)
    with pytest.raises(BudgetExhausted):
        ledger.reserve("attempt3", "A1", "attempt", {}, template="one")
    ledger.close()


def test_exclusive_coordinator_frozen_manifest_and_phase_quota(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = Ledger(path, {"fixture": 1})
    with pytest.raises(RuntimeError, match="already owns"):
        Ledger(path, {"fixture": 1})
    ledger.add_phase("round", {"attempt": 3}, quotas={"one": 1, "two": 1})
    with pytest.raises(ValueError, match="quota"):
        ledger.add_phase("round", {"attempt": 3}, quotas={"one": 2})
    sequence = ["data", "checkpoint", "probes", "committed"]
    with pytest.raises(ValueError, match="skip"):
        ledger.transition("round", sequence, "checkpoint", "a" * 64)
    ledger.transition("round", sequence, "data", "a" * 64)
    ledger.transition("round", sequence, "data", "a" * 64)
    ledger.close()
    with pytest.raises(ValueError, match="manifest"):
        Ledger(path, {"fixture": 2})


def decision():
    templates = []
    for index, (stage, count) in enumerate(
        [("S1", 25), ("S2", 12), ("S2", 13), ("S3", 12), ("S3", 13)]
    ):
        scene, task = starter(stage)
        templates.append(
            {
                "template_id": f"t{index}",
                "accepted_episodes": count,
                "scene": scene.model_dump(mode="json"),
                "task": {
                    k: v
                    for k, v in task.model_dump().items()
                    if k not in {"scene_hash", "schema_version"}
                },
                "teacher": engineering_program(scene, task).model_dump(mode="json"),
            }
        )
    return {
        "schema_version": "astrapush-decision-1",
        "arm": "U",
        "round": 1,
        "allocations": {"S1": 25, "S2": 25, "S3": 25},
        "templates": templates,
        "rationale": "offline contract fixture",
        "predicted_change": "no measured prediction",
        "stage_intent": {"S1": "consolidate", "S2": "advance", "S3": "advance"},
    }


def test_exact_template_quotas_and_no_reallocation_during_repairs():
    original = CurriculumDecision.model_validate(decision())
    original.validate_context(arm="U", round_index=1)
    malformed = decision()
    malformed["templates"][0]["accepted_episodes"] = 24
    with pytest.raises(ValidationError, match="quotas"):
        CurriculumDecision.model_validate(malformed)
    stolen = decision()
    stolen["templates"][1]["accepted_episodes"] += 1
    stolen["templates"][2]["accepted_episodes"] -= 1
    with pytest.raises(ValueError, match="donate"):
        CurriculumDecision.model_validate(stolen).validate_revision(original)
    malformed = copy.deepcopy(decision())
    malformed["report_hash"] = "a" * 64
    with pytest.raises(ValidationError, match="Uniform"):
        CurriculumDecision.model_validate(malformed)


def test_uniform_payload_rejects_performance_and_other_arm_archive():
    kwargs = {
        "arm": "U",
        "round_index": 1,
        "generation_archive": [],
        "validity_failures": [],
    }
    payload = generation_context(**kwargs)
    assert "learner_report" not in payload and "report_hash" not in payload
    with pytest.raises(ValueError, match="never receive"):
        generation_context(**kwargs, report={"successes": 1})
    kwargs["generation_archive"] = [
        {
            "source_arm": "A",
            "scene_hash": "a" * 64,
            "structural_signature": "b" * 64,
            "stage": "S1",
        }
    ]
    with pytest.raises(ValueError, match="other arm"):
        generation_context(**kwargs)
