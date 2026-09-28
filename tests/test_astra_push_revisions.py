"""Repairs retain incurred cost, complete data, frozen contracts and lineage."""

import copy
import json
import sqlite3

import pytest

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.campaign import acquire
from egomimic.experiments.astra_push.collect import completion
from egomimic.experiments.astra_push.continuation import (
    active_ledger,
    open_continuation,
    prepare_continuation,
    verify_ledger_prefix,
)
from egomimic.experiments.astra_push.controller import engineering_program
from egomimic.experiments.astra_push.decisions import CurriculumDecision
from egomimic.experiments.astra_push.ledger import Ledger
from egomimic.experiments.astra_push.render_probe import starter
from egomimic.experiments.astra_push.revisions import (
    RevisionManager,
    acquisition_snapshot,
    validate_exchange,
    validate_revision,
)
from egomimic.experiments.astra_push.schemas import canonical_hash


def decision(arm="U"):
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
                "task": task.model_dump(exclude={"scene_hash", "schema_version"}),
                "teacher": engineering_program(scene, task).model_dump(mode="json"),
            }
        )
    return CurriculumDecision.model_validate(
        {
            "schema_version": "astrapush-decision-1",
            "arm": arm,
            "round": 1,
            "report_hash": "a" * 64 if arm == "A" else None,
            "allocations": {"S1": 25, "S2": 25, "S3": 25},
            "templates": templates,
            "rationale": "Offline unit-test fixture only",
            "predicted_change": "Unmeasured",
            "stage_intent": {"S1": "consolidate", "S2": "advance", "S3": "advance"},
        }
    )


def changed(original, index=1):
    value = original.model_dump(mode="json")
    value["templates"][index]["teacher"]["skills"][0]["speed_m_s"] = 0.09
    return CurriculumDecision.model_validate(value)


def ledger_with_failed_attempt(tmp_path, original):
    ledger = Ledger(tmp_path / "ledger.sqlite", {"fixture": 1})
    phase = f"{original.arm}{original.round}"
    ledger.add_phase(
        phase,
        {"attempt": 300, "training_block": 20},
        quotas={t.template_id: t.accepted_episodes for t in original.templates},
    )
    ledger.reserve(
        f"{phase}-attempt-0001",
        phase,
        "attempt",
        {"old_scene": "immutable"},
        template="t1",
    )
    ledger.complete(
        f"{phase}-attempt-0001",
        status="failed",
        result={"accepted": False, "reason": "teacher_failed"},
    )
    return ledger


@pytest.mark.parametrize("arm", ["U", "A"])
def test_repair_freezes_completed_templates_and_all_quotas(tmp_path, arm):
    original = decision(arm)
    ledger = ledger_with_failed_attempt(tmp_path, original)
    snapshot = acquisition_snapshot(ledger, f"{arm}1")
    assert snapshot["cutover_attempt"] == 1 and snapshot["attempt_cap"] == 300
    assert validate_revision(
        original, changed(original), snapshot, archive=[], bank={"templates": []}
    ) == ["t1"]
    completed = copy.deepcopy(snapshot)
    completed["remaining"]["t1"] = 0
    with pytest.raises(ValueError, match="Completed templates"):
        validate_revision(
            original, changed(original), completed, archive=[], bank={"templates": []}
        )
    stolen = original.model_dump(mode="json")
    stolen["templates"][1]["accepted_episodes"] -= 1
    stolen["templates"][2]["accepted_episodes"] += 1
    with pytest.raises(ValueError, match="donate"):
        validate_revision(
            original,
            CurriculumDecision.model_validate(stolen),
            snapshot,
            archive=[],
            bank={"templates": []},
        )
    ledger.reserve(f"{arm}1-attempt-0002", f"{arm}1", "attempt", {}, template="t1")
    with pytest.raises(ValueError, match="reserved receipt"):
        acquisition_snapshot(ledger, f"{arm}1")
    ledger.close()


def archive_fixture(root, prompt, answer):
    directory = root / "call-002"
    directory.mkdir()
    request = {
        "phase": "U1",
        "request_id": "offline-test-repair",
        "repair_of": "offline-test-original",
        "model": "gpt-6-astra",
        "fork_turns": "none",
        "prompt": {"generation_request": prompt},
    }
    publish_json(directory / "request.json", request)
    publish_json(directory / "raw-response.json", answer.model_dump(mode="json"))
    receipt = {
        "transport": "codex_subagent",
        "valid": True,
        "request_sha256": file_hash(directory / "request.json"),
        "response_sha256": file_hash(directory / "raw-response.json"),
        "answer_hash": canonical_hash(answer),
    }
    publish_json(directory / "receipt.json", receipt)
    return {
        "exchange_directory": "call-002",
        "receipt_sha256": file_hash(directory / "receipt.json"),
    }


def test_revision_cutover_is_immutable_and_resume_replays_no_attempts(
    tmp_path, monkeypatch
):
    original = decision()
    ledger = ledger_with_failed_attempt(tmp_path, original)
    generation = tmp_path / "generation/U1"
    revisions = generation / "revisions"
    revisions.mkdir(parents=True)
    publish_json(
        generation / "request.json",
        {
            "prompt": {
                "context": {
                    "arm": "U",
                    "round": 1,
                    "validity_failures": [],
                    "generation_archive": [],
                },
                "frame_sequences": [],
                "approved_common_examples": [],
            }
        },
    )
    publish_json(
        revisions / "intent-001.json",
        {
            "validity_failures": [
                {
                    "template_id": "t1",
                    "kind": "teacher",
                    "detail_code": "teacher_failed",
                }
            ],
            "teacher_evidence": {"failure": "Approach timed out; offline fixture"},
        },
    )
    revised = changed(original)

    def respond(_seconds):
        directory = revisions / "revision-001"
        request = json.loads((directory / "request.json").read_text())
        assert request["acquisition"]["cutover_attempt"] == 1
        assert "learner_report" not in request["prompt"]["context"]
        ready = archive_fixture(generation, request["prompt"], revised)
        publish_json(directory / "READY.json", ready)

    monkeypatch.setattr("egomimic.experiments.astra_push.revisions.time.sleep", respond)
    manager = RevisionManager(
        generation, original=original, archive=[], bank={"templates": []}
    )
    assert manager.apply(ledger) == revised
    committed = revisions / "revision-001/committed.json"
    digest = file_hash(committed)
    assert len(manager.archive_entries()) == 5  # Preserve original prompt duplicates.
    ledger.reserve(
        "U1-attempt-0002",
        "U1",
        "attempt",
        {"revised_teacher": canonical_hash(revised)},
        template="t1",
    )
    ledger.complete("U1-attempt-0002", status="failed", result={"accepted": False})
    resumed = RevisionManager(
        generation, original=original, archive=[], bank={"templates": []}
    )
    assert resumed.apply(ledger) == revised and file_hash(committed) == digest
    assert acquisition_snapshot(ledger, "U1")["cutover_attempt"] == 2
    # The historical prefix remains exact even after new incurred work.
    assert json.loads(committed.read_text())["acquisition"]["cutover_attempt"] == 1
    raw = generation / "call-002/raw-response.json"
    raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match="bound native request"):
        RevisionManager(
            generation, original=original, archive=[], bank={"templates": []}
        ).apply(ledger)
    ledger.close()


def test_every_revised_geometry_is_retained_in_own_generation_archive(tmp_path):
    original = decision()
    value = original.model_dump(mode="json")
    value["templates"][1]["scene"]["cubes"][0]["placement"]["center_xy"][0] += 0.02
    revised = CurriculumDecision.model_validate(value)
    manager = RevisionManager(
        tmp_path, original=original, archive=[], bank={"templates": []}
    )
    manager.history.append(revised)
    entries = manager.archive_entries()
    assert len(entries) == 6
    assert {row["source_arm"] for row in entries} == {"U"}
    assert revised.templates[1].scene.structural_signature() in {
        row["structural_signature"] for row in entries
    }


def test_revision_exchange_charges_invalid_and_unused_calls(tmp_path):
    root = tmp_path / "generation/U1"
    root.mkdir(parents=True)
    ready = archive_fixture(root, {}, decision())
    for index in range(3, 11):
        directory = root / f"call-{index:03d}"
        directory.mkdir()
        publish_json(directory / "request.json", {"repair_of": None})
    with pytest.raises(ValueError, match="budgets"):
        validate_exchange(root, ready, prompt={}, phase="U1")


def test_versioned_ledger_preserves_all_prior_cost_and_detects_quota_mutation(tmp_path):
    original = decision()
    ledger = ledger_with_failed_attempt(tmp_path, original)
    ledger.reserve("U1-attempt-0002", "U1", "attempt", {"pending": True}, template="t1")
    parent = tmp_path / "initial-ledger.sqlite"
    with sqlite3.connect(parent) as target:
        ledger.connection.backup(target)
    ledger.complete("U1-attempt-0002", status="failed", result={"accepted": False})
    verify_ledger_prefix(
        parent, ledger.path if hasattr(ledger, "path") else tmp_path / "ledger.sqlite"
    )
    ledger.connection.execute(
        "UPDATE phases SET caps=? WHERE name='U1'", ('{"attempt":301}',)
    )
    with pytest.raises(ValueError, match="original phases"):
        verify_ledger_prefix(parent, tmp_path / "ledger.sqlite")
    ledger.close()


def test_active_ledger_cannot_silently_use_old_or_unbound_database(tmp_path):
    assert active_ledger(tmp_path) == tmp_path / "ledger.sqlite"
    directory = tmp_path / "continuations/continuation-001"
    directory.mkdir(parents=True)
    publish_json(
        directory / "receipt.json",
        {"ledger_path": "continuations/continuation-001/ledger.sqlite"},
    )
    publish_json(
        directory / "activation.json",
        {
            "receipt_sha256": file_hash(directory / "receipt.json"),
            "previous_activation_sha256": None,
        },
    )
    assert active_ledger(tmp_path) == directory / "ledger.sqlite"
    (directory / "receipt.json").write_text('{"ledger_path":"ledger.sqlite"}')
    with pytest.raises(ValueError, match="receipt changed"):
        active_ledger(tmp_path)


def test_acquisition_repair_collects_only_remaining_quota_without_refunding_cost(
    tmp_path, monkeypatch
):
    original = decision()
    ledger = ledger_with_failed_attempt(tmp_path, original)
    data = tmp_path / "data/U1"
    data.mkdir(parents=True)
    ordinal = 1
    original_hashes = {}
    for template in original.templates:
        count = template.accepted_episodes - (template.template_id == "t1")
        for _ in range(count):
            ordinal += 1
            op = f"U1-attempt-{ordinal:04d}"
            directory = data / op
            directory.mkdir()
            episode = directory / "episode.hdf5"
            episode.write_bytes(f"offline accepted fixture {op}".encode())
            receipt = {
                "accepted": True,
                "failure": None,
                "episode": {"path": str(episode), "sha256": file_hash(episode)},
            }
            publish_json(directory / "receipt.json", receipt)
            original_hashes[str(episode)] = file_hash(episode)
            ledger.reserve(
                op, "U1", "attempt", {"original": True}, template=template.template_id
            )
            ledger.complete(
                op,
                status="succeeded",
                result=completion(receipt, directory / "receipt.json"),
            )
    assert ordinal == 75
    revised = changed(original)
    generation = tmp_path / "generation/U1"
    revisions = generation / "revisions"
    revisions.mkdir(parents=True)
    publish_json(
        generation / "request.json", {"prompt": {"context": {"validity_failures": []}}}
    )
    publish_json(
        revisions / "intent-001.json",
        {
            "validity_failures": [
                {
                    "template_id": "t1",
                    "kind": "teacher",
                    "detail_code": "teacher_failed",
                }
            ],
            "teacher_evidence": {"failure": "offline fixture"},
        },
    )

    def respond(_seconds):
        directory = revisions / "revision-001"
        prompt = json.loads((directory / "request.json").read_text())["prompt"]
        assert prompt["acquisition_revision"]["acquisition"]["cutover_attempt"] == 75
        assert (
            sum(prompt["acquisition_revision"]["acquisition"]["remaining"].values())
            == 1
        )
        publish_json(
            directory / "READY.json", archive_fixture(generation, prompt, revised)
        )

    class Worker:
        calls = []

        def call(self, operation, **arguments):
            self.calls.append(arguments)
            assert arguments["program"] == revised.templates[1].teacher.model_dump(
                mode="json"
            )
            assert arguments["episode_id"] == "U1-attempt-0076"
            directory = data / arguments["episode_id"]
            directory.mkdir()
            episode = directory / "episode.hdf5"
            episode.write_bytes(b"offline revised fixture")
            receipt = {
                "accepted": True,
                "failure": None,
                "episode": {"path": str(episode), "sha256": file_hash(episode)},
            }
            publish_json(directory / "receipt.json", receipt)
            return receipt

    monkeypatch.setattr("egomimic.experiments.astra_push.revisions.time.sleep", respond)
    worker = Worker()
    manager = RevisionManager(
        generation, original=original, archive=[], bank={"templates": []}
    )
    result = acquire(
        data,
        phase="U1",
        templates=original.templates,
        quotas={t.template_id: t.accepted_episodes for t in original.templates},
        cap=300,
        ledger=ledger,
        simulator=worker,
        arm="U",
        round_index=1,
        revisions=manager,
    )
    assert len(worker.calls) == 1 and result["attempts"] == 76
    assert len(result["accepted"]) == 75
    assert result["teacher_yield"]["t1"] == {"attempts": 13, "accepted": 12}
    assert all(file_hash(path) == digest for path, digest in original_hashes.items())
    assert (
        json.loads((revisions / "revision-001/committed.json").read_text())[
            "acquisition"
        ]["attempt_cap"]
        == 300
    )
    ledger.close()


def test_continuation_requires_new_validation_and_keeps_frozen_state(
    tmp_path, monkeypatch
):
    root = tmp_path / "experiment"
    root.mkdir()
    for path, value in {
        "run-manifest.json": {
            "source_commit": "a" * 40,
            "config": {"attempt_cap": 300},
        },
        "partition-definitions.json": {"immutable": "bank"},
        "frozen-bank/manifest.json": {"exact_states": "frozen"},
        "checkpoints/initial.json": {"initial": "frozen"},
        "pause.json": {"pid": 1, "pending": [], "counts": [], "phases": []},
        "validation.json": {"source_commit": "a" * 40, "conclusion": "success"},
    }.items():
        publish_json(root / path, value)
    parent = root / "parent.sqlite"
    with sqlite3.connect(parent) as db:
        db.executescript(
            "CREATE TABLE metadata (key,value);CREATE TABLE phases (name,caps,quotas);"
            "CREATE TABLE transitions (phase,ordinal,name,evidence_hash);"
            "CREATE TABLE operations (id,status,phase,kind,template);"
        )
    monkeypatch.setattr(
        "egomimic.experiments.astra_push.continuation.source_audit",
        lambda *_: {"frozen": "same"},
    )
    monkeypatch.setattr(
        "egomimic.experiments.astra_push.continuation.verify_paused_process",
        lambda _pause: None,
    )
    arguments = dict(
        root=root,
        name="continuation-001",
        parent_ledger=parent,
        pause_receipt=root / "pause.json",
        execution_source_commit="b" * 40,
        validation_receipt=root / "validation.json",
    )
    with pytest.raises(ValueError, match="own successful validation"):
        prepare_continuation(**arguments)
    (root / "validation.json").write_text(
        json.dumps({"source_commit": "b" * 40, "conclusion": "success"})
    )
    receipt = prepare_continuation(**arguments)
    original_hash = file_hash(parent)
    ledger_path, execution = open_continuation(root, receipt, source_commit="a" * 40)
    assert ledger_path != parent and execution == "b" * 40
    assert file_hash(parent) == original_hash and active_ledger(root) == ledger_path
    (root / "partition-definitions.json").write_text('{"changed":"forbidden"}')
    with pytest.raises(ValueError, match="Frozen continuation artifact"):
        open_continuation(root, receipt, source_commit="a" * 40)
