"""Immutable, quota-preserving acquisition revisions and their cutover receipts.

Only an explicit teacher-validity intent opens a revision wait. Both curriculum
arms use this same mechanism, attempt budget, grammar and controller. Original
decisions and every incurred operation remain unchanged.
"""

import copy
import json
import time
from pathlib import Path

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.decisions import (
    CurriculumDecision,
    ValidityFailure,
)
from egomimic.experiments.astra_push.partitions import check_training_scene, config
from egomimic.experiments.astra_push.provider import strict_json
from egomimic.experiments.astra_push.schemas import canonical_hash


def read(path):
    return strict_json(Path(path).read_text())


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("Published acquisition revision changed")
        return file_hash(path)
    return publish_json(path, value)


def acquisition_snapshot(ledger, phase, *, cutover=None):
    phase_row = ledger.connection.execute(
        "SELECT * FROM phases WHERE name=?", (phase,)
    ).fetchone()
    if phase_row is None or not phase_row["quotas"]:
        raise ValueError("Revision requires a committed phase and template quotas")
    rows = ledger.connection.execute(
        "SELECT * FROM operations WHERE phase=? AND kind='attempt' ORDER BY id",
        (phase,),
    ).fetchall()
    if cutover is None:
        cutover = len(rows)
    if type(cutover) is not int or not 0 <= cutover <= len(rows):
        raise ValueError("Revision cutover is outside the incurred attempt history")
    rows = rows[:cutover]
    if any(row["status"] == "reserved" for row in rows):
        raise ValueError(
            "Recover every reserved receipt before an acquisition revision"
        )
    quotas = json.loads(phase_row["quotas"])
    accepted = {name: 0 for name in quotas}
    for row in rows:
        result = json.loads(row["result"])
        if result.get("accepted"):
            accepted[row["template"]] += 1
    if any(accepted[name] > quota for name, quota in quotas.items()):
        raise ValueError("Incurred accepted data exceeds its committed quota")
    caps = json.loads(phase_row["caps"])
    return {
        "phase": phase,
        "cutover_attempt": cutover,
        "attempt_cap": caps["attempt"],
        "caps": caps,
        "quotas": quotas,
        "accepted": accepted,
        "remaining": {name: quota - accepted[name] for name, quota in quotas.items()},
        "operation_prefix_hash": canonical_hash([dict(row) for row in rows]),
    }


def validate_revision(previous, revised, snapshot, *, archive, bank):
    revised.validate_revision(previous).validate_context(
        arm=previous.arm, round_index=previous.round, report_hash=previous.report_hash
    )
    before = {template.template_id: template for template in previous.templates}
    changed = []
    for template in revised.templates:
        name = template.template_id
        if canonical_hash(template) != canonical_hash(before[name]):
            if snapshot["remaining"][name] == 0:
                raise ValueError("Completed templates are frozen during a revision")
            changed.append(name)
        check_training_scene(template.scene, bank)
    if not changed:
        raise ValueError("An acquisition repair must revise an unfilled template")
    signatures = {
        template.scene.structural_signature() for template in revised.templates
    }
    if len(signatures - {entry["structural_signature"] for entry in archive}) < 2:
        raise ValueError("Revised round lacks two structurally new templates")
    return changed


def validate_exchange(root, ready, *, prompt, phase):
    """Validate the exact native call while charging all retained phase calls."""
    root = Path(root)
    requests = list(root.glob("call-*/request.json"))
    total_calls = config()["prior_generation_calls"] + len(
        list(root.parent.glob("*/call-*/request.json"))
    )
    if not 1 <= len(requests) <= 8 or total_calls > 80:
        raise ValueError("Archived generation calls exceed declared budgets")
    repairs = {}
    for path in requests:
        original = read(path).get("repair_of")
        if original:
            repairs[original] = repairs.get(original, 0) + 1
    if any(count > 2 for count in repairs.values()):
        raise ValueError("More than two repair responses for an original proposal")
    directory = (root / ready["exchange_directory"]).resolve()
    if root.resolve() not in directory.parents or directory.parent != root.resolve():
        raise ValueError("Revision exchange must be an immediate phase archive")
    request, receipt = (
        read(directory / "request.json"),
        read(directory / "receipt.json"),
    )
    raw = directory / "raw-response.json"
    if (
        request["phase"] != phase
        or request["prompt"].get("generation_request") != prompt
        or request["model"] != "gpt-6-astra"
        or request["fork_turns"] != "none"
        or receipt["transport"] != "codex_subagent"
        or not receipt["valid"]
        or receipt["request_sha256"] != file_hash(directory / "request.json")
        or receipt["response_sha256"] != file_hash(raw)
        or ready["receipt_sha256"] != file_hash(directory / "receipt.json")
    ):
        raise ValueError("Revision exchange differs from its bound native request")
    decision = CurriculumDecision.model_validate(read(raw))
    if canonical_hash(decision) != receipt["answer_hash"]:
        raise ValueError("Revision response differs from its validation receipt")
    return decision, request, receipt


class RevisionManager:
    def __init__(self, root, *, original, archive, bank):
        self.root = Path(root)
        self.original = original
        self.current = original
        self.archive = archive
        self.bank = bank
        self.history = [original]
        self.applied = 0

    def apply(self, ledger):
        phase = f"{self.original.arm}{self.original.round}"
        revision_root = self.root / "revisions"
        while True:
            ordinal = self.applied + 1
            intent_path = revision_root / f"intent-{ordinal:03d}.json"
            if not intent_path.exists():
                break
            directory = revision_root / f"revision-{ordinal:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            intent = read(intent_path)
            if set(intent) != {"validity_failures", "teacher_evidence"}:
                raise ValueError("Revision intent contains unsupported feedback fields")
            failures = [
                ValidityFailure.model_validate(item)
                for item in intent["validity_failures"]
            ]
            if not failures or any(
                item.kind not in {"teacher", "reset", "geometry"} for item in failures
            ):
                raise ValueError(
                    "Acquisition revisions need explicit teacher validity feedback"
                )
            committed_path = directory / "committed.json"
            existing = read(committed_path) if committed_path.exists() else None
            request_path = directory / "request.json"
            requested = read(request_path) if request_path.exists() else None
            cutover = requested["acquisition"]["cutover_attempt"] if requested else None
            snapshot = acquisition_snapshot(ledger, phase, cutover=cutover)
            if not existing and acquisition_snapshot(ledger, phase) != snapshot:
                raise ValueError(
                    "New attempts occurred after an uncommitted revision request"
                )
            if snapshot["cutover_attempt"] >= snapshot["attempt_cap"]:
                raise ValueError(
                    "Revision cannot extend an exhausted acquisition budget"
                )
            for failure in failures:
                if snapshot["remaining"].get(failure.template_id, 0) <= 0:
                    raise ValueError(
                        "Validity feedback names a completed or unknown template"
                    )
            prompt = copy.deepcopy(read(self.root / "request.json")["prompt"])
            prompt["context"]["validity_failures"] = [
                item.model_dump(mode="json") for item in failures
            ]
            prompt["acquisition_revision"] = {
                "ordinal": ordinal,
                "original_answer_hash": canonical_hash(self.original),
                "previous_answer_hash": canonical_hash(self.current),
                "previous_decision": self.current.model_dump(mode="json"),
                "acquisition": snapshot,
                "teacher_validity_evidence": intent["teacher_evidence"],
                "instructions": "Repair only unfilled template geometry/task/teacher contracts. Keep every template ID, stage and accepted-episode quota, all stage allocations, and every completed template exactly unchanged. The fixed controller and original attempt cap cannot change. Return the complete CurriculumDecision. This feedback concerns teacher validity only, not learner performance.",
            }
            request = {
                "schema_version": "astrapush-acquisition-revision-1",
                "phase": phase,
                "ordinal": ordinal,
                "original_answer_hash": canonical_hash(self.original),
                "previous_answer_hash": canonical_hash(self.current),
                "intent_sha256": file_hash(intent_path),
                "acquisition": snapshot,
                "prompt": prompt,
            }
            immutable(request_path, request)
            ready_path = directory / "READY.json"
            deadline = time.monotonic() + config()["generation_wait_seconds"]
            print(
                json.dumps(
                    {
                        "event": "awaiting_acquisition_revision",
                        "phase": phase,
                        "request": str(request_path),
                        "cutover_attempt": snapshot["cutover_attempt"],
                    }
                ),
                flush=True,
            )
            while not ready_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Bounded acquisition revision wait expired")
                time.sleep(2)
            ready = read(ready_path)
            revised, native_request, receipt = validate_exchange(
                self.root, ready, prompt=prompt, phase=phase
            )
            if not native_request.get("repair_of"):
                raise ValueError(
                    "An acquisition revision must retain its proposal repair lineage"
                )
            changed = validate_revision(
                self.current, revised, snapshot, archive=self.archive, bank=self.bank
            )
            lineage = {
                "schema_version": "astrapush-acquisition-revision-1",
                "phase": phase,
                "ordinal": ordinal,
                "original_answer_hash": canonical_hash(self.original),
                "previous_answer_hash": canonical_hash(self.current),
                "revised_answer_hash": canonical_hash(revised),
                "request_sha256": file_hash(request_path),
                "native_request_id": native_request["request_id"],
                "native_request_sha256": receipt["request_sha256"],
                "exchange_directory": ready["exchange_directory"],
                "receipt_sha256": ready["receipt_sha256"],
                "acquisition": snapshot,
                "changed_templates": changed,
            }
            immutable(committed_path, lineage)
            self.current = revised
            self.history.append(revised)
            self.applied = ordinal
        return self.current

    def archive_entries(self):
        entries = []
        for index, decision in enumerate(self.history):
            for template in decision.templates:
                value = {
                    "source_arm": decision.arm,
                    "scene_hash": canonical_hash(template.scene),
                    "structural_signature": template.scene.structural_signature(),
                    "stage": template.scene.stage,
                }
                if index == 0 or value not in entries:
                    entries.append(value)
        return entries
