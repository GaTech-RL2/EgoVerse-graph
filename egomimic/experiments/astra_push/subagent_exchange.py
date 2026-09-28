"""File-backed exchange for the user-authorized Codex Astra pilot.

The current Codex session performs the actual spawn. This module neither
pretends to be an API client nor invents provider usage/identity receipts.
"""

import re
from pathlib import Path

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.ledger import Ledger
from egomimic.experiments.astra_push.provider import strict_json
from egomimic.experiments.astra_push.schemas import canonical_hash

MODEL = "gpt-6-astra"
MAX_RESPONSE_BYTES = 32 * 1024
MANIFEST = {
    "version": "astra-hpt-subagent-pilot-20260928-v1",
    "transport": "codex_subagent",
    "model": MODEL,
    "logical_requests_per_phase": 8,
    "repairs_per_proposal": 2,
    "requested_output_tokens": 8192,
    "output_token_cap_enforced": False,
    "provider_billing_available": False,
}


class SubagentExchange:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.root / "generation-ledger.sqlite", MANIFEST)

    def close(self):
        self.ledger.close()

    def prepare(self, request_id, *, phase, prompt, repair_of=None):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", request_id):
            raise ValueError("Request ID must be a safe path component")
        if phase not in {"commissioning", "seed", "fixture"} | {
            f"{arm}{i}" for arm in ("A", "U") for i in range(1, 5)
        }:
            raise ValueError("Unknown experiment phase")
        self.ledger.add_phase(phase, {"generation": 8})
        if repair_of:
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", repair_of):
                raise ValueError("Repair request ID must be a safe path component")
            previous = strict_json((self.root / repair_of / "request.json").read_text())
            if previous["phase"] != phase or previous["repair_of"] is not None:
                raise ValueError("Repair must reference its original phase proposal")
            repairs = [
                strict_json(p.read_text()) for p in self.root.glob("*/request.json")
            ]
            if sum(r["repair_of"] == repair_of for r in repairs) >= 2:
                raise ValueError("At most two repair responses per proposal")
        request = {
            **MANIFEST,
            "request_id": request_id,
            "phase": phase,
            "repair_of": repair_of,
            "fork_turns": "none",
            "reasoning_effort": None,
            "reasoning_effort_policy": "inherit_session",
            "prompt": prompt,
            "usage": None,
            "resolved_provider_snapshot": None,
        }
        total = self.ledger.connection.execute(
            "SELECT COUNT(*) FROM operations WHERE kind='generation'"
        ).fetchone()[0]
        existing = self.ledger.connection.execute(
            "SELECT id FROM operations WHERE id=?", (request_id,)
        ).fetchone()
        if total >= 80 and existing is None:
            raise ValueError(
                "The experiment cannot exceed eighty logical generation calls"
            )
        reserved = self.ledger.reserve(request_id, phase, "generation", request)
        if not reserved["execute"]:
            return self.root / request_id / "request.json"
        directory = self.root / request_id
        directory.mkdir(exist_ok=False)
        publish_json(directory / "request.json", request)
        return directory / "request.json"

    def accept(self, request_id, response_path, *, agent_id, validator):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", request_id):
            raise ValueError("Request ID must be a safe path component")
        directory = self.root / request_id
        request = strict_json((directory / "request.json").read_text())
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("Record the actual Codex spawn receipt's agent ID")
        raw = Path(response_path).read_bytes()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeds the bounded subagent pilot envelope")
        archived = directory / "raw-response.json"
        with archived.open("xb") as stream:
            stream.write(raw)
        receipt = {
            "transport": "codex_subagent",
            "requested_model": request["model"],
            "agent_id": agent_id,
            "request_sha256": file_hash(directory / "request.json"),
            "response_sha256": file_hash(archived),
            "response_bytes": len(raw),
            "usage": None,
            "transport_attempts": None,
            "resolved_provider_snapshot": None,
            "output_token_cap_enforced": False,
        }
        try:
            value = validator(strict_json(raw.decode("utf-8")))
            answer = value.model_dump(mode="json")
            publish_json(directory / "answer.json", answer)
            receipt.update(valid=True, answer_hash=canonical_hash(answer))
            status = "succeeded"
        except (ValueError, TypeError, KeyError) as exc:
            receipt.update(valid=False, validation_error=str(exc))
            status = "failed"
        publish_json(directory / "receipt.json", receipt)
        self.ledger.complete(request_id, status=status, result=receipt)
        return receipt
