"""Single-writer, bounded operations with crash-visible reservations.

Reserve before external work. A pending operation cannot be silently retried;
its saved completion must be recovered or it remains an incurred, unresolved
attempt. Reopening the database never refunds attempts or rewrites outcomes.
"""

import fcntl
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from egomimic.experiments.astra_push.schemas import canonical_hash


class PendingOperation(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, manifest):
        self.owner = os.getpid()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.owner_lock = self.path.with_suffix(self.path.suffix + ".owner").open("a+")
        try:
            fcntl.flock(self.owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner_lock.close()
            raise RuntimeError("A coordinator already owns this ledger") from None
        self.connection = sqlite3.connect(self.path, timeout=0, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS phases (name TEXT PRIMARY KEY, caps TEXT NOT NULL, quotas TEXT);
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, phase TEXT NOT NULL REFERENCES phases(name),
                kind TEXT NOT NULL, template TEXT, input_hash TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT, reserved_at REAL NOT NULL,
                completed_at REAL, UNIQUE(phase,kind,id));
            CREATE TABLE IF NOT EXISTS transitions (
                phase TEXT NOT NULL REFERENCES phases(name), ordinal INTEGER NOT NULL,
                name TEXT NOT NULL, evidence_hash TEXT NOT NULL,
                PRIMARY KEY(phase,ordinal));
        """)
        digest = canonical_hash(manifest)
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='manifest_hash'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata VALUES ('manifest_hash', ?)", (digest,)
            )
        elif row["value"] != digest:
            self.close()
            raise ValueError("Resume manifest differs from the original ledger")

    def close(self):
        self.connection.close()
        self.owner_lock.close()

    def _begin(self):
        if self.owner != os.getpid():
            raise RuntimeError("Only the coordinator process owns the ledger writer")
        self.connection.execute("BEGIN IMMEDIATE")

    def add_phase(self, name, caps, *, quotas=None):
        if (
            not re.fullmatch(r"[a-zA-Z0-9_-]+", name)
            or not caps
            or any(type(v) is not int or v < 0 for v in caps.values())
        ):
            raise ValueError(
                "Phase names and resource caps must be explicit and bounded"
            )
        if quotas is not None and (
            not quotas or any(type(v) is not int or v <= 0 for v in quotas.values())
        ):
            raise ValueError("Every committed template quota must be positive")
        values = (
            json.dumps(caps, sort_keys=True),
            json.dumps(quotas, sort_keys=True) if quotas is not None else None,
        )
        self._begin()
        try:
            row = self.connection.execute(
                "SELECT caps,quotas FROM phases WHERE name=?", (name,)
            ).fetchone()
            if row and tuple(row) != values:
                raise ValueError(
                    "A committed phase cannot change caps or steal template quota"
                )
            if not row:
                self.connection.execute(
                    "INSERT INTO phases VALUES (?,?,?)", (name, *values)
                )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def reserve(self, operation_id, phase, kind, inputs, *, template=None):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", operation_id):
            raise ValueError("Operation IDs must be explicit safe identifiers")
        digest = canonical_hash(inputs)
        self._begin()
        try:
            previous = self.connection.execute(
                "SELECT * FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
            if previous:
                if (
                    previous["phase"],
                    previous["kind"],
                    previous["template"],
                    previous["input_hash"],
                ) != (phase, kind, template, digest):
                    raise ValueError("Operation ID was reused with different inputs")
                if previous["status"] == "reserved":
                    raise PendingOperation(
                        "Recover the original completion; do not repeat external work"
                    )
                result = json.loads(previous["result"])
                self.connection.commit()
                return {
                    "execute": False,
                    "status": previous["status"],
                    "result": result,
                }
            row = self.connection.execute(
                "SELECT * FROM phases WHERE name=?", (phase,)
            ).fetchone()
            if row is None:
                raise ValueError("Phase is not declared")
            caps = json.loads(row["caps"])
            used = self.connection.execute(
                "SELECT COUNT(*) FROM operations WHERE phase=? AND kind=?",
                (phase, kind),
            ).fetchone()[0]
            if kind not in caps or used >= caps[kind]:
                raise BudgetExhausted(f"{phase}/{kind}: {used} reserved operations")
            quotas = json.loads(row["quotas"]) if row["quotas"] else None
            if kind == "attempt" and quotas:
                if template not in quotas:
                    raise ValueError(
                        "Attempt must retain its committed template allocation"
                    )
                accepted = self._accepted(phase, template)
                pending = self.connection.execute(
                    "SELECT COUNT(*) FROM operations WHERE phase=? AND template=? AND kind='attempt' AND status='reserved'",
                    (phase, template),
                ).fetchone()[0]
                if accepted + pending >= quotas[template]:
                    raise BudgetExhausted("Template quota is full or reserved")
            self.connection.execute(
                "INSERT INTO operations VALUES (?,?,?,?,?,'reserved',NULL,?,NULL)",
                (operation_id, phase, kind, template, digest, time.time()),
            )
            self.connection.commit()
            return {"execute": True, "input_hash": digest}
        except BaseException:
            self.connection.rollback()
            raise

    def complete(self, operation_id, *, status, result):
        if (
            status not in {"succeeded", "failed", "abandoned"}
            or type(result) is not dict
        ):
            raise ValueError("An operation needs a concrete completion record")
        encoded = json.dumps(result, sort_keys=True, allow_nan=False)
        self._begin()
        try:
            row = self.connection.execute(
                "SELECT * FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Reserve incurred work before execution")
            if row["status"] != "reserved":
                if row["status"] != status or row["result"] != encoded:
                    raise ValueError("Published operation outcome is immutable")
                self.connection.commit()
                return
            if row["kind"] == "attempt":
                if type(result.get("accepted")) is not bool:
                    raise ValueError(
                        "Attempt completion must explicitly record acceptance"
                    )
                if result["accepted"] and (
                    status != "succeeded"
                    or not re.fullmatch(r"[0-9a-f]{64}", result.get("episode_hash", ""))
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", result.get("semantic_receipt_hash", "")
                    )
                ):
                    raise ValueError(
                        "Accepted attempts need immutable episode and independent semantic receipts"
                    )
            self.connection.execute(
                "UPDATE operations SET status=?,result=?,completed_at=? WHERE id=?",
                (status, encoded, time.time(), operation_id),
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def _accepted(self, phase, template):
        rows = self.connection.execute(
            "SELECT result FROM operations WHERE phase=? AND template=? AND kind='attempt' AND status='succeeded'",
            (phase, template),
        ).fetchall()
        return sum(json.loads(r["result"]).get("accepted") is True for r in rows)

    def phase_complete(self, phase):
        row = self.connection.execute(
            "SELECT quotas FROM phases WHERE name=?", (phase,)
        ).fetchone()
        if row is None or not row["quotas"]:
            raise ValueError("No acquisition quotas were committed")
        quotas = json.loads(row["quotas"])
        return all(self._accepted(phase, key) == quota for key, quota in quotas.items())

    def transition(self, phase, sequence, name, evidence_hash):
        if (
            len(sequence) != len(set(sequence))
            or name not in sequence
            or not re.fullmatch(r"[0-9a-f]{64}", evidence_hash)
        ):
            raise ValueError(
                "Transitions require a unique declared sequence and evidence hash"
            )
        ordinal = sequence.index(name)
        self._begin()
        try:
            rows = self.connection.execute(
                "SELECT * FROM transitions WHERE phase=? ORDER BY ordinal", (phase,)
            ).fetchall()
            if any(r["name"] != sequence[r["ordinal"]] for r in rows):
                raise ValueError("Round state-machine sequence changed")
            if ordinal < len(rows):
                if rows[ordinal]["evidence_hash"] != evidence_hash:
                    raise ValueError("Published transition evidence changed")
            elif ordinal == len(rows):
                self.connection.execute(
                    "INSERT INTO transitions VALUES (?,?,?,?)",
                    (phase, ordinal, name, evidence_hash),
                )
            else:
                raise ValueError("Cannot skip a required round transition")
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def summary(self):
        rows = self.connection.execute(
            "SELECT phase,kind,status,COUNT(*) AS count FROM operations GROUP BY phase,kind,status ORDER BY phase,kind,status"
        ).fetchall()
        return [dict(row) for row in rows]
