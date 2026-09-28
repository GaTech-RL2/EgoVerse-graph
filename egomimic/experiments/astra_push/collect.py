"""Bounded generated-scene commissioning with immutable attempt accounting."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from egomimic.experiments.astra_push.artifacts import (
    file_hash,
    named_seed,
    publish_json,
)
from egomimic.experiments.astra_push.commissioning import CommissioningProposal
from egomimic.experiments.astra_push.ledger import Ledger
from egomimic.experiments.astra_push.provider import strict_json
from egomimic.experiments.astra_push.schemas import canonical_hash
from egomimic.experiments.astra_push.stock_inventory import stock_inventory
from egomimic.experiments.astra_push.subagent_exchange import MANIFEST


def load_generated(directory):
    directory = Path(directory)
    request = strict_json((directory / "request.json").read_text())
    receipt = strict_json((directory / "receipt.json").read_text())
    raw = directory / "raw-response.json"
    if (
        request["phase"] != "commissioning"
        or request["model"] != "gpt-6-astra"
        or request["fork_turns"] != "none"
        or receipt["transport"] != "codex_subagent"
        or not receipt["valid"]
        or receipt["request_sha256"] != file_hash(directory / "request.json")
        or receipt["response_sha256"] != file_hash(raw)
    ):
        raise ValueError("Generated input archive is inconsistent")
    proposal = CommissioningProposal.model_validate(strict_json(raw.read_text()))
    if receipt["answer_hash"] != canonical_hash(proposal):
        raise ValueError("Validated generator proposal changed")
    return proposal, receipt


def collect_attempts(output, proposal, *, executor, attempt_cap=120):
    """Single coordinator; each crash reservation remains an incurred attempt."""
    output = Path(output)
    if not 1 <= attempt_cap <= 120:
        raise ValueError("Commissioning cannot exceed 120 attempts")
    output.mkdir(parents=True, exist_ok=True)
    templates = {t.template_id: t for t in proposal.templates}
    ledger = Ledger(
        output / "acquisition-ledger.sqlite",
        {**MANIFEST, "proposal_hash": canonical_hash(proposal)},
    )
    ledger.add_phase(
        "commissioning",
        {"attempt": attempt_cap},
        quotas={key: 5 for key in templates},
    )
    try:
        # Recover only already-written completions. Missing completions after
        # a crash consume the original slot without rerunning external work.
        previous = ledger.connection.execute(
            "SELECT * FROM operations ORDER BY reserved_at"
        ).fetchall()
        for operation in previous:
            if operation["status"] != "reserved":
                continue
            receipt_path = output / "attempts" / operation["id"] / "receipt.json"
            if receipt_path.exists():
                receipt = strict_json(receipt_path.read_text())
                result = completion(receipt, receipt_path)
                status = "succeeded" if receipt["accepted"] else "failed"
            else:
                result = {"accepted": False, "reason": "interrupted_without_completion"}
                status = "abandoned"
            ledger.complete(operation["id"], status=status, result=result)
        used = len(previous)
        # Stable round-robin ordering prevents early templates consuming the
        # whole cap. Accepted quotas always remain five per scene.
        while used < attempt_cap and not ledger.phase_complete("commissioning"):
            for key, template in templates.items():
                if used >= attempt_cap:
                    break
                if ledger._accepted("commissioning", key) == 5:
                    continue
                attempt_id = f"commissioning-{used + 1:04d}"
                seed = named_seed(f"commissioning:{key}:{used + 1}")
                inputs = {"template_hash": canonical_hash(template), "seed": seed}
                ledger.reserve(
                    attempt_id, "commissioning", "attempt", inputs, template=key
                )
                used += 1
                path = output / "attempts" / attempt_id
                receipt = executor(path, template, seed, attempt_id)
                receipt_path = path / "receipt.json"
                result = completion(receipt, receipt_path)
                ledger.complete(
                    attempt_id,
                    status="succeeded" if receipt["accepted"] else "failed",
                    result=result,
                )
                print(
                    json.dumps(
                        {
                            "attempt": used,
                            "template": key,
                            "stage": template.scene.stage,
                            "accepted": receipt["accepted"],
                            "failure": receipt["failure"],
                            "steps": receipt["executed_steps"],
                            "template_accepted": ledger._accepted("commissioning", key),
                        }
                    ),
                    flush=True,
                )
        records = []
        for operation in ledger.connection.execute(
            "SELECT * FROM operations WHERE status='succeeded' ORDER BY reserved_at"
        ):
            result = json.loads(operation["result"])
            if result.get("accepted"):
                receipt_path = output / "attempts" / operation["id"] / "receipt.json"
                if file_hash(receipt_path) != result["semantic_receipt_hash"]:
                    raise ValueError("Completed semantic receipt changed")
                receipt = strict_json(receipt_path.read_text())
                records.append(
                    {
                        "template_id": operation["template"],
                        "attempt_id": operation["id"],
                        "episode": receipt["episode"],
                    }
                )
        return {
            "complete": ledger.phase_complete("commissioning"),
            "attempts": used,
            "accepted_by_template": {
                key: ledger._accepted("commissioning", key) for key in templates
            },
            "accepted": records,
            "ledger": ledger.summary(),
        }
    finally:
        ledger.close()


def completion(receipt, receipt_path):
    result = {
        "accepted": receipt["accepted"],
        "semantic_receipt_hash": file_hash(receipt_path),
    }
    if receipt["accepted"]:
        record = receipt["episode"]
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Accepted episode changed before ledger commit")
        result["episode_hash"] = record["sha256"]
    return result


def publish_or_verify(path, record):
    path = Path(path)
    if path.exists():
        if strict_json(path.read_text()) != record:
            raise ValueError(f"Immutable commissioning artifact changed: {path.name}")
        return file_hash(path)
    return publish_json(path, record)


def run(output, generation, stock_root, prior_receipt=None):
    from egomimic.experiments.astra_push.teacher_probe import run_teacher_attempt

    output = Path(output)
    started = time.monotonic()
    prior = strict_json(Path(prior_receipt).read_text()) if prior_receipt else None
    prior_attempts = prior["attempts"] if prior else 0
    if type(prior_attempts) is not int or not 0 <= prior_attempts < 120:
        raise ValueError(
            "Prior attempts must retain room inside the shared 120-attempt cap"
        )
    proposal, generator = load_generated(generation)
    inventory = stock_inventory(stock_root, [t.scene for t in proposal.templates])
    publish_or_verify(output / "stock-inventory.json", inventory)
    publish_or_verify(
        output / "generated-proposal.json", proposal.model_dump(mode="json")
    )
    publish_or_verify(output / "generator-receipt.json", generator)
    if prior:
        publish_or_verify(output / "prior-version-receipt.json", prior)
    if not inventory["passed"]:
        raise ValueError("Stock-scene novelty comparison failed; do not collect")

    def execute(path, template, seed, attempt_id):
        return run_teacher_attempt(
            path,
            scene=template.scene,
            task=template.task.bind(template.scene),
            program=template.teacher,
            seed=seed,
            episode_id=attempt_id,
            phase="commissioning",
            teacher_authorship="codex_subagent:gpt-6-astra",
        )

    result = collect_attempts(
        output, proposal, executor=execute, attempt_cap=120 - prior_attempts
    )
    result.update(
        schema_version="astrapush-commissioning-result-1",
        proposal_hash=canonical_hash(proposal),
        generator_receipt_hash=file_hash(output / "generator-receipt.json"),
        inventory_hash=file_hash(output / "stock-inventory.json"),
        production_optimizer_updates=0,
        visual_catalog_version=2,
        prior_attempts=prior_attempts,
        cumulative_attempts=prior_attempts + result["attempts"],
        prior_receipt_sha256=file_hash(prior_receipt) if prior_receipt else None,
    )
    publish_or_verify(output / "acquisition.json", result)
    final_path = output / "receipt.json"
    if final_path.exists():
        previous = strict_json(final_path.read_text())
        if previous["proposal_hash"] != canonical_hash(proposal):
            raise ValueError(
                "Final commissioning result belongs to a different proposal"
            )
        if not previous["complete"]:
            raise RuntimeError(
                "This commissioning attempt is already recorded incomplete"
            )
        return previous
    result["acquisition_complete"] = result["complete"]
    if result["complete"]:
        reloads = []
        for template in proposal.templates:
            accepted = next(
                r
                for r in result["accepted"]
                if r["template_id"] == template.template_id
            )
            attempt = output / "attempts" / accepted["attempt_id"]
            reload_path = attempt / "fresh-process-reload.json"
            if not reload_path.exists():
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "egomimic.experiments.astra_push.calibration",
                            "--restore-bundle",
                            str(attempt / "bundle"),
                            "--state",
                            str(attempt / "initial-full-state"),
                            "--expected",
                            str(attempt / "expected-replay.json"),
                            "--output",
                            str(reload_path),
                        ],
                        check=True,
                        timeout=90,
                    )
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    publish_json(
                        reload_path,
                        {"fresh_process_restore_passed": False, "error": str(exc)},
                    )
            record = strict_json(reload_path.read_text())
            reloads.append(
                {
                    "template": template.template_id,
                    "receipt_hash": file_hash(reload_path),
                    "passed": record["fresh_process_restore_passed"],
                }
            )
        result["reload_records"] = reloads
        result["fresh_process_reloads"] = sum(r["passed"] for r in reloads)
        result["engineering_reload_resets"] = len(reloads)
        result["complete"] = all(r["passed"] for r in reloads)
    result["seconds"] = time.monotonic() - started
    publish_json(output / "receipt.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "accepted"}), flush=True)
    if not result["complete"]:
        raise RuntimeError("Commissioning incomplete within the fixed attempt cap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--stock-root", required=True)
    parser.add_argument("--prior-receipt")
    args = parser.parse_args()
    run(args.output, args.generation, args.stock_root, args.prior_receipt)
