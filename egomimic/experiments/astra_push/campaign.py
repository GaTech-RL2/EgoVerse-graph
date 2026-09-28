"""Single-writer coordinator for the prospectively frozen matched experiment."""

import argparse
import json
import os
import selectors
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_map

from egomimic.experiments.astra_push.artifacts import (
    file_hash,
    named_seed,
    publish_json,
)
from egomimic.experiments.astra_push.collect import completion, load_generated
from egomimic.experiments.astra_push.data import collate_windows
from egomimic.experiments.astra_push.decisions import (
    CurriculumDecision,
    LearnerReport,
    generation_context,
)
from egomimic.experiments.astra_push.learner import Learner, restore_rng, rng_state
from egomimic.experiments.astra_push.ledger import Ledger
from egomimic.experiments.astra_push.partitions import (
    check_training_scene,
    config,
    definitions,
)
from egomimic.experiments.astra_push.provider import strict_json
from egomimic.experiments.astra_push.schemas import canonical_hash
from egomimic.experiments.astra_push.transport import PolicyServer

SEQUENCE = ["decision", "data", "checkpoint", "probes", "committed"]


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Published evidence changed: {path}")
        return file_hash(path)
    return publish_json(path, value)


class Simulator:
    def __init__(self, root):
        self.root = Path(root)
        self.log = (self.root / f"simulator-session-{time.time_ns()}.log").open("x")
        self.process = subprocess.Popen(
            [
                config()["simulator_python"],
                "-m",
                "egomimic.experiments.astra_push.simulation",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            bufsize=0,
        )
        self.counter = 0
        self.buffer = b""

    def call(self, operation, **arguments):
        self.counter += 1
        request_id = f"worker-{self.counter}"
        self.process.stdin.write(
            (
                json.dumps(
                    {"id": request_id, "operation": operation, "arguments": arguments},
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        )
        self.process.stdin.flush()
        # Each worker operation has a finite deadline; all CPU/GPU work belongs
        # to the allocated job, and simulator stdout retains its progress.
        deadline = time.monotonic() + config()["worker_timeout_seconds"]
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            while time.monotonic() < deadline:
                if b"\n" not in self.buffer:
                    if not selector.select(timeout=1):
                        if self.process.poll() is not None:
                            raise RuntimeError("Simulator process exited")
                        continue
                    block = os.read(self.process.stdout.fileno(), 65536)
                    if not block:
                        raise RuntimeError("Simulator pipe closed")
                    self.buffer += block
                    continue
                raw, self.buffer = self.buffer.split(b"\n", 1)
                line = raw.decode() + "\n"
                self.log.write(line)
                self.log.flush()
                if line.startswith("ASTRA_WORKER "):
                    response = json.loads(line.removeprefix("ASTRA_WORKER "))
                    if response["id"] != request_id:
                        raise ValueError("Simulator response ID changed")
                    if not response["ok"]:
                        raise RuntimeError(response["error"])
                    return response["value"]
                if line.startswith('{"event":'):
                    print(line.rstrip(), flush=True)
            raise TimeoutError("Simulator operation exceeded its declared deadline")
        finally:
            selector.close()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=15)
        self.log.close()


def acquire(
    root,
    *,
    phase,
    templates,
    quotas,
    cap,
    ledger,
    simulator,
    arm="common",
    round_index=0,
    revisions=None,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Recover completed attempts, never execute an existing reservation again.
    for operation in ledger.connection.execute(
        "SELECT * FROM operations WHERE phase=? AND kind='attempt' AND status='reserved' ORDER BY reserved_at",
        (phase,),
    ).fetchall():
        path = root / operation["id"] / "receipt.json"
        if path.exists():
            receipt = read(path)
            ledger.complete(
                operation["id"],
                status="succeeded" if receipt["accepted"] else "failed",
                result=completion(receipt, path),
            )
        else:
            ledger.complete(
                operation["id"],
                status="abandoned",
                result={"accepted": False, "reason": "interrupted_without_completion"},
            )
    used = ledger.connection.execute(
        "SELECT COUNT(*) FROM operations WHERE phase=? AND kind='attempt'", (phase,)
    ).fetchone()[0]
    while used < cap and not ledger.phase_complete(phase):
        if revisions is not None:
            templates = revisions.apply(ledger).templates
        for template in templates:
            # A newly requested validity repair must take effect before another
            # reservation, including in the middle of a five-template sweep.
            if revisions is not None:
                current = revisions.apply(ledger)
                template = next(
                    item
                    for item in current.templates
                    if item.template_id == template.template_id
                )
            name = template.template_id
            if ledger._accepted(phase, name) >= quotas[name] or used >= cap:
                continue
            used += 1
            op = f"{phase}-attempt-{used:04d}"
            task = template.task.bind(template.scene)
            args = {
                "output": str(root / op),
                "scene": template.scene.model_dump(mode="json"),
                "task": task.model_dump(mode="json"),
                "program": template.teacher.model_dump(mode="json"),
                "seed": named_seed(op),
                "episode_id": op,
                "phase": "seed" if phase == "seed" else "round",
                "teacher_authorship": "codex_gpt_6_astra_typed_program",
                "arm": arm,
                "round_index": round_index,
            }
            ledger.reserve(op, phase, "attempt", args, template=name)
            receipt = simulator.call("teacher", **args)
            ledger.complete(
                op,
                status="succeeded" if receipt["accepted"] else "failed",
                result=completion(receipt, root / op / "receipt.json"),
            )
            print(
                json.dumps(
                    {
                        "event": "acquisition",
                        "phase": phase,
                        "attempt": used,
                        "template": name,
                        "accepted": receipt["accepted"],
                        "template_accepted": ledger._accepted(phase, name),
                        "quota": quotas[name],
                        "failure": receipt["failure"],
                    }
                ),
                flush=True,
            )
    if not ledger.phase_complete(phase):
        raise RuntimeError(
            f"{phase} acquisition incomplete at {used}/{cap}; no quota transfer or automatic curriculum substitution"
        )
    accepted, yield_summary = [], {}
    for operation in ledger.connection.execute(
        "SELECT * FROM operations WHERE phase=? AND kind='attempt' ORDER BY reserved_at",
        (phase,),
    ).fetchall():
        summary = yield_summary.setdefault(
            operation["template"], {"attempts": 0, "accepted": 0}
        )
        summary["attempts"] += 1
        result = json.loads(operation["result"])
        if result["accepted"]:
            summary["accepted"] += 1
            receipt_path = root / operation["id"] / "receipt.json"
            if file_hash(receipt_path) != result["semantic_receipt_hash"]:
                raise ValueError("Accepted semantic receipt changed")
            receipt = read(receipt_path)
            if file_hash(receipt["episode"]["path"]) != result["episode_hash"]:
                raise ValueError("Acquired episode changed after ledger commitment")
            accepted.append(receipt["episode"])
    result = {
        "phase": phase,
        "accepted": accepted,
        "attempts": used,
        "quotas": quotas,
        "teacher_yield": yield_summary,
    }
    immutable(root / "manifest.json", result)
    return result


def predict(learner, metadata, arrays):
    row = {
        **arrays,
        "proprioception": learner.stats.normalize(arrays["proprioception"]),
        "instruction": metadata["instruction"],
        "actions": np.zeros((10, 7), np.float32),
        "action_valid": np.ones(10, bool),
    }
    batch = collate_windows([row])["libero_push"]
    batch = {k: v for k, v in batch.items() if k not in {"actions", "action_valid"}}
    batch = tree_map(
        lambda x: x.to("cuda") if isinstance(x, torch.Tensor) else x, batch
    )
    with (
        torch.inference_mode(),
        torch.random.fork_rng(devices=[torch.cuda.current_device()]),
    ):
        torch.manual_seed(metadata["policy_seed"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = learner.graph.forward_eval({"libero_push": batch})["libero_push"][
                "pred_action"
            ]
        return result[0].float().cpu().numpy()


def evaluate(root, *, learner, simulator, bank_path, checkpoint, partition):
    root = Path(root)
    result_path = root / "results.json"
    if result_path.exists():
        result = read(result_path)
        if result["checkpoint_hash"] != checkpoint["sha256"] or result[
            "bank_hash"
        ] != file_hash(bank_path):
            raise ValueError("Evaluation checkpoint or frozen bank changed")
        return result
    root.mkdir(parents=True, exist_ok=True)
    state, training = rng_state(), learner.graph.nets.training
    learner.graph.nets.eval()
    try:
        socket_path = f"/tmp/astra-policy-{os.getpid()}.sock"
        with PolicyServer(
            socket_path,
            lambda metadata, arrays: predict(learner, metadata, arrays),
            timeout=config()["policy_timeout_seconds"],
        ):
            result = simulator.call(
                "evaluate",
                bank_path=str(bank_path),
                output=str(root),
                partition=partition,
                checkpoint_hash=checkpoint["sha256"],
                policy_socket=socket_path,
            )
    finally:
        restore_rng(state)
        learner.graph.nets.train(training)
    publish_json(result_path, result)
    return result


def control_report(
    result, *, arm, round_index, previous=None, allocation=None, teacher_yield=None
):
    stages, frames = {}, []
    for stage in ("S1", "S2", "S3"):
        episodes = sorted(
            [e for e in result["episodes"] if e["stage"] == stage],
            key=lambda e: (e["metrics"]["success"], e["case_id"]),
        )
        if len(episodes) != 12:
            raise ValueError(
                "Control report requires exactly twelve episodes per stage"
            )
        counts = {
            key: sum(bool(e["metrics"][key]) for e in episodes)
            for key in (
                "wrong_object",
                "wrong_target",
                "lift_violation",
                "preservation_violation",
            )
        }
        counts.update(
            trials=12,
            successes=sum(e["metrics"]["success"] for e in episodes),
            displacement_failure=sum(e["displacement_failure"] for e in episodes),
            overshoot_failure=sum(e["overshoot_failure"] for e in episodes),
        )
        stages[stage] = counts
        frames.extend(episodes[:2])
    report = LearnerReport(
        schema_version="astrapush-control-report-1",
        partition="training-control",
        arm=arm,
        checkpoint_hash=result["checkpoint_hash"],
        probe_manifest_hash=result["bank_hash"],
        completed_round=round_index,
        stages=stages,
        previous_report_hash=canonical_hash(previous) if previous else None,
        previous_allocation=allocation,
        frame_sequence_hashes=[e["frames_sha256"] for e in frames],
        inferred_error_categories=[
            "wrong_object inferred from preservation displacement",
            "displacement_failure: requested block never approached within target half-width",
            "overshoot_failure: transient containment without sustained success",
        ],
        stage_success_deltas={
            s: stages[s]["successes"] - previous["stages"][s]["successes"]
            for s in stages
        }
        if previous
        else {},
        teacher_yield=teacher_yield or {},
        remaining_rounds=config()["rounds"] - round_index,
        case_successes={
            e["case_id"]: bool(e["metrics"]["success"]) for e in result["episodes"]
        },
        stage_forgetting_counts={
            s: sum(
                previous.get("case_successes", {}).get(e["case_id"], False)
                and not e["metrics"]["success"]
                for e in result["episodes"]
                if e["stage"] == s
            )
            for s in stages
        }
        if previous
        else {},
        remaining_accepted_episodes=(config()["rounds"] - round_index)
        * config()["round_episodes"],
        remaining_optimizer_updates=(config()["rounds"] - round_index)
        * config()["round_updates"],
    )
    return report.model_dump(mode="json"), [
        {
            "path": e["frames_path"],
            "sha256": e["frames_sha256"],
            "case_id": e["case_id"],
            "stage": e["stage"],
        }
        for e in frames
    ]


def await_decision(
    root, *, phase, arm, round_index, report, frames, archive, bank, common_templates
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "request.json"
    context = generation_context(
        arm=arm,
        round_index=round_index,
        generation_archive=archive,
        validity_failures=[],
        report=report if arm == "A" else None,
    )
    prompt = {
        "context": context,
        "approved_common_examples": [
            t.model_dump(mode="json") for t in common_templates
        ],
        "instructions": "Return only a CurriculumDecision JSON. Author exactly five positive-allocation templates totaling 75 accepted episodes. At least two scene structural signatures must be new relative to the supplied arm-local archive. Use canonical instruction grammar, all geometric constraints, the proven five-skill control API and the same fixed controller. You may reuse valid common geometry for some templates. Never change quotas to fit partial acquisition. Adaptive choices must cite the supplied measured training-control report; no deterministic stage schedule is imposed. Uniform gets exactly 25 per stage and no student feedback. Remain concise under 8192 output tokens and 32768 response bytes.",
        "frame_sequences": frames if arm == "A" else [],
    }
    immutable(
        request_path,
        {
            "phase": phase,
            "prompt": prompt,
            "generator_transport": "codex_subagent",
            "requested_model": "gpt-6-astra",
            "fork_turns": "none",
            "reasoning_effort": None,
            "reasoning_effort_policy": "inherit_session",
        },
    )
    deadline = time.monotonic() + config()["generation_wait_seconds"]
    print(
        json.dumps(
            {
                "event": "awaiting_astra",
                "phase": phase,
                "request": str(request_path),
                "request_hash": file_hash(request_path),
            }
        ),
        flush=True,
    )
    ready_path = root / "READY.json"
    while not ready_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "No authorized Astra decision arrived within the bounded wait"
            )
        time.sleep(2)
    ready = read(ready_path)
    phase_calls = len(list(root.glob("*/request.json")))
    total_calls = config()["prior_generation_calls"] + len(
        list(root.parent.glob("*/*/request.json"))
    )
    if not 1 <= phase_calls <= 8 or total_calls > 80:
        raise ValueError(
            "Archived generation calls exceed the declared phase/global caps"
        )
    directory = root / ready["exchange_directory"]
    if root.resolve() not in directory.resolve().parents:
        raise ValueError("Exchange path escapes its phase")
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
        raise ValueError(
            "Astra request/response archive changed or uses another transport"
        )
    decision = CurriculumDecision.model_validate(
        strict_json(raw.read_text())
    ).validate_context(
        arm=arm,
        round_index=round_index,
        report_hash=canonical_hash(report) if arm == "A" else None,
    )
    if canonical_hash(decision) != receipt["answer_hash"]:
        raise ValueError("Validated Astra answer changed")
    signatures = {t.scene.structural_signature() for t in decision.templates}
    if len(signatures - {a["structural_signature"] for a in archive}) < 2:
        raise ValueError("Round lacks two structurally new templates")
    for template in decision.templates:
        check_training_scene(template.scene, bank)
    return decision


def train_phase(
    root,
    *,
    learner,
    records,
    phase,
    updates,
    parent,
    ledger,
    arm=None,
    round_index=None,
):
    root = Path(root)
    paths = sorted(root.glob("step-*.json")) if root.exists() else []
    receipt = read(paths[-1]) if paths else parent
    if receipt:
        learner.restore(receipt)
    learner.configure_data(records, phase, arm=arm, round_index=round_index)
    final = learner.train(root, updates=updates, ledger=ledger)
    immutable(root / "final.json", final)
    return final


def run(output, *, source_commit, stop_after=None, continuation=None):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = config()
    manifest = {"config": cfg, "source_commit": source_commit}
    immutable(root / "run-manifest.json", manifest)
    execution_source_commit = source_commit
    ledger_path = root / "ledger.sqlite"
    if continuation:
        from egomimic.experiments.astra_push.continuation import open_continuation

        ledger_path, execution_source_commit = open_continuation(
            root, continuation, source_commit=source_commit
        )
    ledger = Ledger(ledger_path, manifest)
    simulator = Simulator(root)
    try:
        proposal, _ = load_generated(cfg["generation_archive"])
        bank = definitions(
            root / "partition-definitions.json",
            forbidden_scenes=[t.scene for t in proposal.templates],
        )
        simulator.call(
            "freeze",
            definitions_path=str(root / "partition-definitions.json"),
            output=str(root / "frozen-bank"),
        )
        bank_path = root / "frozen-bank/manifest.json"
        audit = read(Path(cfg["commissioning_evidence"]) / "learner-summary.json")
        normalization_path = (
            Path(cfg["commissioning_evidence"])
            / "commissioning-learner-audit/frozen-proprioception.json"
        )
        if file_hash(normalization_path) != audit["normalization_sha256"]:
            raise ValueError("Corrected commissioning normalizer changed")
        learner = Learner(read(normalization_path), source_commit=source_commit)
        if learner.initial_hash != audit["initial_state_sha256"]:
            raise ValueError(
                "Production random initialization differs from commissioning audit"
            )
        seeds = [t for t in proposal.templates if t.scene.stage == "S1"]
        quotas = {t.template_id: cfg["seed_episodes"] // len(seeds) for t in seeds}
        ledger.add_phase(
            "seed",
            {"attempt": cfg["seed_attempt_cap"], "training_block": 30},
            quotas=quotas,
        )
        ledger.transition(
            "seed",
            SEQUENCE,
            "decision",
            canonical_hash([t.model_dump(mode="json") for t in seeds]),
        )
        initial_path = root / "checkpoints/initial.json"
        if initial_path.exists():
            initial = read(initial_path)
        else:
            initial = learner.snapshot(
                root / "checkpoints" / f"initial-{time.time_ns()}.pt", ledger=ledger
            )
            immutable(initial_path, initial)
        seed_data = acquire(
            root / "data/seed",
            phase="seed",
            templates=seeds,
            quotas=quotas,
            cap=cfg["seed_attempt_cap"],
            ledger=ledger,
            simulator=simulator,
        )
        ledger.transition(
            "seed", SEQUENCE, "data", file_hash(root / "data/seed/manifest.json")
        )
        validation_path = root / "source-validation.json"
        if not validation_path.exists():
            print(
                json.dumps(
                    {
                        "event": "awaiting_source_validation",
                        "source_commit": source_commit,
                        "path": str(validation_path),
                    }
                ),
                flush=True,
            )
        deadline = time.monotonic() + cfg["generation_wait_seconds"]
        while not validation_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Required source validation receipt did not arrive")
            time.sleep(2)
        validation = read(validation_path)
        if (
            validation.get("source_commit") != source_commit
            or validation.get("conclusion") != "success"
        ):
            raise ValueError(
                "Production updates require successful validation of this source"
            )
        warm = train_phase(
            root / "checkpoints/seed",
            learner=learner,
            records=seed_data["accepted"],
            phase="seed",
            updates=cfg["common_updates"],
            parent=initial,
            ledger=ledger,
        )
        ledger.transition("seed", SEQUENCE, "checkpoint", warm["sha256"])
        control = evaluate(
            root / "evaluations/common/training-control",
            learner=learner,
            simulator=simulator,
            bank_path=bank_path,
            checkpoint=warm,
            partition="training-control",
        )
        common_report, common_frames = control_report(
            control,
            arm="common",
            round_index=0,
            teacher_yield=seed_data["teacher_yield"],
        )
        report_hash = immutable(root / "reports/common.json", common_report)
        ledger.transition("seed", SEQUENCE, "probes", report_hash)
        ledger.transition("seed", SEQUENCE, "committed", warm["sha256"])
        if stop_after == "warm-start":
            immutable(
                root / "warm-start-complete.json",
                {"checkpoint": warm, "report": common_report},
            )
            return
        records = {a: list(seed_data["accepted"]) for a in ("U", "A")}
        checkpoints = {a: warm for a in ("U", "A")}
        reports = {a: common_report for a in ("U", "A")}
        frame_banks = {a: common_frames for a in ("U", "A")}
        common_archive = [
            {
                "source_arm": "common",
                "scene_hash": canonical_hash(t.scene),
                "structural_signature": t.scene.structural_signature(),
                "stage": t.scene.stage,
            }
            for t in proposal.templates
        ]
        archives = {a: list(common_archive) for a in ("U", "A")}
        for round_index in range(1, cfg["rounds"] + 1):
            for arm in ("U", "A"):
                phase = f"{arm}{round_index}"
                decision = await_decision(
                    root / "generation" / phase,
                    phase=phase,
                    arm=arm,
                    round_index=round_index,
                    report=reports[arm],
                    frames=frame_banks[arm],
                    archive=archives[arm],
                    bank=bank,
                    common_templates=proposal.templates,
                )
                quotas = {
                    t.template_id: t.accepted_episodes for t in decision.templates
                }
                ledger.add_phase(
                    phase,
                    {"attempt": cfg["round_attempt_cap"], "training_block": 20},
                    quotas=quotas,
                )
                ledger.transition(phase, SEQUENCE, "decision", canonical_hash(decision))
                from egomimic.experiments.astra_push.revisions import RevisionManager

                revisions = RevisionManager(
                    root / "generation" / phase,
                    original=decision,
                    archive=archives[arm],
                    bank=bank,
                )
                data = acquire(
                    root / "data" / phase,
                    phase=phase,
                    templates=decision.templates,
                    quotas=quotas,
                    cap=cfg["round_attempt_cap"],
                    ledger=ledger,
                    simulator=simulator,
                    arm=arm,
                    round_index=round_index,
                    revisions=revisions,
                )
                # Completed phases can be revisited on continuation without
                # entering acquire's attempt loop. Recover their full lineage.
                revisions.apply(ledger)
                records[arm].extend(data["accepted"])
                ledger.transition(
                    phase,
                    SEQUENCE,
                    "data",
                    file_hash(root / "data" / phase / "manifest.json"),
                )
                checkpoints[arm] = train_phase(
                    root / "checkpoints" / phase,
                    learner=learner,
                    records=records[arm],
                    phase=phase,
                    updates=cfg["round_updates"],
                    parent=checkpoints[arm],
                    ledger=ledger,
                    arm=arm,
                    round_index=round_index,
                )
                ledger.transition(
                    phase, SEQUENCE, "checkpoint", checkpoints[arm]["sha256"]
                )
                control = evaluate(
                    root / "evaluations" / phase / "training-control",
                    learner=learner,
                    simulator=simulator,
                    bank_path=bank_path,
                    checkpoint=checkpoints[arm],
                    partition="training-control",
                )
                report, frames = control_report(
                    control,
                    arm=arm,
                    round_index=round_index,
                    previous=reports[arm],
                    allocation=decision.allocations,
                    teacher_yield=data["teacher_yield"],
                )
                report_hash = immutable(root / "reports" / f"{phase}.json", report)
                ledger.transition(phase, SEQUENCE, "probes", report_hash)
                ledger.transition(
                    phase, SEQUENCE, "committed", checkpoints[arm]["sha256"]
                )
                reports[arm], frame_banks[arm] = report, frames
                archives[arm].extend(revisions.archive_entries())
        finals = {
            "warm_start": warm,
            "U_final": checkpoints["U"],
            "A_final": checkpoints["A"],
        }
        immutable(root / "final-checkpoints.json", finals)
        for label, checkpoint in finals.items():
            learner.restore(checkpoint)
            for partition in ("development", "sealed"):
                evaluate(
                    root / "evaluations" / label / partition,
                    learner=learner,
                    simulator=simulator,
                    bank_path=bank_path,
                    checkpoint=checkpoint,
                    partition=partition,
                )
        from egomimic.experiments.astra_push.reporting import report_campaign

        report_campaign(root)
        from egomimic.experiments.astra_push.diagnostic import run_diagnostic

        run_diagnostic(root, learner=learner, final_checkpoint=checkpoints["A"])
        immutable(
            root / "complete.json",
            {
                "status": "complete",
                "main_updates": 5000,
                "diagnostic_updates": 240,
                "ledger": ledger.summary(),
                "ledger_path": str(ledger_path),
                "execution_source_commit": execution_source_commit,
                "checkpoints": finals,
            },
        )
    finally:
        simulator.close()
        ledger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--stop-after", choices=["warm-start"])
    parser.add_argument(
        "--continuation", help="Immutable versioned continuation receipt"
    )
    run(**vars(parser.parse_args()))
