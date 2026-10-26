"""Synthetic local evidence tests; never real native/GPU certification."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from scripts.train.joint.interruption_admission import (
    fingerprint, compatible, completed_producer, current_intent, quiescence,
    resume_native_contract,
)
from scripts.train.joint.production_history import retained_milestones, scheduled_validation, segments
from scripts.train.joint.production_recovery import validate_phase_proofs
from scripts.train.joint.native_recovery import state_digest
from test_interruption import bind, request, milestone, expected, native


def intent(tmp_path):
    value = dict(schema="native-elmo-current-intent-v1", authority="human", status="ACTIVE",
                 authorized_recovery_run_ids=["run"], retired_run_ids=[], hold_recovery=False,
                 source_reference="synthetic-test-fixture", updated_at="2026-10-26T00:00:00Z")
    path = tmp_path / "intent.json"; path.write_text(json.dumps(value))
    return path, value


def test_later_human_intent_precedes_finalized_admission(tmp_path):
    path, value = intent(tmp_path)
    digest = current_intent(path, "run")
    value["retired_run_ids"] = ["run"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): current_intent(path, "run", digest)
    with pytest.raises(ValueError): current_intent(path, "run")


@pytest.mark.parametrize("change", [dict(status="PAUSED"), dict(hold_recovery=True),
                                   dict(authority="agent"), dict(authorized_recovery_run_ids=[])])
def test_intent_cannot_be_inferred_or_overridden(tmp_path, change):
    path, value = intent(tmp_path); value.update(change); path.write_text(json.dumps(value))
    with pytest.raises(ValueError): current_intent(path, "run")


def pair(tmp_path):
    p = request(tmp_path)
    p.update(world_size=4, job_name="test-parent", local_logs=str(tmp_path / "logs1"),
             operational_service=dict(manifest_sha256="old"), lr_policy=dict(peak_lr=.000005,
                 warmup_optimizer_steps=3000, schedule="linear_warmup_then_constant"),
             overrides=["old-output"])
    c = copy.deepcopy(p)
    c.update(job_name="test-child", output=str(tmp_path / "new-output"), local_logs=str(tmp_path / "logs2"),
             operational_service=dict(manifest_sha256="new"), overrides=["new-output"],
             interruption=dict(parent_request=bind(tmp_path / "parent.json", p), checkpoint=dict(global_step=20000), admission=None))
    return p, c


def test_fingerprint_excludes_only_admission_and_science_remains_bound(tmp_path):
    p, c = pair(tmp_path)
    compatible(p, c)
    digest = fingerprint(c)
    c["interruption"]["admission"] = dict(real_completed_producer="future-test-only")
    assert fingerprint(c) == digest
    c["lr_policy"]["peak_lr"] = .0001
    assert fingerprint(c) != digest
    with pytest.raises(ValueError): compatible(p, c)


def test_future_production_native_contract_has_fresh_command_and_exact_resources():
    parent = expected()
    child = dict(job_name="new-scientific-attempt")
    future = {key: value for key, value in parent.items() if key != "JobId"}
    future.update(JobName=child["job_name"], Command="/synthetic-new-attempt/full.sbatch")
    resume_native_contract(future, parent, child)
    for changes in (dict(NumCPUs="16"), dict(NodeList="gpu12"), dict(UserId="other(1235)"),
                    dict(TimeLimit="00:20:00"), dict(Command=parent["Command"]), dict(JobId="123")):
        with pytest.raises(ValueError): resume_native_contract(dict(future, **changes), parent, child)


@pytest.mark.parametrize("field,value", [("parameters", 1), ("world_size", 2), ("target_steps", 600000),
    ("source_revision", "different"), ("data_manifest_sha256", "different"), ("phase", "smoke"),
    ("allocation_stop_step", 200000)])
def test_request_compatibility_rejects_changed_science_or_budget(tmp_path, field, value):
    p, c = pair(tmp_path); c[field] = value
    with pytest.raises(ValueError): compatible(p, c)


def bundle(tmp_path):
    q = dict(producer_native=expected())
    request_binding = bind(tmp_path / "producer.json", q)
    result_binding = bind(tmp_path / "result.json", dict(job_id="123", request_sha256=request_binding["sha256"],
                                                        schema="test-only", status="DONE_NOT_NATIVE_PROOF"))
    n = tmp_path / "native.txt"; n.write_text(native(state="COMPLETED", ExitCode="0:0"))
    h = tmp_path / "host.txt"; h.write_text("0\n")
    return dict(request=request_binding, result=result_binding,
                native=dict(path=str(n), sha256=hashlib.sha256(n.read_bytes()).hexdigest()),
                host_exit=dict(path=str(h), sha256=hashlib.sha256(h.read_bytes()).hexdigest()))


def test_self_authored_status_is_not_native_completion(tmp_path):
    b = bundle(tmp_path)
    completed_producer(b, "test-only", "DONE_NOT_NATIVE_PROOF")
    n = Path(b["native"]["path"]); n.write_text(native(state="FAILED", ExitCode="1:0"))
    b["native"]["sha256"] = hashlib.sha256(n.read_bytes()).hexdigest()
    with pytest.raises(ValueError): completed_producer(b, "test-only", "DONE_NOT_NATIVE_PROOF")


def test_host_exit_also_required(tmp_path):
    b = bundle(tmp_path)
    h = Path(b["host_exit"]["path"]); h.write_text("1\n")
    b["host_exit"]["sha256"] = hashlib.sha256(h.read_bytes()).hexdigest()
    with pytest.raises(ValueError): completed_producer(b, "test-only", "DONE_NOT_NATIVE_PROOF")


def test_quiescence_rechecks_unknown_claims_and_current_writer(tmp_path, monkeypatch):
    import scripts.train.joint.interruption_admission as module
    root = tmp_path / "claims"; root.mkdir()
    spec = dict(user="test", claim_roots=[dict(path=str(root), terminal_directory=str(tmp_path))],
                scientific_job_names=["test-parent"], scientific_name_prefixes=["test-elmo-scientific-"])
    monkeypatch.setattr(module, "command", lambda argv: "")
    assert quiescence(spec, "run")["claims"] == []
    claim = root / ("a" * 64); claim.mkdir()
    (claim / "PREPARED.json").write_text(json.dumps(dict(request_sha256=claim.name)))
    with pytest.raises(FileNotFoundError): quiescence(spec, "run")
    (claim / "PARENT_JOB_ID").write_text("123\n")
    with pytest.raises(ValueError): quiescence(spec, "run")
    monkeypatch.setattr(module, "command", lambda argv: "123|test-parent|RUNNING|run\n")
    assert quiescence(spec, "run", current_job="123", current_request_sha=claim.name)["claims"][0]["current"]
    with pytest.raises(ValueError): quiescence(spec, "run", current_job="123", current_request_sha="b" * 64)
    with pytest.raises(ValueError): quiescence(spec, "run")


def csv_at(q, start, rows):
    path = Path(q["local_logs"]) / "csv" / f"step{start:09d}" / "metrics.csv"
    path.parent.mkdir(parents=True)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "val/a", "val/b"])
        writer.writeheader(); writer.writerows(rows)
    return path


def history(tmp_path):
    p, c = pair(tmp_path)
    (Path(c["output"]) / "checkpoints").mkdir(parents=True)
    milestone(p, 20000); milestone(c, 40000)
    csv_at(p, 0, [dict(step=19999, **{"val/a": .1, "val/b": .2})])
    csv_at(c, 20000, [dict(step=40000, **{"val/a": .05, "val/b": .1})])
    return p, c


def test_history_retains_immutable_prefix_and_new_attempt(tmp_path):
    p, c = history(tmp_path)
    checkpoints, audit = retained_milestones(c, 40000)
    assert set(checkpoints) == {20000, 40000}
    assert checkpoints[20000].parent == Path(p["output"]) / "checkpoints"
    assert checkpoints[40000].parent == Path(c["output"]) / "checkpoints"
    values, logs = scheduled_validation(c, 40000, {"val/a", "val/b"})
    assert set(values) == {20000, 40000} and len(logs) == 2


@pytest.mark.parametrize("damage", ["repeat_boundary", "repeat_metric", "nonfinite", "partial", "off_schedule", "missing"])
def test_validation_merge_rejects_invalid_history(tmp_path, damage):
    p, c = history(tmp_path)
    path = segments(c, 40000)[-1]["csv"]
    if damage == "repeat_boundary": path.write_text("step,val/a,val/b\n20000,0.1,0.1\n40000,0.1,0.1\n")
    elif damage == "repeat_metric": path.write_text("step,val/a,val/b\n40000,0.1,0.1\n40000,0.1,\n")
    elif damage == "nonfinite": path.write_text("step,val/a,val/b\n40000,nan,0.1\n")
    elif damage == "partial": path.write_text("step,val/a,val/b\n40000,0.1,\n")
    elif damage == "off_schedule": path.write_text("step,val/a,val/b\n40001,0.1,0.1\n")
    else: path.unlink()
    with pytest.raises(ValueError): scheduled_validation(c, 40000, {"val/a", "val/b"})


def test_uncommitted_validation_nonfinite_is_not_hidden_by_checkpoint_selection(tmp_path):
    p, c = history(tmp_path)
    path = segments(c, 40000)[0]["csv"]
    with path.open("a") as stream:
        stream.write("40000,nan,0.1\n")
    with pytest.raises(ValueError, match="including uncommitted"):
        scheduled_validation(c, 40000, {"val/a", "val/b"})


def production_rows():
    from test_native_conditional_recovery import phase_rows
    rows = phase_rows()
    names = rows[0][0]["conditional_optimizer_record"]["parameter_names"]
    start = 20000
    for phase, (first, last) in zip(rows, ((start, start + 2), (start + 2, start + 6))):
        for rank, row in enumerate(phase):
            row["initial_parameter_usage"] = dict(shared=first, private=first // 2, unused=0)
            row["conditional_optimizer_record"].update(global_step=last,
                parameter_update_counts=dict(shared=last, private=last // 2, unused=0))
            row["initial_weights_sha256"] = "w" if first == start else "v"
            row["weights_sha256"] = "v" if first == start else "z"
            row["gpu_uuid"] = str(uuid.UUID(int=rank + 1))
    selected = dict(global_step=start, proof=dict(model_state_sha256="w", rank_rng_sha256=["a" * 64] * 4,
        conditional_optimizer_usage_sha256=state_digest(dict(schema="native-conditional-optimizer-usage-v1",
            global_step=start, parameter_names=names, parameter_update_counts=rows[0][0]["initial_parameter_usage"]))))
    return rows, selected


def test_production_origin_has_separate_scope_and_inspected_state_binding():
    rows, selected = production_rows()
    assert validate_phase_proofs(rows, selected)["diagnostic_scope"] == "production-checkpoint-only"
    selected["proof"]["rank_rng_sha256"][0] = "b" * 64
    with pytest.raises(ValueError): validate_phase_proofs(rows, selected)


@pytest.mark.parametrize("damage", ["counts", "weights", "duplicate_gpu", "changed_gpu", "counter_chain"])
def test_actual_diagnostic_must_bind_inspected_state_and_all_ranks(damage):
    rows, selected = production_rows()
    if damage == "counts": selected["proof"]["conditional_optimizer_usage_sha256"] = "wrong"
    elif damage == "weights": rows[0][0]["initial_weights_sha256"] = "wrong"
    elif damage == "duplicate_gpu": rows[0][1]["gpu_uuid"] = rows[0][0]["gpu_uuid"]
    elif damage == "changed_gpu": rows[1][0]["gpu_uuid"] = str(uuid.UUID(int=5))
    else: rows[1][0]["initial_parameter_usage"]["private"] -= 1
    with pytest.raises(ValueError): validate_phase_proofs(rows, selected)


def test_interruption_does_not_require_or_fake_boundary_completion(monkeypatch):
    import scripts.train.joint.interruption_admission as gate
    from scripts.train.joint.continuation import allocation_steps
    seen = []
    monkeypatch.setattr(gate, "admission", lambda request: seen.append(request) or (20000,300000))
    request = dict(target_steps=300000, interruption=dict(checkpoint=dict(global_step=20000)))
    assert allocation_steps(request) == (20000,300000) and seen == [request]
    request["continuation"] = dict(fake="boundary")
    with pytest.raises(ValueError): allocation_steps(request)


def test_configure_resume_uses_selected_production_checkpoint_and_same_history():
    from omegaconf import OmegaConf
    from scripts.train.joint.continuation import configure_allocation
    cfg = OmegaConf.create(dict(logger=dict(csv=dict(save_dir="old",version="old"),wandb=dict(resume="never")),
                               ckpt_path=None, callbacks={}))
    q = dict(local_logs="fresh", target_steps=300000, interruption=dict(checkpoint=dict(path="retained-production.ckpt")))
    configure_allocation(cfg, q, 20000, 300000)
    assert cfg.ckpt_path == "retained-production.ckpt" and cfg.logger.wandb.resume == "must"
    assert cfg.logger.csv.version == "step000020000" and not cfg.callbacks


def test_existing_dispatcher_routes_production_mode_without_legacy_repair(monkeypatch, tmp_path):
    import sys
    import scripts.train.joint.recovery_check as dispatcher
    import scripts.train.joint.production_recovery as production
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(schema="native-elmo-production-recovery-request-v1")))
    calls = []
    monkeypatch.setattr(production, "main", lambda p: calls.append(p))
    monkeypatch.setattr(sys, "argv", ["recovery_check", str(path)])
    dispatcher.main()
    assert calls == [path]
    monkeypatch.setattr(sys, "argv", ["recovery_check", str(path), "--postflight", "123"])
    with pytest.raises(ValueError): dispatcher.main()
