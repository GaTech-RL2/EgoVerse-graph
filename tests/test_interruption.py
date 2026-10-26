"""CPU contract tests; fixtures are deliberately never runtime proof artifacts."""
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.train.joint.native_interruption import (
    fields, native_identity, classify_failure, check_steps, check_queue, read_tail,
    newest_loss, loss_snapshot, destination_probe,
)
from scripts.train.joint.checkpoint_inspection import attempt_chain, checkpoint_census, check_rng, scheduled_filename
from scripts.train.joint.production_recovery import phases, phase_config
from scripts.train.joint.native_recovery import state_digest, summarize_native_phases


def expected():
    return dict(JobId="123", JobName="test-only-native-elmo", UserId="test(1234)", Account="loaner",
                QOS="loaner-normal", Partition="rl2", ReqNodeList="gpu11", NodeList="gpu11", NumCPUs="32",
                MinMemoryNode="192G", TimeLimit="7-00:00:00", Requeue="0", Restarts="0",
                TresPerNode="gres/gpu:h100:4", Command="/synthetic-test-only/full.sbatch")


def native(state="TIMEOUT", **overrides):
    row = dict(expected(), JobState=state, NumNodes="1", NumTasks="1", ExitCode="0:15",
               EndTime="2026-10-26T08:00:00", ReqTRES="cpu=32,mem=192G,node=1,gres/gpu=4,gres/gpu:h100=4",
               AllocTRES="cpu=32,mem=192G,node=1,gres/gpu=4,gres/gpu:h100=4")
    row.update(overrides)
    row.setdefault("JOB_GRES", "gpu:h100:4" if row["TresPerNode"] else "(null)")
    row.setdefault("GRES", "gpu:h100:4(IDX:0-3)" if state == "RUNNING" and row["TresPerNode"] else "")
    return " ".join(k + "=" + v for k, v in row.items())


def test_terminal_raw_not_status_bool():
    record = native_identity(native(), expected(), "TIMEOUT")
    assert classify_failure(record, "stepd terminated job due to time limit") == "TIMEOUT"
    with pytest.raises(ValueError): fields(native() + " JobId=999")


def test_actual_lambda_cpu_record_has_no_tres_per_node_key():
    e = dict(expected(), NumCPUs="4", MinMemoryNode="24G", TimeLimit="00:20:00", TresPerNode="")
    text = native(state="COMPLETED", NumCPUs="4", MinMemoryNode="24G", TimeLimit="00:20:00",
                  TresPerNode="", ReqTRES="cpu=4,mem=24G,node=1,billing=4", AllocTRES="cpu=4,mem=24G,node=1,billing=4",
                  ExitCode="0:0")
    assert native_identity(text, e, "COMPLETED")["ExitCode"] == "0:0"


@pytest.mark.parametrize("change", [dict(JobId="999"), dict(UserId="other(1235)"), dict(Restarts="1"),
    dict(Requeue="1"), dict(TimeLimit="01:00:00"), dict(NodeList="gpu12"), dict(NumCPUs="16"),
    dict(MinMemoryNode="96G"), dict(EndTime="Unknown"), dict(ExitCode=""), dict(NumNodes="2"),
    dict(AllocTRES="cpu=32,mem=192G,node=1,gres/gpu=8,gres/gpu:h100=8"), dict(JobState="COMPLETING")])
def test_actual_native_shape_and_terminal_required(change):
    with pytest.raises(ValueError): native_identity(native(**change), expected(), "TIMEOUT")


@pytest.mark.parametrize("state", ["CANCELLED", "OUT_OF_MEMORY", "COMPLETED", "FAILED"])
def test_no_arbitrary_automatic_restart(state):
    with pytest.raises(ValueError): classify_failure(dict(JobState=state, ExitCode="1:0"), "training stopped")


def test_verified_nccl_crash_and_scientific_veto():
    row = dict(JobState="FAILED", ExitCode="0:6")
    assert classify_failure(row, "Watchdog caught collective operation timeout") == "VERIFIED_NCCL_WORKER_CRASH"
    with pytest.raises(ValueError): classify_failure(row, "Watchdog caught collective operation timeout\nValueError: nonfinite loss")
    with pytest.raises(ValueError): classify_failure(dict(JobState="TIMEOUT", ExitCode="0:15"), "OutOfMemoryError")


@pytest.mark.parametrize("tail", ["SIGABRT rank0 pid123", "exitcode : -6 rank0", "ProcessGroupNCCL terminated"])
def test_generic_abort_is_not_nccl_timeout_evidence(tail):
    with pytest.raises(ValueError): classify_failure(dict(JobState="FAILED", ExitCode="0:6"), tail)


@pytest.mark.parametrize("tail", ["train/loss=nan", "Train/Loss: +Inf", "joint_proprio_mse=-infinity",
                                  "loss tensor(nan)", "train/loss = -nan"])
def test_infrastructure_terminal_does_not_override_nonfinite_loss(tail):
    with pytest.raises(ValueError): classify_failure(dict(JobState="TIMEOUT", ExitCode="0:15"), tail)


def test_lambda_gpu_tres_omission_is_bound_by_actual_gres():
    for state in ("RUNNING", "COMPLETED"):
        raw = native(state=state, ReqTRES="cpu=32,mem=192G,node=1,billing=32",
                     AllocTRES="cpu=32,mem=192G,node=1,billing=32")
        native_identity(raw, expected(), state)
    with pytest.raises(ValueError):
        native_identity(native(state="RUNNING", GRES=""), expected(), "RUNNING")
    for gres in ("gpu:h100:4(IDX:0-2)", "gpu:h100:4(IDX:0-2,2)", "gpu:a100:4(IDX:0-3)"):
        with pytest.raises(ValueError): native_identity(native(state="RUNNING", GRES=gres), expected(), "RUNNING")
    with pytest.raises(ValueError): native_identity(native(JOB_GRES="gpu:h100:8"), expected(), "TIMEOUT")


LOSS_HEADER = "step,Train/loss,Train/joint_image_mse,Train/joint_proprio_mse,Train/joint_action_mse,Train/joint_camera_mse,Train/source_0_loss_action,Train/joint_image_mse/eva_bimanual,Train/joint_proprio_mse/eva_bimanual,Train/joint_camera_mse/eva_bimanual,Train/joint_action_mse/eva_bimanual,lr-AdamW\n"


def loss_row(step, value="1"):
    return str(step) + "," + ",".join([value] * 10) + ",\n"


@pytest.mark.parametrize("bad", ["nan", "Inf", "-inf", "infinity"])
def test_newest_loss_selected_before_finiteness_and_not_lr_row(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        newest_loss(dict(local=dict(header=LOSS_HEADER, tail=loss_row(20) + loss_row(21, bad) + "22,,,,,,,,,,,0.0001\n")))
    result = newest_loss(dict(local=dict(header=LOSS_HEADER, tail=loss_row(20, bad) + loss_row(21))))
    assert result["step"] == 21 and result["finite"] is True


def test_loss_snapshot_uses_exact_parent_namespace_and_bounded_tail(tmp_path):
    parent = dict(local_logs=str(tmp_path / "logs"), output=str(tmp_path / "output"))
    local = Path(parent["local_logs"]) / "csv/step000000000/metrics.csv"
    local.parent.mkdir(parents=True)
    local.write_text(LOSS_HEADER + "".join(loss_row(step) for step in range(100)))
    sources, snapshots = loss_snapshot(parent)
    assert len(snapshots["local"]["tail"].splitlines()) == 64
    assert newest_loss(snapshots)["step"] == 99 and sources["mirror"]["exists"] is False
    with pytest.raises(ValueError, match="absence is not zero-update"):
        newest_loss({})


def test_loss_sources_latest_conflict_and_missing_modality_fail_closed():
    with pytest.raises(ValueError, match="disagree"):
        newest_loss(dict(local=dict(header=LOSS_HEADER, tail=loss_row(1)), mirror=dict(header=LOSS_HEADER, tail=loss_row(1, "2"))))
    with pytest.raises(ValueError, match="lacks"):
        newest_loss(dict(local=dict(header=LOSS_HEADER, tail="1,1,,,,,,,,,,\n")))
    result = newest_loss(dict(local=dict(header=LOSS_HEADER, tail=loss_row(2)), mirror=dict(header=LOSS_HEADER, tail=loss_row(1))))
    assert result["selected_rows"][0]["source"] == "local"


def test_actual_native_csv_human_rows_and_stale_other_embodiment_values():
    lines = (Path(__file__).parent / "fixtures/actual-smoke-first-four-updates.csv").read_text().splitlines()
    columns = next(csv.reader([lines[0]])); index = columns.index("step")
    for step, embodiment in ((0, "human_bimanual"), (1, "human_bimanual"), (2, "eva_bimanual"), (3, "eva_bimanual")):
        rows = [line for line in lines[1:] if next(csv.reader([line]))[index] == str(step)]
        result = newest_loss(dict(actual=dict(header=lines[0], tail="\n".join(rows))))
        assert result["active_embodiment"] == embodiment
        assert result["supervised_action"] == (embodiment == "eva_bimanual")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0.5"])
def test_observed_validation_veto_is_independent_of_latest_finite_training(value):
    lines = (Path(__file__).parent / "fixtures/actual-smoke-first-four-updates.csv").read_text().splitlines()
    columns = next(csv.reader([lines[0]]))
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writerow({"step": "19999", "val/human_bimanual/forecast/proprio_normalized_mse": value})
    snapshots = dict(actual=dict(header=lines[0], tail="\n".join(lines[1:]) + "\n" + buffer.getvalue()))
    if value == "0.5":
        assert newest_loss(snapshots)["finite"]
        assert classify_failure(dict(JobState="TIMEOUT", ExitCode="0:15"), "time limit") == "TIMEOUT"
    else:
        with pytest.raises(ValueError, match="validation is nonfinite"):
            newest_loss(snapshots)


@pytest.mark.parametrize("name", ["epoch_epoch=0000-step_step=000020000.ckpt", "epoch_0000-step_000020000.ckpt"])
def test_maintained_template_metric_insertion_variants(name):
    assert scheduled_filename(name) == 20000
    with pytest.raises(ValueError): scheduled_filename(name.replace(".ckpt", "-v1.ckpt"))


def test_destination_probe_is_preserved_exclusive_and_requires_space(tmp_path):
    evidence = destination_probe(str(tmp_path / "new-run"), "123", 1)
    assert evidence["actual_write_fsync_readback"] and Path(evidence["probe_path"]).is_file()
    with pytest.raises(FileExistsError): destination_probe(str(tmp_path / "new-run"), "123", 1)
    with pytest.raises(ValueError): destination_probe(str(tmp_path / "new-run"), "124", 10**30)


def test_live_steps_and_exact_writer_queue():
    assert check_steps("", "123")
    with pytest.raises(ValueError): check_steps("123.batch|RUNNING\n", "123")
    assert check_queue("124|other-project|RUNNING\n", ["this-writer"])
    with pytest.raises(ValueError): check_queue("125|this-writer|PENDING\n", ["this-writer"])
    assert check_queue("125|this-writer|RUNNING\n", ["this-writer"], "125")
    with pytest.raises(ValueError): check_queue("125|this-writer|PENDING\n", ["this-writer"], "125")


def test_tail_bounded_and_symlinks_rejected(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text("start\n" * 100000 + "\n".join(str(i) for i in range(100)))
    assert read_tail(path).splitlines() == [str(i) for i in range(36, 100)]
    linked = tmp_path / "link"
    linked.symlink_to(path)
    with pytest.raises(ValueError): read_tail(linked)


def request(tmp_path, name="attempt1"):
    output = tmp_path / name
    (output / "checkpoints").mkdir(parents=True)
    return dict(schema="joint-training-v2", phase="full", kind="native-subtask-world-model-v1",
                run_id="synthetic-test-only", source_revision="s", source_manifest_sha256="m",
                data_manifest_sha256="d", parameters=314187064, target_steps=300000, output=str(output))


def bind(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def milestone(q, step, data=b"test-only"):
    p = Path(q["output"]) / "checkpoints" / f"epoch_epoch=0000-step_step={step:09d}.ckpt"
    p.write_bytes(data)
    return p


def test_census_preserves_every_attempt_and_selects_order_not_last_link(tmp_path):
    parent = request(tmp_path)
    old = milestone(parent, 20000)
    newer = milestone(parent, 40000)
    (old.parent / "last.ckpt").symlink_to(old.name)
    (old.parent / "incomplete.tmp").write_bytes(b"")
    child = request(tmp_path, "attempt2")
    child["interruption"] = dict(parent_request=bind(tmp_path / "parent.json", parent))
    milestone(child, 60000)
    census = checkpoint_census(child)
    assert [r["global_step"] for r in census["candidates"]] == [60000, 40000, 20000]
    assert census["preserved_noncheckpoint_entries"][0]["size_bytes"] == 0
    assert newer.exists() and old.exists()


@pytest.mark.parametrize("damage", ["missing_collection", "off_cadence", "symlink", "archive", "identity", "ambiguous_parent"])
def test_census_unknowns_fail_closed(tmp_path, damage):
    q = request(tmp_path)
    path = milestone(q, 20000)
    if damage == "missing_collection": q["output"] = str(tmp_path / "absent")
    elif damage == "off_cadence": milestone(q, 20001)
    elif damage == "symlink": (path.parent / "epoch_epoch=0000-step_step=000040000.ckpt").symlink_to(path)
    elif damage == "archive": q["archive_policy"] = dict(status="PASS")
    elif damage == "identity":
        parent = dict(q, run_id="another")
        q["interruption"] = dict(parent_request=bind(tmp_path / "parent.json", parent))
    else:
        q["interruption"] = q["continuation"] = dict(parent_request={})
    with pytest.raises((ValueError, FileNotFoundError)): checkpoint_census(q)


def rng_fixture():
    row = dict(torch_cpu=torch.get_rng_state(), torch_cuda=[torch.zeros(16, dtype=torch.uint8)],
               numpy=np.random.get_state(), python=random.getstate(), cuda_scope="current_device")
    return dict(schema="mot_rng_state.distributed.v1", world_size=4,
                ranks=[copy.deepcopy(row) for _ in range(4)])


def test_rng_cpu_structure_check_preserves_all_global_rng():
    saved = rng_fixture()
    before = state_digest(saved)
    check_rng(saved, 4)
    assert state_digest(rng_fixture()) == before


@pytest.mark.parametrize("damage", ["missing_rank", "empty_cuda", "float_cuda", "bad_python", "bad_numpy", "bad_cpu", "scope"])
def test_rng_structure_corruption_rejected(damage):
    saved = rng_fixture()
    if damage == "missing_rank": saved["ranks"].pop()
    elif damage == "empty_cuda": saved["ranks"][0]["torch_cuda"] = []
    elif damage == "float_cuda": saved["ranks"][0]["torch_cuda"][0] = torch.zeros(16)
    elif damage == "bad_python": saved["ranks"][0]["python"] = (0, (), None)
    elif damage == "bad_numpy": saved["ranks"][0]["numpy"] = ("invalid",)
    elif damage == "bad_cpu": saved["ranks"][0]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
    else: saved["ranks"][0]["cuda_scope"] = "all_devices"
    with pytest.raises((ValueError, RuntimeError, IndexError)): check_rng(saved, 4)


def original_config():
    return OmegaConf.create(dict(model=dict(loss=dict(kind="fused", cap=5, tau=1, scale=.73)),
        data=dict(sampling=dict(boundary_probability=.25)), paths=dict(output_dir="old"),
        ckpt_path=None, val_at_start=False, val_at_end=False,
        trainer=dict(max_steps=300000, val_check_interval=20000, limit_val_batches=25,
                     num_sanity_val_steps=0, check_val_every_n_epoch=None),
        callbacks=dict(conditional=dict(_target_="egomimic.pl_utils.joint_smoke.ConditionalOptimizerUsage"),
            model_checkpoint=dict(every_n_train_steps=20000, every_n_epochs=0, save_top_k=-1,
                                  save_last="link", dirpath="old", filename="old"),
            wandb_profiler=dict(_target_="do-not-invoke")),
        logger=dict(wandb=dict(id="must-not-write-history"), csv=dict(save_dir="old", version="old")),
        evaluator=dict(pixel_diagnostics=None)))


@pytest.mark.parametrize("origin", [20000, 40000, 280000])
def test_production_phases_preserve_model_data_and_disable_history(tmp_path, origin):
    original = original_config()
    before = OmegaConf.to_container(original, resolve=True)
    for start, stop in phases(origin, 300000):
        cfg = phase_config(original, output=tmp_path, checkpoint="production.ckpt", origin=origin,
                           start=start, stop=stop, target=300000, world=4, proof=tmp_path / "proof")
        assert cfg.model == original.model and cfg.data == original.data
        assert "wandb" not in cfg.logger and "wandb_profiler" not in cfg.callbacks
        assert cfg.trainer.max_steps == origin + 6 and cfg.trainer.val_check_interval == 2
        assert cfg.callbacks.resume_proof.multimodal
        assert ("allocation_boundary" in cfg.callbacks) == (stop == origin + 2)
    assert OmegaConf.to_container(original, resolve=True) == before


@pytest.mark.parametrize("origin,target", [(0,300000), (50,300000), (300000,300000), (20001,300000), (20000,300001)])
def test_smoke_zero_terminal_and_off_schedule_rejected(origin, target):
    with pytest.raises(ValueError): phases(origin, target)


def test_production_diagnostic_rejects_wrong_rank_or_counts(tmp_path):
    for world, start, stop in [(2,20000,20002), (4,20000,20004), (4,50,52)]:
        with pytest.raises(ValueError): phase_config(original_config(), output=tmp_path, checkpoint="p",
            origin=20000, start=start, stop=stop, target=300000, world=world, proof=tmp_path / "proof")


def test_import_does_not_execute_remote_or_load_checkpoint(monkeypatch, tmp_path):
    import scripts.train.joint.production_recovery as module
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    called = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: called.append("unsafe-load"))
    q = dict(schema="native-elmo-production-recovery-request-v1", producer_native=expected())
    path = tmp_path / "unallocated.json"; path.write_text(json.dumps(q))
    with pytest.raises(ValueError): module.main(path)
    assert not called
