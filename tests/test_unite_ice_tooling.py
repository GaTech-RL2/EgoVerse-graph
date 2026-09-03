from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ice.unite import stage_unite_candidates as stager
from scripts.ice.unite import unite_cutover_record_writer as writer
from scripts.ice.unite.requeue_integration_contract import (
    normalize_restart_count,
    terminal_checkpoint_step,
    validate_trace,
)
from scripts.ice.unite.unite_source_terminal_evidence import remote_executable


def _access_selection():
    probe = {
        "remote": "robot@primary.example.edu",
        "attempted_at_utc": "2026-09-03T00:00:00Z",
        "attempt_count": 1,
        "returncode": 0,
        "hostname": "primary.example.edu",
        "username": "robot",
        "stderr_sha256": "a" * 64,
    }
    return {
        "schema_version": 1,
        "status": "READY",
        "probe_context": {
            "cluster": "PACE ICE",
            "slurm_job_id": "123",
            "slurm_node": "node-1",
            "account": "research-account",
            "partition": "ice-cpu",
            "qos": "research-qos",
        },
        "canonical_remote": "robot@primary.example.edu",
        "fallback_remote": "robot@fallback.example.edu",
        "canonical_probe": probe,
        "fallback_probe": None,
        "selected_remote": "robot@primary.example.edu",
        "selection_reason": "canonical_probe_succeeded",
        "completed_at_utc": "2026-09-03T00:00:01Z",
    }


def test_access_selection_has_no_fixed_user_or_account():
    selection = _access_selection()
    assert stager.exact_access_selection(selection) == selection
    assert writer.validate_access_selection(selection) == selection


def test_candidate_row_binds_paths_through_manifest():
    root = "/remote/experiments/arbitrary-run"
    row = {
        "task_id": "shared-n4",
        "global_step": 80000,
        "source_run_root": root,
        "source_checkpoint": f"{root}/checkpoints/epoch-9-step-80000.ckpt",
        "checkpoint_size_bytes": 10,
        "checkpoint_sha256": "a" * 64,
        "source_config": f"{root}/.hydra/config.yaml",
        "config_size_bytes": 20,
        "config_sha256": "b" * 64,
        "semantic_config_sha256": "c" * 64,
        "source_validation_result_sha256": "d" * 64,
    }
    assert stager.exact_row(row) == row
    escaped = dict(row, source_checkpoint="/remote/else/checkpoints/epoch-9-step-80000.ckpt")
    with pytest.raises(ValueError, match="outside the authorized source run"):
        stager.exact_row(escaped)


@pytest.mark.parametrize("value", [None, "0", "1", "002"])
def test_restart_count_accepts_only_absent_or_ascii_digits(value):
    assert normalize_restart_count(value) == int(value or 0)


@pytest.mark.parametrize("value", ["", "+1", "-1", "1.0", "one", "١"])
def test_restart_count_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="ASCII"):
        normalize_restart_count(value)


@pytest.mark.parametrize(
    "name",
    [
        "integration-terminal-epoch=1-step=11.ckpt",
        "integration-terminal-epoch-1-step-11.ckpt",
    ],
)
def test_terminal_parser_accepts_both_lightning_filename_families(name):
    assert terminal_checkpoint_step(name) == 11


def _passing_trace(world_size=2):
    return {
        "schema_version": 1,
        "seed_step": 3,
        "first_attempt_optimizer_steps": [4, 5, 6, 7, 8],
        "signal_checkpoint_step": 8,
        "step_after_signal_executed": False,
        "save_acknowledged_ranks": list(range(world_size)),
        "world_size": world_size,
        "slurm_restarts": 1,
        "resume_checkpoint_step": 8,
        "resumed_optimizer_steps": [9, 10, 11],
        "validation_logged_step": 10,
        "terminal_checkpoint": "integration-terminal-epoch-1-step-11.ckpt",
        "child_complete": True,
        "complete_sentinel": True,
    }


def test_requeue_trace_requires_all_rank_ack_and_absolute_validation_schedule():
    assert validate_trace(_passing_trace())["status"] == "PASS"
    early = _passing_trace()
    early["validation_logged_step"] = 8
    with pytest.raises(ValueError, match="before all three resumed steps"):
        validate_trace(early)
    missing_rank = _passing_trace()
    missing_rank["save_acknowledged_ranks"] = [0]
    with pytest.raises(ValueError, match="every rank"):
        validate_trace(missing_rank)


def test_remote_slurm_binary_is_manifest_supplied():
    assert remote_executable("/cluster/slurm/bin/sacct", "sacct") == (
        "/cluster/slurm/bin/sacct"
    )
    with pytest.raises(Exception):
        remote_executable("sacct", "sacct")


def test_repository_tools_do_not_embed_old_deployment_identity():
    root = Path("scripts/ice/unite")
    forbidden = (
        "/storage/ice1/5/4/",
        "/coc/flash7/",
        "paphiwetsa3",
        "rl2-group",
        "pushshapes-flow-transfer",
        "5629288",
    )
    for path in root.iterdir():
        if path.suffix not in {".py", ".sh", ".sbatch"}:
            continue
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in text, f"{value!r} leaked into {path}"
