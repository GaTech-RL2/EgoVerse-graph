import importlib.util
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).parents[1] / "scripts/train/verify_planar_training_smoke.py"
_SPEC = importlib.util.spec_from_file_location("verify_planar_training_smoke", _PATH)
MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MODULE)


def test_smoke_artifact_selection_uses_latest_numeric_attempt_and_current_seed_bank(tmp_path):
    artifact_root = tmp_path / "validation_predictions/energy_score"
    for restart in (9, 10):
        path = artifact_root / f"job-123-restart-{restart}/epoch-0-step-2/rank-0-batch-0.pt"
        path.parent.mkdir(parents=True)
        torch.save(
            {
                "schema_version": 1, "metric": "EnergyScore@32",
                "execution": {"slurm_job_id": "123", "slurm_restart_count": restart},
                "epoch": 0, "global_step": 2, "rank": 0, "batch_idx": 0,
                "seed_bank": list(range(32)), "seed_bank_sha256": "pinned-seeds",
                "domains": {"usocket": {}},
            },
            path,
        )
    selected, count = MODULE._verified_energy_artifact(
        tmp_path, expected_step=2, seed_bank_sha256="pinned-seeds", domains=("usocket",),
    )
    assert "job-123-restart-10" in selected.parts and count == 2
    with pytest.raises(ValueError, match="seed-bank identity"):
        MODULE._verified_energy_artifact(
            tmp_path, expected_step=2, seed_bank_sha256="different-seeds", domains=("usocket",),
        )
    with pytest.raises(ValueError, match="checkpoint"):
        MODULE._verified_energy_artifact(
            tmp_path, expected_step=3, seed_bank_sha256="pinned-seeds", domains=("usocket",),
        )

    # A later attempt that has not reached the final step must not borrow a
    # completed validation artifact from an earlier execution.
    incomplete = artifact_root / "job-123-restart-11/epoch-0-step-1/rank-0-batch-0.pt"
    incomplete.parent.mkdir(parents=True)
    incomplete.touch()
    with pytest.raises(ValueError, match="latest execution artifact step 1"):
        MODULE._verified_energy_artifact(
            tmp_path, expected_step=2, seed_bank_sha256="pinned-seeds", domains=("usocket",),
        )
