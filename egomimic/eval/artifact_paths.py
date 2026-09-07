"""Shared immutable validation-artifact paths across Slurm attempts."""

import os
from pathlib import Path


def artifact_execution_identity():
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return None
    if not job_id.isdigit():
        raise ValueError("SLURM_JOB_ID must be numeric for artifact provenance")
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if restart_count < 0:
        raise ValueError("SLURM_RESTART_COUNT must be nonnegative")
    return {"slurm_job_id": job_id, "slurm_restart_count": restart_count}


def artifact_destination(root, execution, *, epoch, global_step, rank, batch_idx):
    root = Path(root)
    if execution is not None:
        root = root / (
            f"job-{execution['slurm_job_id']}"
            f"-restart-{execution['slurm_restart_count']}"
        )
    return (
        root
        / f"epoch-{int(epoch)}-step-{int(global_step)}"
        / f"rank-{int(rank)}-batch-{int(batch_idx)}.pt"
    )
