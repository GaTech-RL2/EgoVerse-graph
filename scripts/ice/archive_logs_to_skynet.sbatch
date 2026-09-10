#!/usr/bin/env bash
# Periodic ICE CPU trigger for archive_logs_to_skynet.sh.
# The job requeues itself after each scan because user crontab is unavailable
# on ICE. It consumes one CPU only while scanning/transferring or sleeping.
#SBATCH --job-name=ice-log-archive
#SBATCH --output=/home/hice1/acheluva3/ice/EgoVerse-graph/logs/ice-log-archive-%j.out
#SBATCH --error=/home/hice1/acheluva3/ice/EgoVerse-graph/logs/ice-log-archive-%j.err
#SBATCH --partition=ice-cpu
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail

REPO_ROOT=${ICE_REPO:-/home/hice1/acheluva3/ice/EgoVerse-graph}
ARCHIVER="$REPO_ROOT/scripts/ice/archive_logs_to_skynet.sh"
INTERVAL_SECONDS=${ICE_LOG_ARCHIVE_INTERVAL_SECONDS:-900}

test -x "$ARCHIVER" || chmod +x "$ARCHIVER"
"$ARCHIVER"
sleep "$INTERVAL_SECONDS"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol requeue "$SLURM_JOB_ID"
fi
