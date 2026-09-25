#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --job-name=sync_rl2_stationery
#SBATCH --output=pi05_%x_%j.log

# Mirror the 2026-09-17 rl2 YAM stationery re-upload (task
# 'organize_stationary_updated', 198 processed episodes) into the shared
# dataset folder. None were present on the mirror, so every data config
# resolving against it saw ABC episodes only. Skips already-present episodes.

set -euo pipefail
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-$WT/.venv/bin/python}

# _sync_s3_to_local() shells out to a bare "s5cmd"; it ships in the venv but
# the venv is not activated in a batch shell, so put it on PATH explicitly.
export PATH="$WT/.venv/bin:$PATH"
command -v s5cmd >/dev/null || { echo "s5cmd not on PATH" >&2; exit 1; }

"$PY" -u scripts/pi05/sync_rl2_stationery.py
