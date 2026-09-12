#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=sync_aria_fold_fresh
#SBATCH --output=pi05_%x_%j.log

# Stage 1 of the abc-eva finetune chain: fresh verified copy of the 709 aria fold episodes into a dedicated folder
# into the shared dataset folder (skips already-present episodes).

set -euo pipefail
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-python}

"$PY" -u scripts/pi05/sync_aria_fold_fresh.py
