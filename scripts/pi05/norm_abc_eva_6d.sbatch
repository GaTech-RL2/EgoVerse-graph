#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=norm_abc_eva_6d
#SBATCH --output=pi05_%x_%j.log

# Stage 2 of the abc-eva finetune chain: per-(step,dim) quantile norm stats
# over the wristframe-6D abc eva data (20D native with grippers — no proprio
# padding patch needed, unlike mecka). Output consumed by
# train_abc_eva_ft.sbatch.

set -euo pipefail
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
PY=${PY:-python}

OUT=/storage/project/r-dxu345-0/agao81/norm_stats/abc_eva_6d
mkdir -p "$OUT"

"$PY" -u scripts/pi05/precompute_norm_stats.py \
  --data pi05/abc_eva_pi_6d \
  --model pi05/pi0.5_ft_abc_eva_6d \
  --sample-frac 0.1 \
  --max-samples 2000000 \
  --num-workers 22 \
  --out "$OUT"
