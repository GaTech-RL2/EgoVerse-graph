#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=smoke_ft_init
#SBATCH --output=ft_%x_%j.log

# CPU-only preflight: build both fine-tune models exactly as trainHydra does and
# run the weights-only init against the checkpoint. No GPU, no data. Run here
# rather than on a login node -- the 2.2 GB checkpoint plus two 180M models gets
# the process killed there (it died silently mid-arcdur on 2026-09-17).

set -euo pipefail
WT=${WT:-$(git rev-parse --show-toplevel)}
cd "$WT"
export PATH="$WT/.venv/bin:$PATH" PYTHONPATH="$WT" PYTHONWARNINGS=ignore HYDRA_FULL_ERROR=1
"$WT/.venv/bin/python" -u /storage/home/hcoda1/5/agao81/smoke_ft_init.py
