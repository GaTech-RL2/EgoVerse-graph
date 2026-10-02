#!/bin/bash
#SBATCH -A gts-dxu345-rl2
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=normstats_arcpre
#SBATCH --output=ft_%x_%j.log
# Recompute arcpre_abc's norm stats exactly (pinned 2089-episode pool, seeded sampler). CPU only:
# norm_stats_only returns before any model/trainer is built.
set -euo pipefail
WT=/storage/project/r-dxu345-0/agao81/EgoVerse-graph; cd $WT
OUT=/storage/project/r-dxu345-0/agao81/grid_finals/arcpre_abc_j13308880; mkdir -p $OUT
export PATH="$WT/.venv/bin:$PATH" PYTHONPATH=$WT HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$WT/.venv/bin/python egomimic/trainHydra.py --config-name=train_zarr_cartesian \
  +experiment=yam_arc_grid/arcpre_abc_arcdur data=abc_arc/arcpre_abc_data_pinned2089 \
  norm_stats_only=true norm_stats.save_cache_dir=$OUT norm_stats.num_workers=14 \
  paths.output_dir=$OUT hydra.run.dir=$OUT/hydra
ls -la $OUT
