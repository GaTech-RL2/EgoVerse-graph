#!/bin/bash
#SBATCH --job-name=bench-unite
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/home/hice1/agao81/scratch/logs/bench-unite-%j.log
#SBATCH --exclude=atl1-1-03-010-10-0,atl1-1-03-010-30-0,atl1-1-03-011-3-0,atl1-1-03-011-13-0,atl1-1-03-011-18-0
source /home/hice1/agao81/scratch/EgoVerse-graph-unite/.venv/bin/activate
cd /home/hice1/agao81/scratch/EgoVerse-graph-ct
export PYTHONPATH=$PWD HYDRA_FULL_ERROR=1
python /home/hice1/agao81/scratch/autoresearch/orchestrator-260909-1520/bench_unite.py
