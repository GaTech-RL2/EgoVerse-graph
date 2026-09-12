#!/bin/bash
#SBATCH --job-name=ct3-monitor
#SBATCH --partition=ice-cpu
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/home/hice1/agao81/scratch/logs/ct3-monitor-%j.log
# Protocol (rev-3) rollout monitor; self-chaining. Stop: scancel -n ct3-monitor; touch ~/scratch/rollouts/CT2-rev3/STOP
D=/home/hice1/agao81/scratch/autoresearch/orchestrator-260909-1520
test -e /home/hice1/agao81/scratch/rollouts/CT2-rev3/STOP && exit 0
(sleep $((17*3600+1800)); test -e /home/hice1/agao81/scratch/rollouts/CT2-rev3/STOP || sbatch "$D/monitor_ct3.sbatch") &
exec python3 "$D/monitor_ct3.py" --loop
