#!/bin/bash
# Pack several small-DDP training runs onto ONE exclusive Lambda loaner node (8 x H100).
# Loaner nodes are exclusive, so one run per job would idle most GPUs. Each run is its own srun step of
# GPUS_PER_RUN tasks (default 2, Lightning DDP via the SLURM environment) inside the pytorch container, with its
# own MASTER_PORT (runs share the node and job id), logging to $RUN_ROOT/<name>/.
# Requeue-safe: a run whose checkpoints/last.ckpt exists resumes from it (the experiments keep a rolling
# last.ckpt every 5k steps and a stable W&B id with resume: allow).
#
#   RUNS="exp/a exp/b ..." sbatch --qos=loaner-normal scripts/clusters/lambda/pack_train.sbatch
#   EXTRA="trainer.max_steps=300 ..."   appended to every run (smoke tests)
#   RUN_ROOT=...                          default /workspace/users/$USER/runs
#   GPUS_PER_RUN=2                        must match the experiments' trainer.devices
#   CPUS_PER_TASK=, PORT_BASE=29500       override the CPU share / DDP port range
# To add runs to a RUNNING pack job's spare GPUs, run this file with bash (not sbatch) and SLURM_JOB_ID=<job>
# exported, plus CPUS_PER_TASK within the job's free CPUs and a PORT_BASE unused on that node. Such steps die
# with that job, so queue a follow-up (sbatch --dependency=afterany:<job>) to resume them from last.ckpt.
#
#SBATCH --job-name=pack_train
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=224
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/workspace/users/%u/logs/%x-%j.out
set -euo pipefail

SRC=${SRC:-/workspace/users/$USER/EgoVerse-graph}
IMAGE=${IMAGE:-/workspace/users/$USER/images/nvidia-pytorch-25.06-py3.sqsh}
RUN_ROOT=${RUN_ROOT:-/workspace/users/$USER/runs}
G=${GPUS_PER_RUN:-2}
read -r -a runs <<< "${RUNS:?set RUNS to space-separated +experiment names}"
(( ${#runs[@]} * G <= 8 )) || { echo "${#runs[@]} runs x $G GPUs do not fit one node" >&2; exit 2; }
CPUS=${CPUS_PER_TASK:-$(( 192 / (${#runs[@]} * G) ))}
test -s "$IMAGE" && test -x "$SRC/.venv/bin/python"
echo "=== pack_train job=$SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0} commit=$(git -C "$SRC" rev-parse --short HEAD) runs=${runs[*]}"

i=0
for exp in "${runs[@]}"; do
  name=${exp##*/}
  port=$(( ${PORT_BASE:-29500} + 10 * i )); i=$(( i + 1 ))
  run="$RUN_ROOT/$name"
  mkdir -p "$run"
  resume=""
  if [ -s "$run/checkpoints/last.ckpt" ]; then
    resume="ckpt_path=$run/checkpoints/last.ckpt"
  fi
  echo "--- $name -> $run ${resume:+(resuming)}"
  srun --exact --ntasks=$G --ntasks-per-node=$G --gres=gpu:h100:$G --gpu-bind=none --cpus-per-task=$CPUS --mem=200G \
    --container-image="$IMAGE" --container-mounts=/workspace:/workspace:rw --container-workdir="$SRC" \
    --output="$run/train-%j.log" \
    bash -c "set -euo pipefail
      ulimit -n \$(ulimit -Hn)  # zarr-heavy loaders exhaust the default 1024 fds (norm pass, job 405906)
      export MASTER_PORT=$port PYTHONPATH=$SRC HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=\${WANDB_MODE:-online}
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID}_$name TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}_$name
      [ \"\${SLURM_PROCID:-0}\" = 0 ] && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
      echo rank=\${SLURM_PROCID:-0} local=\${SLURM_LOCALID:-0} nofile=\$(ulimit -n) commit=\$(git -C $SRC rev-parse --short HEAD) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}
      exec $SRC/.venv/bin/python egomimic/trainHydra.py --config-name=train_zarr_cartesian \
        +experiment=$exp paths.output_dir=$run norm_stats.save_cache_dir=$run hydra.run.dir=$run \
        $resume ${EXTRA:-}" &
done
wait
