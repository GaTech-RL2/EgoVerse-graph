#!/bin/bash
# Stationery tempo-bucket copy of the towels394 session's pack_train.sbatch (2545aac6): reuses that session's image + venv READ-ONLY
# (VENV), runs code from a git-archive snapshot (SRC, which has no .git: the commit is read from SRC/COMMIT), writes only under RUN_ROOT.
# Submit with --qos=loaner-low so the towels runs keep priority (they preempt + requeue us).
# Pack several small-DDP training runs onto ONE exclusive Lambda loaner node (8 x H100).
# Loaner nodes are exclusive, so one run per job would idle most GPUs. Each run is its own srun step of
# GPUS_PER_RUN tasks (default 2, Lightning DDP via the SLURM environment) inside the pytorch container, with its
# own MASTER_PORT (runs share the node and job id), logging to $RUN_ROOT/<name>/.
# Requeue-safe: a run whose checkpoints/last.ckpt exists resumes from it (the experiments keep a rolling
# last.ckpt every 5k steps and a stable W&B id with resume: allow).
#
#   RUNS="exp/a exp/b ..." SRC=... RUN_ROOT=... sbatch --qos=loaner-low scripts/e1/stattempo_lambda_pack_train.sbatch
#   EXTRA="trainer.max_steps=300 ..."   appended to every run (smoke tests)
#   RUN_ROOT=...                          required, e.g. /workspace/users/$USER/stattempo/runs
#   GPUS_PER_RUN=1                        must match the experiments' trainer.devices
#
#SBATCH --job-name=stattempo
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=224
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/workspace/users/%u/stattempo/logs/%x-%j.out
set -euo pipefail

SRC=${SRC:?set SRC to the snapshot dir}
VENV=${VENV:-/workspace/users/$USER/EgoVerse-graph/.venv}
IMAGE=${IMAGE:-/workspace/users/$USER/images/nvidia-pytorch-25.06-py3.sqsh}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
G=${GPUS_PER_RUN:-1}
read -r -a runs <<< "${RUNS:?set RUNS to space-separated +experiment names}"
(( ${#runs[@]} * G <= 8 )) || { echo "${#runs[@]} runs x $G GPUs do not fit one node" >&2; exit 2; }
CPUS=$(( 192 / (${#runs[@]} * G) ))
test -s "$IMAGE" && test -e "$VENV/bin/python" && test -s "$SRC/COMMIT"
echo "=== pack_train job=$SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0} commit=$(cat "$SRC/COMMIT") runs=${runs[*]}"

i=0
for exp in "${runs[@]}"; do
  name=${exp##*/}
  port=$(( 29500 + 10 * i )); i=$(( i + 1 ))
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
      echo rank=\${SLURM_PROCID:-0} local=\${SLURM_LOCALID:-0} nofile=\$(ulimit -n) commit=\$(cat $SRC/COMMIT) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}
      exec $VENV/bin/python egomimic/trainHydra.py --config-name=train_zarr_cartesian \
        +experiment=$exp paths.output_dir=$run norm_stats.save_cache_dir=$run hydra.run.dir=$run \
        $resume ${EXTRA:-}" &
done
wait
