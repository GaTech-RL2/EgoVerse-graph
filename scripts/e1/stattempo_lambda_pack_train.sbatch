#!/bin/bash
# Stationery tempo-bucket copy of the towels394 session's pack_train.sbatch (2545aac6): reuses that session's image + venv READ-ONLY
# (VENV), runs code from a git-archive snapshot (SRC, which has no .git: the commit is read from SRC/COMMIT), writes only under RUN_ROOT.
# QOS: --qos=loaner-low kept the 9-25 towels runs ahead of us; loaner-high (slowpace366, 2026-10-05) preempts lower-QOS jobs.
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
#   Fewer than 8 GPUs (2026-10-06): pipe the script without its --exclusive/--gres/--cpus-per-task lines and pass e.g.
#   --gres=gpu:h100:4 --cpus-per-task=112 on the sbatch line; CPUs per run follow the allocation.
#   RUNS="exp/a exp/b+exp/c"             a '+' group shares ONE GPU step (GPUS_PER_RUN=1 only): both runs train on the same
#                                         GPU, each with its own port and log; CPUs (6/7 of the allocation / runs) and memory (180G / run)
#                                         are split per run, so 10 runs fit 8 GPUs (2026-10-05).
#
#SBATCH --job-name=stattempo
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --mem-per-gpu=225G  # the partition default DefMemPerGPU=256000 MB x 8 exceeds a node: without it every submit fails "Requested node configuration is not available" (2026-10-05)
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
read -r -a steps <<< "${RUNS:?set RUNS to space-separated +experiment names}"
(( ${#steps[@]} * G <= 8 )) || { echo "${#steps[@]} GPU steps x $G GPUs do not fit one node" >&2; exit 2; }
nruns=$(tr ' +' '\n\n' <<< "$RUNS" | grep -c .)
(( nruns == ${#steps[@]} || G == 1 )) || { echo "'+' GPU sharing needs GPUS_PER_RUN=1" >&2; exit 2; }
CPR=$(( ${SLURM_CPUS_ON_NODE:-224} * 6 / 7 / (nruns * G) ))   # CPUs per run (per task): 6/7 of the allocation (192 of a full node)
test -s "$IMAGE" && test -e "$VENV/bin/python" && test -s "$SRC/COMMIT"
echo "=== pack_train job=$SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0} commit=$(cat "$SRC/COMMIT") steps=${steps[*]}"

i=0
for step in "${steps[@]}"; do
  IFS=+ read -r -a exps <<< "$step"
  k=${#exps[@]}
  body=""
  for exp in "${exps[@]}"; do
    name=${exp##*/}
    port=$(( 29500 + 10 * i )); i=$(( i + 1 ))
    run="$RUN_ROOT/$name"
    mkdir -p "$run"
    resume=""
    if [ -s "$run/checkpoints/last.ckpt" ]; then
      resume="ckpt_path=$run/checkpoints/last.ckpt"
    fi
    echo "--- $name -> $run ${resume:+(resuming)} (GPU step: ${exps[*]##*/})"
    body+="
      ( export MASTER_PORT=$port TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${SLURM_JOB_ID}_$name TRITON_CACHE_DIR=/tmp/tr_${SLURM_JOB_ID}_$name
        echo rank=\${SLURM_PROCID:-0} local=\${SLURM_LOCALID:-0} nofile=\$(ulimit -n) commit=\$(cat $SRC/COMMIT) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} gpu_shared_by=$k
        [ \"\${SLURM_PROCID:-0}\" = 0 ] && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
        exec $VENV/bin/python egomimic/trainHydra.py --config-name=train_zarr_cartesian \
          +experiment=$exp paths.output_dir=$run norm_stats.save_cache_dir=$run hydra.run.dir=$run \
          $resume ${EXTRA:-} ) >> $run/train-$SLURM_JOB_ID.log 2>&1 &"
  done
  srun --exact --ntasks=$G --ntasks-per-node=$G --gres=gpu:h100:$G --gpu-bind=none --cpus-per-task=$(( CPR * k )) --mem=$(( 180 * k ))G \
    --container-image="$IMAGE" --container-mounts=/workspace:/workspace:rw --container-workdir="$SRC" \
    --output="$RUN_ROOT/step-%j-$i.log" \
    bash -c "set -euo pipefail
      ulimit -n \$(ulimit -Hn)  # zarr-heavy loaders exhaust the default 1024 fds (norm pass, job 405906)
      export PYTHONPATH=$SRC HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=\${WANDB_MODE:-online}
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      $body
      wait" &
done
wait
