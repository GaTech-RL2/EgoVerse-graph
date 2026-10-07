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
#   RUNTIME=enroot (2026-10-06)       run each step's commands through `enroot start` on a container this job creates once
#                                         in node-local /tmp, instead of pyxis. Added when a storage migration removed the
#                                         /ephemeral/loaner-enroot paths the cluster's enroot.conf points at (pyxis then fails
#                                         every container with "mkdir: cannot create directory '/ephemeral/loaner-enroot'").
#   STAGE_TAR=<dataset .tar> DATA_DIR=<path the configs read>   (2026-10-07, Aidan's Lambda rule, vault PACE.md)
#                                         extract the dataset tar to node-local /tmp in ONE tar operation before training,
#                                         and mount that copy over DATA_DIR in the container, so training never reads
#                                         thousands of small files off /workspace NFS (which took nodes down 10-06).
#   Fewer than 8 GPUs (2026-10-06): pipe the script without its --exclusive/--gres/--cpus-per-task lines and pass e.g.
#   --gres=gpu:h100:4 --cpus-per-task=112 on the sbatch line; CPUs per run follow the allocation.
#   RUNS="exp/a exp/b+exp/c"             a '+' group shares ONE GPU step (GPUS_PER_RUN=1 only): both runs train on the same
#                                         GPU, each with its own port and log; CPUs (6/7 of the allocation / runs) and memory (180G / run)
#                                         are split per run, so 10 runs fit 8 GPUs (2026-10-05).
#
#SBATCH --job-name=stattempo
#SBATCH --account=loaner
#SBATCH --partition=rl2  # 2026-10-07: the loaner partition is gone; rl2 is the one that allows account loaner
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
echo "=== pack_train job=$SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0} commit=$(cat "$SRC/COMMIT") runtime=${RUNTIME:-pyxis} steps=${steps[*]}"
cleanup() { [ -n "${CNAME:-}" ] && enroot remove -f "$CNAME" >/dev/null 2>&1; rm -rf "${E:-/nonexistent}" "${L:-/nonexistent}"; }
trap cleanup EXIT
STAGED=""
if [ -n "${STAGE_TAR:-}" ]; then
  DATA_DIR=${DATA_DIR:?set DATA_DIR to the dataset path the configs read}
  L=/tmp/$USER-data-$SLURM_JOB_ID
  need_kb=$(( $(stat -c %s "$STAGE_TAR") / 1024 + 10 * 1024 * 1024 ))  # the tar's size + 10 GB headroom
  avail_kb=$(df -Pk /tmp | awk 'NR == 2 {print $4}')
  (( avail_kb > need_kb )) || { echo "not enough node-local disk for $STAGE_TAR: ${avail_kb} KB free, need ${need_kb}" >&2; exit 3; }
  mkdir -p "$L"; t0=$(date +%s)
  tar -xf "$STAGE_TAR" -C "$L"
  STAGED=$L/$(basename "$DATA_DIR"); test -d "$STAGED"
  echo "staged $STAGE_TAR -> $STAGED ($(du -sh "$L" | cut -f1), $(ls "$STAGED" | wc -l) entries) in $(( $(date +%s) - t0 )) s; mounted over $DATA_DIR"
  export EXTRA_MOUNT="$STAGED:$DATA_DIR"
fi
if [ "${RUNTIME:-pyxis}" = enroot ]; then
  E=/tmp/$USER-enroot-$SLURM_JOB_ID
  export ENROOT_RUNTIME_PATH=$E/run ENROOT_CACHE_PATH=$E/cache ENROOT_DATA_PATH=$E/data ENROOT_TEMP_PATH=$E/tmp
  mkdir -p $E/run $E/cache $E/data $E/tmp
  CNAME=stattempo_$SLURM_JOB_ID
  enroot create -f -n $CNAME "$IMAGE" > /dev/null
  # one wrapper per job: start the shared container with this step's GPU / Slurm / home variables, run the step script
  cat > "$RUN_ROOT/enroot_step-$SLURM_JOB_ID.sh" <<'WRAP'
#!/bin/bash
args=()
for v in $(compgen -e | grep -E '^(SLURM_.*|CUDA_VISIBLE_DEVICES|HOME|USER|LOGNAME|WANDB_.*)$'); do args+=(--env "$v"); done
m=(--mount /workspace:/workspace)
case "$HOME" in /workspace/*) ;; *) [ -d "$HOME" ] && m+=(--mount "$HOME:$HOME") ;; esac  # W&B reads ~/.netrc (pyxis mounted home)
[ -n "${EXTRA_MOUNT:-}" ] && m+=(--mount "$EXTRA_MOUNT")  # the node-local dataset copy, over its /workspace path
exec enroot start --rw "${args[@]}" "${m[@]}" "$1" bash "$2"
WRAP
fi

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
  stepfile="$RUN_ROOT/step-$SLURM_JOB_ID-$i.sh"
  cat > "$stepfile" <<STEP
set -euo pipefail
cd $SRC
ulimit -n \$(ulimit -Hn)  # zarr-heavy loaders exhaust the default 1024 fds (norm pass, job 405906)
export PYTHONPATH=$SRC HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 WANDB_MODE=\${WANDB_MODE:-online}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$body
wait
STEP
  common=(--exact --ntasks=$G --ntasks-per-node=$G --gres=gpu:h100:$G --gpu-bind=none --cpus-per-task=$(( CPR * k )) --mem=$(( 180 * k ))G
          --output="$RUN_ROOT/step-%j-$i.log")
  if [ "${RUNTIME:-pyxis}" = enroot ]; then
    srun "${common[@]}" bash "$RUN_ROOT/enroot_step-$SLURM_JOB_ID.sh" "$CNAME" "$stepfile" &
    sleep 20  # stagger: simultaneous enroot starts of the shared container race for its rootfs lock (job 408657: 2 of 8 steps died)
  else
    srun "${common[@]}" --container-image="$IMAGE" --container-mounts=/workspace:/workspace:rw${STAGED:+,$STAGED:$DATA_DIR:ro} --container-workdir="$SRC" bash "$stepfile" &
  fi
done
wait
