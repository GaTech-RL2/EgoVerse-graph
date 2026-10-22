#!/bin/bash
set -euo pipefail
root=${1:?Operations root}
request=${2:?Exact request}
source "$root/launcher_common.sh"
runtime_exports=$(python3 "$root/lambda_runtime.py" "$request" --inside)
eval "$runtime_exports"
export PIXEL_FA4_ROOT="$JOINT_FA4_ROOT" PIXEL_FA3_ROOT="$JOINT_FA3_ROOT"
export PIXEL_TASK="$JOINT_RUNTIME_TASK"
export PYTHONPATH="$root:$PIXEL_SOURCE_ROOT:$PIXEL_FA3_ROOT:$PIXEL_FA4_ROOT/runtime/nvidia_cutlass_dsl/dsl_packages:$PIXEL_FA4_ROOT/torch-runtime:$PIXEL_FA4_ROOT/runtime:$JOINT_EXTRA_PYTHONPATH"
export PATH="$(dirname "$JOINT_ACTUAL_PYTHON"):$PATH"
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
cache="$JOINT_NODE_SCRATCH/jit/${SLURM_JOB_ID}"
export CUTE_DSL_CACHE_DIR="$cache/cute" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
export TMPDIR="$JOINT_JOB_TMP" XDG_CACHE_HOME="$cache/cache"
export WANDB_CACHE_DIR="$cache/cache/wandb" WANDB_DATA_DIR="$cache/data/wandb"
mkdir -p "$CUTE_DSL_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
phase=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["phase"])' "$request")
[[ "$phase" == smoke || "$phase" == full || "$phase" == recovery ]]
export PIXEL_WANDB_NETRC=/run/pixel-wandb.netrc
exec "$JOINT_BOOTSTRAP_PYTHON" "$root/runtime_bootstrap.py" "$root/run_full.py" "$request"
